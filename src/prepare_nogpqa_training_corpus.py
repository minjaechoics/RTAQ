"""Build a strictly GPQA-free recovery-training corpus.

The source datasets already contain solutions/reasoning.  This script does
not generate new labels: it selects complete, reasonably concise records,
rejects lexical repetition, renders them with the Qwen3.6 chat template, and
writes one variable-length sequence per document.  A separate validation
split is stratified by source.
"""

from __future__ import annotations

import os
RTAQ_ROOT = os.environ.get("RTAQ_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(RTAQ_ROOT)
MODEL = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))   # tokenizer / chat template
HF_DATA = ROOT / "data/hf_datasets"                                      # scripts/01_prepare_corpus.sh downloads here

QUOTAS = {
    "openr1_math_verified": 4096,
    "nemotron_math_v3_verified": 2048,
    "nemotron_ptd_math": 2048,
    "nemotron_ptd_code": 2048,
    "nemotron_ptd_stem": 2048,
    "nemotron_science": 1024,
    "nemotron_ptd_chat": 1024,
    "nemotron_chat_reasoning_off": 1024,
    "cnn_dailymail_anchor": 1024,
}


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    words = re.findall(r"\S+", text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def assistant_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        parts.append(message.get("reasoning_content") or "")
        parts.append(message.get("content") or "")
    return "\n".join(parts).strip()


def all_text(messages: list[dict]) -> str:
    return "\n".join(
        str(message.get("content") or "") + "\n" + str(message.get("reasoning_content") or "")
        for message in messages
    )


def normalize_messages(messages) -> list[dict]:
    result = []
    for message in messages or []:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        item = {"role": role, "content": str(message.get("content") or "")}
        reasoning = message.get("reasoning_content")
        if role == "assistant" and reasoning:
            item["reasoning_content"] = str(reasoning)
        result.append(item)
    return result


def render_candidate(tokenizer, source: str, source_id: str, messages, metadata: dict, args):
    messages = normalize_messages(messages)
    if not messages or not any(m["role"] == "user" for m in messages):
        return None, "no_user"
    answer = assistant_text(messages)
    if not answer:
        return None, "no_assistant"
    text = all_text(messages)
    # Strict data-level exclusion.  This also excludes dataset-card-like rows
    # that discuss GPQA rather than silently allowing them into the corpus.
    if re.search(r"\bgpqa\b", text, flags=re.I):
        return None, "gpqa_text"
    rep8 = repeated_ngram_ratio(answer)
    if rep8 >= args.max_rep8:
        return None, "rep8"
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            preserve_thinking=True,
        )
    except Exception:
        return None, "template_error"
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if not (args.min_tokens <= len(input_ids) <= args.max_tokens):
        return None, "length"
    return {
        "source": source,
        "source_id": str(source_id),
        "input_ids": list(map(int, input_ids)),
        "seq_len": len(input_ids),
        "rep8": rep8,
        "metadata": metadata,
    }, None


def parquet_rows(pattern: str):
    for path in sorted(glob.glob(pattern)):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prompt_fingerprint(messages) -> str:
    prompt = "\n".join(
        str(message.get("content") or "") for message in messages or []
        if message.get("role") == "user"
    )
    prompt = re.sub(r"\s+", " ", prompt).strip().lower()
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def take(source, rows, quota, converter, tokenizer, args, rejects, seen_prompts):
    selected = []
    for index, row in enumerate(rows):
        converted = converter(row, index)
        if converted is None:
            rejects[source]["converter"] += 1
            continue
        source_id, messages, metadata = converted
        fingerprint = prompt_fingerprint(messages)
        if fingerprint in seen_prompts:
            rejects[source]["duplicate_prompt"] += 1
            continue
        item, reason = render_candidate(tokenizer, source, source_id, messages, metadata, args)
        if item is None:
            rejects[source][reason] += 1
            continue
        seen_prompts.add(fingerprint)
        item["prompt_sha256"] = fingerprint
        selected.append(item)
        if len(selected) % 512 == 0:
            print(f"[{source}] selected={len(selected)}/{quota} scanned={index + 1}", flush=True)
        if len(selected) >= quota:
            break
    if len(selected) != quota:
        raise RuntimeError(f"{source}: selected {len(selected)} but quota is {quota}")
    return selected


def openr1_converter(row, index):
    messages = row.get("messages")
    if not messages:
        return None
    assistant = [m.get("content") or "" for m in messages if m.get("role") == "assistant"][-1]
    generations = row.get("generations") or []
    try:
        generation_index = generations.index(assistant)
    except ValueError:
        return None
    complete = row.get("is_reasoning_complete") or []
    correct_verify = row.get("correctness_math_verify") or []
    correct_judge = row.get("correctness_llama") or []
    is_complete = generation_index < len(complete) and bool(complete[generation_index])
    is_correct = (
        generation_index < len(correct_verify) and bool(correct_verify[generation_index])
    ) or (
        generation_index < len(correct_judge) and bool(correct_judge[generation_index])
    )
    if not (is_complete and is_correct):
        return None
    return row.get("uuid", index), messages, {
        "answer": row.get("answer"), "problem_type": row.get("problem_type"),
        "source_dataset": row.get("source"), "correctness_count": row.get("correctness_count"),
        "selected_generation_index": generation_index,
    }


def standard_converter(row, index):
    return row.get("uuid", index), row.get("messages"), {
        key: row.get(key) for key in ("license", "generator", "category", "reasoning", "used_in")
        if key in row
    }


def math_v3_rows():
    # Streaming avoids downloading the 154 GB monolithic JSONL merely to use
    # a small, rigorously filtered recovery-training subset.
    from datasets import load_dataset

    yield from load_dataset("nvidia/Nemotron-SFT-Math-v3", split="train", streaming=True)


def math_v3_converter(row, index):
    if row.get("tool_usage") != "without Python TIR":
        return None
    if not row.get("expected_answer"):
        return None
    return row.get("uuid", index), row.get("messages"), {
        "expected_answer": row.get("expected_answer"),
        "data_source": row.get("data_source"),
        "tool_usage": row.get("tool_usage"),
        "license": row.get("license"),
    }


def cnn_converter(row, index):
    article = row.get("article") or ""
    summary = row.get("highlights") or ""
    messages = [
        {"role": "user", "content": "Summarize the following article accurately and concisely.\n\n" + article},
        {"role": "assistant", "content": summary},
    ]
    return row.get("id", index), messages, {"role": "non_reasoning_language_anchor"}


def science_rows():
    mcq = jsonl_rows(HF_DATA / "nemotron_science_v1/data/MCQ.jsonl")
    rqa = jsonl_rows(HF_DATA / "nemotron_science_v1/data/RQA.jsonl")
    # Fixed alternation prevents one subset from dominating the first 1024
    # accepted rows while retaining deterministic, resumable preparation.
    while True:
        produced = False
        for iterator in (mcq, rqa):
            try:
                yield next(iterator)
                produced = True
            except StopIteration:
                pass
        if not produced:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--openr1_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_rep8", type=float, default=0.06)
    parser.add_argument("--validation", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rejects = defaultdict(Counter)
    seen_prompts = set()
    rows = []

    rows += take(
        "openr1_math_verified",
        parquet_rows(str(args.openr1_dir / "data/*.parquet")),
        QUOTAS["openr1_math_verified"], openr1_converter, tokenizer, args, rejects, seen_prompts,
    )
    rows += take(
        "nemotron_math_v3_verified", math_v3_rows(), QUOTAS["nemotron_math_v3_verified"],
        math_v3_converter, tokenizer, args, rejects, seen_prompts,
    )
    for category in ("math", "code", "stem", "chat"):
        source = f"nemotron_ptd_{category}"
        rows += take(
            source,
            parquet_rows(str(HF_DATA / f"nemotron_ptd_v2/data/{category}-*.parquet")),
            QUOTAS[source], standard_converter, tokenizer, args, rejects, seen_prompts,
        )
    rows += take(
        "nemotron_science", science_rows(), QUOTAS["nemotron_science"],
        standard_converter, tokenizer, args, rejects, seen_prompts,
    )
    rows += take(
        "nemotron_chat_reasoning_off",
        jsonl_rows(HF_DATA / "nemotron_sft_chat_v2/data/reasoning_off.jsonl"),
        QUOTAS["nemotron_chat_reasoning_off"], standard_converter, tokenizer, args, rejects, seen_prompts,
    )
    rows += take(
        "cnn_dailymail_anchor",
        parquet_rows(str(HF_DATA / "cnn_dailymail/3.0.0/train-*.parquet")),
        QUOTAS["cnn_dailymail_anchor"], cnn_converter, tokenizer, args, rejects, seen_prompts,
    )

    rng = random.Random(args.seed)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    validation = []
    remainder = args.validation
    sources = sorted(by_source)
    for position, source in enumerate(sources):
        group = by_source[source]
        rng.shuffle(group)
        n = remainder if position == len(sources) - 1 else round(args.validation * len(group) / len(rows))
        n = min(n, len(group) - 1)
        validation.extend(group[:n])
        by_source[source] = group[n:]
        remainder -= n
    train = [row for source in sources for row in by_source[source]]
    rng.shuffle(train)
    rng.shuffle(validation)

    for name, values in (("train", train), ("validation", validation)):
        target = args.output_dir / f"{name}.jsonl"
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(target)

    metadata = {
        "format": "qwen36_recovery_nogpqa_v1",
        "gpqa_records": 0,
        "target_documents": sum(QUOTAS.values()),
        "train_documents": len(train),
        "validation_documents": len(validation),
        "train_tokens": sum(row["seq_len"] for row in train),
        "validation_tokens": sum(row["seq_len"] for row in validation),
        "source_counts_total": dict(Counter(row["source"] for row in rows)),
        "source_counts_train": dict(Counter(row["source"] for row in train)),
        "source_counts_validation": dict(Counter(row["source"] for row in validation)),
        "filter": {"min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "max_rep8": args.max_rep8},
        "rejections": {source: dict(counts) for source, counts in rejects.items()},
        "bf16_cot_generation_required": False,
        "note": "Every selected reasoning source already provides CoT/solution text; regenerating it would add unverified synthetic trajectories.",
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / ".dataset_done").touch()
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

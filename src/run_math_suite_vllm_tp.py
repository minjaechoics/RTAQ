#!/usr/bin/env python3
"""MATH-500 / GSM8K / AIME 2026 / Omni-MATH full sets for a Qwen3.6-35B-A3B checkpoint on one vLLM engine.

Every problem of every requested benchmark is queued at once on a single tensor-parallel engine, so
continuous batching refills a slot the moment any problem finishes and no benchmark pays an engine restart.
Prompts use the repo's hendrycks_math500 / aime26 instructions (final answer in \\boxed{}) through the chat
template, decoded greedily; the tasks' "Problem:" / "Question:" stop strings are dropped because a thinking model
restates the problem mid-reasoning. Answers are scored with math_verify on the text after </think> (a capped response on its last \\boxed{}). Each continuation is cached the
moment it finishes, so an interrupted run resumes, and a benchmark is scored in a separate process as soon as
its last problem finishes, so scoring never stalls generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BENCHMARKS = ("gsm8k", "math500", "aime25", "aime26", "omni_math")
# The repo's AIME task configs open with "Question:" where its MATH-500 config uses "Problem:".
PROMPT_PREFIX = {"aime25": "Question: ", "aime26": "Question: "}
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    os.replace(temporary, path)


def difficulty_group(value) -> str:
    try:
        return f"difficulty{int(float(value))}"
    except (TypeError, ValueError):
        return "difficulty_unknown"


def load_problems(name: str) -> list[dict]:
    from datasets import load_dataset

    if name == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": row["unique_id"], "problem": row["problem"], "gold": row["answer"],
                 "group": f"level{row['level']}"} for row in rows]
    if name == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        return [{"id": f"gsm8k/{index}", "problem": row["question"],
                 "gold": row["answer"].split("####")[-1].strip().replace(",", ""), "group": "all"}
                for index, row in enumerate(rows)]
    if name in ("aime25", "aime26"):
        rows = load_dataset(f"math-ai/{name}", split="test")
        return [{"id": f"{name}/{row['id']}", "problem": row["problem"], "gold": str(row["answer"]), "group": "all"}
                for row in rows]
    if name == "omni_math":
        rows = load_dataset("KbsdJames/Omni-MATH", split="test")
        return [{"id": f"omni_math/{index}", "problem": row["problem"], "gold": str(row["answer"] or ""),
                 "group": difficulty_group(row["difficulty"])} for index, row in enumerate(rows)]
    raise ValueError(f"unknown benchmark {name!r}; expected one of {BENCHMARKS}")


def build_scorer():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    prediction_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=True, basic_latex=True, boxed="all", units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        ),
        ExprExtractionConfig(),
    ]

    boxed_config = prediction_config[:1]

    def score(gold: str, text: str) -> dict:
        gold_parsed = []
        if gold.strip():
            gold_parsed = parse(f"${gold}$") or parse(gold)
        if "</think>" in text:
            # The final answer follows the reasoning block.
            predicted = parse(text.split("</think>")[-1], extraction_config=prediction_config)
        else:
            # A capped response never finished reasoning: take only its last \boxed{} (as GPQA takes the last
            # "(X)"). The bare-number fallback misparses looping markdown ("64.\n\n    *" -> 5) and all-boxed
            # normalization joins repeated boxes ("460,460").
            last_box = text.rfind("\\boxed")
            predicted = parse(text[last_box:], extraction_config=boxed_config) if last_box >= 0 else []
        return {
            "gold_parseable": bool(gold_parsed),
            "prediction": str(predicted[-1]) if predicted else None,
            "correct": bool(gold_parsed and predicted and verify(gold_parsed, predicted)),
        }

    return score


def prepare(names: list[str], model_dir: str, max_gen_toks: int, limit: int | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    problems: dict[str, list[dict]] = {}
    for name in names:
        items = load_problems(name)
        if limit:
            items = items[:limit]
        for item in items:
            prompt = PROMPT_PREFIX.get(name, "Problem: ") + item["problem"] + PROMPT_SUFFIX
            item["context"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
            item["key"] = hashlib.sha256(json.dumps([item["context"], max_gen_toks]).encode()).hexdigest()
        problems[name] = items
        print(f"[suite] {name}: {len(items)} problems", flush=True)
    return tokenizer, problems


def cache_path(output_dir: Path, name: str, item: dict) -> Path:
    return output_dir / name / "generations" / f"{item['key']}.json"


def write_scores(name: str, items: list[dict], output_dir: Path, max_gen_toks: int,
                 wall_clock_s: float | None) -> None:
    score = build_scorer()
    missing = [item["id"] for item in items if not cache_path(output_dir, name, item).exists()]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} generations missing, e.g. {missing[:3]}")
    rows = []
    for item in items:
        record = json.loads(cache_path(output_dir, name, item).read_text())
        rows.append({
            "id": item["id"], "group": item["group"], "gold": item["gold"], **score(item["gold"], record["text"]),
            "num_generated_tokens": record["num_generated_tokens"], "finish_reason": record["finish_reason"],
            "think_closed": "</think>" in record["text"],
        })
    bench_dir = output_dir / name
    with open(bench_dir / "scores.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokens = [row["num_generated_tokens"] for row in rows]
    scoreable = [row for row in rows if row["gold_parseable"]]
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["correct"])
    correct = sum(row["correct"] for row in rows)
    summary = {
        "benchmark": name,
        "num_problems": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "gold_unparseable": len(rows) - len(scoreable),
        "accuracy_scoreable": sum(row["correct"] for row in scoreable) / max(len(scoreable), 1),
        "no_extracted_answer": sum(row["prediction"] is None for row in rows),
        "cap_hits": sum(row["finish_reason"] == "length" for row in rows),
        "think_close_count": sum(row["think_closed"] for row in rows),
        "mean_generated_tokens": statistics.mean(tokens),
        "median_generated_tokens": statistics.median(tokens),
        "max_generated_tokens": max(tokens),
        "total_generated_tokens": sum(tokens),
        "max_gen_toks": max_gen_toks,
        "wall_clock_s": wall_clock_s,
        "scorer": "math_verify on the text after </think>; capped responses on their last \\boxed{}",
        "by_group": {group: {"n": len(values), "accuracy": sum(values) / len(values)}
                     for group, values in sorted(groups.items())},
    }
    atomic_json(bench_dir / "summary.json", summary)
    print(f"[score] {name}: {correct}/{len(rows)} = {summary['accuracy'] * 100:.2f}% "
          f"(scoreable {summary['accuracy_scoreable'] * 100:.2f}%) cap_hits={summary['cap_hits']} "
          f"no_answer={summary['no_extracted_answer']} mean_tokens={summary['mean_generated_tokens']:.0f}", flush=True)


def dry_run(problems: dict[str, list[dict]]) -> None:
    score = build_scorer()
    for name, items in problems.items():
        print(f"=== {name} context[0] ===\n{items[0]['context']}", flush=True)
        right = [score(item["gold"], f"<think>\nwork\n</think>\n\nThe final answer is $\\boxed{{{item['gold']}}}$.")
                 for item in items]
        wrong = score(items[0]["gold"], "<think>\nwork\n</think>\n\nThe final answer is $\\boxed{123456789}$.")
        print(f"[dry] {name}: gold-as-answer scored correct {sum(r['correct'] for r in right)}/{len(items)}, "
              f"gold unparseable {sum(not r['gold_parseable'] for r in right)}, "
              f"wrong answer scored correct={wrong['correct']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--max_gen_toks", type=int, default=40000)
    parser.add_argument("--max_model_len", type=int, default=None, help="default: max_gen_toks + 2048")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_num_seqs", type=int, default=512)
    parser.add_argument("--moe_backend", default="triton",
                        help="triton avoids the FlashInfer CUTLASS JIT build (OOM-killed with 4 TP workers)")
    parser.add_argument("--limit", type=int, default=None, help="first N problems per benchmark (smoke tests)")
    parser.add_argument("--dry_run", action="store_true", help="build prompts and self-check the scorer; no GPU")
    parser.add_argument("--score_only", action="store_true", help="score cached generations and exit")
    parser.add_argument("--wall_clock_s", type=float, default=None)
    args = parser.parse_args()

    names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    output_dir = Path(args.output_dir)
    tokenizer, problems = prepare(names, args.model_dir, args.max_gen_toks, args.limit)

    if args.dry_run:
        dry_run(problems)
        return
    if args.score_only:
        for name in names:
            write_scores(name, problems[name], output_dir, args.max_gen_toks, args.wall_clock_s)
        return

    pending = [(name, item) for name in names for item in problems[name]
               if not cache_path(output_dir, name, item).exists()]
    remaining = defaultdict(int)
    for name, _ in pending:
        remaining[name] += 1
    print(f"[suite] {sum(len(v) for v in problems.values()) - len(pending)} cached, {len(pending)} to generate "
          f"({dict(remaining)})", flush=True)

    scorers: list[subprocess.Popen] = []

    def spawn_scorer(name: str, wall_clock_s: float | None) -> None:
        command = [sys.executable, "-u", os.path.abspath(__file__), "--model_dir", args.model_dir,
                   "--output_dir", args.output_dir, "--benchmarks", name, "--max_gen_toks", str(args.max_gen_toks),
                   "--score_only"]
        if args.limit:
            command += ["--limit", str(args.limit)]
        if wall_clock_s is not None:
            command += ["--wall_clock_s", f"{wall_clock_s:.1f}"]
        scorers.append(subprocess.Popen(command))

    for name in names:
        if remaining[name] == 0:
            spawn_scorer(name, None)

    if pending:
        from vllm import LLM, SamplingParams

        llm_kwargs = dict(
            model=args.model_dir,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len or args.max_gen_toks + 2048,
            max_num_seqs=min(args.max_num_seqs, len(pending)),
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        if args.moe_backend:
            llm_kwargs["moe_backend"] = args.moe_backend
        print(f"[vllm] LLM kwargs: {llm_kwargs}", flush=True)
        engine = LLM(**llm_kwargs).llm_engine

        running = {}
        for index, (name, item) in enumerate(pending):
            request_id = f"{item['key']}-{index}"
            prompt_ids = tokenizer(item["context"], add_special_tokens=False)["input_ids"]
            engine.add_request(request_id, {"prompt_token_ids": prompt_ids},
                               SamplingParams(temperature=0.0, max_tokens=args.max_gen_toks))
            running[request_id] = (name, item, time.time())
        print(f"[vllm] {len(running)} requests queued", flush=True)

        finished = 0
        generated_tokens: dict[str, int] = {}
        loop_started = last_report = time.time()
        last_generated = 0
        while engine.has_unfinished_requests():
            for output in engine.step():
                generated_tokens[output.request_id] = len(output.outputs[0].token_ids)
                if not output.finished:
                    continue
                name, item, submitted = running.pop(output.request_id)
                completion = output.outputs[0]
                record = {"id": item["id"], "text": completion.text,
                          "num_generated_tokens": len(completion.token_ids),
                          "finish_reason": completion.finish_reason, "latency_s": time.time() - submitted}
                atomic_json(cache_path(output_dir, name, item), record)
                remaining[name] -= 1
                finished += 1
                print(f"[vllm] finished {finished}/{len(pending)} {name} left={remaining[name]} "
                      f"tokens={record['num_generated_tokens']} reason={record['finish_reason']} "
                      f"latency={record['latency_s']:.0f}s", flush=True)
                if remaining[name] == 0:
                    spawn_scorer(name, time.time() - loop_started)
            now = time.time()
            if now - last_report >= 30:
                generated = sum(generated_tokens.values())
                print(f"[tps] t={now - loop_started:.0f}s unfinished={len(running)} finished={finished} "
                      f"generated={generated} window={(generated - last_generated) / (now - last_report):.0f} tok/s "
                      f"avg={generated / (now - loop_started):.0f} tok/s "
                      f"left={ {name: remaining[name] for name in names} }", flush=True)
                last_report, last_generated = now, generated

    failed = [process.args for process in scorers if process.wait() != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} scoring process(es) failed")
    print("[suite] done: " + ", ".join(
        f"{name}={json.loads((output_dir / name / 'summary.json').read_text())['accuracy'] * 100:.2f}%"
        for name in names), flush=True)


if __name__ == "__main__":
    main()

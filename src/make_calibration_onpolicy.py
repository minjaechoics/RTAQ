#!/usr/bin/env python3
"""Rewrite the calibration candidate pool on-policy: the target model answers the pool's own prompts.

`prepare_bipea_v3_data.py prepare` builds a pool of 1,024 documents whose assistant turns were written by whatever
models produced the source datasets. Quantizing against that text calibrates the quantizer on states the target
model does not actually visit. This step keeps the prompts and the source stratification and replaces every
assistant turn with the target model's own BF16 continuation, sampled under the evaluation protocol; the routing
scan and coverage selection that follow then run over text the model itself produced.

Measured on NVIDIA Nemotron-3-Nano-30B-A3B, this raises the per-block correlation between the calibration routing
load and the model's routing load on its own MATH-500 reasoning traces from 0.51 to 0.71, and is worth +8.3 AIME25
and +7.8 AIME26 points on the final 2-bit checkpoint -- the largest single-factor effect in the ablations.

No benchmark prompt is involved: the pool is the GPQA-free training pool of step 1.

  usage: make_calibration_onpolicy.py --output data/calibration --model <bf16 model dir> [--tensor_parallel_size N]

Reads  <output>/calibration_candidates.jsonl   (written by prepare_bipea_v3_data.py prepare)
Writes <output>/calibration_candidates.jsonl   (on-policy; the originals move to calibration_candidates_source.jsonl)
       <output>/onpolicy_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from prepare_bipea_v3_data import assistant_loss_mask  # noqa: E402  (same mask contract as the source pool)

TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)
# The evaluation protocol, so the calibration states are the ones the benchmarks sample from. Override per model.
SAMPLING = dict(
    temperature=float(os.environ.get("RTAQ_TEMPERATURE", "0.7")),
    top_p=float(os.environ.get("RTAQ_TOP_P", "0.8")),
    top_k=int(os.environ.get("RTAQ_TOP_K", "20")),
    presence_penalty=float(os.environ.get("RTAQ_PRESENCE_PENALTY", "1.5")),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def to_prompt(row: dict, tokenizer, seq: int) -> dict:
    """Prompt ids for one pool document: every turn up to the last assistant turn, which is the one to re-answer.

    Earlier assistant turns are context and keep their text, minus their thinking blocks -- which is what the chat
    template itself does for history. `thinking` records whether the turn being replaced had a thinking block, so a
    reasoning-off document stays reasoning-off.
    """
    text = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
    turns = TURN.findall(text)
    if not any(role == "assistant" for role, _ in turns):  # prompt alone fills the document; keep it as it is
        return {"prompt_ids": row["input_ids"][:seq], "thinking": False, "passthrough": True}
    last = max(index for index, (role, _) in enumerate(turns) if role == "assistant")
    answer = turns[last][1]
    thought = THINK.search(answer)
    thinking = bool(thought and thought.group(1).strip()) or ("<think>" in answer and "</think>" not in answer)
    messages = [{"role": role, "content": THINK.sub("", body) if role == "assistant" else body}
                for role, body in turns[:last]]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=thinking)
    except TypeError:  # a thinking-only model's template takes no switch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"], "thinking": thinking,
            "passthrough": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="the calibration directory of step 2")
    parser.add_argument("--model", required=True, help="BF16 model directory of the model being quantized")
    parser.add_argument("--seq", type=int, default=2048, help="document length, matching prepare's calibration cap")
    parser.add_argument("--min_new", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="render the prompts and print a few; no GPU")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source_path = args.output / "calibration_candidates_source.jsonl"
    pool_path = args.output / "calibration_candidates.jsonl"
    if source_path.exists():                      # resumable: the source pool is only ever written once
        rows = read_jsonl(source_path)
    else:
        rows = read_jsonl(pool_path)
        write_jsonl(source_path, rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end == tokenizer.unk_token_id:
        im_end = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id[0]
        newline: list[int] = []
    else:
        newline = tokenizer("\n", add_special_tokens=False)["input_ids"]

    jobs = [{**row, **to_prompt(row, tokenizer, args.seq)} for row in rows]
    log(f"{len(jobs)} pool documents; thinking on in {sum(j['thinking'] for j in jobs)}; "
        f"longest prompt {max(len(j['prompt_ids']) for j in jobs)} tokens")
    if args.dry_run:
        for job in jobs[:: max(1, len(jobs) // 4)]:
            text = tokenizer.decode(job["prompt_ids"], skip_special_tokens=False)
            print(f"--- {job['source']} thinking={job['thinking']} tokens={len(job['prompt_ids'])}\n"
                  f"{text[:200]!r} ... {text[-80:]!r}")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=max(len(j["prompt_ids"]) for j in jobs) + args.seq,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=True)
    params = [SamplingParams(n=1, max_tokens=max(args.min_new, args.seq - len(j["prompt_ids"])),
                             seed=int(hashlib.sha256(f"{args.seed}/{j['candidate_index']}".encode()).hexdigest()[:8], 16),
                             **SAMPLING) for j in jobs]
    live = [i for i, j in enumerate(jobs) if not j["passthrough"]]
    outputs = dict(zip(live, llm.generate([{"prompt_token_ids": jobs[i]["prompt_ids"]} for i in live],
                                          [params[i] for i in live], use_tqdm=False)))

    rebuilt, finished = [], 0
    for index, job in enumerate(jobs):
        completion = outputs[index].outputs[0] if index in outputs else None
        new = list(completion.token_ids) if completion else []
        stopped = bool(completion) and completion.finish_reason == "stop"
        if stopped:  # close the turn exactly as the chat template would, so the loss mask terminates
            new = (new if new and new[-1] == im_end else new + [im_end]) + newline
        ids = (job["prompt_ids"] + new)[: args.seq]
        mask = assistant_loss_mask(ids)
        finished += bool(stopped and len(job["prompt_ids"]) + len(new) <= args.seq)
        rebuilt.append({"source": job["source"], "source_id": job.get("source_id"),
                        "source_line_no": job.get("source_line_no"), "candidate_index": job["candidate_index"],
                        "prompt_sha256": hashlib.sha256(bytes(str(job["prompt_ids"]), "utf8")).hexdigest(),
                        "input_ids": ids, "attention_mask": [1] * len(ids), "loss_mask": mask,
                        "sequence_tokens": len(ids), "assistant_loss_tokens": int(sum(mask))})

    write_jsonl(pool_path, rebuilt)
    manifest = {
        "model": str(args.model), "documents": len(rebuilt), "seq": args.seq, "sampling": SAMPLING,
        "seed": args.seed, "finished_inside_seq": finished,
        "assistant_loss_tokens": sum(r["assistant_loss_tokens"] for r in rebuilt),
        "source_pool": str(source_path),
        "note": "assistant turns regenerated by the target model; prompts and source stratification unchanged",
    }
    (args.output / "onpolicy_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"rewrote {len(rebuilt)} documents on-policy ({finished} finished inside {args.seq} tokens, "
        f"{manifest['assistant_loss_tokens']} assistant loss tokens) -> {pool_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score a qwen_glmstyle_proxy_v1 GPQA-30 run with the rules of summarize_glm52_gpqa.py (GLM-5.2).

The correct letter is reconstructed from sample.doc["Correct Answer"] and the choices displayed in
the prompt, never from the stored target.  Direct scoring takes the last "answer is (X)" in the
raw response.  Flexible scoring applies the task's flexible-extract rule (last "(X)" in the
response).  Also reports responses without an explicit final answer, cap hits (>= cap - 10
generated tokens, GLM's operational threshold), </think> emission and generated-token statistics.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

TASK = "gpqa_diamond_cot_zeroshot"
FINAL_RE = re.compile(r"(?:the\s+)?answer\s+is\s*\(?\s*([ABCD])\s*\)?", re.I)
CHOICE_RE = re.compile(r"^\(([ABCD])\)\s*(.*?)(?=^\([ABCD]\)|^Let's think)", re.M | re.S)
FLEXIBLE_RE = re.compile(r"\(([A-D])\)", re.I)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def prompt_target(sample: dict) -> str:
    prompt = sample["arguments"][0][0]
    displayed = CHOICE_RE.findall(prompt)
    # Case-insensitive matching first; choices that differ only by case (GPQA-198 doc 191: q/l^2 vs q/L^2)
    # are ambiguous there, so retry case-sensitively before giving up.
    for normalize in (normalized, lambda text: re.sub(r"\s+", " ", text).strip()):
        correct = normalize(sample["doc"]["Correct Answer"])
        choices = {letter: normalize(value) for letter, value in displayed}
        exact = [letter for letter, value in choices.items() if value == correct]
        if len(exact) == 1:
            return exact[0]
        contained = [letter for letter, value in choices.items() if correct in value or value in correct]
        if len(contained) == 1:
            return contained[0]
    raise ValueError(f"Cannot remap target for doc_id={sample.get('doc_id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--num-docs", type=int, default=30)
    parser.add_argument("--max-gen-toks", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="used only for answers that lack vLLM's generated-token count")
    args = parser.parse_args()

    tokenizer = None
    rows = []
    for index in range(args.num_docs):
        path = args.result_dir / TASK / "answers" / f"q{index:05d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        sample = record["sample"]
        response = sample["resps"][0][0]
        target = prompt_target(sample)
        finals = FINAL_RE.findall(response)
        direct = finals[-1].upper() if finals else None
        flexible_matches = FLEXIBLE_RE.findall(response)
        flexible = flexible_matches[-1].upper() if flexible_matches else None
        tokens = record.get("num_generated_tokens")
        if tokens is None:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            tokens = len(tokenizer.encode(response, add_special_tokens=False))
        rows.append({
            "index": index,
            "target": target,
            "stored_target": sample.get("target"),
            "direct": direct,
            "flexible": flexible,
            "lm_eval_flexible_exact_match": record["metrics"].get("exact_match,flexible-extract"),
            "tokens": int(tokens),
            "finish_reason": record.get("finish_reason"),
            "has_think_close": "</think>" in response,
        })

    count = len(rows)
    tokens = [row["tokens"] for row in rows]
    summary = {
        "result_dir": str(args.result_dir.resolve()),
        "completed": count,
        "planned": args.num_docs,
        "max_gen_toks": args.max_gen_toks,
        "prompt_remapped": True,
        "stored_target_mismatches": sum(
            re.sub(r"[^A-D]", "", str(row["stored_target"]).upper()) != row["target"] for row in rows
        ),
        "direct_explicit_correct": sum(row["direct"] == row["target"] for row in rows),
        "flexible_correct": sum(row["flexible"] == row["target"] for row in rows),
        "no_explicit_final": sum(row["direct"] is None for row in rows),
        "cap_hits": sum(row["tokens"] >= args.max_gen_toks - 10 for row in rows),
        "think_close_count": sum(row["has_think_close"] for row in rows),
        "mean_generated_tokens": statistics.fmean(tokens) if tokens else None,
        "median_generated_tokens": statistics.median(tokens) if tokens else None,
        "max_generated_tokens": max(tokens) if tokens else None,
        "ambiguous_or_conflicting": [
            {key: row[key] for key in ("index", "target", "direct", "flexible", "tokens")}
            for row in rows if row["direct"] is None or row["direct"] != row["flexible"]
        ],
        "rules": "GLM summarize_glm52_gpqa.py: prompt-remapped target, last 'answer is (X)' direct, "
                 "last '(X)' flexible, cap at >= max_gen_toks - 10 tokens",
        "per_question": rows,
    }
    for key in ("direct_explicit_correct", "flexible_correct", "no_explicit_final", "cap_hits", "think_close_count"):
        summary[f"{key}_rate"] = summary[key] / count if count else None
    # Names read by the supervisor and the report.
    summary.update({
        "answered": count,
        "direct_correct": summary["direct_explicit_correct"],
        "avg_generated_tokens": summary["mean_generated_tokens"],
        "cap_rate": summary["cap_hits_rate"],
        "think_close_rate": summary["think_close_count_rate"],
        "strict_correct": sum(float(json.loads((args.result_dir / TASK / "answers" / f"q{row['index']:05d}.json").read_text())
                                    ["metrics"].get("exact_match,strict-match", 0) or 0) > 0 for row in rows),
    })
    (args.result_dir / "gpqa_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[score] completed={count}/{args.num_docs} direct={summary['direct_explicit_correct']} "
          f"flexible={summary['flexible_correct']} no_final={summary['no_explicit_final']} "
          f"cap_hits={summary['cap_hits']} think_close={summary['think_close_count']} "
          f"mean_tokens={summary['mean_generated_tokens']} stored_target_mismatches={summary['stored_target_mismatches']}",
          flush=True)


if __name__ == "__main__":
    main()

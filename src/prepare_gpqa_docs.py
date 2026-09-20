#!/usr/bin/env python3
"""Rebuild the GPQA-Diamond prompt cache that `run_sampled_eval_vllm.py --benchmark gpqa198` reads.

The 198 questions come from the gated dataset Idavidrein/gpqa (accept its terms on the Hub, then `hf auth login`);
nothing from it is stored in this repository. What is shipped is assets/gpqa198_choice_order.json: for every question,
which of the dataset's four answer fields was shown as (A), (B), (C) and (D) in the paper's runs (the order produced by
lm_eval's gpqa_diamond_cot_zeroshot shuffle), so the prompts are reproduced byte for byte and every model is scored on
the same choice order. The prompt is the zero-shot CoT template of that task; the user turn is wrapped in the chat
template of --model-dir (the evaluator re-renders it for the model under test with --gpqa_rerender).

Each q{idx:03d}.json holds {"doc_idx", "sample": {"doc", "doc_id", "target", "arguments": [[prompt, gen_kwargs]]}}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEAD = "What is the correct answer to this question:"
TAIL = ("\nLet's think step by step, then state your final answer clearly on its own line in the exact format: "
        "The answer is (X), where X is A, B, C, or D.")


def preprocess(text):  # lm_eval.tasks.gpqa.zeroshot.utils.preprocess
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--order", type=Path, required=True, help="assets/gpqa198_choice_order.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", required=True, help="tokenizer whose chat template wraps the user turn")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    order = json.loads(args.order.read_text())
    rows = load_dataset(order["dataset"], order["config"], split=order["split"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx_str, letters in sorted(order["choice_order"].items(), key=lambda kv: int(kv[0])):
        idx = int(idx_str)
        if args.limit and written >= args.limit:
            break
        row = rows[idx]
        choices = [preprocess(row[order["fields"][letter]]) for letter in letters]
        user = HEAD + row["Question"] + "\nChoices:\n" + "\n".join(f"({L}) {c}" for L, c in zip("ABCD", choices)) + TAIL
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        answer = "(" + "ABCD"[letters.index("C")] + ")"
        doc = {k: row[k] for k in ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")}
        record = {"doc_idx": idx, "sample": {"doc": doc, "doc_id": idx, "target": answer, "arguments": [[prompt, dict(order["gen_kwargs"])]]}}
        (args.out / f"q{idx:03d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
        written += 1
    print(f"wrote {written} GPQA-Diamond prompt records to {args.out}", flush=True)


if __name__ == "__main__":
    main()

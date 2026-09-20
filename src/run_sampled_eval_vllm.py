#!/usr/bin/env python3
"""Sampled avg@k evaluation (GPQA-198, AIME 2026, GSM8K or LiveCodeBench code generation) with loop-abort, on one TP vLLM engine.

Every (problem, sample) pair is its own request with its own seed, so continuous batching refills a slot the moment
any sample finishes and a single looping sample can be aborted without touching its siblings. Prompts are exactly
the ones the greedy runs used: GPQA contexts come from the recorded lm_eval doc cache, AIME/GSM8K contexts from
run_math_suite_vllm_tp's chat-templated prompt. Each sample is cached as it finishes, so an interrupted run resumes.

Loop abort: every --loop_check_every generated tokens, the last --loop_window tokens are tested for exact
periodicity. A sample is aborted (finish_reason "loop") only when some period p <= --loop_max_period repeats at
least --loop_min_repeats times and the trailing max(p * min_repeats, --loop_min_span) tokens agree with themselves
shifted by p in >= --loop_match of positions. That is a near-verbatim cycle thousands of tokens long; loops of that
kind were never escaped in the greedy runs, so the abort saves compute without changing a score. Paraphrasing loops
are deliberately not caught.

Scoring is unchanged from the greedy runs: GPQA takes the last "(X)" (flexible) and the last "answer is X" (direct)
anywhere in the text; AIME uses math_verify on the text after </think>, or on the last \\boxed{} of an unfinished
sample. A loop-aborted or capped sample is scored the same way. avg@k is the mean over all k*N samples.

LiveCodeBench (lcb_codegen) uses lcb_codegen.py: the frozen problem file (--lcb_problems), the official LCB prompt and
code extraction, and the official test runner executed in parallel worker processes; per-sample verdicts are cached in
<output_dir>/grades. Run generation with --no_score and grading with --score_only so no test code executes next to a
live engine. avg@k is LCB pass@1 over k samples; maj@k is not defined for code and is reported as null.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TWLA = Path(__file__).resolve().parent


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    os.replace(temporary, path)


def find_loop(tail: np.ndarray, max_period: int, min_repeats: int, min_span: int, match: float):
    """Return (period, agreement) if the end of `tail` is a near-exact cycle, else None."""
    n = int(tail.shape[0])
    if n < min_span:
        return None
    hits = np.nonzero(tail[:-1] == tail[-1])[0]
    periods = np.unique((n - 1) - hits)
    periods = periods[(periods >= 1) & (periods <= max_period)]
    for period in periods.tolist():
        span = max(period * min_repeats, min_span)
        if span > n:
            continue
        window = tail[-span:]
        agreement = float(np.mean(window[period:] == window[:-period]))
        if agreement >= match:
            return period, agreement
    return None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_items(args, tokenizer) -> list[dict]:
    items = []
    if args.benchmark == "gpqa198":
        from lm_eval.models.utils import handle_stop_sequences

        grader = load_module("gpqa_grader", TWLA / "score_gpqa_qwen_glmstyle_proxy_v1.py")
        eos = tokenizer.decode(tokenizer.eos_token_id, skip_special_tokens=False)
        for path in sorted(Path(args.gpqa_doc_cache).glob("q*.json"))[: args.limit or None]:
            record = json.loads(path.read_text())
            sample = record["sample"]
            context, gen_kwargs = sample["arguments"][0]
            if args.gpqa_rerender:  # the cache holds Qwen-templated prompts; re-render the user turn for this model
                head, tail = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant"
                assert context.startswith(head) and tail in context, "unexpected cached GPQA prompt format"
                user = context[len(head):context.rindex(tail)]
                context = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                        add_generation_prompt=True)
            items.append({"id": f"gpqa/{int(record['doc_idx'])}", "context": context,
                          "stop": handle_stop_sequences(gen_kwargs.get("until"), eos=eos),
                          "gold": grader.prompt_target(sample)})
    elif args.benchmark in ("aime25", "aime26", "gsm8k"):
        suite = load_module("math_suite", TWLA / "run_math_suite_vllm_tp.py")
        for problem in suite.load_problems(args.benchmark)[: args.limit or None]:
            prompt = suite.PROMPT_PREFIX.get(args.benchmark, "Problem: ") + problem["problem"] + suite.PROMPT_SUFFIX
            context = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                    add_generation_prompt=True)
            items.append({"id": problem["id"], "context": context, "stop": None, "gold": problem["gold"]})
    elif args.benchmark == "lcb_codegen":
        import lcb_codegen

        for problem in lcb_codegen.load_problems(Path(args.lcb_problems))[: args.limit or None]:
            context = tokenizer.apply_chat_template(lcb_codegen.prompt_messages(problem), tokenize=False,
                                                    add_generation_prompt=True)
            items.append({"id": f"lcb/{problem['question_id']}", "context": context, "stop": None,
                          "gold": problem["question_id"], "difficulty": problem["difficulty"],
                          "platform": problem["platform"]})
    else:
        raise ValueError(args.benchmark)
    return items


def sampling_signature(args) -> dict:
    return {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "presence_penalty": args.presence_penalty, "max_tokens": args.max_gen_toks, "seed": args.seed}


def sample_key(item: dict, sample: int, signature: dict) -> str:
    return hashlib.sha256(json.dumps([item["context"], item["stop"], signature, sample]).encode()).hexdigest()


def sample_seed(base: int, key: str) -> int:
    return (base * 1_000_003 + int(key[:12], 16)) % (2**31 - 1)


def build_grader(benchmark: str):
    if benchmark == "gpqa198":
        grader = load_module("gpqa_grader", TWLA / "score_gpqa_qwen_glmstyle_proxy_v1.py")

        def grade(gold: str, text: str) -> dict:
            flexible = grader.FLEXIBLE_RE.findall(text)
            direct = grader.FINAL_RE.findall(text)
            flexible = flexible[-1].upper() if flexible else None
            direct = direct[-1].upper() if direct else None
            return {"correct": flexible == gold, "flexible": flexible, "direct": direct,
                    "direct_correct": direct == gold}
        return grade
    suite = load_module("math_suite", TWLA / "run_math_suite_vllm_tp.py")
    score = suite.build_scorer()

    def grade(gold: str, text: str) -> dict:
        result = score(gold, text)
        return {"correct": result["correct"], "prediction": result["prediction"]}
    return grade


def write_summary(args, items: list[dict], out: Path, signature: dict) -> dict:
    lcb = args.benchmark == "lcb_codegen"
    records = []
    for item in items:
        for sample in range(args.num_samples):
            key = sample_key(item, sample, signature)
            path = out / "generations" / f"{key}.json"
            if not path.exists():
                raise RuntimeError(f"missing generation {item['id']} sample {sample}")
            records.append((item, sample, key, json.loads(path.read_text())))
    incomplete = [key for _, _, key, record in records if record["finish_reason"] == "abort"]
    if incomplete:  # samples cut off by an engine shutdown are not results (caches written before the 2026-09-20 fix)
        raise RuntimeError(f"{len(incomplete)} cached samples have finish_reason=abort; delete them and resume")
    if lcb:
        import lcb_codegen

        grades = lcb_codegen.grade_samples(Path(args.lcb_problems),
                                           [{"key": key, "question_id": item["gold"], "text": record["text"]}
                                            for item, _, key, record in records],
                                           out / "grades", workers=args.lcb_workers, timeout=args.lcb_timeout,
                                           memory_gb=args.lcb_memory_gb)

        def verdict(item: dict, key: str, record: dict) -> dict:
            fields = ("correct", "code_found", "tests_passed", "tests_total", "error_code", "error_message")
            return {"difficulty": item["difficulty"], "platform": item["platform"],
                    **{field: grades[key].get(field) for field in fields}}
    else:
        grade = build_grader(args.benchmark)

        def verdict(item: dict, key: str, record: dict) -> dict:
            return grade(item["gold"], record["text"])
    rows = []
    for item, sample, key, record in records:
        rows.append({"id": item["id"], "sample": sample, "gold": item["gold"], **verdict(item, key, record),
                     "num_generated_tokens": record["num_generated_tokens"],
                     "finish_reason": record["finish_reason"], "think_closed": "</think>" in record["text"],
                     "loop_period": record.get("loop_period")})
    with open(out / "scores.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_items, k = len(items), args.num_samples
    per_run = [sum(r["correct"] for r in rows if r["sample"] == s) / n_items for s in range(k)]
    by_item = defaultdict(list)
    for row in rows:
        by_item[row["id"]].append(row)
    majority = None if lcb else 0
    for group in ([] if lcb else by_item.values()):
        answers = [r.get("flexible", r.get("prediction")) for r in group if r.get("flexible", r.get("prediction"))]
        if answers:
            top = Counter(answers).most_common(1)[0][0]
            majority += any(r["correct"] for r in group if r.get("flexible", r.get("prediction")) == top)
    reasons = Counter(r["finish_reason"] for r in rows)
    terminated = [r for r in rows if r["finish_reason"] == "stop"]
    summary = {
        "benchmark": args.benchmark, "model_dir": args.model_dir, "problems": n_items, "samples_per_problem": k,
        "sampling": signature,
        "loop_abort": {"check_every": args.loop_check_every, "window": args.loop_window,
                       "max_period": args.loop_max_period, "min_repeats": args.loop_min_repeats,
                       "min_span": args.loop_min_span, "match": args.loop_match},
        f"avg@{k}": statistics.mean(per_run),
        "per_sample_run_accuracy": per_run,
        "per_sample_run_std": statistics.pstdev(per_run) if k > 1 else 0.0,
        f"maj@{k}": majority / n_items if majority is not None else None,
        f"pass@{k}": sum(any(r["correct"] for r in group) for group in by_item.values()) / n_items,
        "finish_reasons": dict(reasons),
        "termination_rate": len(terminated) / len(rows),
        "think_close_rate": sum(r["think_closed"] for r in rows) / len(rows),
        "accuracy_when_terminated": (sum(r["correct"] for r in terminated) / len(terminated)) if terminated else None,
        "mean_generated_tokens": statistics.mean(r["num_generated_tokens"] for r in rows),
        "mean_tokens_when_terminated": (statistics.mean(r["num_generated_tokens"] for r in terminated)
                                        if terminated else None),
    }
    if args.benchmark == "gpqa198":
        summary[f"direct_avg@{k}"] = sum(r["direct_correct"] for r in rows) / len(rows)
    if lcb:
        manifest = Path(args.lcb_problems).with_suffix(".manifest.json")
        summary["lcb"] = {"problems_file": str(args.lcb_problems), "timeout_s": args.lcb_timeout,
                          "memory_headroom_gb": args.lcb_memory_gb,
                          "manifest": ({key: value for key, value in json.loads(manifest.read_text()).items()
                                        if key != "question_ids"} if manifest.exists() else None)}
        summary[f"strict_avg@{k}"] = sum(r["correct"] and r["think_closed"] for r in rows) / len(rows)
        summary["code_found_rate"] = sum(bool(r["code_found"]) for r in rows) / len(rows)
        summary["by_difficulty"] = {
            level: {"problems": len({r["id"] for r in group}), f"avg@{k}": sum(r["correct"] for r in group) / len(group),
                    "termination_rate": sum(r["finish_reason"] == "stop" for r in group) / len(group)}
            for level in ("easy", "medium", "hard")
            if (group := [r for r in rows if r["difficulty"] == level])}
    atomic_json(out / "summary.json", summary)
    majority_text = f"{summary[f'maj@{k}'] * 100:.1f}%" if majority is not None else "n/a"
    print(f"[score] {args.benchmark} avg@{k}={summary[f'avg@{k}'] * 100:.2f}% runs="
          f"{[round(v * 100, 1) for v in per_run]} maj@{k}={majority_text} "
          f"termination={summary['termination_rate'] * 100:.1f}% reasons={dict(reasons)} "
          f"mean_tokens={summary['mean_generated_tokens']:.0f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark", choices=("gpqa198", "aime25", "aime26", "gsm8k", "lcb_codegen"), required=True)
    parser.add_argument("--gpqa_doc_cache", default=None)
    parser.add_argument("--gpqa_rerender", action="store_true",
                        help="rebuild each cached GPQA prompt with this model's chat template (the cache was rendered "
                             "for Qwen); leave off for Qwen so existing results stay byte-identical")
    parser.add_argument("--lcb_problems", default=str(Path(os.environ.get("RTAQ_ROOT") or TWLA.parent) / "data" / "livecodebench" / "lcb_codegen_v6_2502_2505.jsonl"))
    parser.add_argument("--lcb_workers", type=int, default=16)
    parser.add_argument("--lcb_timeout", type=int, default=6, help="official per-test timeout (s)")
    parser.add_argument("--lcb_memory_gb", type=float, default=8.0)
    parser.add_argument("--max_gen_toks", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, default=None, help="default: max_gen_toks + 2048")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop_check_every", type=int, default=1024)
    parser.add_argument("--loop_window", type=int, default=16384)
    parser.add_argument("--loop_max_period", type=int, default=4096)
    parser.add_argument("--loop_min_repeats", type=int, default=4)
    parser.add_argument("--loop_min_span", type=int, default=2048)
    parser.add_argument("--loop_match", type=float, default=0.98)
    parser.add_argument("--no_loop_abort", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--moe_backend", default="triton")
    parser.add_argument("--disable_flashinfer_autotune", action="store_true",
                        help="skip FlashInfer warmup autotuning after a tuned cache has been created")
    parser.add_argument("--limit", type=int, default=None, help="first N problems (smoke tests)")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--no_score", action="store_true", help="generate only; score later with --score_only")
    args = parser.parse_args()
    if args.benchmark == "gpqa198" and not args.gpqa_doc_cache:
        parser.error("--gpqa_doc_cache is required for gpqa198")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    items = prepare_items(args, tokenizer)
    signature = sampling_signature(args)
    out = Path(args.output_dir)
    atomic_json(out / "run_config.json", {"args": vars(args), "sampling": signature, "problems": len(items)})
    print(f"[eval] {args.benchmark}: {len(items)} problems x {args.num_samples} samples, sampling={signature}", flush=True)
    if args.score_only:
        write_summary(args, items, out, signature)
        return

    pending = [(item, sample) for item in items for sample in range(args.num_samples)
               if not (out / "generations" / f"{sample_key(item, sample, signature)}.json").exists()]
    print(f"[eval] {len(items) * args.num_samples - len(pending)} cached, {len(pending)} to generate", flush=True)
    if pending:
        from vllm import LLM, SamplingParams

        llm_kwargs = dict(model=args.model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len or args.max_gen_toks + 2048,
                          max_num_seqs=min(args.max_num_seqs, len(pending)),
                          tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True,
                          limit_mm_per_prompt={"image": 0, "video": 0})
        if args.moe_backend:
            llm_kwargs["moe_backend"] = args.moe_backend
        if args.disable_flashinfer_autotune:
            llm_kwargs["enable_flashinfer_autotune"] = False
        print(f"[vllm] LLM kwargs: {llm_kwargs}", flush=True)
        engine = LLM(**llm_kwargs).llm_engine

        running = {}
        for item, sample in pending:
            key = sample_key(item, sample, signature)
            params = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                    presence_penalty=args.presence_penalty, max_tokens=args.max_gen_toks,
                                    stop=item["stop"], seed=sample_seed(args.seed, key))
            prompt_ids = tokenizer(item["context"], add_special_tokens=False)["input_ids"]
            engine.add_request(key, {"prompt_token_ids": prompt_ids}, params)
            running[key] = (item, sample, time.time())
        print(f"[vllm] {len(running)} requests queued", flush=True)

        def save(key: str, text: str, token_count: int, reason: str, loop=None) -> None:
            item, sample, submitted = running.pop(key)
            record = {"id": item["id"], "sample": sample, "text": text, "num_generated_tokens": token_count,
                      "finish_reason": reason, "latency_s": time.time() - submitted}
            if loop:
                record["loop_period"], record["loop_agreement"] = loop
            atomic_json(out / "generations" / f"{key}.json", record)
            finished[reason] += 1
            print(f"[vllm] finished {sum(finished.values())}/{len(pending)} {item['id']}#{sample} "
                  f"tokens={token_count} reason={reason}" + (f" period={loop[0]}" if loop else ""), flush=True)

        finished: Counter = Counter()
        engine_aborted: list[str] = []
        checked: dict[str, int] = {}
        generated: dict[str, int] = {}
        started = last_report = time.time()
        last_generated = 0
        while engine.has_unfinished_requests():
            aborted = []
            for output in engine.step():
                key = output.request_id
                if key not in running:
                    continue
                completion = output.outputs[0]
                count = len(completion.token_ids)
                generated[key] = count
                if output.finished:
                    if completion.finish_reason == "abort":
                        # the engine was shut down under us (SIGTERM, a killed worker): the sample is incomplete, so
                        # it must not be cached as a result; leave it for the resumed run
                        engine_aborted.append(key)
                        running.pop(key)
                        continue
                    save(key, completion.text, count, completion.finish_reason or "stop")
                    continue
                if args.no_loop_abort or count - checked.get(key, 0) < args.loop_check_every:
                    continue
                checked[key] = count
                tail = np.asarray(completion.token_ids[-args.loop_window:], dtype=np.int64)
                loop = find_loop(tail, args.loop_max_period, args.loop_min_repeats, args.loop_min_span,
                                 args.loop_match)
                if loop:
                    save(key, completion.text, count, "loop", loop)
                    aborted.append(key)
            if aborted:
                engine.abort_request(aborted)
            now = time.time()
            if now - last_report >= 30:
                total = sum(generated.values())
                print(f"[tps] t={now - started:.0f}s unfinished={len(running)} finished={dict(finished)} "
                      f"generated={total} window={(total - last_generated) / (now - last_report):.0f} tok/s", flush=True)
                last_report, last_generated = now, total
        if engine_aborted or running:
            print(f"[eval] INCOMPLETE: {len(engine_aborted)} samples aborted by an engine shutdown, {len(running)} never "
                  f"finished; nothing was cached for them and no summary is written. Re-run to resume.", flush=True)
            sys.exit(3)
    if args.no_score:
        print("[eval] generation complete; scoring deferred (--no_score)", flush=True)
        return
    write_summary(args, items, out, signature)


if __name__ == "__main__":
    main()

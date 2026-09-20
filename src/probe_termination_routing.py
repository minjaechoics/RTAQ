#!/usr/bin/env python3
"""Which layers/experts fire when the model decides to stop reasoning?

Teacher-forces a saved response (prompt + generated text) through the model in chunks and records, for every
position, which of the 256 routed experts each of the 40 layers' top-8 gate selected -- the same hook as
debug_expert_routing_qwen36.py (Qwen3_5MoeTopKRouter on model.model.language_model.layers[i].mlp.gate), but over a
recorded trace instead of a fresh generation, so a 30-problem sweep is one forward pass per problem.

Positions are then split into classes and compared:
  close_think  : the position that predicts </think> (the decision to stop reasoning)
  pre_eos      : the last position of the response (the one that predicts the EOS the run actually emitted)
  loop_onset   : where an exact token loop starts, for responses that hit the cap (from --loop_rows)
  reasoning    : sampled positions inside the thinking block, as the baseline
Per (layer, expert) it reports selection frequency in each class and the lift over the reasoning baseline, so the
experts that carry termination can be ranked and given more bits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

THINK_CLOSE_ID = 248069
EOS_IDS = (248046, 248044)


class RoutingTracer:
    """Collects router_indices for EVERY position of each forward call, one entry per layer call."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.calls: list[np.ndarray] = []

    def hook(self, module, args, output):
        _, _, router_indices = output
        self.calls.append(router_indices.detach().to(torch.int16).cpu().numpy())

    def reset(self):
        self.calls = []

    def finalize(self, n_tokens: int) -> np.ndarray:
        """-> (n_tokens, n_layers, top_k) int16."""
        assert len(self.calls) == self.n_layers, f"{len(self.calls)} router calls != {self.n_layers} layers"
        stacked = np.stack(self.calls, axis=1)          # (tokens, layers, top_k)
        assert stacked.shape[0] == n_tokens, f"{stacked.shape[0]} routed positions != {n_tokens} tokens"
        return stacked


def load_records(source_dir: Path, name: str) -> dict:
    out = {}
    for path in (source_dir / name / "generations").glob("*.json"):
        record = json.loads(path.read_text())
        out[record["id"]] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", default=None, help="default: model_dir")
    parser.add_argument("--source_dir", type=Path, required=True, help="math-suite output dir holding generations")
    parser.add_argument("--benchmark", default="aime26")
    parser.add_argument("--source_format", choices=("math_suite", "gpqa", "sampled"), default="math_suite",
                        help="gpqa reads <source_dir>/gpqa_diamond_cot_zeroshot/answers/q*.json instead; sampled reads "
                             "<source_dir>/generations/*.json written by run_sampled_eval_vllm.py (one record per "
                             "problem and sample; prompts rebuilt with its prepare_items for --benchmark)")
    parser.add_argument("--gpqa_doc_cache", default=None, help="sampled + gpqa198: the cached GPQA docs of the eval run")
    parser.add_argument("--gpqa_rerender", action="store_true",
                        help="sampled + gpqa198: re-render the cached prompts with this tokenizer's chat template")
    parser.add_argument("--skip_capped", action="store_true",
                        help="skip responses that hit the generation cap: they hold no termination position")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--reasoning_samples", type=int, default=200, help="baseline positions per response")
    parser.add_argument("--loop_rows", type=Path, default=None, help="json with per-problem loop_start (optional)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=("qwen36", "qwen3_moe", "nemotron_h", "qwen3_next"), default="qwen36",
                        help="qwen36: Qwen3.6 MoE (router at language_model.layers[i].mlp.gate, chunked forward with "
                             "KV cache); nemotron_h: NVIDIA Nemotron 3 Nano hybrid Mamba/attention/MoE (router at "
                             "model.layers[i].mixer.gate of the MoE blocks only; whole sequence in one forward, "
                             "since the Mamba cache does not support chunked teacher forcing)")
    args = parser.parse_args()

    import importlib.util
    TWLA = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("math_suite", TWLA / "run_math_suite_vllm_tp.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    import sys
    sys.path.insert(0, str(TWLA))
    from streaming_load import load_pretrained_streaming

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.model_dir)
    if args.source_format == "gpqa":
        items, records = {}, {}
        for path in sorted((args.source_dir / "gpqa_diamond_cot_zeroshot" / "answers").glob("q*.json")):
            record = json.loads(path.read_text())
            sample = record["sample"]
            problem_id = f"gpqa/{int(record['doc_idx'])}"
            items[problem_id] = {"id": problem_id, "context": sample["arguments"][0][0]}
            records[problem_id] = {"id": problem_id, "text": sample["resps"][0][0],
                                   "num_generated_tokens": record.get("num_generated_tokens") or 0,
                                   "finish_reason": record.get("finish_reason") or "stop"}
    elif args.source_format == "sampled":
        import types
        spec2 = importlib.util.spec_from_file_location("sampled_eval", TWLA / "run_sampled_eval_vllm.py")
        sampled = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(sampled)
        ns = types.SimpleNamespace(benchmark=args.benchmark, limit=0, gpqa_doc_cache=args.gpqa_doc_cache,
                                   gpqa_rerender=args.gpqa_rerender,
                                   lcb_problems=str(TWLA / "data" / "livecodebench" / "lcb_codegen_v6_2502_2505.jsonl"))
        contexts = {item["id"]: item["context"] for item in sampled.prepare_items(ns, tokenizer)}
        items, records = {}, {}
        for path in sorted((args.source_dir / "generations").glob("*.json")):
            record = json.loads(path.read_text())
            key = f"{record['id']}#{record['sample']}"
            items[key] = {"id": key, "context": contexts[record["id"]]}
            records[key] = record
    else:
        _, problems = suite.prepare([args.benchmark], args.tokenizer_dir or args.model_dir, 70000, None)
        items = {item["id"]: item for item in problems[args.benchmark]}
        records = load_records(args.source_dir, args.benchmark)
    if args.skip_capped:
        before = len(records)
        records = {k: v for k, v in records.items() if v["finish_reason"] != "length"}
        print(f"[probe] skipping capped responses: {before} -> {len(records)} problems", flush=True)
    loop_start = {}
    if args.loop_rows and args.loop_rows.exists():
        for row in json.loads(args.loop_rows.read_text()):
            if row.get("loop_start") is not None:
                loop_start[row.get("id") or f"{args.benchmark}/{row['index']}"] = row["loop_start"]

    print(f"loading {args.model_dir} on {args.device} (bf16)", flush=True)
    if args.arch == "qwen36":
        model = load_pretrained_streaming(args.model_dir, args.device, dtype=torch.bfloat16)
        model.eval()
        text_model = model.model.language_model
        n_layers = text_model.config.num_hidden_layers
        routers = [text_model.layers[i].mlp.gate for i in range(n_layers)]
        for router in routers:
            assert isinstance(router, Qwen3_5MoeTopKRouter)
        n_experts, top_k = 256, 8
        think_close_id = THINK_CLOSE_ID
    elif args.arch == "qwen3_moe":
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="balanced" if args.device in ("auto", "balanced") else {"": args.device},
        )
        model.eval()
        routers = [m for _, m in model.named_modules() if isinstance(m, Qwen3MoeTopKRouter)]
        if not routers:
            raise RuntimeError("No Qwen3MoeTopKRouter modules found")
        n_layers = len(routers)
        n_experts, top_k = int(routers[0].num_experts), int(routers[0].top_k)
        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        print(
            f"[probe] qwen3_moe: {n_layers} MoE layers, {n_experts} experts, "
            f"top-{top_k}, </think> id {think_close_id}",
            flush=True,
        )
    elif args.arch == "qwen3_next":
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter

        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                     device_map=args.device if args.device in ("auto", "balanced") else {"": args.device})
        model.eval()
        routers = [m for _, m in model.named_modules() if isinstance(m, Qwen3NextTopKRouter)]
        n_layers = len(routers)
        n_experts, top_k = int(routers[0].num_experts), int(routers[0].top_k)
        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        args.chunk = 0  # one forward per response; the gated-delta-net state cannot be fed chunk by chunk
        print(f"[probe] qwen3_next: {n_layers} MoE layers, {n_experts} experts, top-{top_k}, </think> id {think_close_id}",
              flush=True)
    else:
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHTopkRouter
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = model.to(args.device).eval()
        routers = [m for _, m in model.named_modules() if isinstance(m, NemotronHTopkRouter)]  # MoE blocks only
        n_layers = len(routers)
        n_experts, top_k = int(routers[0].weight.shape[0]), int(routers[0].top_k)
        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        args.chunk = 0  # one forward per response; the hybrid Mamba cache cannot be fed chunk by chunk
        print(f"[probe] nemotron_h: {n_layers} MoE blocks, {n_experts} experts, top-{top_k}, </think> id {think_close_id}",
              flush=True)
    input_device = next(model.parameters()).device  # a device_map-spread model takes its inputs on the first shard
    tracer = RoutingTracer(n_layers)
    handles = [router.register_forward_hook(tracer.hook) for router in routers]

    rng = random.Random(args.seed)
    counts = {cls: np.zeros((n_layers, n_experts), dtype=np.int64) for cls in ("close_think", "pre_eos", "loop_onset", "reasoning")}
    # Keep the raw per-position selections (layers x top_k) so the maps can be plotted unaggregated.
    raw_selections = {cls: [] for cls in ("close_think", "pre_eos", "loop_onset")}
    raw_ids = {cls: [] for cls in ("close_think", "pre_eos", "loop_onset")}
    totals = Counter()
    per_problem = []
    chosen_ids = sorted(records)[: args.limit or None]

    for problem_id in chosen_ids:
        record = records[problem_id]
        item = items[problem_id]
        prompt_ids = tokenizer(item["context"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        ids = prompt_ids + response_ids
        offset = len(prompt_ids)

        positions: dict[str, list[int]] = defaultdict(list)
        think_positions = [i for i, t in enumerate(response_ids) if t == think_close_id]
        if think_positions:
            positions["close_think"].append(offset + think_positions[0] - 1)
        if record["finish_reason"] != "length":
            positions["pre_eos"].append(len(ids) - 1)
        if problem_id in loop_start:
            positions["loop_onset"].append(offset + int(loop_start[problem_id]) - 1)
        think_end = offset + (think_positions[0] if think_positions else len(response_ids))
        pool = range(offset + 32, max(offset + 33, think_end - 1))
        positions["reasoning"] = rng.sample(list(pool), min(args.reasoning_samples, max(len(pool), 1))) if len(pool) > 1 else []

        wanted = {p for ps in positions.values() for p in ps if 0 <= p < len(ids)}
        if not wanted:
            print(f"  {problem_id}: no usable positions, skipped", flush=True)
            continue

        selections: dict[int, np.ndarray] = {}
        past = None
        with torch.no_grad():
            if args.chunk <= 0:  # whole sequence in one forward, no cache; keep one logit row (vocab x tokens is GBs)
                tracer.reset()
                model(input_ids=torch.tensor([ids], device=input_device), use_cache=False, logits_to_keep=1)
                routed = tracer.finalize(len(ids))
                for pos in wanted:
                    selections[pos] = routed[pos]
            else:
                for start in range(0, len(ids), args.chunk):
                    block = ids[start:start + args.chunk]
                    tracer.reset()
                    out = model(input_ids=torch.tensor([block], device=input_device), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    routed = tracer.finalize(len(block))
                    for pos in wanted:
                        if start <= pos < start + len(block):
                            selections[pos] = routed[pos - start]
        del past
        torch.cuda.empty_cache()

        for cls, cls_positions in positions.items():
            for pos in cls_positions:
                sel = selections.get(pos)
                if sel is None:
                    continue
                for layer in range(n_layers):
                    counts[cls][layer, sel[layer]] += 1
                totals[cls] += 1
                if cls in raw_selections:
                    raw_selections[cls].append(sel.astype(np.int16))
                    raw_ids[cls].append(problem_id)
        per_problem.append({"id": problem_id, "finish_reason": record["finish_reason"],
                            "tokens": record["num_generated_tokens"],
                            "positions": {cls: ps for cls, ps in positions.items() if cls != "reasoning"},
                            "reasoning_sampled": len(positions["reasoning"])})
        print(f"  {problem_id}: {len(ids)} tokens, classes " +
              ", ".join(f"{cls}={len(ps)}" for cls, ps in positions.items() if ps), flush=True)

    for handle in handles:
        handle.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = {f"raw_{cls}": (np.stack(v) if v else np.zeros((0, n_layers, top_k), dtype=np.int16))
           for cls, v in raw_selections.items()}
    np.savez_compressed(args.out.with_suffix(".npz"), **{f"counts_{cls}": counts[cls] for cls in counts}, **raw,
                        **{f"rawids_{cls}": np.array(v, dtype=object) for cls, v in raw_ids.items()},
                        totals=np.array([totals[cls] for cls in ("close_think", "pre_eos", "loop_onset", "reasoning")]))
    report = {"model_dir": args.model_dir, "benchmark": args.benchmark, "arch": args.arch, "n_layers": n_layers,
              "n_experts": n_experts, "top_k": top_k, "totals": dict(totals), "per_problem": per_problem}

    base = counts["reasoning"] / max(totals["reasoning"], 1)
    for cls in ("close_think", "pre_eos", "loop_onset"):
        if not totals[cls]:
            continue
        freq = counts[cls] / totals[cls]
        lift = np.where(base > 0, freq / np.maximum(base, 1e-9), np.where(freq > 0, np.inf, 0.0))
        ranked = [(int(l), int(e), float(freq[l, e]), float(base[l, e]), float(lift[l, e]))
                  for l, e in zip(*np.where(freq >= 0.5))]
        ranked.sort(key=lambda r: (-r[4], -r[2]))
        report[f"top_{cls}"] = [{"layer": l, "expert": e, "freq": f, "baseline": b, "lift": (None if np.isinf(li) else li)}
                                for l, e, f, b, li in ranked[:40]]
        layer_shift = [float(np.abs(freq[l] - base[l]).sum() / 2) for l in range(n_layers)]
        report[f"layer_shift_{cls}"] = layer_shift
        order = np.argsort(layer_shift)[::-1][:10]
        print(f"[{cls}] positions={totals[cls]} | layers with the largest routing shift vs reasoning: " +
              ", ".join(f"L{int(i)}={layer_shift[int(i)]:.2f}" for i in order), flush=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out} and {args.out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()

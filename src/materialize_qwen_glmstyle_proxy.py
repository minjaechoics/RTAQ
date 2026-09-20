#!/usr/bin/env python3
"""Materialize a qwen_glmstyle_proxy_v1 expert allocation as a bf16 Qwen3.6-35B-A3B checkpoint.

Nothing is re-quantized.  For each decoder layer, every routed expert's fused gate_up_proj and
its down_proj are decoded from the one stored level-bank variant its (layer, expert) allocation
names, and the layer's two expert tensors are written as one safetensors shard.  Every other
tensor (attention, shared expert, router, embeddings, norms, LM head, vision tower, MTP head)
stays BF16 from the original checkpoint; those shards are kept once in a shared non-expert base
and hard-linked into each checkpoint.  Tensor names, shapes and dtypes match the original
release, so vLLM and transformers load the result exactly like the BF16 model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qwen_glmstyle_proxy_v1_bank_io import (
    NUM_EXPERTS,
    NUM_LAYERS,
    RUN_ID,
    atomic_json,
    decode_range,
    describe_variant,
)

ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
ROUTED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
PROJECTIONS = ("gate_up_proj", "down_proj")


def contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(values))
    runs = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def expert_shard_name(layer: int) -> str:
    return f"model-experts-layer{layer:02d}.safetensors"


def build_nonexpert_base(model_dir: Path, base: Path) -> dict:
    if (base / ".done").exists():
        return json.loads((base / "nonexpert_index.json").read_text())
    if base.exists():
        shutil.rmtree(base)
    partial = base.with_name(base.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        if not ROUTED.match(key):
            by_file[filename].append(key)
    files = sorted(by_file)
    index = {"source": str(model_dir.resolve()), "tensor_bytes": 0, "weight_map": {}}
    for number, filename in enumerate(files, 1):
        name = f"model-nonexpert-{number:05d}-of-{len(files):05d}.safetensors"
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in by_file[filename]}
        index["tensor_bytes"] += sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, str(partial / name), metadata={"format": "pt"})
        index["weight_map"].update({key: name for key in tensors})
        print(f"[base] {name} <- {filename} tensors={len(tensors)}", flush=True)
        del tensors
    atomic_json(partial / "nonexpert_index.json", index)
    (partial / ".done").touch()
    os.replace(partial, base)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B")))
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--bank-sources", type=Path, default=None, help="default: <bank-dir>/bank_sources.json")
    parser.add_argument("--levels-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nonexpert-base", type=Path, required=True)
    parser.add_argument("--case", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    started = time.time()
    if (args.out_dir / ".quant_done").exists():
        print(f"[complete] already materialized: {args.out_dir}", flush=True)
        return

    raw = json.loads(args.levels_json.read_text())
    expected_keys = {f"{layer}:{expert}" for layer in range(NUM_LAYERS) for expert in range(NUM_EXPERTS)}
    if set(raw) != expected_keys:
        missing, extra = expected_keys - set(raw), set(raw) - expected_keys
        raise ValueError(f"allocation must name every (layer, expert) once: missing={sorted(missing)[:3]} "
                         f"extra={sorted(extra)[:3]}")
    assignment = [[int(raw[f"{layer}:{expert}"]) for expert in range(NUM_EXPERTS)] for layer in range(NUM_LAYERS)]
    digest = hashlib.sha256(json.dumps(assignment).encode()).hexdigest()

    sources_path = args.bank_sources or args.bank_dir / "bank_sources.json"
    sources_payload = json.loads(sources_path.read_text())
    sources = {
        int(layer): {int(level): Path(path) for level, path in levels.items()}
        for layer, levels in sources_payload["layers"].items()
    }
    for layer in range(NUM_LAYERS):
        absent = sorted(set(assignment[layer]) - set(sources.get(layer, {})))
        if absent:
            raise ValueError(f"layer {layer} assigns levels {absent} that the bank does not hold")

    original_map = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    routed_keys = [key for key in original_map if ROUTED.match(key)]
    if len(routed_keys) != 2 * NUM_LAYERS:
        raise RuntimeError(f"expected {2 * NUM_LAYERS} routed expert tensors, found {len(routed_keys)}")
    shapes = {}
    for projection in PROJECTIONS:
        key = f"model.language_model.layers.0.mlp.experts.{projection}"
        with safe_open(str(args.model_dir / original_map[key]), framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(key)
            if tensor_slice.get_dtype() != "BF16":
                raise RuntimeError(f"{key} is {tensor_slice.get_dtype()}, expected BF16")
            shapes[projection] = tuple(tensor_slice.get_shape())

    base_index = build_nonexpert_base(args.model_dir, args.nonexpert_base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / ".materialize_progress.json"
    progress = {"assignment_sha256": digest, "layers": {}}
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous.get("assignment_sha256") == digest:
            progress = previous

    for layer in range(NUM_LAYERS):
        shard = args.out_dir / expert_shard_name(layer)
        if str(layer) in progress["layers"] and shard.exists():
            continue
        layer_started = time.time()
        gate_up = torch.empty(shapes["gate_up_proj"], dtype=torch.bfloat16, device=args.device)
        down = torch.empty(shapes["down_proj"], dtype=torch.bfloat16, device=args.device)
        filled = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
        stored_levels: dict[int, dict[str, int]] = {}
        groups: dict[int, list[int]] = defaultdict(list)
        for expert, level in enumerate(assignment[layer]):
            groups[level].append(expert)
        for level, experts in sorted(groups.items()):
            path = sources[layer][level]
            info = describe_variant(path, level)
            for start, end in contiguous_runs(experts):
                for chunk_start in range(start, end, 32):
                    chunk_end = min(chunk_start + 32, end)
                    decoded_gate_up, decoded_down = decode_range(path, level, chunk_start, chunk_end, args.device)
                    gate_up[chunk_start:chunk_end].copy_(decoded_gate_up)
                    down[chunk_start:chunk_end].copy_(decoded_down)
                    filled[chunk_start:chunk_end] = True
                    del decoded_gate_up, decoded_down
            for expert in experts:
                stored_levels[expert] = info["projection_levels"]
        if not bool(filled.all()):
            raise RuntimeError(f"layer {layer}: experts {torch.nonzero(~filled).flatten().tolist()[:5]} were not filled")
        untied = [
            expert for expert in range(NUM_EXPERTS)
            if not stored_levels[expert]["gate_up"] == stored_levels[expert]["down"] == assignment[layer][expert]
        ]
        if untied:
            raise RuntimeError(f"layer {layer}: projection levels differ for experts {untied[:5]}, e.g. "
                               f"{stored_levels[untied[0]]} vs assigned K={assignment[layer][untied[0]]}")
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        tensors = {prefix + "gate_up_proj": gate_up.cpu(), prefix + "down_proj": down.cpu()}
        temporary = shard.with_name(shard.name + ".tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, shard)
        del tensors, gate_up, down
        torch.cuda.empty_cache()
        progress["layers"][str(layer)] = {
            "seconds": time.time() - layer_started,
            "level_counts": {str(k): len(v) for k, v in sorted(groups.items())},
            "projection_levels_tied": True,
        }
        atomic_json(progress_path, progress)
        print(f"[materialize] layer={layer}/{NUM_LAYERS - 1} levels={dict(sorted(Counter(assignment[layer]).items()))} "
              f"seconds={time.time() - layer_started:.1f}", flush=True)

    for name in sorted(set(base_index["weight_map"].values())):
        target = args.out_dir / name
        if not target.exists():
            try:
                os.link(args.nonexpert_base / name, target)
            except OSError:
                shutil.copy2(args.nonexpert_base / name, target)
    weight_map = dict(base_index["weight_map"])
    expert_bytes = 0
    for layer in range(NUM_LAYERS):
        for projection in PROJECTIONS:
            weight_map[f"model.language_model.layers.{layer}.mlp.experts.{projection}"] = expert_shard_name(layer)
            expert_bytes += math.prod(shapes[projection]) * 2
    if set(weight_map) != set(original_map):
        raise RuntimeError("materialized tensor names differ from the original checkpoint")
    atomic_json(args.out_dir / "model.safetensors.index.json", {
        "metadata": {"total_size": int(base_index["tensor_bytes"] + expert_bytes)},
        "weight_map": dict(sorted(weight_map.items())),
    })
    for path in sorted(args.model_dir.iterdir()):
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, args.out_dir / path.name)

    flat = [level for row in assignment for level in row]
    bank_metadata = {}
    proxy_costs = args.bank_dir / "proxy_costs.json"
    if proxy_costs.exists():
        bank_metadata = json.loads(proxy_costs.read_text())["metadata"]
    atomic_json(args.out_dir / "quantization_stats.json", {
        "run_id": RUN_ID,
        "case": args.case,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds_last_invocation": time.time() - started,
        "layer_seconds_total": sum(float(item["seconds"]) for item in progress["layers"].values()),
        "bank_dir": str(args.bank_dir.resolve()),
        "bank_sources": str(sources_path.resolve()),
        "bank_scheme": bank_metadata.get("scheme"),
        "levels_json": str(args.levels_json.resolve()),
        "assignment_sha256": digest,
        "routed_experts_only": True,
        "dense_dtype": "bfloat16",
        "activation_bits": 16,
        "allocation_units": len(flat),
        "average_logical_bits": sum(math.log2(level) for level in flat) / len(flat),
        "average_packed_code_bits": sum(2 if level <= 4 else 4 for level in flat) / len(flat),
        "level_counts": {str(k): v for k, v in sorted(Counter(flat).items())},
        "layer_mean_logical_bits": [
            sum(math.log2(level) for level in row) / len(row) for row in assignment
        ],
        "projection_tie_check": {
            "ok": True,
            "units_checked": len(flat),
            "rule": "fused gate_up_proj (gate+up) and down_proj of each (layer, expert) come from one "
                    "variant file; each projection's stored K (codebook width or level record) must "
                    "equal the assigned K, otherwise materialization aborts",
        },
        "nonexpert_base": str(args.nonexpert_base.resolve()),
    })
    (args.out_dir / ".quant_done").touch()
    print(f"[complete] checkpoint={args.out_dir} seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()

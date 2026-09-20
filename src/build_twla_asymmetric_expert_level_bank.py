#!/usr/bin/env python3
"""Build a reusable hybrid TWLA level bank for Qwen3.6 routed experts.

Level 3 preserves the original symmetric TWLA ternary path.  Levels 4 and
above are independently computed from the untouched BF16 expert weight with
an asymmetric non-uniform DP codebook.  It can be built in disjoint layer
ranges by several GPUs without shared-file writes.
"""

from __future__ import annotations

from streaming_load import load_pretrained_streaming

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

from build_twla_expert_level_bank import (
    write_layer_level as write_symmetric_layer_level,
)
from quantize.E2M_ATQ_asymmetric_codebook import (
    asymmetric_codebook_quantize,
    kronecker_factorize,
)
from quantize_qwen36_experts import load_qwen36_calibset
from quantize_qwen36_experts_bidirectional_rd import (
    collect_layer_moments,
    model_forward_kwargs,
    run_layer,
)


ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
DEFAULT_MODEL = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))
DEFAULT_DATA = None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bank_path(bank_dir: Path, layer: int, level: int) -> Path:
    return bank_dir / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"


def parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not levels or levels[0] < 2 or levels[-1] > 256:
        raise ValueError("levels must be within 2..256")
    return levels


def allocate_buffers(experts, level: int) -> dict[str, torch.Tensor]:
    gate_up = experts.gate_up_proj
    down = experts.down_proj
    gu_n1, gu_n2 = kronecker_factorize(int(gate_up.shape[-1]))
    dn_n1, dn_n2 = kronecker_factorize(int(down.shape[-1]))
    count = int(experts.num_experts)
    return {
        "gate_up_codes": torch.empty(tuple(gate_up.shape), dtype=torch.uint8),
        "gate_up_codebook": torch.empty((count, level), dtype=torch.float16),
        "gate_up_codebook_priors": torch.empty((count, level), dtype=torch.float16),
        "gate_up_mu": torch.empty((count, gate_up.shape[1]), dtype=torch.bfloat16),
        "gate_up_alpha": torch.empty((count, gate_up.shape[1]), dtype=torch.bfloat16),
        "gate_up_rotation_left": torch.empty((count, gu_n1, gu_n1), dtype=torch.float16),
        "gate_up_rotation_right": torch.empty((count, gu_n2, gu_n2), dtype=torch.float16),
        "down_codes": torch.empty(tuple(down.shape), dtype=torch.uint8),
        "down_codebook": torch.empty((count, level), dtype=torch.float16),
        "down_codebook_priors": torch.empty((count, level), dtype=torch.float16),
        "down_mu": torch.empty((count, down.shape[1]), dtype=torch.bfloat16),
        "down_alpha": torch.empty((count, down.shape[1]), dtype=torch.bfloat16),
        "down_rotation_left": torch.empty((count, dn_n1, dn_n1), dtype=torch.float16),
        "down_rotation_right": torch.empty((count, dn_n2, dn_n2), dtype=torch.float16),
    }


def store_result(buffers: dict[str, torch.Tensor], prefix: str, expert: int, result) -> None:
    buffers[f"{prefix}_codes"][expert].copy_(result.codes.cpu())
    buffers[f"{prefix}_codebook"][expert].copy_(result.codebook.to(torch.float16).cpu())
    buffers[f"{prefix}_codebook_priors"][expert].copy_(
        result.codebook_priors.to(torch.float16).cpu()
    )
    buffers[f"{prefix}_mu"][expert].copy_(result.mu.to(torch.bfloat16).cpu())
    buffers[f"{prefix}_alpha"][expert].copy_(result.alpha.to(torch.bfloat16).cpu())
    buffers[f"{prefix}_rotation_left"][expert].copy_(
        result.rotation_left.to(torch.float16).cpu()
    )
    buffers[f"{prefix}_rotation_right"][expert].copy_(
        result.rotation_right.to(torch.float16).cpu()
    )


def quantize_one(weight: torch.Tensor, moment: torch.Tensor | None, level: int, args):
    source = weight.float().to(args.device)
    if moment is None:
        moment = torch.zeros(
            (source.shape[1], source.shape[1]), device=args.device, dtype=torch.float32
        )
    else:
        moment = moment.to(args.device).float()
    return asymmetric_codebook_quantize(
        source,
        moment,
        num_levels=level,
        histogram_bins=args.histogram_bins,
        histogram_clip_quantile=args.histogram_clip_quantile,
        rotation_iters=args.rotation_iters,
        rotation_learning_rate=args.rotation_learning_rate,
        rotation_peak_sigma=args.rotation_peak_sigma,
        rotation_occupancy_beta=args.rotation_occupancy_beta,
        assignment_iters=args.assignment_iters,
        reg_lambda=args.reg_lambda,
        degenerate_tau=args.degenerate_tau,
    )


def write_layer_level(path: Path, experts, moments: dict, level: int, args) -> dict:
    if path.exists():
        return {"resumed": True, "path": str(path), "bytes": path.stat().st_size}
    buffers = allocate_buffers(experts, level)
    zero_hit = 0
    squared_error = 0.0
    weighted_error = 0.0
    started = time.time()
    for expert in range(int(experts.num_experts)):
        entry = moments.get(expert)
        if entry is None:
            zero_hit += 1
        gate_result = quantize_one(
            experts.gate_up_proj.data[expert], None if entry is None else entry[0], level, args
        )
        store_result(buffers, "gate_up", expert, gate_result)
        squared_error += float(gate_result.stats["weight_squared_error"])
        weighted_error += float(gate_result.stats["activation_weighted_error"])
        del gate_result

        down_result = quantize_one(
            experts.down_proj.data[expert], None if entry is None else entry[1], level, args
        )
        store_result(buffers, "down", expert, down_result)
        squared_error += float(down_result.stats["weight_squared_error"])
        weighted_error += float(down_result.stats["activation_weighted_error"])
        del down_result
        if (expert + 1) % 16 == 0:
            print(
                f"[bank] level={level} experts={expert + 1}/{int(experts.num_experts)} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.contiguous() for key, value in buffers.items()}, str(temporary))
    os.replace(temporary, path)
    stats = {
        "resumed": False,
        "path": str(path),
        "bytes": path.stat().st_size,
        "zero_hit_experts": zero_hit,
        "weight_squared_error": squared_error,
        "activation_weighted_error": weighted_error,
        "elapsed_seconds": time.time() - started,
    }
    del buffers
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def write_hybrid_layer_level(
    path: Path,
    experts,
    moments: dict,
    level: int,
    args,
) -> dict:
    """Write original symmetric ternary at K=3 and asymmetric DP above it."""
    if path.exists():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            is_asymmetric = "gate_up_codebook" in handle.keys()
        expected_asymmetric = level >= 4
        if is_asymmetric != expected_asymmetric:
            expected = "asymmetric DP" if expected_asymmetric else "symmetric ternary"
            raise RuntimeError(
                f"incompatible existing bank entry {path}: expected {expected}; "
                "rebuild this level instead of resuming it"
            )

    if level == 3:
        stats = write_symmetric_layer_level(
            path,
            experts,
            moments,
            level,
            args.euclidean_iters,
            args.device,
        )
        stats["quantization_scheme"] = "original_symmetric_twla_ternary"
        return stats

    stats = write_layer_level(path, experts, moments, level, args)
    stats["quantization_scheme"] = "asymmetric_histogram_dp_fixed_codebook"
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--levels", default="3,4,5")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True, help="exclusive")
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--nsamples", type=int, default=115)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=260910)
    parser.add_argument("--histogram-bins", type=int, default=2048)
    parser.add_argument("--histogram-clip-quantile", type=float, default=1e-4)
    parser.add_argument("--rotation-iters", type=int, default=30)
    parser.add_argument("--rotation-learning-rate", type=float, default=1e-2)
    parser.add_argument("--rotation-peak-sigma", type=float, default=0.25)
    parser.add_argument("--rotation-occupancy-beta", type=float, default=0.05)
    parser.add_argument("--assignment-iters", type=int, default=5)
    parser.add_argument("--euclidean-iters", type=int, default=15)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--degenerate-tau", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=20)
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    if not 0 <= args.layer_start < args.layer_end <= 40:
        raise ValueError("layer range must satisfy 0 <= start < end <= 40")
    torch.set_num_threads(args.num_threads)
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "scheme": "hybrid_symmetric_ternary_asymmetric_dp_codebook_bank",
        "model_dir": str(args.model_dir.resolve()),
        "calibset": str(args.calibset.resolve()),
        "levels": list(levels),
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "histogram_bins": args.histogram_bins,
        "rotation_iters": args.rotation_iters,
        "rotation_learning_rate": args.rotation_learning_rate,
        "rotation_peak_sigma": args.rotation_peak_sigma,
        "rotation_occupancy_beta": args.rotation_occupancy_beta,
        "assignment_iters": args.assignment_iters,
        "ternary_euclidean_iters": args.euclidean_iters,
        "routed_experts_only": True,
        "level_3_quantizer": "original symmetric TWLA ternary E2M-ATQ",
        "level_4_plus_quantizer": "asymmetric histogram-DP fixed-codebook KOTMS E2M",
        "codebook_sharing": "one per expert matrix; gate_up and down independent",
        "rotation_sharing": "one Kronecker pair per expert matrix",
        "affine_granularity": "row-wise mu and alpha",
        "source_for_every_level": "untouched BF16",
    }
    manifest_path = args.bank_dir / f"bank_manifest_{args.shard_id}.json"
    manifest = {"status": "building", "run_config": config, "layers": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_config") != config:
            raise ValueError(f"incompatible existing manifest: {manifest_path}")
        manifest = previous
        manifest["status"] = "building"
    atomic_json(manifest_path, manifest)

    print(f"[load] master={args.model_dir} device={args.device}", flush=True)
    model = load_pretrained_streaming(args.model_dir, args.device, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    text_model.config.use_cache = False
    trainloader = load_qwen36_calibset(
        str(args.calibset), args.nsamples, args.seed, args.seqlen
    )
    if len(trainloader) != args.nsamples:
        raise RuntimeError(f"expected {args.nsamples} blocks, got {len(trainloader)}")
    hidden_states = [
        text_model.embed_tokens(input_ids.to(args.device)).cpu()
        for input_ids, _ in trainloader
    ]
    common_kwargs = model_forward_kwargs(
        text_model, hidden_states[0].to(args.device), args.device
    )

    started = time.time()
    for layer_idx, layer in enumerate(text_model.layers):
        if layer_idx >= args.layer_end:
            break
        if layer_idx < args.layer_start:
            hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
            print(f"[prefix] layer={layer_idx}/39 replayed", flush=True)
            continue
        layer_started = time.time()
        print(f"[bank] layer={layer_idx}/39 collecting BF16 routed moments", flush=True)
        moments = collect_layer_moments(layer, hidden_states, common_kwargs, args.device)
        layer_stats = {"experts_hit": len(moments), "levels": {}}
        for level in levels:
            path = bank_path(args.bank_dir, layer_idx, level)
            stats = write_hybrid_layer_level(
                path, layer.mlp.experts, moments, level, args
            )
            layer_stats["levels"][str(level)] = stats
            print(
                f"[bank] layer={layer_idx}/39 level={level} "
                f"resumed={stats['resumed']} seconds={stats.get('elapsed_seconds', 0):.1f}",
                flush=True,
            )
        hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
        del moments
        torch.cuda.empty_cache()
        layer_stats["elapsed_seconds"] = time.time() - layer_started
        manifest["layers"][str(layer_idx)] = layer_stats
        atomic_json(manifest_path, manifest)
        print(
            f"[bank] layer={layer_idx}/39 complete "
            f"elapsed_seconds={time.time() - started:.1f}",
            flush=True,
        )

    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.time() - started
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(manifest_path, manifest)
    marker = args.bank_dir / f".bank_shard_{args.shard_id}_done"
    marker.touch()
    print(f"[complete] marker={marker} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a qwen_glmstyle_proxy_v1 routed-expert level bank and its activation-weighted proxy costs.

Three schemes share this builder:
  asymmetric        K=3 original symmetric TWLA ternary, K=4..9 asymmetric histogram-DP codebook
                    with the codebook-aware Kronecker rotation (build_twla_asymmetric_expert_level_bank.py)
  asymmetric_norot  the same, but without rotation training: histogram -> DP codebook -> Stage I code
                    assignment -> Stage II relocation, identity rotation.  Equals
                    asymmetric_codebook_quantize(rotation_iters=0); the histogram and the DP (run once at
                    the largest level, backtracked for every level) are shared across levels.
  symmetric         K=3..9 symmetric TWLA (twla_parameters of build_twla_expert_level_bank.py)
Variants that bipea_asymmetric_dp_codebook_level_bank_v1 already holds under the same model,
calibration set and quantizer settings are reused in place; the rest are written here with
4-bit packed codes.  Every routed expert then gets one proxy cost per level,

    C[e,K] = sum_{p in gate_up, down} tr((W_p - Q_K(W_p)) S_p (W_p - Q_K(W_p))^T) / T

where S_p is the second moment of the expert's routed BF16 inputs (the same moments the bank
quantizers use, i.e. the existing activation_weighted_error), T is the number of calibration
tokens, and Q_K is the stored variant decoded exactly as the materializer installs it.  gate_up
is the fused gate+up matrix, so its cost already covers both halves.
"""

from __future__ import annotations

import os

from streaming_load import load_pretrained_streaming

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_twla_asymmetric_expert_level_bank import allocate_buffers, quantize_one, store_result
from quantize.E2M_ATQ_asymmetric_codebook import (
    AsymmetricCodebookResult,
    _weighted_interval_sse,
    activation_aware_relocation_with_rotation,
    activation_weighted_histogram,
    euclidean_code_assignment,
    kronecker_factorize,
)
from build_twla_expert_level_bank import allocate_level_buffers, twla_parameters
from qwen_glmstyle_proxy_v1_bank_io import (
    ASYMMETRIC_VARIANT,
    HYBRID_SCHEME,
    NUM_EXPERTS,
    NUM_LAYERS,
    RUN_ID,
    SYMMETRIC_SCHEME,
    atomic_json,
    bank_path,
    decode_range,
    describe_variant,
    save_variant,
)
from quantize_qwen36_experts import load_qwen36_calibset
from quantize_qwen36_experts_bidirectional_rd import (
    collect_layer_moments,
    model_forward_kwargs,
    run_layer,
)

ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
DEFAULT_MODEL = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))
DEFAULT_DATA = None            # --calibset is required
DEFAULT_LEGACY_BANK = None     # no bank to reuse in a fresh build; --reuse-bank-dir opts in
LEGACY_MANIFESTS = ("bank_manifest_layers00_39.json", "bank_manifest_layers00_39.levels3to7.json.bak")
COST_DEFINITION = (
    "C[e,K] = sum over gate_up_proj and down_proj of tr((W - Q_K) S (W - Q_K)^T) / T, "
    "S = sum of outer products of the routed BF16 inputs of that projection over the calibration "
    "tokens (collect_layer_moments), T = calibration tokens, Q_K = stored bank variant decoded to bf16"
)


def cost_path(bank_dir: Path, layer: int) -> Path:
    return bank_dir / "proxy_costs_by_layer" / f"layer_{layer:02d}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=("asymmetric", "asymmetric_norot", "symmetric"), required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--reuse-bank-dir", type=Path, default=DEFAULT_LEGACY_BANK)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibset", type=Path, default=DEFAULT_DATA, required=DEFAULT_DATA is None)
    parser.add_argument("--levels", default="3,4,5,6,7,8,9")
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
    parser.add_argument("--layer-end", type=int, default=NUM_LAYERS,
                        help="exclusive; below 40 only for tests, which skip proxy_costs.json assembly")
    parser.add_argument("--cost-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.levels = tuple(sorted({int(item) for item in args.levels.split(",") if item.strip()}))
    if not args.levels or args.levels[0] < 3 or args.levels[-1] > 15:
        raise ValueError("levels must lie within 3..15")
    if not 1 <= args.layer_end <= NUM_LAYERS:
        raise ValueError("--layer-end must lie within 1..40")
    return args


def legacy_config(args) -> dict:
    """run_config recorded by build_twla_asymmetric_expert_level_bank.py, minus its level list."""
    return {
        "scheme": "hybrid_symmetric_ternary_asymmetric_dp_codebook_bank",
        "model_dir": str(args.model_dir.resolve()),
        "calibset": str(args.calibset.resolve()),
        "layer_start": 0,
        "layer_end": NUM_LAYERS,
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


def legacy_variants(args) -> tuple[dict[tuple[int, int], Path], list[str]]:
    """Variants of the legacy bank built under this run's model, data and quantizer settings."""
    found: dict[tuple[int, int], Path] = {}
    notes: list[str] = []
    if args.reuse_bank_dir is None:
        return found, notes
    if args.histogram_clip_quantile != 1e-4 or args.reg_lambda != 1e-4 or args.degenerate_tau != 1e-8:
        notes.append("quantizer defaults changed; legacy variants not reused")
        return found, notes
    expected = legacy_config(args)
    for name in LEGACY_MANIFESTS:
        path = args.reuse_bank_dir / name
        if not path.exists():
            notes.append(f"{name}: absent")
            continue
        manifest = json.loads(path.read_text())
        config = dict(manifest.get("run_config", {}))
        levels = {int(level) for level in config.pop("levels", [])}
        differing = sorted(key for key in set(config) | set(expected) if config.get(key) != expected.get(key))
        if differing:
            notes.append(f"{name}: incompatible run_config keys {differing}")
            continue
        count = 0
        for layer, stats in manifest.get("layers", {}).items():
            for level in stats.get("levels", {}):
                file = bank_path(args.reuse_bank_dir, int(layer), int(level))
                if int(level) in levels and file.exists():
                    found[(int(layer), int(level))] = file
                    count += 1
        notes.append(f"{name}: compatible, {count} completed layer-level files")
    return found, notes


def expects_asymmetric(args, level: int) -> bool:
    return args.scheme in ("asymmetric", "asymmetric_norot") and level >= 4


def reuses_legacy(args, level: int) -> bool:
    """Legacy K=3 is the shared ternary; legacy K=4,5 carry a trained rotation, so only the
    rotation scheme may reuse them."""
    return level == 3 or (args.scheme == "asymmetric" and level >= 4)


def scheme_name(args) -> str:
    if args.scheme == "symmetric":
        return SYMMETRIC_SCHEME
    if args.scheme == "asymmetric_norot":
        return HYBRID_SCHEME + "_norot"
    return HYBRID_SCHEME


def dp_codebooks_all_levels(centers: torch.Tensor, masses: torch.Tensor, levels) -> dict:
    """optimal_weighted_scalar_codebook for every requested K from one DP run at max(levels).

    Row k of the DP depends only on rows below k, so back[k, n] for k < k_max is what a run with
    num_levels=k would produce; each level is backtracked and gauge-fixed exactly as the original.
    """
    x = centers.detach().double().cpu().numpy()
    w = masses.detach().double().cpu().numpy()
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[keep], w[keep]
    k_max = int(max(levels))
    if x.size < k_max:
        raise ValueError(f"histogram has {x.size} non-empty bins, fewer than K={k_max}")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    n = int(x.size)
    p0 = np.concatenate(([0.0], np.cumsum(w, dtype=np.float64)))
    p1 = np.concatenate(([0.0], np.cumsum(w * x, dtype=np.float64)))
    p2 = np.concatenate(([0.0], np.cumsum(w * x * x, dtype=np.float64)))
    previous = np.full(n + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    back = np.full((k_max + 1, n + 1), -1, dtype=np.int32)
    totals = {}
    for k in range(1, k_max + 1):
        current = np.full(n + 1, np.inf, dtype=np.float64)

        def solve(left: int, right: int, opt_left: int, opt_right: int) -> None:
            if left > right:
                return
            middle = (left + right) // 2
            lo = max(k - 1, opt_left)
            hi = min(middle - 1, opt_right)
            candidates = np.arange(lo, hi + 1, dtype=np.int64)
            values = previous[candidates] + _weighted_interval_sse(p0, p1, p2, candidates, middle)
            best_offset = int(np.argmin(values))
            best = int(candidates[best_offset])
            current[middle] = float(values[best_offset])
            back[k, middle] = best
            solve(left, middle - 1, opt_left, best)
            solve(middle + 1, right, best, opt_right)

        solve(k, n, k - 1, n - 1)
        previous = current
        totals[k] = float(previous[n])

    out_device = centers.device
    out_dtype = centers.dtype if centers.is_floating_point() else torch.float32
    result = {}
    for level in levels:
        segments = []
        end = n
        for k in range(int(level), 0, -1):
            start = int(back[k, end])
            if start < 0:
                raise RuntimeError("failed to backtrack 1-D k-means DP")
            segments.append((start, end))
            end = start
        segments.reverse()
        codebook = []
        cluster_mass = []
        for start, end in segments:
            mass = p0[end] - p0[start]
            codebook.append((p1[end] - p1[start]) / mass)
            cluster_mass.append(mass)
        codebook_np = np.asarray(codebook, dtype=np.float64)
        prior_np = np.asarray(cluster_mass, dtype=np.float64)
        prior_np /= prior_np.sum()
        mean = float(np.sum(prior_np * codebook_np))
        variance = float(np.sum(prior_np * (codebook_np - mean) ** 2))
        if not np.isfinite(variance) or variance <= 1e-20:
            raise FloatingPointError("degenerate codebook variance")
        codebook_np = (codebook_np - mean) / np.sqrt(variance)
        result[int(level)] = (
            torch.as_tensor(codebook_np, device=out_device, dtype=out_dtype),
            torch.as_tensor(prior_np, device=out_device, dtype=out_dtype),
            totals[int(level)],
        )
    return result


@torch.no_grad()
def quantize_norot_levels(weight: torch.Tensor, moment: torch.Tensor | None, levels, args) -> dict:
    """asymmetric_codebook_quantize(rotation_iters=0) for several K, sharing histogram and DP."""
    source = weight.float().to(args.device)
    if moment is None:
        metric = torch.zeros((source.shape[1], source.shape[1]), device=args.device, dtype=torch.float32)
    else:
        metric = moment.to(args.device).float()
    centers, masses, _hist_mu, _hist_alpha = activation_weighted_histogram(
        source, metric, bins=args.histogram_bins, clip_quantile=args.histogram_clip_quantile
    )
    codebooks = dp_codebooks_all_levels(centers, masses, levels)
    n1, n2 = kronecker_factorize(int(source.shape[1]))
    left = torch.eye(n1, device=source.device, dtype=source.dtype)
    right = torch.eye(n2, device=source.device, dtype=source.dtype)
    results = {}
    for level in levels:
        codebook, priors, histogram_sse = codebooks[int(level)]
        codes, level_values, mu0, alpha0 = euclidean_code_assignment(
            source, codebook, iters=args.assignment_iters
        )
        reconstructed, mu, alpha = activation_aware_relocation_with_rotation(
            source, level_values, metric, left, right, mu0, alpha0,
            reg_lambda=args.reg_lambda, degenerate_tau=args.degenerate_tau,
        )
        error = source - reconstructed
        results[int(level)] = AsymmetricCodebookResult(
            reconstructed=reconstructed,
            codes=codes.to(torch.uint8),
            codebook=codebook,
            codebook_priors=priors,
            mu=mu,
            alpha=alpha,
            rotation_left=left,
            rotation_right=right,
            stats={
                "num_levels": int(level),
                "histogram_bins": int(args.histogram_bins),
                "histogram_nonempty_bins": int(centers.numel()),
                "histogram_dp_sse": float(histogram_sse),
                "weight_squared_error": float(error.square().sum().double().cpu()),
                "activation_weighted_error": float((error * (metric @ error.t()).t()).sum().double().cpu()),
                "rotation_iters": 0,
            },
        )
    return results


def resolve_sources(args, legacy: dict) -> tuple[dict, dict]:
    sources, origin = {}, {}
    for layer in range(NUM_LAYERS):
        for level in args.levels:
            new = bank_path(args.bank_dir, layer, level)
            if new.exists():
                sources[(layer, level)], origin[(layer, level)] = new, "written"
            elif (layer, level) in legacy and reuses_legacy(args, level):
                sources[(layer, level)], origin[(layer, level)] = legacy[(layer, level)], "legacy"
            else:
                sources[(layer, level)], origin[(layer, level)] = new, "build"
    return sources, origin


@torch.no_grad()
def build_norot_variants(paths: dict, experts, moments: dict, levels, args) -> dict:
    """Build every pending asymmetric_norot level of one layer in a single expert pass."""
    started = time.time()
    count = int(experts.num_experts)
    levels = sorted(int(level) for level in levels)
    buffers = {level: allocate_buffers(experts, level) for level in levels}
    for expert in range(count):
        entry = moments.get(expert)
        for index, prefix, weights in ((0, "gate_up", experts.gate_up_proj), (1, "down", experts.down_proj)):
            results = quantize_norot_levels(weights.data[expert], None if entry is None else entry[index], levels, args)
            for level, result in results.items():
                store_result(buffers[level], prefix, expert, result)
            del results
        if (expert + 1) % 64 == 0:
            print(f"[bank] norot levels={levels} experts={expert + 1}/{count} seconds={time.time() - started:.1f}", flush=True)
    stats = {}
    for level in levels:
        save_variant(paths[level], buffers[level], level, ASYMMETRIC_VARIANT + "_norot")
        stats[str(level)] = {
            "level": level,
            "variant": ASYMMETRIC_VARIANT + "_norot",
            "path": str(paths[level]),
            "bytes": paths[level].stat().st_size,
            "zero_hit_experts": count - len(moments),
            "seconds": (time.time() - started) / len(levels),
        }
    del buffers
    gc.collect()
    torch.cuda.empty_cache()
    return stats


# No torch.no_grad here: the asymmetric quantizer trains its Kronecker rotation with autograd.
def build_variant(path: Path, experts, moments: dict, level: int, args) -> dict:
    started = time.time()
    count = int(experts.num_experts)
    asymmetric = expects_asymmetric(args, level)
    buffers = allocate_buffers(experts, level) if asymmetric else allocate_level_buffers(experts)
    for expert in range(count):
        entry = moments.get(expert)
        for index, prefix, weights in ((0, "gate_up", experts.gate_up_proj), (1, "down", experts.down_proj)):
            if asymmetric:
                result = quantize_one(weights.data[expert], None if entry is None else entry[index], level, args)
                store_result(buffers, prefix, expert, result)
                del result
            else:
                weight = weights.data[expert].float()
                moment = (
                    torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device=weight.device)
                    if entry is None else entry[index]
                )
                codes, mu, alpha = twla_parameters(weight, moment, level, args.euclidean_iters)
                buffers[f"{prefix}_codes"][expert].copy_(codes.cpu())
                buffers[f"{prefix}_mu"][expert].copy_(mu.to(torch.bfloat16).cpu())
                buffers[f"{prefix}_alpha"][expert].copy_(alpha.to(torch.bfloat16).cpu())
                del weight, moment, codes, mu, alpha
        if asymmetric and (expert + 1) % 32 == 0:
            print(f"[bank] level={level} experts={expert + 1}/{count} seconds={time.time() - started:.1f}", flush=True)
    save_variant(path, buffers, level, ASYMMETRIC_VARIANT if asymmetric else SYMMETRIC_SCHEME)
    del buffers
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "level": level,
        "variant": ASYMMETRIC_VARIANT if asymmetric else SYMMETRIC_SCHEME,
        "path": str(path),
        "bytes": path.stat().st_size,
        "zero_hit_experts": count - len(moments),
        "seconds": time.time() - started,
    }


@torch.no_grad()
def layer_costs(experts, moments: dict, sources: dict[int, Path], levels, tokens: int, args) -> dict:
    count = int(experts.num_experts)
    projection = {str(expert): {} for expert in range(count)}
    clamped = 0
    for level in levels:
        for start in range(0, count, args.cost_chunk):
            end = min(start + args.cost_chunk, count)
            decoded = decode_range(sources[level], level, start, end, args.device)
            parts = {expert: {} for expert in range(start, end)}
            for index, prefix, weights in ((0, "gate_up", experts.gate_up_proj), (1, "down", experts.down_proj)):
                error = weights.data[start:end].float() - decoded[index].float()
                width = error.shape[-1]
                metric = torch.stack([
                    moments[expert][index].float() if expert in moments
                    else torch.zeros((width, width), dtype=torch.float32, device=error.device)
                    for expert in range(start, end)
                ])
                values = (torch.bmm(error, metric) * error).sum(dim=(1, 2)).double().cpu().tolist()
                for expert, value in zip(range(start, end), values):
                    if value < 0:
                        clamped += 1
                        value = 0.0
                    parts[expert][prefix] = value / tokens
                del error, metric
            for expert, value in parts.items():
                projection[str(expert)][str(level)] = value
            del decoded
    costs = {
        expert: {level: value["gate_up"] + value["down"] for level, value in by_level.items()}
        for expert, by_level in projection.items()
    }
    hits = {str(expert): int(moments[expert][2]) if expert in moments else 0 for expert in range(count)}
    return {"costs": costs, "projection_costs": projection, "hits": hits, "negative_values_clamped": clamped}


def cost_file_complete(path: Path, levels, sources: dict[int, Path]) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        payload.get("levels") == list(levels)
        and payload.get("sources") == {str(level): str(sources[level]) for level in levels}
        and len(payload.get("costs", {})) == NUM_EXPERTS
    )


def assemble(args, sources: dict, tokens: int) -> None:
    costs = {}
    for layer in range(NUM_LAYERS):
        payload = json.loads(cost_path(args.bank_dir, layer).read_text())
        if payload["levels"] != list(args.levels) or len(payload["costs"]) != NUM_EXPERTS:
            raise RuntimeError(f"layer {layer} proxy costs are incomplete")
        for expert in range(NUM_EXPERTS):
            costs[f"{layer}:{expert}"] = payload["costs"][str(expert)]
    for (layer, level), path in sources.items():
        info = describe_variant(path, level)
        if info["asymmetric"] != expects_asymmetric(args, level):
            raise RuntimeError(f"{path} has the wrong quantizer for layer {layer} level {level}")
    metadata = {
        "model": str(args.model_dir.resolve()),
        "num_hidden_layers": NUM_LAYERS,
        "first_sparse_layer": 0,
        "num_experts": NUM_EXPERTS,
        "levels": list(args.levels),
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "symmetric": args.scheme == "symmetric",
        "routed_experts_only": True,
        "three_projections_tied": True,
        "scheme": scheme_name(args),
        "rotation": "none (identity; Stage I code assignment + Stage II relocation only)"
                    if args.scheme == "asymmetric_norot" else
                    ("codebook-aware Kronecker rotation for K>=4" if args.scheme == "asymmetric" else "none"),
        "run_id": RUN_ID,
        "calibset": str(args.calibset.resolve()),
        "calibration_tokens": tokens,
        "cost_definition": COST_DEFINITION,
        "gate_up_fused": True,
        "bank_dir": str(args.bank_dir.resolve()),
        "bank_sources": "bank_sources.json",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(args.bank_dir / "proxy_costs.json", {"metadata": metadata, "costs": costs})
    atomic_json(args.bank_dir / "bank_sources.json", {
        "scheme": metadata["scheme"],
        "layers": {
            str(layer): {str(level): str(sources[(layer, level)]) for level in args.levels}
            for layer in range(NUM_LAYERS)
        },
    })


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    legacy, notes = legacy_variants(args)
    sources, origin = resolve_sources(args, legacy)
    for (layer, level), kind in sorted(origin.items()):
        if kind != "build":
            info = describe_variant(sources[(layer, level)], level)
            if info["asymmetric"] != expects_asymmetric(args, level):
                raise RuntimeError(f"{sources[(layer, level)]} has the wrong quantizer for this scheme")
    print(f"[plan] scheme={args.scheme} bank={args.bank_dir}", flush=True)
    for note in notes:
        print(f"[plan] legacy {note}", flush=True)
    for level in args.levels:
        kinds = [origin[(layer, level)] for layer in range(args.layer_end)]
        print(f"[plan] level={level} legacy={kinds.count('legacy')} written={kinds.count('written')} "
              f"to_build={kinds.count('build')}", flush=True)
    if args.plan_only:
        return

    config = {
        "run_id": RUN_ID,
        "scheme": scheme_name(args),
        "model_dir": str(args.model_dir.resolve()),
        "calibset": str(args.calibset.resolve()),
        "reuse_bank_dir": None if args.reuse_bank_dir is None else str(args.reuse_bank_dir.resolve()),
        "levels": list(args.levels),
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "histogram_bins": args.histogram_bins,
        "histogram_clip_quantile": args.histogram_clip_quantile,
        "rotation_iters": args.rotation_iters,
        "rotation_learning_rate": args.rotation_learning_rate,
        "rotation_peak_sigma": args.rotation_peak_sigma,
        "rotation_occupancy_beta": args.rotation_occupancy_beta,
        "assignment_iters": args.assignment_iters,
        "euclidean_iters": args.euclidean_iters,
        "reg_lambda": args.reg_lambda,
        "degenerate_tau": args.degenerate_tau,
        "cost_definition": COST_DEFINITION,
        "code_storage": "4-bit packed for variants written here; legacy uint8 read in place",
    }
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.bank_dir / "bank_manifest.json"
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
    if len(text_model.layers) != NUM_LAYERS:
        raise RuntimeError(f"expected {NUM_LAYERS} decoder layers, found {len(text_model.layers)}")
    trainloader = load_qwen36_calibset(str(args.calibset), args.nsamples, args.seed, args.seqlen)
    if len(trainloader) != args.nsamples:
        raise RuntimeError(f"expected {args.nsamples} blocks, got {len(trainloader)}")
    hidden_states = [text_model.embed_tokens(input_ids.to(args.device)).cpu() for input_ids, _ in trainloader]
    tokens = sum(int(hidden.shape[0] * hidden.shape[1]) for hidden in hidden_states)
    common_kwargs = model_forward_kwargs(text_model, hidden_states[0].to(args.device), args.device)

    started = time.time()
    for layer_idx, layer in enumerate(text_model.layers):
        if layer_idx >= args.layer_end:
            break
        experts = layer.mlp.experts
        if int(experts.num_experts) != NUM_EXPERTS:
            raise RuntimeError(f"layer {layer_idx} has {experts.num_experts} routed experts")
        layer_sources = {level: sources[(layer_idx, level)] for level in args.levels}
        pending = [level for level in args.levels if not layer_sources[level].exists()]
        if not pending and cost_file_complete(cost_path(args.bank_dir, layer_idx), args.levels, layer_sources):
            hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
            print(f"[bank] layer={layer_idx}/39 resumed", flush=True)
            continue

        layer_started = time.time()
        print(f"[bank] layer={layer_idx}/39 collecting BF16 routed moments", flush=True)
        moments = collect_layer_moments(layer, hidden_states, common_kwargs, args.device)
        built = {}
        norot_levels = [level for level in pending if args.scheme == "asymmetric_norot" and level >= 4]
        if norot_levels:
            built.update(build_norot_variants({level: layer_sources[level] for level in norot_levels}, experts, moments,
                                              norot_levels, args))
            print(f"[bank] layer={layer_idx}/39 levels={norot_levels} built (no rotation) "
                  f"seconds={sum(built[str(level)]['seconds'] for level in norot_levels):.1f}", flush=True)
        for level in pending:
            if level in norot_levels:
                continue
            built[str(level)] = build_variant(layer_sources[level], experts, moments, level, args)
            print(f"[bank] layer={layer_idx}/39 level={level} built seconds={built[str(level)]['seconds']:.1f}", flush=True)
        cost_started = time.time()
        payload = layer_costs(experts, moments, layer_sources, args.levels, tokens, args)
        atomic_json(cost_path(args.bank_dir, layer_idx), {
            "layer": layer_idx,
            "levels": list(args.levels),
            "calibration_tokens": tokens,
            "sources": {str(level): str(layer_sources[level]) for level in args.levels},
            **payload,
            "cost_seconds": time.time() - cost_started,
        })
        hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
        del moments
        torch.cuda.empty_cache()
        manifest["layers"][str(layer_idx)] = {
            "built": built,
            "origins": {str(level): origin[(layer_idx, level)] for level in args.levels},
            "experts_hit": sum(1 for hits in payload["hits"].values() if hits),
            "negative_values_clamped": payload["negative_values_clamped"],
            "cost_seconds": time.time() - cost_started,
            "elapsed_seconds": time.time() - layer_started,
        }
        atomic_json(manifest_path, manifest)
        print(f"[bank] layer={layer_idx}/39 complete layer_seconds={time.time() - layer_started:.1f} "
              f"elapsed_seconds={time.time() - started:.1f}", flush=True)

    if args.layer_end < NUM_LAYERS:
        print(f"[complete] test build stopped at layer {args.layer_end}; proxy_costs.json not assembled", flush=True)
        return
    assemble(args, sources, tokens)
    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(manifest_path, manifest)
    (args.bank_dir / ".bank_done").touch()
    print(f"[complete] bank={args.bank_dir} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a reusable expert-level TWLA codebook bank for Qwen3.6.

Every routed expert is quantized from the same untouched BF16 weight and the
same calibration-induced second moment at each integer codebook size in the
configured ladder.  Codes, row offsets, and row scales are stored rather than
dense reconstructions.  The bank therefore supports exact, reversible
expert-level hot swaps during validation-NLL search without rerunning TWLA for
each lambda or each binary-search branch.

The initial 5-level checkpoint is materialized from the very same stored bank,
so it is bit-for-bit consistent with the optimizer's starting state.
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
from transformers import AutoModelForImageTextToText, AutoTokenizer

from quantize import E2M_ATQ as ternary
from quantize import E2M_ATQ_bidirectional as multilevel
from quantize_qwen36_experts_5level import _CalibratingExperts, load_qwen36_calibset


ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
DEFAULT_MODEL = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))
DEFAULT_DATA = None
DEFAULT_BANK = None
DEFAULT_INITIAL = None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bank_path(bank_dir: Path, layer: int, level: int) -> Path:
    return bank_dir / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"


@torch.no_grad()
def twla_parameters(
    weight: torch.Tensor,
    moment: torch.Tensor,
    level: int,
    euclidean_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if level == 3:
        codes, mu0, alpha0 = ternary.euclidean_warm_start(
            weight, iters=euclidean_iters
        )
        mu, alpha = ternary.manifold_relocation(
            weight, codes, moment, mu0, alpha0
        )
        stored_codes = (codes + 1.0).round().to(torch.uint8)
    else:
        codes, mu0, alpha0 = multilevel.euclidean_warm_start(
            weight, num_levels=level, iters=euclidean_iters
        )
        mu, alpha = multilevel.manifold_relocation(
            weight, codes, moment, mu0, alpha0
        )
        center = (level - 1) / 2.0
        stored_codes = (codes + center).round().to(torch.uint8)
    return stored_codes, mu, alpha


@torch.no_grad()
def collect_layer(
    layer,
    hidden_states: list[torch.Tensor],
    common_kwargs: dict,
    device: str,
) -> tuple[dict, list[torch.Tensor]]:
    experts = layer.mlp.experts
    accum: dict = {}
    calibrating = _CalibratingExperts(experts, accum)
    original_forward = experts.forward
    experts.forward = calibrating
    outputs = []
    mask_key = "linear_attn_mask" if (getattr(layer, "block_type", None) or layer.layer_type) == "linear_attention" else "causal_mask"
    try:
        for hidden in hidden_states:
            output = layer(
                hidden.to(device),
                position_embeddings=common_kwargs["position_embeddings"],
                attention_mask=common_kwargs[mask_key],
                position_ids=common_kwargs["text_position_ids"],
                past_key_values=None,
                use_cache=False,
            )
            outputs.append(output.cpu())
    finally:
        experts.forward = original_forward
    return accum, outputs


def allocate_level_buffers(experts) -> dict[str, torch.Tensor]:
    gu = experts.gate_up_proj
    dn = experts.down_proj
    return {
        "gate_up_codes": torch.empty(tuple(gu.shape), dtype=torch.uint8),
        "gate_up_mu": torch.empty((gu.shape[0], gu.shape[1]), dtype=torch.bfloat16),
        "gate_up_alpha": torch.empty((gu.shape[0], gu.shape[1]), dtype=torch.bfloat16),
        "down_codes": torch.empty(tuple(dn.shape), dtype=torch.uint8),
        "down_mu": torch.empty((dn.shape[0], dn.shape[1]), dtype=torch.bfloat16),
        "down_alpha": torch.empty((dn.shape[0], dn.shape[1]), dtype=torch.bfloat16),
    }


@torch.no_grad()
def write_layer_level(
    path: Path,
    experts,
    accum: dict,
    level: int,
    euclidean_iters: int,
    device: str,
) -> dict:
    if path.exists():
        return {"level": level, "path": str(path), "resumed": True}
    buffers = allocate_level_buffers(experts)
    zero_hit = 0
    squared_error = 0.0
    for expert_idx in range(int(experts.num_experts)):
        entry = accum.get(expert_idx)
        if entry is None:
            zero_hit += 1
            s_gu = torch.zeros(
                (experts.gate_up_proj.shape[-1],) * 2,
                dtype=torch.float32,
                device=device,
            )
            s_dn = torch.zeros(
                (experts.down_proj.shape[-1],) * 2,
                dtype=torch.float32,
                device=device,
            )
        else:
            s_gu, s_dn, _hits = entry

        gu = experts.gate_up_proj.data[expert_idx].float()
        codes, mu, alpha = twla_parameters(
            gu, s_gu, level, euclidean_iters
        )
        reconstruction = mu[:, None] + alpha[:, None] * (
            codes.float() - (level - 1) / 2.0
        )
        squared_error += float((gu - reconstruction).square().sum().double().cpu())
        buffers["gate_up_codes"][expert_idx].copy_(codes.cpu())
        buffers["gate_up_mu"][expert_idx].copy_(mu.to(torch.bfloat16).cpu())
        buffers["gate_up_alpha"][expert_idx].copy_(alpha.to(torch.bfloat16).cpu())
        del gu, codes, mu, alpha, reconstruction

        dn = experts.down_proj.data[expert_idx].float()
        codes, mu, alpha = twla_parameters(
            dn, s_dn, level, euclidean_iters
        )
        reconstruction = mu[:, None] + alpha[:, None] * (
            codes.float() - (level - 1) / 2.0
        )
        squared_error += float((dn - reconstruction).square().sum().double().cpu())
        buffers["down_codes"][expert_idx].copy_(codes.cpu())
        buffers["down_mu"][expert_idx].copy_(mu.to(torch.bfloat16).cpu())
        buffers["down_alpha"][expert_idx].copy_(alpha.to(torch.bfloat16).cpu())
        del dn, codes, mu, alpha, reconstruction
        if entry is None:
            del s_gu, s_dn

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.contiguous() for key, value in buffers.items()}, str(temporary))
    os.replace(temporary, path)
    size = path.stat().st_size
    del buffers
    gc.collect()
    return {
        "level": level,
        "path": str(path),
        "bytes": size,
        "zero_hit_experts": zero_hit,
        "weight_squared_error": squared_error,
        "resumed": False,
    }


@torch.no_grad()
def install_uniform_level(layer, path: Path, level: int, device: str, chunk: int = 32) -> None:
    experts = layer.mlp.experts
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    center = (level - 1) / 2.0
    for start in range(0, int(experts.num_experts), chunk):
        end = min(start + chunk, int(experts.num_experts))
        codes = tensors["gate_up_codes"][start:end].to(device).float().sub_(center)
        mu = tensors["gate_up_mu"][start:end].to(device).float()
        alpha = tensors["gate_up_alpha"][start:end].to(device).float()
        reconstruction = mu[:, :, None] + alpha[:, :, None] * codes
        experts.gate_up_proj.data[start:end].copy_(reconstruction.to(torch.bfloat16))
        del codes, mu, alpha, reconstruction

        codes = tensors["down_codes"][start:end].to(device).float().sub_(center)
        mu = tensors["down_mu"][start:end].to(device).float()
        alpha = tensors["down_alpha"][start:end].to(device).float()
        reconstruction = mu[:, :, None] + alpha[:, :, None] * codes
        experts.down_proj.data[start:end].copy_(reconstruction.to(torch.bfloat16))
        del codes, mu, alpha, reconstruction
    del tensors
    torch.cuda.empty_cache()


def parse_levels(value: str, require_initial: bool) -> tuple[int, ...]:
    levels = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    # Level 2 is the symmetric binary endpoint of the same multilevel TWLA
    # parameterization.  The original experiment only built levels 3..16,
    # but low-to-high initialization experiments also need a durable level-2
    # bank entry.  Keep the default ladder unchanged while allowing an
    # explicit --levels 2 shard to extend an existing bank.
    if not levels or levels[0] < 2 or levels[-1] > 16:
        raise ValueError("levels must be within 2..16")
    if require_initial and 5 not in levels:
        raise ValueError("the shard materializing the initial checkpoint must include level 5")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--levels", default=",".join(map(str, range(3, 17))))
    parser.add_argument("--initial-level", type=int, default=5)
    parser.add_argument("--nsamples", type=int, default=115)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=260910)
    parser.add_argument("--euclidean-iters", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--coordinated-shard",
        action="store_true",
        help="write a shard-specific manifest/marker; the supervisor creates .bank_done after all shards",
    )
    parser.add_argument("--shard-id", default="full")
    parser.add_argument(
        "--skip-initial-checkpoint",
        action="store_true",
        help="build only this level subset; another coordinated shard owns the level-5 checkpoint",
    )
    args = parser.parse_args()
    levels = parse_levels(args.levels, require_initial=not args.skip_initial_checkpoint)
    if args.initial_level != 5:
        raise ValueError("this experiment's fixed initial level is 5")
    torch.set_num_threads(args.num_threads)
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    args.initial_checkpoint.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_dir": str(args.model_dir.resolve()),
        "calibset": str(args.calibset.resolve()),
        "levels": list(levels),
        "initial_level": args.initial_level,
        "initial_logical_bits": math.log2(args.initial_level),
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "euclidean_iters": args.euclidean_iters,
        "twla_stages": [1, 2],
        "routed_experts_only": True,
        "scale_granularity": "one mu and alpha per output row",
        "allocation_unit": "one routed expert; gate_up_proj and down_proj tied",
    }
    manifest_path = args.bank_dir / (
        f"bank_manifest_{args.shard_id}.json" if args.coordinated_shard else "bank_manifest.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_config") != run_config:
            raise ValueError("bank directory contains a different run configuration")
    else:
        manifest = {"status": "building", "run_config": run_config, "layers": {}}
        atomic_json(manifest_path, manifest)

    print(f"[bank] loading BF16 master {args.model_dir} -> {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_pretrained_streaming(args.model_dir, args.device, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    text_model.config.use_cache = False
    trainloader = load_qwen36_calibset(
        str(args.calibset), args.nsamples, args.seed, args.seqlen
    )
    if len(trainloader) != args.nsamples:
        raise RuntimeError(f"expected {args.nsamples} calibration blocks, got {len(trainloader)}")

    from transformers.masking_utils import create_causal_mask

    hidden_states = [
        text_model.embed_tokens(ids.to(args.device)).cpu() for ids, _ in trainloader
    ]
    sample = hidden_states[0].to(args.device)
    position_ids = torch.arange(args.seqlen, device=args.device).view(1, 1, -1).expand(4, 1, -1)
    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=sample,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids[0],
    )
    common_kwargs = {
        "causal_mask": causal_mask,
        "linear_attn_mask": None,
        "text_position_ids": position_ids[0],
        "position_embeddings": text_model.rotary_emb(sample, position_ids[1:]),
    }

    total_layers = int(text_model.config.num_hidden_layers)
    if args.max_layers is not None:
        total_layers = min(total_layers, args.max_layers)
    started = time.time()
    for layer_idx in range(total_layers):
        layer_started = time.time()
        layer = text_model.layers[layer_idx]
        print(f"[bank] layer={layer_idx}/39 collecting routed moments", flush=True)
        accum, next_hidden_states = collect_layer(
            layer, hidden_states, common_kwargs, args.device
        )
        del hidden_states
        hidden_states = next_hidden_states
        layer_stats = {"experts_hit": len(accum), "levels": {}}
        for level in levels:
            output_path = bank_path(args.bank_dir, layer_idx, level)
            level_started = time.time()
            stats = write_layer_level(
                output_path,
                layer.mlp.experts,
                accum,
                level,
                args.euclidean_iters,
                args.device,
            )
            stats["elapsed_seconds"] = time.time() - level_started
            layer_stats["levels"][str(level)] = stats
            print(
                f"[bank] layer={layer_idx}/39 level={level} resumed={stats['resumed']} "
                f"seconds={stats['elapsed_seconds']:.1f}",
                flush=True,
            )
        del accum
        torch.cuda.empty_cache()
        if not args.skip_initial_checkpoint:
            install_uniform_level(
                layer,
                bank_path(args.bank_dir, layer_idx, args.initial_level),
                args.initial_level,
                args.device,
            )
        layer_stats["elapsed_seconds"] = time.time() - layer_started
        manifest["layers"][str(layer_idx)] = layer_stats
        atomic_json(manifest_path, manifest)
        print(
            f"[bank] layer={layer_idx}/39 complete elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    if total_layers != int(text_model.config.num_hidden_layers):
        print("[bank] debug max-layers run: checkpoint not saved", flush=True)
        return

    if not args.skip_initial_checkpoint:
        print(f"[bank] saving initial 5-level checkpoint {args.initial_checkpoint}", flush=True)
        model.save_pretrained(args.initial_checkpoint)
        tokenizer.save_pretrained(args.initial_checkpoint)
        atomic_json(args.initial_checkpoint / "twla_quant_stats.json", {
            "scheme": "uniform_5level_TWLA_from_shared_level_bank",
            "num_levels": 5,
            "logical_routed_expert_bits": math.log2(5),
            "calibset": str(args.calibset.resolve()),
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "twla_stages": [1, 2],
            "routed_experts_only": True,
            "bank_dir": str(args.bank_dir.resolve()),
        })
        (args.initial_checkpoint / ".quant_done").touch()
    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["elapsed_seconds"] = time.time() - started
    manifest["initial_checkpoint"] = (
        None if args.skip_initial_checkpoint else str(args.initial_checkpoint.resolve())
    )
    atomic_json(manifest_path, manifest)
    marker = (
        args.bank_dir / f".bank_shard_{args.shard_id}_done"
        if args.coordinated_shard
        else args.bank_dir / ".bank_done"
    )
    marker.touch()
    print(
        f"[bank] complete elapsed={manifest['elapsed_seconds']:.1f}s "
        f"initial_checkpoint={args.initial_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()

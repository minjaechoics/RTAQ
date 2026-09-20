"""Five-sweep bidirectional mixed-precision TWLA allocation for Qwen3.6.

Initialization reproduces the prior layer-ablation prior: decoder layers
0,2,24,28,30,35,38,39 use 16 levels (4 bits), while every other routed
expert starts with literal TWLA ternary (3 levels, log2(3) bits).

Each sweep loads the untouched BF16 master, replays calibration through the
CURRENT mixed-precision prefix, and measures both neighboring changes with
the same activation-weighted objective ``||(W-Wq)A||^2``.  Cheap demotions
fund valuable promotions under the initial fixed bit budget.  The precision
ladder is 3/4/8/16 levels.  Fractional average precision is obtained by
assigning different equal-weight row groups to adjacent ladder entries; no
fractional codebook and no pruning are used.

The checkpoint is a dense BF16 fake-quantized artifact, consistent with the
other Qwen3.6 experiments in this repository.  Only routed-expert gate/up and
down matrices are quantized; activations and all other weights stay BF16.
"""

from __future__ import annotations

import os
RTAQ_ROOT = os.environ.get("RTAQ_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from streaming_load import load_pretrained_streaming

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from quantize.E2M_ATQ import e2m_atq_quantize as ternary_e2m_atq
from quantize.E2M_ATQ_bidirectional import (
    activation_weighted_row_error,
    e2m_atq_quantize_levels,
)
from quantize_qwen36_experts import _CalibratingExperts, load_qwen36_calibset


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))
DEFAULT_CALIBSET = None
SENSITIVE_LAYERS = frozenset((0, 2, 24, 28, 30, 35, 38, 39))
LEVEL_LADDER = (3, 4, 8, 16)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def save_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def group_id(layer: int, expert: int, matrix: str, row_start: int, row_end: int) -> str:
    return f"L{layer:02d}.E{expert:03d}.{matrix}.R{row_start:04d}-{row_end:04d}"


def row_groups(nrows: int, ncols: int, target_group_weights: int) -> list[tuple[int, int]]:
    rows_per_group = max(1, int(target_group_weights) // int(ncols))
    return [
        (start, min(start + rows_per_group, nrows))
        for start in range(0, nrows, rows_per_group)
    ]


def initial_level(layer_idx: int, state: dict | None = None) -> int:
    if state is not None and bool(state.get("initial_all_ternary", False)):
        return 3
    return 16 if int(layer_idx) in SENSITIVE_LAYERS else 3


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "status": "scoring",
            "sweeps_completed": 0,
            "levels": {},
            "history": [],
            "seen_map_hashes": [],
            "initial_budget_bits": None,
        }
    state = json.loads(path.read_text())
    if int(state.get("version", -1)) != 1:
        raise ValueError(f"unsupported state version in {path}")
    return state


def resolved_level(state: dict, key: str, layer_idx: int) -> int:
    return int(state["levels"].get(key, initial_level(layer_idx, state)))


def map_hash(levels: dict[str, int]) -> str:
    packed = json.dumps(sorted((key, int(value)) for key, value in levels.items()))
    return hashlib.sha256(packed.encode()).hexdigest()


def weighted_bits(state: dict, specs: dict[str, dict]) -> tuple[float, float, int]:
    total_weights = sum(int(spec["num_weights"]) for spec in specs.values())
    total_bits = sum(
        int(spec["num_weights"])
        * math.log2(resolved_level(state, key, int(spec["layer"])))
        for key, spec in specs.items()
    )
    return total_bits / max(total_weights, 1), total_bits, total_weights


def packed_bpw_estimate(state: dict, specs: dict[str, dict]) -> float:
    """Projected real storage BPW for the assigned routed-expert codes.

    Ternary uses the repository's five-trits-per-byte representation (1.6
    code bits/weight).  Power-of-two levels use literal 2/3/4-bit codes.
    Per-output-row fp16 mu and fp16 alpha add 32 bits per row.  Group-map
    metadata is excluded because it is negligible and format-dependent.
    """
    code_bits = {3: 1.6, 4: 2.0, 8: 3.0, 16: 4.0}
    total_weights = 0
    total_bits = 0.0
    seen_rows: set[tuple[int, int, str, int, int]] = set()
    for key, spec in specs.items():
        level = resolved_level(state, key, int(spec["layer"]))
        weights = int(spec["num_weights"])
        total_weights += weights
        total_bits += weights * code_bits[level]
        row_key = (
            int(spec["layer"]),
            int(spec["expert"]),
            str(spec["matrix"]),
            int(spec["row_start"]),
            int(spec["row_end"]),
        )
        if row_key not in seen_rows:
            total_bits += 32.0 * (int(spec["row_end"]) - int(spec["row_start"]))
            seen_rows.add(row_key)
    return total_bits / max(total_weights, 1)


def model_forward_kwargs(text_model, sample_inputs_embeds: torch.Tensor, device: str) -> dict:
    from transformers.masking_utils import create_causal_mask

    config = text_model.config
    seqlen = sample_inputs_embeds.shape[1]
    position_ids = torch.arange(seqlen, device=device).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids = position_ids[0]
    mrope_position_ids = position_ids[1:]
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=sample_inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    return {
        "causal_mask": causal_mask,
        "linear_attn_mask": None,
        "text_position_ids": text_position_ids,
        "position_embeddings": text_model.rotary_emb(sample_inputs_embeds, mrope_position_ids),
    }


@torch.no_grad()
def run_layer(layer, hidden_states, common_kwargs, device: str) -> list[torch.Tensor]:
    mask_key = "linear_attn_mask" if (getattr(layer, "block_type", None) or layer.layer_type) == "linear_attention" else "causal_mask"
    outputs = []
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
    return outputs


@torch.no_grad()
def collect_layer_moments(layer, hidden_states, common_kwargs, device: str) -> dict:
    experts = layer.mlp.experts
    accum: dict = {}
    calibrating = _CalibratingExperts(experts, accum)
    original_forward = experts.forward
    experts.forward = calibrating
    try:
        run_layer(layer, hidden_states, common_kwargs, device)
    finally:
        experts.forward = original_forward
    return accum


@torch.no_grad()
def quantize_candidate(
    weight: torch.Tensor, moment: torch.Tensor, level: int, euclidean_iters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(level) == 3:
        reconstructed = ternary_e2m_atq(weight, moment, euclidean_iters=euclidean_iters)
    else:
        reconstructed = e2m_atq_quantize_levels(
            weight,
            moment,
            num_levels=int(level),
            euclidean_iters=euclidean_iters,
        )
    row_error = activation_weighted_row_error(weight, reconstructed, moment)
    if not bool(torch.isfinite(row_error).all()):
        raise FloatingPointError(f"non-finite activation-weighted error at level={level}")
    return reconstructed, row_error


@torch.no_grad()
def process_matrix(
    *,
    weight: torch.Tensor,
    moment: torch.Tensor | None,
    layer_idx: int,
    expert_idx: int,
    matrix_name: str,
    state: dict,
    specs: dict[str, dict],
    actions_up: list[dict],
    actions_down: list[dict],
    args,
    score_neighbors: bool,
) -> torch.Tensor:
    # Zero-hit experts have no activation evidence for ranking, but they are
    # still materialized at their assigned precision via Stage-I's Euclidean
    # fallback.  This keeps the logical budget structural and deterministic
    # even if routing changes between sweeps.
    has_activation_evidence = moment is not None
    if moment is None:
        moment = torch.zeros(
            (weight.shape[1], weight.shape[1]),
            dtype=torch.float32,
            device=weight.device,
        )

    nrows, ncols = map(int, weight.shape)
    groups = row_groups(nrows, ncols, args.target_group_weights)
    current_groups: list[tuple[str, int, int, int]] = []
    required_levels: set[int] = set()
    for start, end in groups:
        key = group_id(layer_idx, expert_idx, matrix_name, start, end)
        specs[key] = {
            "layer": layer_idx,
            "expert": expert_idx,
            "matrix": matrix_name,
            "row_start": start,
            "row_end": end,
            "num_weights": (end - start) * ncols,
        }
        level = resolved_level(state, key, layer_idx)
        if level not in LEVEL_LADDER:
            raise ValueError(f"invalid level={level} for {key}")
        current_groups.append((key, start, end, level))
        required_levels.add(level)
        if score_neighbors and has_activation_evidence:
            index = LEVEL_LADDER.index(level)
            if index > 0:
                required_levels.add(LEVEL_LADDER[index - 1])
            if index + 1 < len(LEVEL_LADDER):
                required_levels.add(LEVEL_LADDER[index + 1])

    reconstructions: dict[int, torch.Tensor] = {}
    errors: dict[int, torch.Tensor] = {}
    for level in sorted(required_levels):
        reconstructed, row_error = quantize_candidate(
            weight, moment, level, args.euclidean_iters
        )
        reconstructions[level] = reconstructed
        errors[level] = row_error

    mixed = torch.empty_like(weight)
    for key, start, end, level in current_groups:
        mixed[start:end] = reconstructions[level][start:end]
        if not score_neighbors or not has_activation_evidence:
            continue
        current_error = float(errors[level][start:end].sum().double().cpu())
        index = LEVEL_LADDER.index(level)
        spec = specs[key]
        num_weights = int(spec["num_weights"])
        if index > 0:
            lower = LEVEL_LADDER[index - 1]
            lower_error = float(errors[lower][start:end].sum().double().cpu())
            freed_bits = num_weights * (math.log2(level) - math.log2(lower))
            distortion_cost = max(0.0, lower_error - current_error)
            actions_down.append(
                {
                    "group": key,
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "matrix": matrix_name,
                    "from_levels": level,
                    "to_levels": lower,
                    "from_bits": math.log2(level),
                    "to_bits": math.log2(lower),
                    "freed_bits": freed_bits,
                    "distortion_cost": distortion_cost,
                    "cost_per_bit": distortion_cost / max(freed_bits, args.score_epsilon),
                }
            )
        if index + 1 < len(LEVEL_LADDER):
            upper = LEVEL_LADDER[index + 1]
            upper_error = float(errors[upper][start:end].sum().double().cpu())
            spent_bits = num_weights * (math.log2(upper) - math.log2(level))
            distortion_gain = max(0.0, current_error - upper_error)
            actions_up.append(
                {
                    "group": key,
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "matrix": matrix_name,
                    "from_levels": level,
                    "to_levels": upper,
                    "from_bits": math.log2(level),
                    "to_bits": math.log2(upper),
                    "spent_bits": spent_bits,
                    "distortion_gain": distortion_gain,
                    "gain_per_bit": distortion_gain / max(spent_bits, args.score_epsilon),
                }
            )
    del reconstructions, errors
    return mixed


@torch.no_grad()
def run_sweep(
    state: dict, args, *, score_neighbors: bool, save_model: bool
) -> tuple[dict, list[dict], list[dict], dict]:
    sweep_number = int(state["sweeps_completed"]) + (1 if score_neighbors else 0)
    phase = f"score_sweep_{sweep_number}" if score_neighbors else "final_materialization"
    print(f"[{phase}] loading untouched BF16 master onto {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_pretrained_streaming(args.model_dir, args.device, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    trainloader = load_qwen36_calibset(
        str(args.calibset), nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen
    )
    if not trainloader:
        raise RuntimeError("calibration loader returned zero samples")
    print(f"[{phase}] samples={len(trainloader)} seqlen={args.seqlen}", flush=True)
    hidden_states = [
        text_model.embed_tokens(input_ids.to(args.device)).cpu()
        for input_ids, _ in trainloader
    ]
    common_kwargs = model_forward_kwargs(
        text_model, hidden_states[0].to(args.device), args.device
    )
    specs: dict[str, dict] = {}
    actions_up: list[dict] = []
    actions_down: list[dict] = []
    layer_stats: list[dict] = []
    started = time.time()
    for layer_idx, layer in enumerate(text_model.layers):
        layer_started = time.time()
        if layer_idx in args.fixed_bf16_layers:
            # These layers are an immutable high-precision anchor.  They are
            # excluded from both the allocator's logical budget and its
            # promotion/demotion candidates, and their routed experts remain
            # byte-for-byte BF16 from the untouched master.
            hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
            layer_stats.append(
                {
                    "layer": layer_idx,
                    "fixed_bf16": True,
                    "zero_hit_experts_euclidean_fallback": 0,
                }
            )
            print(
                f"[{phase}] layer={layer_idx + 1}/{len(text_model.layers)} fixed_bf16=true "
                f"layer_s={time.time() - layer_started:.1f} elapsed_s={time.time() - started:.1f}",
                flush=True,
            )
            continue
        # Pass 1 observes the current layer input.  gate_up's moment depends
        # only on that input, so it is exact for the current mixed prefix.
        accum_gate = collect_layer_moments(layer, hidden_states, common_kwargs, args.device)
        # The BF16 down moments from this pass are not the moments of the
        # current quantized gate/up path and will be recollected below.
        for entry in accum_gate.values():
            entry[1] = None
        torch.cuda.empty_cache()
        experts = layer.mlp.experts
        dead = 0
        for expert_idx in range(int(experts.num_experts)):
            entry = accum_gate.get(expert_idx)
            if entry is None:
                dead += 1
                S_gate_up = None
            else:
                S_gate_up, _discarded_down_moment, _hits = entry
            gate_up = experts.gate_up_proj.data[expert_idx].float()
            mixed_gate_up = process_matrix(
                weight=gate_up,
                moment=S_gate_up,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                matrix_name="gate_up_proj",
                state=state,
                specs=specs,
                actions_up=actions_up,
                actions_down=actions_down,
                args=args,
                score_neighbors=score_neighbors,
            )
            experts.gate_up_proj.data[expert_idx] = mixed_gate_up.to(
                experts.gate_up_proj.dtype
            )
            del gate_up, mixed_gate_up

        # Keep only one full per-expert moment table resident at a time.
        del accum_gate
        torch.cuda.empty_cache()

        # Pass 2 runs with CURRENT-map gate_up reconstructions installed.
        # This makes down_proj's activation moment reflect the nonlinear
        # gate/up product of the current mixed-precision layer, rather than
        # the BF16 gate/up activations used by the older one-pass driver.
        accum_down = collect_layer_moments(layer, hidden_states, common_kwargs, args.device)
        # Only down moments are consumed in the second half.
        for entry in accum_down.values():
            entry[0] = None
        torch.cuda.empty_cache()
        for expert_idx in range(int(experts.num_experts)):
            entry = accum_down.get(expert_idx)
            S_down = None if entry is None else entry[1]
            down = experts.down_proj.data[expert_idx].float()
            mixed_down = process_matrix(
                weight=down,
                moment=S_down,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                matrix_name="down_proj",
                state=state,
                specs=specs,
                actions_up=actions_up,
                actions_down=actions_down,
                args=args,
                score_neighbors=score_neighbors,
            )
            experts.down_proj.data[expert_idx] = mixed_down.to(experts.down_proj.dtype)
            del down, mixed_down
        hidden_states = run_layer(layer, hidden_states, common_kwargs, args.device)
        del accum_down
        torch.cuda.empty_cache()
        layer_stats.append({"layer": layer_idx, "zero_hit_experts_euclidean_fallback": dead})
        print(
            f"[{phase}] layer={layer_idx + 1}/{len(text_model.layers)} zero_hit={dead} "
            f"layer_s={time.time() - layer_started:.1f} elapsed_s={time.time() - started:.1f}",
            flush=True,
        )
    average_bits, total_bits, total_weights = weighted_bits(state, specs)
    metadata = {
        "phase": phase,
        "calibset": str(args.calibset),
        "samples": len(trainloader),
        "seqlen": args.seqlen,
        "average_bits": average_bits,
        "total_logical_bits": total_bits,
        "quantized_active_weights": total_weights,
        "target_group_weights": args.target_group_weights,
        "level_ladder": list(LEVEL_LADDER),
        "fixed_bf16_layers_0base": sorted(args.fixed_bf16_layers),
        "layer_stats": layer_stats,
        "elapsed_seconds": time.time() - started,
    }
    if save_model:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        precision_payload = {
            "scheme": (
                "bidirectional_activation_rd_fixed_sensitive_bf16"
                if args.fixed_bf16_layers
                else "bidirectional_activation_rd_fixed_budget"
            ),
            "initial_sensitive_layers_0base": sorted(SENSITIVE_LAYERS),
            "level_ladder": list(LEVEL_LADDER),
            "fixed_bf16_layers_0base": sorted(args.fixed_bf16_layers),
            "initial_checkpoint": (
                None if args.initial_checkpoint is None else str(args.initial_checkpoint)
            ),
            "effective_average_bits_active_quantized_weights": average_bits,
            "packed_bpw_estimate": packed_bpw_estimate(state, specs),
            "initial_budget_bits": state["initial_budget_bits"],
            "resolved_levels": {
                key: resolved_level(state, key, int(spec["layer"]))
                for key, spec in specs.items()
            },
            "group_specs": specs,
            "state": state,
            "metadata": metadata,
            "activation_bits": 16,
            "twla_stages": [1, 2],
            "routed_experts_only": True,
            "allocation_unit": args.allocation_unit,
            "gate_up_down_tied_per_expert": args.allocation_unit == "expert",
            "expert_prior": (
                None if args.expert_prior is None else str(args.expert_prior)
            ),
            "prior_exponent": args.prior_exponent,
            "zero_hit_experts": "assigned precision via Euclidean Stage-I fallback",
            "packed": False,
            "pruning": False,
        }
        atomic_json(args.out_dir / "precision_map.json", precision_payload)
        print(f"[{phase}] saving checkpoint={args.out_dir}", flush=True)
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        (args.out_dir / ".quant_done").touch()
    del hidden_states, common_kwargs, text_model, model
    gc.collect()
    torch.cuda.empty_cache()
    return specs, actions_up, actions_down, metadata


def select_swaps(
    state: dict,
    specs: dict[str, dict],
    actions_up: list[dict],
    actions_down: list[dict],
    args,
) -> dict:
    average_before, total_before, total_weights = weighted_bits(state, specs)
    layer_weights: dict[int, int] = {}
    for spec in specs.values():
        layer = int(spec["layer"])
        layer_weights[layer] = layer_weights.get(layer, 0) + int(spec["num_weights"])
    if state["initial_budget_bits"] is None:
        # With immutable BF16 anchors, every allocatable layer starts at the
        # bottom of the 3/4/8/16 ladder.  A strictly fixed ternary budget has
        # no possible first demotion and therefore cannot bootstrap an
        # exchange.  Reserve one small layer-local tranche for the first
        # promotion; all subsequent sweeps redistribute inside this fixed
        # enlarged budget.
        bootstrap = 0.0
        if args.bootstrap_exchange_bits > 0 and layer_weights:
            ordered = sorted(layer_weights.values())
            representative_layer_weights = ordered[len(ordered) // 2]
            # Add one group's maximum 3->4 discretization increment so the
            # bundle that crosses the target is never rejected solely due to
            # row-group granularity.
            max_group_weights = max(int(spec["num_weights"]) for spec in specs.values())
            granularity_margin = max_group_weights * (
                math.log2(LEVEL_LADDER[1]) - math.log2(LEVEL_LADDER[0])
            )
            bootstrap = (
                float(args.bootstrap_exchange_bits) * representative_layer_weights
                + granularity_margin
            )
        state["initial_budget_bits"] = total_before + bootstrap
        state["bootstrap_budget_bits"] = bootstrap
    fixed_budget = float(state["initial_budget_bits"])
    bank_before = max(0.0, fixed_budget - total_before)
    up_by_layer: dict[int, list[dict]] = {}
    down_by_layer: dict[int, list[dict]] = {}
    for action in actions_up:
        if action["distortion_gain"] > 0:
            up_by_layer.setdefault(int(action["layer"]), []).append(action)
    for action in actions_down:
        down_by_layer.setdefault(int(action["layer"]), []).append(action)

    # One outer sweep performs ONE interpretable layer-to-layer exchange.
    # Within each selected layer, equal-weight row groups implement an
    # approximately 0.25-bit layer-average move while retaining real
    # 3/4/8/16-level codebooks for each materialized group.
    promotion_bundles: list[dict] = []
    for layer, layer_actions in up_by_layer.items():
        target = float(args.round_exchange_bits) * layer_weights[layer]
        chosen: list[dict] = []
        spent = gain = 0.0
        for action in sorted(
            layer_actions,
            key=lambda item: (-item["gain_per_bit"], item["group"]),
        ):
            chosen.append(action)
            spent += float(action["spent_bits"])
            gain += float(action["distortion_gain"])
            if spent + args.score_epsilon >= target:
                break
        if chosen and spent > 0:
            promotion_bundles.append(
                {"layer": layer, "actions": chosen, "spent_bits": spent, "gain": gain}
            )

    best: dict | None = None
    for promotion in promotion_bundles:
        required = max(0.0, float(promotion["spent_bits"]) - bank_before)
        if required <= args.score_epsilon:
            candidates = [(None, [], 0.0, 0.0)]
        else:
            candidates = []
            for layer, layer_actions in down_by_layer.items():
                if layer == promotion["layer"]:
                    continue
                chosen: list[dict] = []
                freed = cost = 0.0
                for action in sorted(
                    layer_actions,
                    key=lambda item: (item["cost_per_bit"], item["group"]),
                ):
                    chosen.append(action)
                    freed += float(action["freed_bits"])
                    cost += float(action["distortion_cost"])
                    if freed + args.score_epsilon >= required:
                        break
                if freed + args.score_epsilon >= required:
                    candidates.append((layer, chosen, freed, cost))
        for demotion_layer, selected_down, freed, cost in candidates:
            net = float(promotion["gain"]) - cost
            required_net = args.min_relative_gain * max(
                float(promotion["gain"]), cost, args.score_epsilon
            )
            if net <= required_net:
                continue
            proposal = {
                "promotion_layer": int(promotion["layer"]),
                "demotion_layer": None if demotion_layer is None else int(demotion_layer),
                "selected_up": promotion["actions"],
                "selected_down": selected_down,
                "spent_bits": float(promotion["spent_bits"]),
                "freed_bits": freed,
                "predicted_gain": float(promotion["gain"]),
                "predicted_cost": cost,
                "predicted_net": net,
            }
            if best is None or (proposal["predicted_net"], -proposal["spent_bits"]) > (
                best["predicted_net"], -best["spent_bits"]
            ):
                best = proposal

    if best is None:
        return {
            "changed": False,
            "reason": "no_profitable_budget_neutral_swap",
            "average_before": average_before,
            "average_after": average_before,
            "selected_up": [],
            "selected_down": [],
        }

    selected_up = best["selected_up"]
    selected_down = best["selected_down"]
    current_resolved = {
        key: resolved_level(state, key, int(spec["layer"])) for key, spec in specs.items()
    }
    current_hash = map_hash(current_resolved)
    seen_hashes = set(state.get("seen_map_hashes", []))
    seen_hashes.add(current_hash)
    proposed = dict(current_resolved)
    for action in selected_down:
        proposed[action["group"]] = int(action["to_levels"])
    for action in selected_up:
        proposed[action["group"]] = int(action["to_levels"])
    proposed_hash = map_hash(proposed)
    if proposed_hash in seen_hashes:
        return {
            "changed": False,
            "reason": "precision_map_cycle_prevented",
            "average_before": average_before,
            "average_after": average_before,
            "selected_up": [],
            "selected_down": [],
        }
    state["levels"] = proposed
    state["seen_map_hashes"] = sorted(seen_hashes | {proposed_hash})
    average_after, total_after, _ = weighted_bits(state, specs)
    if total_after > fixed_budget + max(1.0, args.score_epsilon):
        raise RuntimeError(
            f"allocator violated fixed budget: after={total_after} budget={fixed_budget}"
        )
    return {
        "changed": True,
        "reason": "profitable_swaps_applied",
        "average_before": average_before,
        "average_after": average_after,
        "total_bits_before": total_before,
        "total_bits_after": total_after,
        "fixed_budget_bits": fixed_budget,
        "promotion_layer": best["promotion_layer"],
        "demotion_layer": best["demotion_layer"],
        "selected_up_count": len(selected_up),
        "selected_down_count": len(selected_down),
        "freed_bits": best["freed_bits"],
        "spent_bits": best["spent_bits"],
        "unspent_bank_bits": fixed_budget - total_after,
        "predicted_distortion_gain": best["predicted_gain"],
        "predicted_distortion_cost": best["predicted_cost"],
        "predicted_net_improvement": best["predicted_net"],
        "selected_up": selected_up,
        "selected_down": selected_down,
    }


def _expert_action_bundles(actions: list[dict], direction: str, args) -> list[dict]:
    """Tie gate_up_proj and down_proj into one indivisible expert action."""
    grouped: dict[tuple[int, int, int, int], list[dict]] = {}
    for action in actions:
        key = (
            int(action["layer"]),
            int(action["expert"]),
            int(action["from_levels"]),
            int(action["to_levels"]),
        )
        grouped.setdefault(key, []).append(action)
    bundles = []
    for (layer, expert, from_level, to_level), members in grouped.items():
        matrices = {member["matrix"] for member in members}
        if matrices != {"gate_up_proj", "down_proj"}:
            # With one row-group per matrix this normally means the expert
            # had no calibration evidence; it must not be ranked from a
            # half-observed MLP.
            continue
        expert_key = f"L{layer:02d}.E{expert:03d}"
        prior = float(args.expert_prior_map.get(expert_key, 1.0))
        multiplier = prior ** float(args.prior_exponent)
        if direction == "up":
            bits = sum(float(member["spent_bits"]) for member in members)
            raw = sum(float(member["distortion_gain"]) for member in members)
            score = raw * multiplier
        else:
            bits = sum(float(member["freed_bits"]) for member in members)
            raw = sum(float(member["distortion_cost"]) for member in members)
            score = raw * multiplier
        bundles.append(
            {
                "expert_key": expert_key,
                "layer": layer,
                "expert": expert,
                "from_levels": from_level,
                "to_levels": to_level,
                "members": members,
                "bits": bits,
                "raw_score": raw,
                "score": score,
                "score_per_bit": score / max(bits, args.score_epsilon),
                "prior": prior,
            }
        )
    return bundles


def select_expert_swaps(
    state: dict,
    specs: dict[str, dict],
    actions_up: list[dict],
    actions_down: list[dict],
    args,
) -> dict:
    """Multi-action low->high / high->low exchange at routed-expert grain.

    Unlike the legacy layer-bundle selector, the first sweep can promote
    thousands of experts into an explicit global budget.  Later sweeps may
    demote several cheap experts to fund one valuable 4->8 or 8->16 step.
    This removes the previous two-round/one-layer bottleneck while keeping
    both fused expert matrices tied at exactly the same precision.
    """
    average_before, total_before, total_weights = weighted_bits(state, specs)
    target_budget = float(args.target_average_bits) * total_weights
    if state["initial_budget_bits"] is None:
        if target_budget + args.score_epsilon < total_before:
            raise ValueError(
                f"target average {args.target_average_bits} is below initial "
                f"average {average_before}"
            )
        state["initial_budget_bits"] = target_budget
        state["bootstrap_budget_bits"] = target_budget - total_before
    fixed_budget = float(state["initial_budget_bits"])
    bank = max(0.0, fixed_budget - total_before)
    sweep_cap = max(0.0, float(args.round_exchange_bits) * total_weights)
    ups = sorted(
        _expert_action_bundles(actions_up, "up", args),
        key=lambda item: (-item["score_per_bit"], -item["score"], item["expert_key"]),
    )
    downs = sorted(
        _expert_action_bundles(actions_down, "down", args),
        key=lambda item: (item["score_per_bit"], item["score"], item["expert_key"]),
    )
    selected_up: list[dict] = []
    selected_down: list[dict] = []
    touched: set[str] = set()
    promotion_bits = 0.0
    predicted_gain = 0.0
    predicted_cost = 0.0

    # Phase A: low -> high, consuming any unspent global budget.  Continue
    # past an oversized action because a later smaller action may still fit.
    for up in ups:
        if up["expert_key"] in touched:
            continue
        if promotion_bits + up["bits"] > sweep_cap + args.score_epsilon:
            continue
        if up["bits"] > bank + args.score_epsilon:
            continue
        selected_up.append(up)
        touched.add(up["expert_key"])
        bank -= up["bits"]
        promotion_bits += up["bits"]
        predicted_gain += up["score"]

    # Phase B: high -> low creates a bank, immediately followed by the best
    # still-available low -> high move.  Multiple demotions can fund one
    # high-value promotion and many such exchanges may occur in one sweep.
    used_down: set[str] = set()
    for up in ups:
        if up["expert_key"] in touched:
            continue
        if promotion_bits + up["bits"] > sweep_cap + args.score_epsilon:
            continue
        needed = max(0.0, up["bits"] - bank)
        funding = []
        freed = 0.0
        cost = 0.0
        if needed > args.score_epsilon:
            for down in downs:
                key = down["expert_key"]
                if key in touched or key in used_down or key == up["expert_key"]:
                    continue
                funding.append(down)
                freed += down["bits"]
                cost += down["score"]
                if freed + args.score_epsilon >= needed:
                    break
        if freed + bank + args.score_epsilon < up["bits"]:
            continue
        required_net = args.min_relative_gain * max(up["score"], cost, args.score_epsilon)
        if up["score"] - cost <= required_net:
            continue
        for down in funding:
            selected_down.append(down)
            used_down.add(down["expert_key"])
            touched.add(down["expert_key"])
        selected_up.append(up)
        touched.add(up["expert_key"])
        bank += freed - up["bits"]
        promotion_bits += up["bits"]
        predicted_gain += up["score"]
        predicted_cost += cost

    if not selected_up and not selected_down:
        return {
            "changed": False,
            "reason": "no_profitable_expert_granular_swap",
            "average_before": average_before,
            "average_after": average_before,
            "packed_bpw_estimate": packed_bpw_estimate(state, specs),
            "selected_up": [],
            "selected_down": [],
        }

    proposed = {
        key: resolved_level(state, key, int(spec["layer"])) for key, spec in specs.items()
    }
    current_hash = map_hash(proposed)
    seen = set(state.get("seen_map_hashes", [])) | {current_hash}
    for bundle in selected_down:
        for member in bundle["members"]:
            proposed[member["group"]] = int(member["to_levels"])
    for bundle in selected_up:
        for member in bundle["members"]:
            proposed[member["group"]] = int(member["to_levels"])
    proposed_hash = map_hash(proposed)
    if proposed_hash in seen:
        return {
            "changed": False,
            "reason": "expert_precision_map_cycle_prevented",
            "average_before": average_before,
            "average_after": average_before,
            "packed_bpw_estimate": packed_bpw_estimate(state, specs),
            "selected_up": [],
            "selected_down": [],
        }
    state["levels"] = proposed
    state["seen_map_hashes"] = sorted(seen | {proposed_hash})
    average_after, total_after, _ = weighted_bits(state, specs)
    packed_after = packed_bpw_estimate(state, specs)
    if total_after > fixed_budget + max(1.0, args.score_epsilon):
        raise RuntimeError(
            f"expert allocator violated fixed budget: after={total_after} budget={fixed_budget}"
        )
    if packed_after > 2.0 + 1e-9:
        raise RuntimeError(f"packed routed-expert BPW target violated: {packed_after}")
    return {
        "changed": True,
        "reason": "expert_granular_profitable_swaps_applied",
        "average_before": average_before,
        "average_after": average_after,
        "packed_bpw_estimate": packed_after,
        "total_bits_before": total_before,
        "total_bits_after": total_after,
        "fixed_budget_bits": fixed_budget,
        "target_average_bits": args.target_average_bits,
        "selected_up_count": len(selected_up),
        "selected_down_count": len(selected_down),
        "selected_up_matrix_groups": sum(len(item["members"]) for item in selected_up),
        "selected_down_matrix_groups": sum(len(item["members"]) for item in selected_down),
        "promotion_bits": promotion_bits,
        "unspent_bank_bits": fixed_budget - total_after,
        "predicted_distortion_gain": predicted_gain,
        "predicted_distortion_cost": predicted_cost,
        "predicted_net_improvement": predicted_gain - predicted_cost,
        "selected_up": selected_up,
        "selected_down": selected_down,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--calibset", type=Path, default=DEFAULT_CALIBSET)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--euclidean_iters", type=int, default=15)
    parser.add_argument("--target_group_weights", type=int, default=262144)
    parser.add_argument("--round_exchange_bits", type=float, default=0.25)
    parser.add_argument("--min_relative_gain", type=float, default=1e-6)
    parser.add_argument("--score_epsilon", type=float, default=1e-30)
    parser.add_argument("--max_sweeps", type=int, default=5)
    parser.add_argument(
        "--allocation_unit",
        choices=("layer", "expert"),
        default="layer",
        help="Legacy layer bundles or tied gate_up+down routed-expert bundles.",
    )
    parser.add_argument(
        "--initial_all_ternary",
        action="store_true",
        help="Start every routed expert at 3 levels instead of the legacy sensitive-layer prior.",
    )
    parser.add_argument(
        "--target_average_bits",
        type=float,
        default=1.94,
        help="Global logical routed-expert budget used by --allocation_unit expert.",
    )
    parser.add_argument(
        "--expert_prior",
        type=Path,
        default=None,
        help="JSON produced by bipea_2bit_tools.py select; used only as a multiplier on RD.",
    )
    parser.add_argument(
        "--prior_exponent",
        type=float,
        default=1.0,
        help="Strength of the empirically selected expert-sensitivity prior (0 disables it).",
    )
    parser.add_argument(
        "--fixed_bf16_layers",
        default="",
        help="Comma-separated 0-based decoder layers excluded from quantization/allocation.",
    )
    parser.add_argument(
        "--bootstrap_exchange_bits",
        type=float,
        default=0.0,
        help="One-time layer-average bit tranche used when all movable groups start ternary.",
    )
    parser.add_argument("--initial_checkpoint", type=Path, default=None)
    parser.add_argument("--num_threads", type=int, default=None)
    args = parser.parse_args()
    args.expert_prior_map = {}
    if args.expert_prior is not None:
        prior_payload = json.loads(args.expert_prior.read_text())
        if int(prior_payload.get("gpqa_records", -1)) != 0:
            raise ValueError("expert prior is not certified GPQA-clean")
        args.expert_prior_map = {
            str(key): float(value) for key, value in prior_payload["prior"].items()
        }
    args.fixed_bf16_layers = frozenset(
        int(item) for item in args.fixed_bf16_layers.split(",") if item.strip()
    )
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if args.max_sweeps < 1:
        raise ValueError("--max_sweeps must be positive")
    if args.allocation_unit == "expert":
        if not args.initial_all_ternary:
            raise ValueError("expert allocation requires --initial_all_ternary")
        if args.target_average_bits > 1.94 + 1e-12:
            raise ValueError(
                "--target_average_bits > 1.94 lacks the conservative packed-BPW<=2 margin"
            )
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_dir() or not (
            args.initial_checkpoint / ".quant_done"
        ).exists():
            raise FileNotFoundError(
                f"initial checkpoint is missing or incomplete: {args.initial_checkpoint}"
            )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "state.json"
    state = load_state(state_path)
    state["initial_all_ternary"] = bool(args.initial_all_ternary)
    run_config = {
        "model_dir": str(args.model_dir.resolve()),
        "calibset": str(args.calibset.resolve()),
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "euclidean_iters": args.euclidean_iters,
        "target_group_weights": args.target_group_weights,
        "round_exchange_bits": args.round_exchange_bits,
        "max_sweeps": args.max_sweeps,
        "allocation_unit": args.allocation_unit,
        "initial_all_ternary": args.initial_all_ternary,
        "target_average_bits": args.target_average_bits,
        "expert_prior": (
            None if args.expert_prior is None else str(args.expert_prior.resolve())
        ),
        "prior_exponent": args.prior_exponent,
        "level_ladder": list(LEVEL_LADDER),
        "sensitive_layers": sorted(SENSITIVE_LAYERS),
        "fixed_bf16_layers": sorted(args.fixed_bf16_layers),
        "bootstrap_exchange_bits": args.bootstrap_exchange_bits,
        "initial_checkpoint": (
            None if args.initial_checkpoint is None else str(args.initial_checkpoint.resolve())
        ),
    }
    if state.get("run_config") is None:
        state["run_config"] = run_config
        atomic_json(state_path, state)
    elif state["run_config"] != run_config:
        raise ValueError(
            "work_dir contains state from a different configuration; "
            "use a fresh work_dir rather than silently resuming it"
        )
    print(
        f"Bidirectional RD start: sensitive={sorted(SENSITIVE_LAYERS)} "
        f"fixed_bf16={sorted(args.fixed_bf16_layers)} "
        f"ladder={LEVEL_LADDER} max_sweeps={args.max_sweeps} "
        f"exchange_cap={args.round_exchange_bits} avg-bit/sweep pruning=OFF",
        flush=True,
    )
    while state["status"] == "scoring" and state["sweeps_completed"] < args.max_sweeps:
        specs, actions_up, actions_down, metadata = run_sweep(
            state, args, score_neighbors=True, save_model=False
        )
        selector = select_expert_swaps if args.allocation_unit == "expert" else select_swaps
        decision = selector(state, specs, actions_up, actions_down, args)
        state["sweeps_completed"] += 1
        state["history"].append(
            {
                "sweep": state["sweeps_completed"],
                **{
                    key: value
                    for key, value in decision.items()
                    if key not in {"selected_up", "selected_down"}
                },
                "calibration_elapsed_seconds": metadata["elapsed_seconds"],
            }
        )
        save_gzip_json(
            args.work_dir / f"sweep_{state['sweeps_completed']:02d}_actions.json.gz",
            {
                "decision": decision,
                "actions_up": actions_up,
                "actions_down": actions_down,
                "metadata": metadata,
            },
        )
        print(
            f"[allocation] sweep={state['sweeps_completed']} "
            f"avg={decision['average_before']:.6f}->{decision['average_after']:.6f} "
            f"up={decision.get('selected_up_count', 0)} "
            f"down={decision.get('selected_down_count', 0)} "
            f"reason={decision['reason']}",
            flush=True,
        )
        if not decision["changed"]:
            state["status"] = "ready_to_materialize"
            state["stop_reason"] = decision["reason"]
        atomic_json(state_path, state)
    if state["status"] == "scoring":
        state["status"] = "ready_to_materialize"
        state["stop_reason"] = "max_sweeps"
        atomic_json(state_path, state)
    if state["status"] == "ready_to_materialize":
        specs, _up, _down, metadata = run_sweep(
            state, args, score_neighbors=False, save_model=True
        )
        average, total_bits, total_weights = weighted_bits(state, specs)
        completed_state = dict(state)
        completed_state["status"] = "complete"
        completed_state["final_average_bits"] = average
        completed_state["final_packed_bpw_estimate"] = packed_bpw_estimate(state, specs)
        completed_state["final_total_logical_bits"] = total_bits
        completed_state["final_quantized_active_weights"] = total_weights
        completed_state["final_materialization_seconds"] = metadata["elapsed_seconds"]
        # Commit the externally checked summary first.  If interrupted
        # before state.json is committed, resume safely repeats final
        # materialization instead of getting stuck in a false complete state.
        atomic_json(args.out_dir / "bidirectional_rd_summary.json", completed_state)
        atomic_json(state_path, completed_state)
        state = completed_state
    elif state["status"] == "complete":
        # Idempotently repair a missing public summary from an interrupted
        # older launch whose state had already reached complete.
        atomic_json(args.out_dir / "bidirectional_rd_summary.json", state)
    print(
        f"ALL DONE status={state['status']} sweeps={state['sweeps_completed']} "
        f"average_bits={state.get('final_average_bits')} checkpoint={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

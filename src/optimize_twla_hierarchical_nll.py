#!/usr/bin/env python3
"""Alternating validation-NLL optimization of routed-expert TWLA levels.

The optimization unit is one (layer, routed expert) pair; gate_up_proj and
down_proj always move together by one codebook level.  Branch decisions use
the GPQA-free probe validation split, while the selected leaf must improve the
independent convergence split before it is committed.  The final split is
read exactly once after optimization and never influences allocation.
"""

from __future__ import annotations

from streaming_load import load_pretrained_streaming

import argparse
import gc
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from quantize.E2M_ATQ_asymmetric_codebook import decode_asymmetric_representation

try:
    from branch_bound_search import GroupMeasurement, maximize_with_branch_bound
except ImportError:  # Backward-compatible standalone V2 code bundle.
    GroupMeasurement = None
    maximize_with_branch_bound = None

try:
    from taylor_fisher_proxy import (
        collect_taylor_fisher_scores,
        objective_gain_from_proxy,
    )
except ImportError:  # Only required by the proxy_batch search mode.
    collect_taylor_fisher_scores = None
    objective_gain_from_proxy = None


ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
# This module is imported for its bank installer and validation-NLL helpers (fisher_anchor.py select); its own CLI
# (the earlier validation-NLL search) takes every path explicitly.
DEFAULT_INITIAL = None
DEFAULT_BANK = None
DEFAULT_DATA = None
DEFAULT_ANCHOR_PRIOR = None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def bank_path(bank_dir: Path, layer: int, level: int) -> Path:
    return bank_dir / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"


def contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
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


@torch.no_grad()
def install_range(experts, path: Path, level: int, start: int, end: int, device: str) -> None:
    """Decode one contiguous expert range directly from a memory-mapped bank."""
    center = (level - 1) / 2.0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        asymmetric = "gate_up_codebook" in handle.keys()
        gu_codes_slice = handle.get_slice("gate_up_codes")
        gu_mu_slice = handle.get_slice("gate_up_mu")
        gu_alpha_slice = handle.get_slice("gate_up_alpha")
        dn_codes_slice = handle.get_slice("down_codes")
        dn_mu_slice = handle.get_slice("down_mu")
        dn_alpha_slice = handle.get_slice("down_alpha")
        if asymmetric:
            gu_codebook_slice = handle.get_slice("gate_up_codebook")
            gu_left_slice = handle.get_slice("gate_up_rotation_left")
            gu_right_slice = handle.get_slice("gate_up_rotation_right")
            dn_codebook_slice = handle.get_slice("down_codebook")
            dn_left_slice = handle.get_slice("down_rotation_left")
            dn_right_slice = handle.get_slice("down_rotation_right")
        for chunk_start in range(start, end, 32):
            chunk_end = min(chunk_start + 32, end)
            codes = gu_codes_slice[chunk_start:chunk_end].to(device)
            mu = gu_mu_slice[chunk_start:chunk_end].to(device).float()
            alpha = gu_alpha_slice[chunk_start:chunk_end].to(device).float()
            if asymmetric:
                reconstructed = decode_asymmetric_representation(
                    codes,
                    gu_codebook_slice[chunk_start:chunk_end].to(device).float(),
                    mu,
                    alpha,
                    gu_left_slice[chunk_start:chunk_end].to(device).float(),
                    gu_right_slice[chunk_start:chunk_end].to(device).float(),
                )
            else:
                reconstructed = mu[:, :, None] + alpha[:, :, None] * (
                    codes.float() - center
                )
            experts.gate_up_proj.data[chunk_start:chunk_end].copy_(
                reconstructed.to(experts.gate_up_proj.dtype)
            )
            del codes, mu, alpha, reconstructed

            codes = dn_codes_slice[chunk_start:chunk_end].to(device)
            mu = dn_mu_slice[chunk_start:chunk_end].to(device).float()
            alpha = dn_alpha_slice[chunk_start:chunk_end].to(device).float()
            if asymmetric:
                reconstructed = decode_asymmetric_representation(
                    codes,
                    dn_codebook_slice[chunk_start:chunk_end].to(device).float(),
                    mu,
                    alpha,
                    dn_left_slice[chunk_start:chunk_end].to(device).float(),
                    dn_right_slice[chunk_start:chunk_end].to(device).float(),
                )
            else:
                reconstructed = mu[:, :, None] + alpha[:, :, None] * (
                    codes.float() - center
                )
            experts.down_proj.data[chunk_start:chunk_end].copy_(
                reconstructed.to(experts.down_proj.dtype)
            )
            del codes, mu, alpha, reconstructed


@torch.no_grad()
def install_assignments(
    text_model,
    assignments: dict[tuple[int, int], int],
    bank_dir: Path,
    device: str,
) -> None:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (layer, expert), level in assignments.items():
        grouped[(int(layer), int(level))].append(int(expert))
    for (layer, level), experts_to_set in sorted(grouped.items()):
        path = bank_path(bank_dir, layer, level)
        if not path.exists():
            raise FileNotFoundError(f"missing TWLA bank entry: {path}")
        module = text_model.layers[layer].mlp.experts
        for start, end in contiguous_runs(experts_to_set):
            install_range(module, path, level, start, end, device)
    torch.cuda.empty_cache()


def load_validation(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"input_ids", "attention_mask", "loss_mask"}
    if not required.issubset(payload):
        raise ValueError(f"{path} lacks {sorted(required - set(payload))}")
    return payload


def load_stratified_screen(
    tensor_path: Path,
    metadata_path: Path,
    divisor: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Take a deterministic source-stratified subset of a validation split.

    The full probe split remains the authority for leaf ranking.  This smaller
    split is used only for broad branch screening, where evaluating the entire
    probe for every branch would dominate the search cost.
    """
    if divisor < 1:
        raise ValueError("screen divisor must be positive")
    payload = load_validation(tensor_path)
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    if len(rows) != int(payload["input_ids"].shape[0]):
        raise ValueError("probe tensor and JSONL metadata have different row counts")
    totals = Counter(str(row["source"]) for row in rows)
    quotas = {source: max(1, count // divisor) for source, count in totals.items()}
    used: Counter[str] = Counter()
    indices: list[int] = []
    for index, row in enumerate(rows):
        source = str(row["source"])
        if used[source] < quotas[source]:
            indices.append(index)
            used[source] += 1
    if used != Counter(quotas):
        raise RuntimeError(f"could not construct screen split: used={used} quotas={quotas}")
    selected = torch.tensor(indices, dtype=torch.long)
    tensor_keys = ("input_ids", "attention_mask", "loss_mask")
    return {
        key: payload[key].index_select(0, selected) for key in tensor_keys
    }, indices


@torch.inference_mode()
def validation_nll(model, payload: dict[str, torch.Tensor], device: str) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    rows = int(payload["input_ids"].shape[0])
    for row_idx in range(rows):
        length = int(payload["attention_mask"][row_idx].sum())
        ids = payload["input_ids"][row_idx, :length].to(device).unsqueeze(0)
        loss_mask = payload["loss_mask"][row_idx, 1:length].to(device).bool()
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits[0, :-1]
        labels = ids[0, 1:]
        positions = loss_mask.nonzero(as_tuple=False).flatten()
        for start in range(0, int(positions.numel()), 128):
            chosen = positions[start : start + 128]
            scores = logits.index_select(0, chosen).float()
            targets = labels.index_select(0, chosen)
            selected = scores.gather(1, targets[:, None]).squeeze(1)
            total_nll += float((torch.logsumexp(scores, dim=1) - selected).double().sum().cpu())
            total_tokens += int(chosen.numel())
            del scores, targets, selected
        del ids, loss_mask, output, logits, labels, positions
        if (row_idx + 1) % 8 == 0 or row_idx + 1 == rows:
            print(
                f"[validation] documents={row_idx + 1}/{rows} loss_tokens={total_tokens}",
                flush=True,
            )
    if total_tokens == 0:
        raise RuntimeError("validation split has zero selected loss tokens")
    return total_nll / total_tokens, total_tokens


def average_bits(levels: list[list[int]]) -> float:
    values = [math.log2(level) for row in levels for level in row]
    return sum(values) / len(values)


def smooth_dual_update(
    lambda_value: float,
    residual: float,
    tolerance: float,
    base_lr: float,
    min_lr_ratio: float,
    full_lr_error: float,
    max_step: float,
    lambda_min: float,
    lambda_max: float,
) -> dict[str, float | bool]:
    """Return a deadbanded, distance-adaptive lambda controller update.

    The controller is continuous at the tolerance boundary: only the residual
    outside the deadband contributes to the step.  Both the effective learning
    rate and the resulting lambda step shrink smoothly near the bit target.
    """

    excess_error = max(abs(residual) - tolerance, 0.0)
    if excess_error == 0.0:
        effective_lr = 0.0
        raw_step = 0.0
    else:
        scale_denominator = max(full_lr_error - tolerance, 1e-12)
        distance_scale = min(excess_error / scale_denominator, 1.0)
        effective_lr = base_lr * (
            min_lr_ratio + (1.0 - min_lr_ratio) * distance_scale
        )
        raw_step = math.copysign(effective_lr * excess_error, residual)
    clipped_step = max(-max_step, min(max_step, raw_step))
    updated = max(lambda_min, min(lambda_max, lambda_value + clipped_step))
    actual_step = updated - lambda_value
    return {
        "excess_error": excess_error,
        "effective_lr": effective_lr,
        "raw_step": raw_step,
        "clipped_step": clipped_step,
        "actual_step": actual_step,
        "lambda_after": updated,
        "changed": abs(actual_step) > 1e-12,
    }


def clipped_dual_update(
    lambda_value: float,
    residual: float,
    tolerance: float,
    response_gain: float,
    min_step: float,
    max_step: float,
    lambda_min: float,
    lambda_max: float,
) -> dict[str, float | bool]:
    """Deadbanded dual update that is fast far away and bounded near target.

    Outside the target deadband, the step magnitude is

        min(max_step, max(min_step, response_gain * (abs(residual)-tolerance))).

    The minimum step prevents repeated expensive searches at an unchanged
    allocation, while the maximum step prevents a large initial residual from
    producing a destructive lambda jump.
    """

    excess_error = max(abs(residual) - tolerance, 0.0)
    if excess_error == 0.0:
        requested_magnitude = 0.0
        raw_step = 0.0
    else:
        requested_magnitude = max(min_step, response_gain * excess_error)
        raw_step = math.copysign(min(max_step, requested_magnitude), residual)
    updated = max(lambda_min, min(lambda_max, lambda_value + raw_step))
    actual_step = updated - lambda_value
    return {
        "excess_error": excess_error,
        "effective_lr": response_gain,
        "requested_magnitude": requested_magnitude,
        "raw_step": raw_step,
        "clipped_step": raw_step,
        "actual_step": actual_step,
        "lambda_after": updated,
        "changed": abs(actual_step) > 1e-12,
    }


def objective(
    nll: float,
    levels: list[list[int]],
    lambda_value: float,
    reference_level: int = 5,
) -> tuple[float, float, float]:
    bits = average_bits(levels)
    increase = bits - math.log2(reference_level)
    return nll + lambda_value * increase, bits, increase


def moved_map(
    levels: list[list[int]],
    units: list[tuple[int, int]],
    direction: str,
    min_level: int,
    max_level: int,
) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    candidate = [row[:] for row in levels]
    assignments = {}
    step = -1 if direction == "down" else 1
    for layer, expert in units:
        current = int(levels[layer][expert])
        target = current + step
        if target < min_level or target > max_level:
            continue
        candidate[layer][expert] = target
        assignments[(layer, expert)] = target
    return candidate, assignments


def assigned_map(
    levels: list[list[int]],
    assignments: dict[tuple[int, int], int],
    min_level: int,
    max_level: int,
) -> list[list[int]]:
    """Return a copied level map with arbitrary one-step assignments applied."""

    candidate = [row[:] for row in levels]
    for (layer, expert), target in assignments.items():
        current = int(levels[layer][expert])
        target = int(target)
        if not min_level <= target <= max_level:
            raise ValueError(f"assignment {(layer, expert)} -> K={target} is out of range")
        if abs(target - current) != 1:
            raise ValueError(
                f"assignment {(layer, expert)} must move exactly one level: "
                f"K={current} -> K={target}"
            )
        candidate[layer][expert] = target
    return candidate


def proxy_loss_delta(item: dict, first_weight: float, fisher_weight: float) -> float:
    """Predicted NLL change for one atomic level move (negative is better)."""

    return (
        first_weight * float(item["first_order_loss_delta"])
        + fisher_weight * float(item["fisher_second_order_loss_delta"])
    )


def build_exchange_bundles(
    down_ranked: list[dict],
    up_ranked: list[dict],
    *,
    total_units: int,
    pool_size: int,
    max_per_side: int,
    bundle_limit: int,
    balance_tolerance: float,
    beam_width: int,
    first_weight: float,
    fisher_weight: float,
) -> list[dict]:
    """Build approximately bit-neutral cross-level down/up exchange bundles.

    Fisher/Taylor values only rank and combine candidates.  Exact validation
    NLL still decides whether a returned bundle is committed.  Unlike a
    same-transition swap, the donor and receiver counts may differ: their
    actual log2(level) changes are balanced by a small bounded beam search.
    """

    if min(total_units, pool_size, max_per_side, bundle_limit, beam_width) < 1:
        return []
    raw_tolerance = balance_tolerance * total_units

    donors = []
    for item in down_ranked:
        current = int(item["current_level"])
        target = int(item["target_level"])
        saving = math.log2(current) - math.log2(target)
        if saving <= 0:
            continue
        damage = proxy_loss_delta(item, first_weight, fisher_weight)
        donors.append({
            "unit": tuple(item["unit"]),
            "from_level": current,
            "to_level": target,
            "bits": saving,
            "loss_delta": damage,
            "efficiency": damage / saving,
        })
    donors.sort(
        key=lambda item: (
            item["efficiency"],
            item["loss_delta"],
            item["unit"],
        )
    )
    donors = donors[:pool_size]

    receivers = []
    for item in up_ranked:
        current = int(item["current_level"])
        target = int(item["target_level"])
        cost = math.log2(target) - math.log2(current)
        if cost <= 0:
            continue
        recovery = -proxy_loss_delta(item, first_weight, fisher_weight)
        if recovery <= 0:
            continue
        receivers.append({
            "unit": tuple(item["unit"]),
            "from_level": current,
            "to_level": target,
            "bits": cost,
            "recovery": recovery,
            "efficiency": recovery / cost,
        })
    receivers.sort(
        key=lambda item: (
            item["efficiency"],
            item["recovery"],
            tuple(-value for value in item["unit"]),
        ),
        reverse=True,
    )
    receivers = receivers[:pool_size]
    if not donors or not receivers:
        return []

    def match_donors(required: float, forbidden: set[tuple[int, int]]) -> list[dict]:
        # Each state is (saved bits, predicted NLL damage, donor indices).
        # Rounding only merges near-identical partial budgets; final admission
        # always checks the unrounded bit balance.
        states: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
        for donor_index, donor in enumerate(donors):
            if donor["unit"] in forbidden:
                continue
            expanded = list(states)
            for saved, damage, chosen in states:
                if len(chosen) >= max_per_side:
                    continue
                new_saved = saved + float(donor["bits"])
                if new_saved > required + raw_tolerance:
                    continue
                expanded.append((
                    new_saved,
                    damage + float(donor["loss_delta"]),
                    chosen + (donor_index,),
                ))
            best_by_bucket: dict[tuple[int, int], tuple[float, float, tuple[int, ...]]] = {}
            for state in expanded:
                saved, damage, chosen = state
                key = (len(chosen), round(saved / 0.0025))
                incumbent = best_by_bucket.get(key)
                if incumbent is None or damage < incumbent[1]:
                    best_by_bucket[key] = state
            states = sorted(
                best_by_bucket.values(),
                key=lambda state: (
                    abs(required - state[0]),
                    state[1],
                    len(state[2]),
                ),
            )[:beam_width]
        feasible = [
            state for state in states
            if state[2] and abs(state[0] - required) <= raw_tolerance
        ]
        return sorted(feasible, key=lambda state: (state[1], abs(state[0] - required)))[:4]

    bundles: list[dict] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    # Small offsets preserve alternative receiver compositions when the best
    # prefix has no feasible cross-level bit match.
    for offset in range(min(4, len(receivers))):
        for count in range(1, min(max_per_side, len(receivers) - offset) + 1):
            chosen_receivers = receivers[offset : offset + count]
            receiver_units = {item["unit"] for item in chosen_receivers}
            required = sum(float(item["bits"]) for item in chosen_receivers)
            recovery = sum(float(item["recovery"]) for item in chosen_receivers)
            for saved, damage, donor_indices in match_donors(required, receiver_units):
                chosen_donors = [donors[index] for index in donor_indices]
                predicted_gain = recovery - damage
                if predicted_gain <= 0:
                    continue
                assignments = {
                    item["unit"]: int(item["to_level"]) for item in chosen_donors
                }
                assignments.update({
                    item["unit"]: int(item["to_level"])
                    for item in chosen_receivers
                })
                signature = tuple(sorted(
                    (unit[0], unit[1], target)
                    for unit, target in assignments.items()
                ))
                if signature in seen:
                    continue
                seen.add(signature)
                bundles.append({
                    "assignments": assignments,
                    "donors": [item["unit"] for item in chosen_donors],
                    "receivers": [item["unit"] for item in chosen_receivers],
                    "donor_savings": saved,
                    "receiver_cost": required,
                    "average_bit_delta": (required - saved) / total_units,
                    "predicted_nll_gain": predicted_gain,
                })
    bundles.sort(
        key=lambda item: (
            item["predicted_nll_gain"],
            -abs(item["average_bit_delta"]),
            -len(item["assignments"]),
        ),
        reverse=True,
    )
    return bundles[:bundle_limit]


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


class Optimizer:
    def __init__(self, args, model, tokenizer, state: dict):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.text_model = model.model.language_model
        self.state = state
        self.probe = load_validation(args.data_dir / "probe_validation.pt")
        self.screen, self.screen_indices = load_stratified_screen(
            args.data_dir / "probe_validation.pt",
            args.data_dir / "probe_validation.jsonl",
            args.screen_divisor,
        )
        self.micro, self.micro_indices = load_stratified_screen(
            args.data_dir / "probe_validation.pt",
            args.data_dir / "probe_validation.jsonl",
            args.proxy_micro_divisor,
        )
        self.convergence = load_validation(args.data_dir / "convergence_validation.pt")
        self.final = load_validation(args.data_dir / "final_validation.pt")
        self.evaluations_path = args.work_dir / "objective_evaluations.jsonl"
        self.anchor_prior: dict[tuple[int, int], float] = {}
        if args.bb_anchor_prior is not None and args.bb_anchor_prior.exists():
            raw = json.loads(args.bb_anchor_prior.read_text()).get("prior", {})
            for key, value in raw.items():
                try:
                    layer_text, expert_text = str(key).split(".")
                    unit = (int(layer_text[1:]), int(expert_text[1:]))
                    self.anchor_prior[unit] = float(value)
                except (TypeError, ValueError, IndexError):
                    continue

    def restore(self, assignments: dict[tuple[int, int], int]) -> None:
        current = {
            unit: int(self.state["levels"][unit[0]][unit[1]]) for unit in assignments
        }
        install_assignments(
            self.text_model, current, self.args.bank_dir, self.args.device
        )

    def score_group(
        self,
        units: list[tuple[int, int]],
        direction: str,
        split_name: str,
        stage: str,
        group_index: int,
    ) -> dict | None:
        candidate_levels, assignments = moved_map(
            self.state["levels"],
            units,
            direction,
            self.args.min_level,
            self.args.max_level,
        )
        if not assignments:
            return None
        install_assignments(
            self.text_model, assignments, self.args.bank_dir, self.args.device
        )
        started = time.time()
        payloads = {
            "micro": self.micro,
            "screen": self.screen,
            "probe": self.probe,
            "convergence": self.convergence,
        }
        if split_name not in payloads:
            raise ValueError(f"unknown validation split {split_name}")
        payload = payloads[split_name]
        try:
            nll, tokens = validation_nll(self.model, payload, self.args.device)
        finally:
            self.restore(assignments)
        value, bits, increase = objective(
            nll,
            candidate_levels,
            self.args.lambda_value,
            self.args.initial_level,
        )
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "attempt": self.state["attempts_completed"] + 1,
            "direction": direction,
            "split": split_name,
            "stage": stage,
            "group_index": group_index,
            "unit_count": len(assignments),
            "unit_first": list(min(assignments)),
            "unit_last": list(max(assignments)),
            "validation_nll": nll,
            "validation_loss_tokens": tokens,
            "average_logical_bits": bits,
            "bit_delta_from_initial_level": increase,
            "initial_level": self.args.initial_level,
            "lambda": self.args.lambda_value,
            "objective": value,
            "elapsed_seconds": time.time() - started,
        }
        append_jsonl(self.evaluations_path, record)
        print(
            f"[objective] direction={direction} stage={stage} group={group_index} "
            f"units={len(assignments)} nll={nll:.8f} bits={bits:.8f} J={value:.8f}",
            flush=True,
        )
        return {**record, "assignments": assignments, "candidate_levels": candidate_levels}

    def score_assignments(
        self,
        assignments: dict[tuple[int, int], int],
        split_name: str,
        stage: str,
        group_index: int,
        direction: str = "exchange",
    ) -> dict | None:
        """Measure an arbitrary set of one-level expert moves exactly."""

        assignments = {
            tuple(unit): int(target)
            for unit, target in assignments.items()
            if int(self.state["levels"][unit[0]][unit[1]]) != int(target)
        }
        if not assignments:
            return None
        candidate_levels = assigned_map(
            self.state["levels"],
            assignments,
            self.args.min_level,
            self.args.max_level,
        )
        payloads = {
            "micro": self.micro,
            "screen": self.screen,
            "probe": self.probe,
            "convergence": self.convergence,
        }
        if split_name not in payloads:
            raise ValueError(f"unknown validation split {split_name}")
        install_assignments(
            self.text_model, assignments, self.args.bank_dir, self.args.device
        )
        started = time.time()
        try:
            nll, tokens = validation_nll(
                self.model, payloads[split_name], self.args.device
            )
        finally:
            self.restore(assignments)
        value, bits, increase = objective(
            nll,
            candidate_levels,
            self.args.lambda_value,
            self.args.initial_level,
        )
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "attempt": self.state["attempts_completed"] + 1,
            "direction": direction,
            "split": split_name,
            "stage": stage,
            "group_index": group_index,
            "unit_count": len(assignments),
            "unit_first": list(min(assignments)),
            "unit_last": list(max(assignments)),
            "validation_nll": nll,
            "validation_loss_tokens": tokens,
            "average_logical_bits": bits,
            "bit_delta_from_initial_level": increase,
            "initial_level": self.args.initial_level,
            "lambda": self.args.lambda_value,
            "objective": value,
            "elapsed_seconds": time.time() - started,
        }
        append_jsonl(self.evaluations_path, record)
        print(
            f"[objective] direction={direction} stage={stage} group={group_index} "
            f"units={len(assignments)} nll={nll:.8f} bits={bits:.8f} J={value:.8f}",
            flush=True,
        )
        return {
            **record,
            "assignments": assignments,
            "candidate_levels": candidate_levels,
        }

    def choose(
        self,
        groups: list[list[tuple[int, int]]],
        direction: str,
        stage: str,
        split_name: str = "probe",
    ) -> dict | None:
        candidates = []
        for group_index, units in enumerate(groups):
            score = self.score_group(
                units, direction, split_name, stage, group_index
            )
            if score is not None:
                candidates.append(score)
        if not candidates:
            return None
        # In both directions, the group producing the lowest resulting J is
        # the branch worth descending.  Choosing the largest-loss branch in
        # high->low would intentionally demote the most damaging experts.
        return min(candidates, key=lambda item: (item["objective"], item["group_index"]))

    def hierarchical_leaf(self, direction: str) -> dict | None:
        layers = list(range(40))
        branch = self.choose(
            [[(layer, expert) for layer in group for expert in range(256)]
             for group in chunks(layers, 20)],
            direction,
            "layers_20",
        )
        if branch is None:
            return None
        selected_layers = sorted({unit[0] for unit in branch["assignments"]})

        for group_size, stage in ((4, "layers_4"), (2, "layers_2"), (1, "layers_1")):
            branch = self.choose(
                [[(layer, expert) for layer in group for expert in range(256)]
                 for group in chunks(selected_layers, group_size)],
                direction,
                stage,
            )
            if branch is None:
                return None
            selected_layers = sorted({unit[0] for unit in branch["assignments"]})

        layer = selected_layers[0]
        selected_experts = list(range(256))
        for group_size in (128, 64, 32, 16, 8, 4, 2, 1):
            branch = self.choose(
                [[(layer, expert) for expert in group]
                 for group in chunks(selected_experts, group_size)],
                direction,
                f"experts_{group_size}",
            )
            if branch is None:
                return None
            selected_experts = sorted({unit[1] for unit in branch["assignments"]})
        return branch

    def verified_beam_leaf(self, direction: str) -> dict | None:
        """Find a leaf with all-layer screening and multipath expert search.

        Unlike the legacy procedure, no irreversible layer binary decision is
        made and expert IDs are not assumed to have a meaningful contiguous
        order.  Every layer is screened individually; several layers and tree
        branches survive; deterministic shuffled trees provide independent
        paths.  Only singleton leaves are ranked on the full probe split.
        """
        layer_scores = []
        for layer in range(40):
            score = self.score_group(
                [(layer, expert) for expert in range(256)],
                direction,
                "screen",
                "verified_layers_all40",
                layer,
            )
            if score is not None:
                layer_scores.append(score)
        if not layer_scores:
            return None
        layer_scores.sort(key=lambda item: (item["objective"], item["group_index"]))
        selected_layers = [
            int(item["group_index"]) for item in layer_scores[: self.args.top_layers]
        ]
        print(f"[verified] selected_layers={selected_layers}", flush=True)

        leaf_screen_scores: dict[tuple[int, int], dict] = {}
        for layer in selected_layers:
            eligible = []
            step = -1 if direction == "down" else 1
            for expert in range(256):
                target = int(self.state["levels"][layer][expert]) + step
                if self.args.min_level <= target <= self.args.max_level:
                    eligible.append((layer, expert))
            if not eligible:
                continue
            for seed_index, base_seed in enumerate(self.args.beam_seeds):
                ordered = eligible[:]
                seed = (
                    int(base_seed)
                    + 1000003 * int(self.state["attempts_completed"])
                    + 1009 * layer
                    + (0 if direction == "down" else 49999)
                )
                random.Random(seed).shuffle(ordered)
                active = [ordered]
                depth = 0
                while active and any(len(node) > 1 for node in active):
                    children: list[list[tuple[int, int]]] = []
                    terminal: list[list[tuple[int, int]]] = []
                    for node in active:
                        if len(node) == 1:
                            terminal.append(node)
                            continue
                        middle = (len(node) + 1) // 2
                        children.extend((node[:middle], node[middle:]))
                    scored = []
                    for child_index, child in enumerate(children):
                        score = self.score_group(
                            child,
                            direction,
                            "screen",
                            f"verified_L{layer:02d}_seed{seed_index}_depth{depth}",
                            child_index,
                        )
                        if score is not None:
                            scored.append(score)
                    scored.sort(key=lambda item: (item["objective"], item["group_index"]))
                    survivors = [
                        list(item["assignments"].keys())
                        for item in scored[: self.args.beam_width]
                    ]
                    active = terminal + survivors
                    depth += 1
                for node in active:
                    if len(node) != 1:
                        continue
                    unit = node[0]
                    screen_score = self.score_group(
                        [unit],
                        direction,
                        "screen",
                        f"verified_L{layer:02d}_seed{seed_index}_leaf",
                        unit[1],
                    )
                    if screen_score is not None:
                        leaf_screen_scores[unit] = screen_score

        if not leaf_screen_scores:
            return None
        full_probe_scores = []
        for leaf_index, unit in enumerate(sorted(leaf_screen_scores)):
            score = self.score_group(
                [unit], direction, "probe", "verified_leaf_full_probe", leaf_index
            )
            if score is not None:
                score["screen_validation_nll"] = leaf_screen_scores[unit]["validation_nll"]
                full_probe_scores.append(score)
        if not full_probe_scores:
            return None
        return min(
            full_probe_scores,
            key=lambda item: (item["objective"], item["group_index"]),
        )

    @staticmethod
    def branch_bound_split(
        units: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        """Split layers first, then experts within a singleton layer.

        This preserves the experiment's semantic hierarchy: the first query
        compares the front/back 20-layer groups, and no group crosses a layer
        boundary once a single layer has been reached.
        """

        if len(units) < 2:
            raise ValueError("cannot split a singleton branch-bound node")
        layers = sorted({layer for layer, _ in units})
        if len(layers) > 1:
            middle = (len(layers) + 1) // 2
            left_layers = set(layers[:middle])
            left = tuple(unit for unit in units if unit[0] in left_layers)
            right = tuple(unit for unit in units if unit[0] not in left_layers)
            return left, right
        middle = (len(units) + 1) // 2
        return units[:middle], units[middle:]

    def select_epsilon_anchors(
        self,
        eligible: tuple[tuple[int, int], ...],
    ) -> list[tuple[int, int]]:
        """Select prior-high and prior-middle anchors with layer diversity."""

        count = min(int(self.args.bb_epsilon_anchors), len(eligible))
        if count <= 0:
            return []
        scored = sorted(
            eligible,
            key=lambda unit: (self.anchor_prior.get(unit, 0.0), unit),
            reverse=True,
        )
        high_count = (count + 1) // 2
        middle_count = count - high_count
        middle = len(scored) // 2
        ranks = {unit: index for index, unit in enumerate(scored)}
        middle_pool = sorted(
            scored,
            key=lambda unit: (
                abs(ranks[unit] - middle),
                unit,
            ),
        )

        chosen: list[tuple[int, int]] = []
        used_layers: set[int] = set()

        def take(pool: list[tuple[int, int]], wanted: int) -> None:
            for require_new_layer in (True, False):
                for unit in pool:
                    if len(chosen) >= wanted or unit in chosen:
                        continue
                    if require_new_layer and unit[0] in used_layers:
                        continue
                    chosen.append(unit)
                    used_layers.add(unit[0])
                if len(chosen) >= wanted:
                    break

        take(scored, high_count)
        take(middle_pool, high_count + middle_count)
        take(list(eligible), count)
        return chosen[:count]

    def containing_hierarchy_groups(
        self,
        eligible: tuple[tuple[int, int], ...],
        anchor: tuple[int, int],
    ) -> list[tuple[tuple[int, int], ...]]:
        """Return the layer and half-layer nodes actually used by this tree."""

        current = eligible
        path: list[tuple[tuple[int, int], ...]] = []
        while len(current) > 1:
            left, right = self.branch_bound_split(current)
            current = left if anchor in left else right
            path.append(current)
        within_layer = [
            group
            for group in path
            if len(group) > 1 and len({layer for layer, _ in group}) == 1
        ]
        return within_layer[: int(self.args.bb_epsilon_groups_per_anchor)]

    def calibrate_branch_bound_epsilon(
        self,
        eligible: tuple[tuple[int, int], ...],
        direction: str,
        evaluate,
    ) -> tuple[dict, list[tuple[tuple[int, int], GroupMeasurement]]]:
        """Estimate epsilon from singleton/group ratios on real search nodes."""

        anchors = self.select_epsilon_anchors(eligible)
        ratios: list[float] = []
        observations: list[dict] = []
        unsafe_denominators = 0
        initial_leaves: list[tuple[tuple[int, int], GroupMeasurement]] = []

        for anchor_index, anchor in enumerate(anchors):
            leaf = evaluate(
                (anchor,),
                0,
                anchor_index,
                "bb_epsilon_anchor_leaf",
            )
            initial_leaves.append((anchor, leaf))
            for group_index, group in enumerate(
                self.containing_hierarchy_groups(eligible, anchor)
            ):
                group_measurement = evaluate(
                    group,
                    group_index + 1,
                    anchor_index * self.args.bb_epsilon_groups_per_anchor + group_index,
                    f"bb_epsilon_anchor_group_{group_index + 1}",
                )
                ratio = None
                if leaf.score > 0.0:
                    if group_measurement.score > self.args.bb_epsilon_denominator_floor:
                        ratio = float(leaf.score / group_measurement.score)
                        ratios.append(ratio)
                    else:
                        unsafe_denominators += 1
                observations.append({
                    "anchor": list(anchor),
                    "group_size": len(group),
                    "leaf_utility": leaf.score,
                    "group_utility": group_measurement.score,
                    "ratio": ratio,
                })

        max_ratio = max(ratios, default=1.0)
        epsilon = max(0.0, max_ratio - 1.0) * self.args.bb_epsilon_safety_factor
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "attempt": self.state["attempts_completed"] + 1,
            "direction": direction,
            "anchor_source": (
                str(self.args.bb_anchor_prior.resolve())
                if self.args.bb_anchor_prior is not None
                and self.args.bb_anchor_prior.exists()
                else "deterministic_fallback"
            ),
            "anchors": [list(unit) for unit in anchors],
            "anchor_count": len(anchors),
            "groups_per_anchor": self.args.bb_epsilon_groups_per_anchor,
            "valid_ratios": len(ratios),
            "unsafe_denominators": unsafe_denominators,
            "max_observed_leaf_over_group_ratio": max_ratio,
            "safety_factor": self.args.bb_epsilon_safety_factor,
            "epsilon": epsilon,
            "observations": observations,
        }
        append_jsonl(self.args.work_dir / "branch_bound_epsilon.jsonl", summary)
        print(f"[branch-bound-epsilon] summary={json.dumps(summary, sort_keys=True)}", flush=True)
        return summary, initial_leaves

    def branch_bound_leaf(self, direction: str) -> dict | None:
        """Search every eligible expert with retained group-sum bounds.

        Let ``a_e = J(current) - J(move e)`` be the atomic objective utility
        to maximize. A group query measures the analogous joint utility. Ten
        prior-stratified singleton anchors and their actual layer/half-layer
        groups estimate ``epsilon`` from ``max a(e)/a(G)``. The controller then
        uses that envelope, retains provisional prunes in a deferred heap, and
        adaptively enlarges epsilon when descendants violate an ancestor bound.

        With a configured atomic lower bound L, the upper bound is

            group_utility - (group_size - 1) * L
            + epsilon * max(group_utility, 0) + safety_margins.

        L=0 and a positive group utility reduce this to
        ``(1 + epsilon) * group_utility``. Nonpositive denominators are always
        expanded rather than pruned.
        """

        if GroupMeasurement is None or maximize_with_branch_bound is None:
            raise ImportError(
                "branch_bound search requires branch_bound_search.py next to the optimizer"
            )
        step = -1 if direction == "down" else 1
        eligible = tuple(
            (layer, expert)
            for layer in range(40)
            for expert in range(256)
            if self.args.min_level
            <= int(self.state["levels"][layer][expert]) + step
            <= self.args.max_level
        )
        if not eligible:
            return None

        split_name = self.args.bb_split
        baseline_nll = float(self.state[f"{split_name}_nll"])
        baseline_objective, _, _ = objective(
            baseline_nll,
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        total_units = sum(len(row) for row in self.state["levels"])

        def atomic_bit_reward(unit: tuple[int, int]) -> float:
            """Known J improvement contributed by one unit's level change."""

            layer, expert = unit
            current = int(self.state["levels"][layer][expert])
            target = current + step
            return self.args.lambda_value * (
                math.log2(current) - math.log2(target)
            ) / total_units

        measurement_cache: dict[
            tuple[tuple[int, int], ...], GroupMeasurement
        ] = {}
        unique_model_queries = 0

        def evaluate(
            units: tuple[tuple[int, int], ...],
            depth: int,
            query_index: int,
            stage: str,
        ) -> GroupMeasurement:
            nonlocal unique_model_queries
            cached = measurement_cache.get(units)
            if cached is not None:
                return cached
            score = self.score_group(
                list(units),
                direction,
                split_name,
                stage,
                query_index,
            )
            if score is None:
                raise RuntimeError("eligible branch-bound group produced no assignments")
            # A joint group move contains the bit reward once per member. Using
            # that joint objective as an upper bound makes large groups scale
            # with |G| and reduces B&B to an exhaustive scan at high lambda.
            # The search target, however, is ONE atomic move. Keep the measured
            # group loss utility, but add only the largest known one-unit bit
            # reward in this node. At a singleton this is exactly the original
            # objective improvement (up to floating-point roundoff).
            joint_objective_utility = baseline_objective - float(score["objective"])
            loss_utility = baseline_nll - float(score["validation_nll"])
            max_atomic_bit_reward = max(atomic_bit_reward(unit) for unit in units)
            utility = loss_utility + max_atomic_bit_reward
            if len(units) == 1 and not math.isclose(
                utility, joint_objective_utility, rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    "singleton branch-bound utility disagrees with objective improvement"
                )
            score["baseline_objective"] = baseline_objective
            score["joint_group_objective_utility"] = joint_objective_utility
            score["group_loss_utility"] = loss_utility
            score["max_atomic_bit_reward"] = max_atomic_bit_reward
            score["group_utility"] = utility
            measurement = GroupMeasurement(score=utility, payload=score)
            measurement_cache[units] = measurement
            unique_model_queries += 1
            return measurement

        def measure(
            units: tuple[tuple[int, int], ...], depth: int, query_index: int
        ) -> GroupMeasurement:
            return evaluate(
                units,
                depth,
                query_index,
                f"branch_bound_depth_{depth:02d}",
            )

        persisted_epsilon = self.state["branch_bound_epsilon"].get(direction)
        if persisted_epsilon is None:
            calibration, initial_leaves = self.calibrate_branch_bound_epsilon(
                eligible,
                direction,
                evaluate,
            )
            calibration["reused_from_prior_search"] = False
        else:
            calibration = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "attempt": self.state["attempts_completed"] + 1,
                "direction": direction,
                "epsilon": float(persisted_epsilon),
                "reused_from_prior_search": True,
                "reason": (
                    "reuse the largest calibrated/adaptive envelope observed "
                    "for this direction; the current search can only enlarge it"
                ),
            }
            initial_leaves = []
            print(
                f"[branch-bound-epsilon] direction={direction} "
                f"reused={persisted_epsilon:.10f}",
                flush=True,
            )
        calibration_model_queries = unique_model_queries

        def event(kind, node, incumbent) -> None:
            if kind not in {"defer", "reopen", "prune", "incumbent", "epsilon_update"}:
                return
            unit = list(node.units[0]) if len(node.units) == 1 else None
            print(
                f"[branch-bound] event={kind} depth={node.depth} "
                f"units={len(node.units)} upper={node.upper_bound:.10f} "
                f"incumbent={incumbent} unit={unit}",
                flush=True,
            )

        result = maximize_with_branch_bound(
            eligible,
            measure,
            split=self.branch_bound_split,
            leaf_lower_bound=self.args.bb_leaf_lower_bound,
            interaction_slack=self.args.bb_interaction_slack,
            interaction_slack_per_unit=self.args.bb_interaction_slack_per_unit,
            measurement_margin=self.args.bb_measurement_margin,
            initial_epsilon=calibration["epsilon"],
            adaptive_epsilon=self.args.bb_adaptive_epsilon,
            epsilon_safety_factor=self.args.bb_epsilon_safety_factor,
            epsilon_denominator_floor=self.args.bb_epsilon_denominator_floor,
            initial_leaves=initial_leaves,
            keep_ties=self.args.bb_keep_ties,
            on_event=event,
        )
        self.state["branch_bound_epsilon"][direction] = max(
            float(calibration["epsilon"]),
            float(result.epsilon_final),
        )
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "attempt": self.state["attempts_completed"] + 1,
            "direction": direction,
            "split": split_name,
            "eligible_units": len(eligible),
            "best_unit": list(result.best_unit) if result.best_unit is not None else None,
            "best_atomic_utility": result.best_score,
            "group_queries": result.group_queries,
            "unique_model_queries": unique_model_queries,
            "calibration_model_queries": calibration_model_queries,
            "expanded_groups": result.expanded_groups,
            "deferred_events": result.deferred_events,
            "reopened_groups": result.reopened_groups,
            "reopened_units": result.reopened_units,
            "pruned_groups": result.pruned_groups,
            "pruned_units": result.pruned_units,
            "lower_bound_violations": result.lower_bound_violations,
            "nonpositive_group_expansions": result.nonpositive_group_expansions,
            "exhaustive": result.exhaustive,
            "epsilon_calibrated": calibration["epsilon"],
            "epsilon_reused_from_prior_search": calibration[
                "reused_from_prior_search"
            ],
            "epsilon_final": result.epsilon_final,
            "epsilon_updates": result.epsilon_updates,
            "max_observed_descendant_over_ancestor_ratio": result.max_observed_ratio,
            "epsilon_safety_factor": self.args.bb_epsilon_safety_factor,
            "epsilon_denominator_floor": self.args.bb_epsilon_denominator_floor,
            "adaptive_epsilon": self.args.bb_adaptive_epsilon,
            "leaf_lower_bound": self.args.bb_leaf_lower_bound,
            "interaction_slack": self.args.bb_interaction_slack,
            "interaction_slack_per_unit": self.args.bb_interaction_slack_per_unit,
            "measurement_margin": self.args.bb_measurement_margin,
            "group_score_definition": (
                "joint_group_loss_utility + max_singleton_bit_reward_in_group"
            ),
            "prune_rule": "defer upper<=incumbent_lower" if not self.args.bb_keep_ties else "defer upper<incumbent_lower",
        }
        append_jsonl(self.args.work_dir / "branch_bound_searches.jsonl", summary)
        print(f"[branch-bound] summary={json.dumps(summary, sort_keys=True)}", flush=True)
        if result.best_payload is None or result.best_unit is None:
            return None

        leaf = result.best_payload
        if split_name == "screen":
            full_probe = self.score_group(
                [result.best_unit],
                direction,
                "probe",
                "branch_bound_leaf_full_probe",
                0,
            )
            if full_probe is None:
                return None
            full_probe["screen_validation_nll"] = leaf["validation_nll"]
            leaf = full_probe
        leaf["branch_bound_summary"] = summary
        return leaf

    def taylor_fisher_shortlist(self, direction: str) -> list[dict]:
        """Rank every eligible atomic move with a current-state proxy.

        This is a screening device, not a commit criterion.  The actual TWLA
        target weights are injected into the routed-expert forward, while one
        short validation sequence supplies the signed first-order term and a
        diagonal empirical-Fisher second-order term.  Exact NLL funnels below
        decide which candidates can reach the independent convergence split.
        """

        if collect_taylor_fisher_scores is None or objective_gain_from_proxy is None:
            raise ImportError("Taylor--Fisher search requires taylor_fisher_proxy.py")
        step = -1 if direction == "down" else 1
        eligible = [
            (layer, expert)
            for layer in range(40)
            for expert in range(256)
            if self.args.min_level
            <= int(self.state["levels"][layer][expert]) + step
            <= self.args.max_level
        ]
        if not eligible:
            return []

        torch.cuda.reset_peak_memory_stats(self.args.device)
        raw, proxy_summary = collect_taylor_fisher_scores(
            model=self.model,
            text_model=self.text_model,
            payload=self.probe,
            levels=self.state["levels"],
            direction=direction,
            bank_dir=self.args.bank_dir,
            device=self.args.device,
            min_level=self.args.min_level,
            max_level=self.args.max_level,
            max_tokens=self.args.proxy_max_tokens,
            rows=self.args.proxy_rows,
            row_offset=(
                self.state["attempts_completed"]
                + (0 if direction == "down" else self.args.proxy_rows * 4)
            ),
        )
        total_units = sum(len(row) for row in self.state["levels"])
        ranked = []
        for unit in eligible:
            layer, expert = unit
            current_level = int(self.state["levels"][layer][expert])
            target_level = current_level + step
            values = raw.get(unit, {})
            first = float(values.get("first_order", 0.0))
            fisher = float(values.get("fisher_second_order", 0.0))
            predicted_loss_delta = (
                self.args.proxy_first_weight * first
                + self.args.proxy_fisher_weight * fisher
            )
            gain = objective_gain_from_proxy(
                first_order=first,
                fisher_second_order=fisher,
                current_level=current_level,
                target_level=target_level,
                lambda_value=self.args.lambda_value,
                total_units=total_units,
                first_weight=self.args.proxy_first_weight,
                fisher_weight=self.args.proxy_fisher_weight,
            )
            ranked.append({
                "unit": unit,
                "current_level": current_level,
                "target_level": target_level,
                "first_order_loss_delta": first,
                "fisher_second_order_loss_delta": fisher,
                "predicted_loss_delta": predicted_loss_delta,
                "predicted_objective_gain": gain,
                "routing_count": int(values.get("routing_count", 0)),
                "mean_gate_score": (
                    float(values.get("gate_score_sum", 0.0))
                    / max(int(values.get("routing_count", 0)), 1)
                ),
                "prior": self.anchor_prior.get(unit, 0.0),
            })
        ranked.sort(
            key=lambda item: (
                item["predicted_objective_gain"],
                item["prior"],
                -item["unit"][0],
                -item["unit"][1],
            ),
            reverse=True,
        )

        round_index = self.state["attempts_completed"] + 1
        saved = ranked[: min(self.args.proxy_shortlist, len(ranked))]
        proxy_summary.update({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "batch_search": round_index,
            "eligible_units": len(eligible),
            "shortlist_size": len(saved),
            "first_weight": self.args.proxy_first_weight,
            "fisher_weight": self.args.proxy_fisher_weight,
            "top_candidates": [
                {**item, "unit": list(item["unit"])} for item in saved
            ],
        })
        proxy_dir = self.args.work_dir / "proxy_rounds"
        atomic_json(
            proxy_dir / f"round_{round_index:04d}_{direction}.json",
            proxy_summary,
        )
        print(
            f"[proxy] direction={direction} eligible={len(eligible)} "
            f"covered={proxy_summary['covered_experts']} "
            f"shortlist={len(saved)} elapsed={proxy_summary['elapsed_seconds']:.1f}s "
            f"top_gain={saved[0]['predicted_objective_gain']:.10f}",
            flush=True,
        )
        return ranked

    def proxy_batch_attempt(self, direction: str) -> int:
        """Screen broadly, then commit up to N exact improving moves."""

        batch_index = self.state["attempts_completed"] + 1
        print(f"[proxy-batch] number={batch_index} direction={direction}", flush=True)
        ranked = self.taylor_fisher_shortlist(direction)
        if not ranked:
            self.state["attempts_completed"] += 1
            self.state.setdefault("proxy_batch_history", []).append({
                "batch": batch_index,
                "direction": direction,
                "eligible_units": 0,
                "accepted_updates": 0,
            })
            atomic_json(self.args.work_dir / "state.json", self.state)
            return 0

        shortlist_count = min(self.args.proxy_shortlist, len(ranked))
        candidates = [item["unit"] for item in ranked[:shortlist_count]]
        outside = [item["unit"] for item in ranked[shortlist_count:]]
        audit_count = min(self.args.proxy_audit, len(outside))
        if audit_count:
            seed = (
                self.args.proxy_seed
                + 1000003 * batch_index
                + (0 if direction == "down" else 49999)
            )
            audits = random.Random(seed).sample(outside, audit_count)
            candidates.extend(audits)
        print(
            f"[proxy-funnel] stage=shortlist proxy={shortlist_count} "
            f"random_audit={audit_count} total={len(candidates)}",
            flush=True,
        )

        # Refresh split baselines at the current mixed-level state.  They are
        # useful both for diagnostics and for a fair gain column in summaries.
        split_payloads = {
            "micro": self.micro,
            "screen": self.screen,
            "probe": self.probe,
        }
        baselines = {}
        for split_name, payload in split_payloads.items():
            print(f"[proxy-baseline] split={split_name}", flush=True)
            baselines[split_name], _ = validation_nll(
                self.model, payload, self.args.device
            )
        self.state["micro_nll"] = baselines["micro"]
        self.state["screen_nll"] = baselines["screen"]
        self.state["probe_nll"] = baselines["probe"]

        funnel = [
            ("micro", self.args.proxy_stage1_keep),
            ("screen", self.args.proxy_stage2_keep),
            ("probe", self.args.proxy_finalists),
        ]
        stage_summaries = []
        for stage_number, (split_name, keep) in enumerate(funnel, start=1):
            scored = []
            for candidate_index, unit in enumerate(candidates):
                score = self.score_group(
                    [unit],
                    direction,
                    split_name,
                    f"proxy_funnel_{stage_number}_{split_name}",
                    candidate_index,
                )
                if score is not None:
                    baseline_j, _, _ = objective(
                        baselines[split_name],
                        self.state["levels"],
                        self.args.lambda_value,
                        self.args.initial_level,
                    )
                    score["exact_objective_gain"] = baseline_j - score["objective"]
                    scored.append(score)
            scored.sort(
                key=lambda item: (
                    item["objective"],
                    item["group_index"],
                )
            )
            candidates = [
                next(iter(item["assignments"]))
                for item in scored[: min(keep, len(scored))]
            ]
            stage_summary = {
                "split": split_name,
                "evaluated": len(scored),
                "kept": len(candidates),
                "best_exact_objective_gain": (
                    scored[0]["exact_objective_gain"] if scored else None
                ),
            }
            stage_summaries.append(stage_summary)
            print(f"[proxy-funnel] summary={json.dumps(stage_summary)}", flush=True)
            if not candidates:
                break

        accepted = 0
        consecutive_failures = 0
        final_decisions = []
        for candidate_index, unit in enumerate(candidates):
            if accepted >= self.args.proxy_accept_limit:
                break
            convergence = self.score_group(
                [unit],
                direction,
                "convergence",
                "proxy_batch_confirmation",
                candidate_index,
            )
            if convergence is None:
                continue
            current_j, current_bits, current_increase = objective(
                self.state["convergence_nll"],
                self.state["levels"],
                self.args.lambda_value,
                self.args.initial_level,
            )
            improvement = current_j - convergence["objective"]
            decision = {
                "attempt": batch_index,
                "cycle": self.state["cycles_completed"] + 1,
                "direction": direction,
                "unit": list(unit),
                "from_level": self.state["levels"][unit[0]][unit[1]],
                "to_level": convergence["assignments"][unit],
                "convergence_nll_before": self.state["convergence_nll"],
                "convergence_nll_after_candidate": convergence["validation_nll"],
                "objective_before": current_j,
                "objective_after_candidate": convergence["objective"],
                "objective_improvement": improvement,
                "average_bits_before": current_bits,
                "bit_increase_before": current_increase,
                "changed": False,
            }
            if improvement > self.args.min_improvement:
                install_assignments(
                    self.text_model,
                    convergence["assignments"],
                    self.args.bank_dir,
                    self.args.device,
                )
                layer, expert = unit
                self.state["levels"][layer][expert] = convergence["assignments"][unit]
                self.state["convergence_nll"] = convergence["validation_nll"]
                accepted += 1
                consecutive_failures = 0
                decision["changed"] = True
                self.state["phase_updates"] += 1
            else:
                consecutive_failures += 1
            decision["average_bits_after"] = average_bits(self.state["levels"])
            decision["phase_update"] = self.state["phase_updates"]
            self.state["history"].append(decision)
            final_decisions.append(decision)
            atomic_json(self.args.work_dir / "state.json", self.state)
            print(
                f"[proxy-decision] batch={batch_index} candidate={candidate_index} "
                f"unit={unit} changed={decision['changed']} "
                f"improvement={improvement:.10f} accepted={accepted} "
                f"fail_streak={consecutive_failures}",
                flush=True,
            )
            if consecutive_failures >= self.args.proxy_failure_limit:
                break

        self.state["attempts_completed"] += 1
        self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
        batch_summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "batch": batch_index,
            "cycle": self.state["cycles_completed"] + 1,
            "direction": direction,
            "proxy_shortlist": shortlist_count,
            "random_audit": audit_count,
            "funnel": stage_summaries,
            "confirmed_candidates": len(final_decisions),
            "accepted_updates": accepted,
            "ending_failure_streak": consecutive_failures,
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
        }
        self.state.setdefault("proxy_batch_history", []).append(batch_summary)
        append_jsonl(self.args.work_dir / "proxy_batches.jsonl", batch_summary)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(f"[proxy-batch] summary={json.dumps(batch_summary)}", flush=True)
        return accepted

    def proxy_prefix_attempt(self, direction: str) -> int:
        """Commit at most one exact-improving prefix of a proxy ranking.

        A single Taylor--Fisher pass orders atomic moves.  Probe NLL evaluates
        only cumulative prefixes of that order, then the best few prefix sizes
        are checked on the independent convergence split.  This deliberately
        trades exact coordinate ordering for far fewer full-model evaluations.
        """

        search_index = self.state["attempts_completed"] + 1
        print(
            f"[proxy-prefix] number={search_index} direction={direction}",
            flush=True,
        )
        ranked = self.taylor_fisher_shortlist(direction)
        shortlist = ranked[: min(self.args.proxy_shortlist, len(ranked))]
        if self.args.prefix_positive_only:
            shortlist = [
                item for item in shortlist
                if item["predicted_objective_gain"] > 0.0
            ]
        candidates = [item["unit"] for item in shortlist]

        def finish_empty(reason: str) -> int:
            self.state["attempts_completed"] += 1
            self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
            summary = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "search": search_index,
                "cycle": self.state["cycles_completed"] + 1,
                "direction": direction,
                "lambda": self.args.lambda_value,
                "reason": reason,
                "ranked_units": len(ranked),
                "shortlisted_units": len(candidates),
                "accepted_updates": 0,
                "average_bits": average_bits(self.state["levels"]),
                "convergence_nll": self.state["convergence_nll"],
            }
            self.state.setdefault("proxy_prefix_history", []).append(summary)
            self.state["history"].append({**summary, "changed": False})
            append_jsonl(self.args.work_dir / "proxy_prefix_searches.jsonl", summary)
            atomic_json(self.args.work_dir / "state.json", self.state)
            print(f"[proxy-prefix] summary={json.dumps(summary)}", flush=True)
            return 0

        if not candidates:
            return finish_empty("no_proxy_positive_candidate")

        candidate_limit = len(candidates)
        active_prefix_sizes = list(self.args.prefix_sizes)
        prefix_regime = "fixed"
        if self.args.adaptive_prefix_sizes and self.args.adaptive_lambda:
            target_distance = abs(
                average_bits(self.state["levels"]) - self.args.target_average_bits
            )
            if target_distance <= self.args.bit_tolerance:
                prefix_cap = self.args.prefix_near_max
                prefix_regime = "near"
            elif target_distance <= self.args.prefix_mid_threshold:
                prefix_cap = self.args.prefix_mid_max
                prefix_regime = "mid"
            else:
                prefix_cap = candidate_limit
                prefix_regime = "far"
            active_prefix_sizes = [
                size for size in active_prefix_sizes if size <= prefix_cap
            ]
        requested_sizes = {
            min(int(size), candidate_limit)
            for size in active_prefix_sizes
            if int(size) > 0
        }
        if not self.args.adaptive_prefix_sizes:
            requested_sizes.add(candidate_limit)
        elif candidate_limit <= max(active_prefix_sizes, default=0):
            requested_sizes.add(candidate_limit)
        probe_scores: dict[int, dict] = {}
        # Tiered evaluation (opt-in): scan prefix sizes on a small split, re-score the
        # finalists on a larger one, and confirm only the best on the convergence split.
        scan_split = self.args.prefix_scan_split
        finalist_split = self.args.prefix_finalist_split

        def score_probe(size: int) -> None:
            if size <= 0 or size in probe_scores:
                return
            score = self.score_group(
                candidates[:size],
                direction,
                scan_split,
                "proxy_prefix_probe",
                size,
            )
            if score is not None:
                probe_scores[size] = score

        for size in sorted(requested_sizes):
            score_probe(size)
        if not probe_scores:
            return finish_empty("no_eligible_prefix")

        current_probe_j, _, _ = objective(
            self.state[f"{scan_split}_nll"],
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        # Refine only around the current best prefix.  Zero is an explicit
        # candidate, so a small profitable prefix can still be found when all
        # coarse prefix sizes are worse than the current allocation.
        for _ in range(self.args.prefix_refine_rounds):
            measured = sorted({0, *probe_scores})
            best_size = min(
                measured,
                key=lambda size: (
                    current_probe_j if size == 0 else probe_scores[size]["objective"],
                    size,
                ),
            )
            position = measured.index(best_size)
            intervals = []
            if position > 0:
                intervals.append((measured[position - 1], best_size))
            if position + 1 < len(measured):
                intervals.append((best_size, measured[position + 1]))
            new_sizes = {
                (left + right) // 2
                for left, right in intervals
                if right - left > 1
            }
            if not new_sizes:
                break
            for size in sorted(new_sizes):
                score_probe(size)

        ordered_probe = sorted(
            probe_scores.items(),
            key=lambda item: (item[1]["objective"], item[0]),
        )
        finalist_sizes = [
            size for size, _ in ordered_probe[: self.args.prefix_convergence_finalists]
        ]
        tier_scores: dict[int, dict] = {}
        if finalist_split != "convergence" and finalist_split != scan_split:
            for size in finalist_sizes:
                score = self.score_group(
                    candidates[:size],
                    direction,
                    finalist_split,
                    "proxy_prefix_finalist",
                    size,
                )
                if score is not None:
                    tier_scores[size] = score
            confirm_sizes = [
                size for size, _ in sorted(
                    tier_scores.items(), key=lambda item: (item[1]["objective"], item[0])
                )[: self.args.prefix_confirm_count]
            ]
        else:
            confirm_sizes = finalist_sizes
        convergence_scores = []
        for size in confirm_sizes:
            score = self.score_group(
                candidates[:size],
                direction,
                "convergence",
                "proxy_prefix_confirmation",
                size,
            )
            if score is not None:
                convergence_scores.append((size, score))

        current_j, current_bits, current_increase = objective(
            self.state["convergence_nll"],
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        best_size = 0
        best_score = None
        if convergence_scores:
            best_size, best_score = min(
                convergence_scores,
                key=lambda item: (item[1]["objective"], item[0]),
            )
        improvement = (
            current_j - best_score["objective"] if best_score is not None else 0.0
        )
        changed = bool(best_score is not None and improvement > self.args.min_improvement)
        # A rejected prefix is restored (score_group restores) and retried at half its size.
        shrink_retries = 0
        while (
            not changed and best_size > 1
            and shrink_retries < self.args.prefix_shrink_retries
        ):
            shrink_retries += 1
            size = max(1, best_size // 2)
            score = self.score_group(
                candidates[:size],
                direction,
                "convergence",
                "proxy_prefix_shrink_retry",
                size,
            )
            if score is None:
                break
            convergence_scores.append((size, score))
            best_size, best_score = size, score
            improvement = current_j - best_score["objective"]
            changed = bool(improvement > self.args.min_improvement)
        accepted = 0
        if changed:
            install_assignments(
                self.text_model,
                best_score["assignments"],
                self.args.bank_dir,
                self.args.device,
            )
            self.state["levels"] = best_score["candidate_levels"]
            if best_size in tier_scores and finalist_split == "probe":
                self.state["probe_nll"] = tier_scores[best_size]["validation_nll"]
            elif scan_split == "probe" and best_size in probe_scores:
                self.state["probe_nll"] = probe_scores[best_size]["validation_nll"]
            if scan_split == "screen" and best_size in probe_scores:
                self.state["screen_nll"] = probe_scores[best_size]["validation_nll"]
            self.state["convergence_nll"] = best_score["validation_nll"]
            accepted = len(best_score["assignments"])
            self.state["phase_updates"] += accepted

        self.state["attempts_completed"] += 1
        self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "search": search_index,
            "cycle": self.state["cycles_completed"] + 1,
            "direction": direction,
            "lambda": self.args.lambda_value,
            "ranked_units": len(ranked),
            "shortlisted_units": len(candidates),
            "positive_only": self.args.prefix_positive_only,
            "prefix_regime": prefix_regime,
            "active_prefix_sizes": sorted(requested_sizes),
            "scan_split": scan_split,
            "finalist_split": finalist_split,
            "finalist_prefixes": [
                {"size": size, "objective": score["objective"]}
                for size, score in sorted(tier_scores.items())
            ],
            "shrink_retries": shrink_retries,
            "probe_prefixes": [
                {
                    "size": size,
                    "objective": score["objective"],
                    "objective_improvement": current_probe_j - score["objective"],
                }
                for size, score in sorted(probe_scores.items())
            ],
            "convergence_prefixes": [
                {
                    "size": size,
                    "objective": score["objective"],
                    "objective_improvement": current_j - score["objective"],
                }
                for size, score in convergence_scores
            ],
            "selected_prefix": best_size if changed else 0,
            "objective_before": current_j,
            "objective_after_candidate": (
                best_score["objective"] if best_score is not None else current_j
            ),
            "objective_improvement": improvement,
            "average_bits_before": current_bits,
            "bit_increase_before": current_increase,
            "accepted_updates": accepted,
            "changed": changed,
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
        }
        self.state.setdefault("proxy_prefix_history", []).append(summary)
        self.state["history"].append(summary)
        append_jsonl(self.args.work_dir / "proxy_prefix_searches.jsonl", summary)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(f"[proxy-prefix] summary={json.dumps(summary)}", flush=True)
        return accepted

    def tf_trust_attempt(self, direction: str) -> int:
        """Commit one Taylor--Fisher trust-region batch without a prefix-size NLL scan.

        One ranking (gradient rows from the probe split) orders the one-level moves.  The top
        positive-gain moves, at most ``--tf-trust-moves``, form a single batch.  With
        ``--tf-verify-split convergence`` the batch is committed only when the independent
        convergence objective decreases; a rejected batch is retried at half its size up to
        ``--tf-trust-retries`` times.  ``none`` commits the batch unchecked.
        """

        search_index = self.state["attempts_completed"] + 1
        print(f"[tf-trust] number={search_index} direction={direction}", flush=True)
        ranked = self.taylor_fisher_shortlist(direction)
        positive = [item for item in ranked if item["predicted_objective_gain"] > 0.0]
        batch = positive[: self.args.tf_trust_moves]
        candidates = [item["unit"] for item in batch]
        predicted = [float(item["predicted_objective_gain"]) for item in batch]
        current_j, current_bits, current_increase = objective(
            self.state["convergence_nll"],
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        checks = []
        committed = None
        size = len(candidates)
        while size > 0:
            if self.args.tf_verify_split == "none":
                candidate_levels, assignments = moved_map(
                    self.state["levels"],
                    candidates[:size],
                    direction,
                    self.args.min_level,
                    self.args.max_level,
                )
                if assignments:
                    committed = {
                        "assignments": assignments,
                        "candidate_levels": candidate_levels,
                        "validation_nll": self.state["convergence_nll"],
                    }
                break
            score = self.score_group(
                candidates[:size],
                direction,
                self.args.tf_verify_split,
                "tf_trust_check",
                size,
            )
            if score is None:
                break
            improvement = current_j - score["objective"]
            checks.append({
                "size": size,
                "objective": score["objective"],
                "objective_improvement": improvement,
                "predicted_objective_gain": sum(predicted[:size]),
            })
            if improvement > self.args.min_improvement:
                committed = score
                break
            if len(checks) > self.args.tf_trust_retries or size == 1:
                break
            size = max(1, size // 2)

        accepted = 0
        if committed is not None:
            install_assignments(
                self.text_model,
                committed["assignments"],
                self.args.bank_dir,
                self.args.device,
            )
            self.state["levels"] = committed["candidate_levels"]
            self.state["convergence_nll"] = committed["validation_nll"]
            accepted = len(committed["assignments"])
            self.state["phase_updates"] += accepted

        self.state["attempts_completed"] += 1
        self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "search": search_index,
            "cycle": self.state["cycles_completed"] + 1,
            "direction": direction,
            "lambda": self.args.lambda_value,
            "ranked_units": len(ranked),
            "positive_units": len(positive),
            "batch_cap": self.args.tf_trust_moves,
            "verify_split": self.args.tf_verify_split,
            "checks": checks,
            "selected_prefix": accepted,
            "objective_before": current_j,
            "objective_after_candidate": checks[-1]["objective"] if checks else current_j,
            "objective_improvement": checks[-1]["objective_improvement"] if checks else 0.0,
            "average_bits_before": current_bits,
            "bit_increase_before": current_increase,
            "accepted_updates": accepted,
            "changed": accepted > 0,
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
        }
        self.state.setdefault("tf_trust_history", []).append(summary)
        self.state["history"].append(summary)
        append_jsonl(self.args.work_dir / "tf_trust_searches.jsonl", summary)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(f"[tf-trust] summary={json.dumps(summary)}", flush=True)
        return accepted

    def fast_target_prefix_attempt(self, direction: str) -> int:
        """Use one proxy ranking and one exact finalist to approach the bit target."""

        search_index = self.state["attempts_completed"] + 1
        current_bits = average_bits(self.state["levels"])
        residual = current_bits - self.args.target_average_bits
        ranked = self.taylor_fisher_shortlist(direction)
        shortlist = [
            item for item in ranked[: self.args.proxy_shortlist]
            if item["predicted_objective_gain"] > 0.0
        ]
        units = [item["unit"] for item in shortlist]
        requested_sizes = sorted({
            min(size, len(units))
            for size in self.args.fast_prefix_sizes
            if size > 0 and units
        })
        target_distance = abs(residual)
        trust_radius = max(
            self.args.bit_tolerance,
            self.args.fast_prefix_trust_ratio * target_distance,
        )
        micro_scores: list[tuple[int, dict]] = []
        rejected_sizes: list[dict] = []
        for size in requested_sizes:
            candidate_levels, assignments = moved_map(
                self.state["levels"],
                units[:size],
                direction,
                self.args.min_level,
                self.args.max_level,
            )
            if not assignments:
                continue
            candidate_bits = average_bits(candidate_levels)
            movement = abs(candidate_bits - current_bits)
            candidate_distance = abs(candidate_bits - self.args.target_average_bits)
            if movement > trust_radius + 1e-12:
                rejected_sizes.append({"size": size, "reason": "trust_radius"})
                continue
            if candidate_distance > max(target_distance, self.args.bit_tolerance) + 1e-12:
                rejected_sizes.append({"size": size, "reason": "moves_away_from_target"})
                continue
            score = self.score_group(
                units[:size],
                direction,
                "micro",
                "fast_target_micro",
                size,
            )
            if score is not None:
                micro_scores.append((size, score))

        current_micro_j, _, _ = objective(
            self.state["micro_nll"],
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        improving_micro = [
            item for item in micro_scores
            if current_micro_j - item[1]["objective"] > self.args.min_improvement
        ]
        improving_micro.sort(key=lambda item: (item[1]["objective"], item[0]))
        finalist = improving_micro[: self.args.fast_convergence_finalists]
        convergence_scores: list[tuple[int, dict]] = []
        for size, _ in finalist:
            score = self.score_group(
                units[:size],
                direction,
                "convergence",
                "fast_target_confirmation",
                size,
            )
            if score is not None:
                convergence_scores.append((size, score))

        current_convergence_j, _, _ = objective(
            self.state["convergence_nll"],
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        best_size = 0
        best_score = None
        if convergence_scores:
            best_size, best_score = min(
                convergence_scores,
                key=lambda item: (item[1]["objective"], item[0]),
            )
        exact_gain = (
            current_convergence_j - best_score["objective"]
            if best_score is not None else 0.0
        )
        changed = bool(best_score is not None and exact_gain > self.args.min_improvement)
        accepted = 0
        if changed:
            install_assignments(
                self.text_model,
                best_score["assignments"],
                self.args.bank_dir,
                self.args.device,
            )
            self.state["levels"] = best_score["candidate_levels"]
            selected_micro = next(
                score for size, score in micro_scores if size == best_size
            )
            self.state["micro_nll"] = selected_micro["validation_nll"]
            self.state["convergence_nll"] = best_score["validation_nll"]
            accepted = len(best_score["assignments"])
            self.state["phase_updates"] += accepted

        self.state["attempts_completed"] += 1
        self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "search": search_index,
            "cycle": self.state["cycles_completed"] + 1,
            "mode": "fast_target",
            "direction": direction,
            "lambda": self.args.lambda_value,
            "ranked_units": len(ranked),
            "proxy_positive_units": len(units),
            "requested_prefix_sizes": requested_sizes,
            "rejected_prefixes": rejected_sizes,
            "trust_radius_bits": trust_radius,
            "micro_prefixes": [
                {
                    "size": size,
                    "objective": score["objective"],
                    "objective_gain": current_micro_j - score["objective"],
                }
                for size, score in micro_scores
            ],
            "convergence_prefixes": [
                {
                    "size": size,
                    "objective": score["objective"],
                    "objective_gain": current_convergence_j - score["objective"],
                }
                for size, score in convergence_scores
            ],
            "selected_prefix": best_size if changed else 0,
            "objective_gain": exact_gain,
            "accepted_updates": accepted,
            "changed": changed,
            "average_bits_before": current_bits,
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
        }
        self.state.setdefault("fast_target_history", []).append(summary)
        self.state["history"].append(summary)
        append_jsonl(self.args.work_dir / "fast_target_searches.jsonl", summary)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(f"[fast-target] summary={json.dumps(summary)}", flush=True)
        return accepted

    def exchange_bundle_attempt(self) -> int:
        """Try one near-bit-neutral bundle and accept it by convergence NLL."""

        search_index = self.state["attempts_completed"] + 1
        current_bits = average_bits(self.state["levels"])
        down_ranked = self.taylor_fisher_shortlist("down")
        up_ranked = self.taylor_fisher_shortlist("up")
        bundles = build_exchange_bundles(
            down_ranked,
            up_ranked,
            total_units=sum(len(row) for row in self.state["levels"]),
            pool_size=self.args.exchange_pool_size,
            max_per_side=self.args.exchange_max_per_side,
            bundle_limit=self.args.exchange_bundle_limit,
            balance_tolerance=self.args.exchange_balance_tolerance,
            beam_width=self.args.exchange_beam_width,
            first_weight=self.args.proxy_first_weight,
            fisher_weight=self.args.proxy_fisher_weight,
        )
        allowed_bundles = []
        for bundle in bundles:
            candidate_levels = assigned_map(
                self.state["levels"],
                bundle["assignments"],
                self.args.min_level,
                self.args.max_level,
            )
            candidate_bits = average_bits(candidate_levels)
            if abs(candidate_bits - self.args.target_average_bits) <= self.args.bit_tolerance:
                allowed_bundles.append(bundle)

        micro_scores: list[tuple[dict, dict]] = []
        for index, bundle in enumerate(
            allowed_bundles[: self.args.exchange_micro_finalists]
        ):
            score = self.score_assignments(
                bundle["assignments"],
                "micro",
                "exchange_micro",
                index,
            )
            if score is not None:
                micro_scores.append((bundle, score))
        improving_micro = [
            item for item in micro_scores
            if self.state["micro_nll"] - item[1]["validation_nll"]
            > self.args.min_improvement
        ]
        improving_micro.sort(
            key=lambda item: (
                item[1]["validation_nll"],
                -item[0]["predicted_nll_gain"],
            )
        )
        convergence_scores: list[tuple[dict, dict]] = []
        for index, (bundle, _) in enumerate(
            improving_micro[: self.args.exchange_convergence_finalists]
        ):
            score = self.score_assignments(
                bundle["assignments"],
                "convergence",
                "exchange_confirmation",
                index,
            )
            if score is not None:
                convergence_scores.append((bundle, score))
        best_bundle = None
        best_score = None
        if convergence_scores:
            best_bundle, best_score = min(
                convergence_scores,
                key=lambda item: item[1]["validation_nll"],
            )
        nll_gain = (
            self.state["convergence_nll"] - best_score["validation_nll"]
            if best_score is not None else 0.0
        )
        changed = bool(best_score is not None and nll_gain > self.args.min_improvement)
        accepted = 0
        if changed:
            install_assignments(
                self.text_model,
                best_score["assignments"],
                self.args.bank_dir,
                self.args.device,
            )
            self.state["levels"] = best_score["candidate_levels"]
            selected_micro = next(
                score for bundle, score in micro_scores if bundle is best_bundle
            )
            self.state["micro_nll"] = selected_micro["validation_nll"]
            self.state["convergence_nll"] = best_score["validation_nll"]
            accepted = len(best_score["assignments"])

        self.state["attempts_completed"] += 1
        self.state["proxy_rounds_completed"] = self.state["attempts_completed"]
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "search": search_index,
            "cycle": self.state["cycles_completed"] + 1,
            "mode": "bit_neutral_exchange",
            "lambda_frozen": self.args.lambda_value,
            "candidate_bundles": len(bundles),
            "target_compatible_bundles": len(allowed_bundles),
            "micro_measured": len(micro_scores),
            "convergence_measured": len(convergence_scores),
            "selected_donors": (
                [list(unit) for unit in best_bundle["donors"]]
                if changed else []
            ),
            "selected_receivers": (
                [list(unit) for unit in best_bundle["receivers"]]
                if changed else []
            ),
            "selected_average_bit_delta": (
                best_bundle["average_bit_delta"] if changed else 0.0
            ),
            "predicted_nll_gain": (
                best_bundle["predicted_nll_gain"] if best_bundle else None
            ),
            "exact_convergence_nll_gain": nll_gain,
            "accepted_updates": accepted,
            "changed": changed,
            "average_bits_before": current_bits,
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
        }
        self.state.setdefault("exchange_history", []).append(summary)
        self.state["history"].append(summary)
        append_jsonl(self.args.work_dir / "exchange_searches.jsonl", summary)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(f"[exchange] summary={json.dumps(summary)}", flush=True)
        return accepted

    def run_fast_target_exchange(self) -> None:
        """Approach the target with J, then refine at fixed bits with NLL swaps."""

        self.state.setdefault("fast_stage", "target")
        self.state.setdefault("exchange_no_change_rounds", 0)
        self.state.setdefault("fast_target_history", [])
        self.state.setdefault("exchange_history", [])
        while self.state["cycles_completed"] < self.args.max_cycles:
            bits_now = average_bits(self.state["levels"])
            at_target = (
                abs(bits_now - self.args.target_average_bits)
                <= self.args.bit_tolerance
            )
            if at_target:
                # Close an already-started down/up target cycle before the
                # controller is frozen for the exchange phase.
                if (
                    self.state.get("fast_stage") == "target"
                    and self.state["next_direction"] == "up"
                ):
                    self.finish_direction_phase("up")
                self.state["fast_stage"] = "exchange"
                self.state["consecutive_no_change_cycles"] = 0
                atomic_json(self.args.work_dir / "state.json", self.state)
                accepted = self.exchange_bundle_attempt()
                self.state["cycles_completed"] += 1
                if accepted:
                    self.state["exchange_no_change_rounds"] = 0
                else:
                    self.state["exchange_no_change_rounds"] += 1
                self.state["cycle_history"].append({
                    "cycle": self.state["cycles_completed"],
                    "stage": "exchange",
                    "changed": accepted > 0,
                    "accepted_updates": accepted,
                    "average_bits": average_bits(self.state["levels"]),
                    "convergence_nll": self.state["convergence_nll"],
                    "lambda": self.args.lambda_value,
                })
                atomic_json(self.args.work_dir / "state.json", self.state)
                if (
                    self.state["exchange_no_change_rounds"]
                    >= self.args.exchange_no_change_rounds
                ):
                    self.state["consecutive_no_change_cycles"] = self.args.no_change_pairs
                    self.state["fast_stop_reason"] = (
                        f"{self.args.exchange_no_change_rounds}_exchange_rounds_"
                        "without_an_accepted_bundle"
                    )
                    break
                continue

            self.state["fast_stage"] = "target"
            direction = self.state["next_direction"]
            while (
                abs(
                    average_bits(self.state["levels"])
                    - self.args.target_average_bits
                ) > self.args.bit_tolerance
                and self.fast_target_prefix_attempt(direction) > 0
            ):
                pass
            self.finish_direction_phase(direction)
            if (
                self.state["consecutive_no_change_cycles"]
                >= self.args.no_change_pairs
                and abs(
                    average_bits(self.state["levels"])
                    - self.args.target_average_bits
                ) > self.args.bit_tolerance
            ):
                self.state["fast_stop_reason"] = "target_unreachable_within_lambda_bounds"
                break

    def attempt(self, direction: str) -> bool:
        print(
            f"[attempt] number={self.state['attempts_completed'] + 1} direction={direction}",
            flush=True,
        )
        if self.args.search_mode == "branch_bound":
            leaf = self.branch_bound_leaf(direction)
        elif self.args.search_mode == "verified_beam":
            leaf = self.verified_beam_leaf(direction)
        else:
            leaf = self.hierarchical_leaf(direction)
        changed = False
        decision = {"direction": direction, "changed": False}
        if leaf is not None:
            unit = next(iter(leaf["assignments"]))
            convergence = self.score_group(
                [unit], direction, "convergence", "leaf_confirmation", 0
            )
            assert convergence is not None
            current_j, current_bits, current_increase = objective(
                self.state["convergence_nll"],
                self.state["levels"],
                self.args.lambda_value,
                self.args.initial_level,
            )
            improvement = current_j - convergence["objective"]
            decision.update({
                "unit": list(unit),
                "from_level": self.state["levels"][unit[0]][unit[1]],
                "to_level": convergence["assignments"][unit],
                "probe_nll": leaf["validation_nll"],
                "convergence_nll_before": self.state["convergence_nll"],
                "convergence_nll_after_candidate": convergence["validation_nll"],
                "objective_before": current_j,
                "objective_after_candidate": convergence["objective"],
                "objective_improvement": improvement,
                "average_bits_before": current_bits,
                "bit_increase_before": current_increase,
            })
            if "branch_bound_summary" in leaf:
                decision["branch_bound"] = leaf["branch_bound_summary"]
            if improvement > self.args.min_improvement:
                install_assignments(
                    self.text_model,
                    convergence["assignments"],
                    self.args.bank_dir,
                    self.args.device,
                )
                layer, expert = unit
                self.state["levels"][layer][expert] = convergence["assignments"][unit]
                self.state["probe_nll"] = leaf["validation_nll"]
                if "screen_validation_nll" in leaf:
                    self.state["screen_nll"] = leaf["screen_validation_nll"]
                self.state["convergence_nll"] = convergence["validation_nll"]
                changed = True
                decision["changed"] = True

        self.state["attempts_completed"] += 1
        if changed:
            self.state["phase_updates"] += 1
        decision["attempt"] = self.state["attempts_completed"]
        decision["cycle"] = self.state["cycles_completed"] + 1
        decision["phase_update"] = self.state["phase_updates"]
        decision["average_bits_after"] = average_bits(self.state["levels"])
        self.state["history"].append(decision)
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(
            f"[decision] attempt={self.state['attempts_completed']} direction={direction} "
            f"changed={changed} bits={decision['average_bits_after']:.8f}",
            flush=True,
        )
        return changed

    def finish_direction_phase(self, direction: str) -> None:
        updates = int(self.state["phase_updates"])
        phase = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cycle": self.state["cycles_completed"] + 1,
            "direction": direction,
            "accepted_updates": updates,
            "searches_completed_total": self.state["attempts_completed"],
            "average_bits": average_bits(self.state["levels"]),
            "convergence_nll": self.state["convergence_nll"],
            "lambda": self.args.lambda_value,
        }
        self.state["phase_history"].append(phase)
        self.state["phase_updates"] = 0
        if direction == "down":
            self.state["cycle_had_change"] = updates > 0
            self.state["next_direction"] = "up"
        else:
            cycle_changed = bool(self.state["cycle_had_change"] or updates > 0)
            self.state["cycles_completed"] += 1
            bits_now = average_bits(self.state["levels"])
            dual_requires_more = False
            dual_update = None
            if self.args.adaptive_lambda:
                lambda_before = float(self.args.lambda_value)
                residual = bits_now - self.args.target_average_bits
                previous_residual = self.state.get("previous_dual_residual")
                crossed_target = bool(
                    previous_residual is not None
                    and float(previous_residual) * residual < 0.0
                )
                if self.args.stabilized_adaptive_lambda and cycle_changed:
                    # An up phase can make a new down move profitable through
                    # interactions. Repeat the full pair at the same lambda
                    # before allowing the controller to move lambda.
                    controller = {
                        "excess_error": max(
                            abs(residual) - self.args.bit_tolerance, 0.0
                        ),
                        "effective_lr": 0.0,
                        "raw_step": 0.0,
                        "actual_step": 0.0,
                        "lambda_after": lambda_before,
                        "changed": False,
                    }
                    lambda_after = lambda_before
                    dual_requires_more = True
                    deferred_reason = "fixed_lambda_inner_pair_changed"
                elif self.args.stabilized_adaptive_lambda:
                    if self.args.clipped_adaptive_lambda:
                        controller = clipped_dual_update(
                            lambda_value=lambda_before,
                            residual=residual,
                            tolerance=self.args.bit_tolerance,
                            response_gain=self.args.dual_response_gain,
                            min_step=self.args.dual_min_step,
                            max_step=(
                                self.args.dual_max_step * 0.5
                                if crossed_target else self.args.dual_max_step
                            ),
                            lambda_min=self.args.lambda_min,
                            lambda_max=self.args.lambda_max,
                        )
                    else:
                        controller = smooth_dual_update(
                            lambda_value=lambda_before,
                            residual=residual,
                            tolerance=self.args.bit_tolerance,
                            base_lr=self.args.dual_lr,
                            min_lr_ratio=self.args.dual_min_lr_ratio,
                            full_lr_error=self.args.dual_full_lr_error,
                            max_step=self.args.dual_max_step,
                            lambda_min=self.args.lambda_min,
                            lambda_max=self.args.lambda_max,
                        )
                    lambda_after = float(controller["lambda_after"])
                    dual_requires_more = bool(controller["changed"])
                    deferred_reason = None
                    self.state["previous_dual_residual"] = residual
                else:
                    lambda_after = min(
                        self.args.lambda_max,
                        max(
                            self.args.lambda_min,
                            lambda_before + self.args.dual_lr * residual,
                        ),
                    )
                    controller = {
                        "excess_error": max(
                            abs(residual) - self.args.bit_tolerance, 0.0
                        ),
                        "effective_lr": self.args.dual_lr,
                        "raw_step": self.args.dual_lr * residual,
                        "actual_step": lambda_after - lambda_before,
                        "lambda_after": lambda_after,
                        "changed": abs(lambda_after - lambda_before) > 1e-12,
                    }
                    dual_requires_more = bool(
                        residual > self.args.bit_tolerance
                        or (
                            residual < -self.args.bit_tolerance
                            and lambda_before > self.args.lambda_min + 1e-12
                        )
                    )
                    deferred_reason = None
                    self.state["previous_dual_residual"] = residual
                dual_update = {
                    "cycle": self.state["cycles_completed"],
                    "average_bits": bits_now,
                    "target_average_bits": self.args.target_average_bits,
                    "residual": residual,
                    "lambda_before": lambda_before,
                    "lambda_after": lambda_after,
                    "dual_lr": self.args.dual_lr,
                    "effective_dual_lr": controller["effective_lr"],
                    "excess_error": controller["excess_error"],
                    "raw_lambda_step": controller["raw_step"],
                    "actual_lambda_step": controller["actual_step"],
                    "clipped_adaptive_lambda": self.args.clipped_adaptive_lambda,
                    "dual_min_step": (
                        self.args.dual_min_step
                        if self.args.clipped_adaptive_lambda else None
                    ),
                    "dual_response_gain": (
                        self.args.dual_response_gain
                        if self.args.clipped_adaptive_lambda else None
                    ),
                    "crossed_target": crossed_target,
                    "deferred_reason": deferred_reason,
                    "bit_tolerance": self.args.bit_tolerance,
                    "requires_more_search": dual_requires_more,
                }
                self.args.lambda_value = lambda_after
                self.state["current_lambda"] = lambda_after
                self.state.setdefault("lambda_history", []).append(dual_update)
                print(f"[dual] {json.dumps(dual_update)}", flush=True)
            self.state["consecutive_no_change_cycles"] = (
                0
                if cycle_changed or dual_requires_more
                else self.state["consecutive_no_change_cycles"] + 1
            )
            self.state["cycle_history"].append({
                "cycle": self.state["cycles_completed"],
                "changed": cycle_changed,
                "average_bits": bits_now,
                "convergence_nll": self.state["convergence_nll"],
                "lambda": self.args.lambda_value,
                "dual_update": dual_update,
            })
            self.state["cycle_had_change"] = False
            self.state["next_direction"] = "down"
        atomic_json(self.args.work_dir / "state.json", self.state)
        print(
            f"[phase-complete] cycle={phase['cycle']} direction={direction} "
            f"accepted_updates={updates} next={self.state['next_direction']}",
            flush=True,
        )

    def run(self) -> None:
        if self.args.search_mode == "fast_target_exchange":
            self.run_fast_target_exchange()
        else:
            while (
                self.state["cycles_completed"] < self.args.max_cycles
                and self.state["consecutive_no_change_cycles"] < self.args.no_change_pairs
            ):
                direction = self.state["next_direction"]
                if self.args.search_mode == "proxy_prefix":
                    if self.args.prefix_until_no_change:
                        # Recompute the proxy after every accepted batch and stay
                        # in this direction until exact convergence NLL rejects it.
                        while self.proxy_prefix_attempt(direction) > 0:
                            pass
                    else:
                        self.proxy_prefix_attempt(direction)
                elif self.args.search_mode == "tf_trust":
                    self.tf_trust_attempt(direction)
                elif self.args.search_mode == "proxy_batch":
                    # One Taylor--Fisher ranking feeds several exact coordinate
                    # updates.  Refresh after each accepted batch because all
                    # remaining marginal effects are then stale.
                    while self.proxy_batch_attempt(direction) > 0:
                        pass
                else:
                    # Stay in one direction and re-search after every accepted
                    # expert move. Switch only when the best move fails.
                    while self.attempt(direction):
                        pass
                self.finish_direction_phase(direction)

        if self.args.audit_only:
            self.state["status"] = "audit_complete"
            self.state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            atomic_json(self.args.work_dir / "state.json", self.state)
            (self.args.work_dir / ".audit_done").touch()
            print("[complete] audit-only search finished without checkpoint save", flush=True)
            return

        if self.state.get("fast_stop_reason"):
            stop_reason = self.state["fast_stop_reason"]
        elif self.state["consecutive_no_change_cycles"] >= self.args.no_change_pairs:
            stop_reason = (
                f"{self.args.no_change_pairs}_full_down_up_cycles_without_an_accepted_update"
            )
        else:
            stop_reason = "max_down_up_cycles"
        self.state["status"] = "final_validation"
        self.state["stop_reason"] = stop_reason
        atomic_json(self.args.work_dir / "state.json", self.state)
        final_nll, final_tokens = validation_nll(
            self.model, self.final, self.args.device
        )
        final_j, bits, increase = objective(
            final_nll,
            self.state["levels"],
            self.args.lambda_value,
            self.args.initial_level,
        )
        self.state.update({
            "status": "saving",
            "final_validation_nll": final_nll,
            "final_validation_loss_tokens": final_tokens,
            "final_objective": final_j,
            "final_average_logical_bits": bits,
            "final_bit_delta_from_initial_level": increase,
        })
        atomic_json(self.args.work_dir / "state.json", self.state)

        print(f"[save] checkpoint={self.args.out_dir}", flush=True)
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.args.out_dir)
        self.tokenizer.save_pretrained(self.args.out_dir)
        precision = {
            "scheme": f"alternating_{self.args.search_mode}_validation_nll_twla",
            "objective": (
                "validation_token_NLL + lambda * "
                "(average_log2_levels - log2(initial_level))"
            ),
            "lambda": self.args.lambda_value,
            "initial_lambda": self.state.get(
                "initial_lambda", self.args.lambda_value
            ),
            "adaptive_lambda": self.args.adaptive_lambda,
            "target_average_bits": (
                self.args.target_average_bits if self.args.adaptive_lambda else None
            ),
            "dual_lr": self.args.dual_lr if self.args.adaptive_lambda else None,
            "stabilized_adaptive_lambda": (
                self.args.stabilized_adaptive_lambda
                if self.args.adaptive_lambda else False
            ),
            "clipped_adaptive_lambda": (
                self.args.clipped_adaptive_lambda
                if self.args.adaptive_lambda else False
            ),
            "dual_max_step": (
                self.args.dual_max_step if self.args.adaptive_lambda else None
            ),
            "dual_min_step": (
                self.args.dual_min_step if self.args.adaptive_lambda else None
            ),
            "dual_response_gain": (
                self.args.dual_response_gain if self.args.adaptive_lambda else None
            ),
            "dual_min_lr_ratio": (
                self.args.dual_min_lr_ratio if self.args.adaptive_lambda else None
            ),
            "dual_full_lr_error": (
                self.args.dual_full_lr_error if self.args.adaptive_lambda else None
            ),
            "lambda_bounds": (
                [self.args.lambda_min, self.args.lambda_max]
                if self.args.adaptive_lambda else None
            ),
            "bit_tolerance": (
                self.args.bit_tolerance if self.args.adaptive_lambda else None
            ),
            "initial_level": self.args.initial_level,
            "level_bounds": [self.args.min_level, self.args.max_level],
            "routed_experts_only": True,
            "allocation_unit": "one layer routed expert",
            "gate_up_down_tied_per_expert": True,
            "optimization_schedule": (
                "down_to_convergence_then_up_to_convergence; repeat full cycles"
            ),
            "completed_cycles": self.state["cycles_completed"],
            "atomic_searches": self.state["attempts_completed"],
            "levels": self.state["levels"],
            "final_average_logical_bits": bits,
            "activation_bits": 16,
            "packed": False,
            "bank_dir": str(self.args.bank_dir.resolve()),
            "initial_checkpoint": str(self.args.initial_checkpoint.resolve()),
            "search_mode": self.args.search_mode,
            "screen_divisor": self.args.screen_divisor,
            "screen_indices": self.screen_indices,
            "top_layers": self.args.top_layers,
            "beam_width": self.args.beam_width,
            "beam_seeds": self.args.beam_seeds,
        }
        if self.args.search_mode == "proxy_batch":
            precision["taylor_fisher_proxy_batch"] = {
                "role": "candidate ranking only; every commit uses exact convergence NLL",
                "formula": (
                    "g_dot_delta_h + 0.5 * delta_h_dot_diag_empirical_fisher_dot_delta_h"
                ),
                "shortlist": self.args.proxy_shortlist,
                "random_audit": self.args.proxy_audit,
                "funnel": [
                    ["micro", self.args.proxy_stage1_keep],
                    ["screen", self.args.proxy_stage2_keep],
                    ["probe", self.args.proxy_finalists],
                    ["convergence", self.args.proxy_accept_limit],
                ],
                "max_tokens": self.args.proxy_max_tokens,
                "rows": self.args.proxy_rows,
                "first_weight": self.args.proxy_first_weight,
                "fisher_weight": self.args.proxy_fisher_weight,
                "failure_limit": self.args.proxy_failure_limit,
                "micro_indices": self.micro_indices,
            }
        if self.args.search_mode == "proxy_prefix":
            precision["taylor_fisher_proxy_prefix"] = {
                "role": "proxy-ordered group line search with exact convergence commit",
                "formula": (
                    "g_dot_delta_h + 0.5 * delta_h_dot_diag_empirical_fisher_dot_delta_h"
                ),
                "shortlist": self.args.proxy_shortlist,
                "positive_only": self.args.prefix_positive_only,
                "probe_prefix_sizes": self.args.prefix_sizes,
                "adaptive_prefix_sizes": self.args.adaptive_prefix_sizes,
                "prefix_caps": {
                    "near": self.args.prefix_near_max,
                    "mid": self.args.prefix_mid_max,
                    "mid_threshold": self.args.prefix_mid_threshold,
                },
                "probe_refine_rounds": self.args.prefix_refine_rounds,
                "convergence_finalists": self.args.prefix_convergence_finalists,
                "schedule": (
                    "repeat prefix search to directional convergence"
                    if self.args.prefix_until_no_change
                    else "one prefix search per direction per cycle"
                ),
                "max_tokens": self.args.proxy_max_tokens,
                "rows": self.args.proxy_rows,
                "first_weight": self.args.proxy_first_weight,
                "fisher_weight": self.args.proxy_fisher_weight,
            }
        if self.args.search_mode == "tf_trust":
            precision["tf_trust"] = {
                "role": (
                    "one Taylor--Fisher ranking per phase; the top positive-gain one-level "
                    "moves form one batch committed after an independent split check"
                ),
                "formula": (
                    "g_dot_delta_h + 0.5 * delta_h_dot_diag_empirical_fisher_dot_delta_h"
                ),
                "gradient_split": "probe",
                "batch_cap": self.args.tf_trust_moves,
                "halving_retries": self.args.tf_trust_retries,
                "verify_split": self.args.tf_verify_split,
                "max_tokens": self.args.proxy_max_tokens,
                "rows": self.args.proxy_rows,
                "first_weight": self.args.proxy_first_weight,
                "fisher_weight": self.args.proxy_fisher_weight,
            }
        if self.args.search_mode == "fast_target_exchange":
            precision["fast_target_exchange"] = {
                "target_phase": (
                    "Taylor--Fisher ranks all eligible one-level moves; only positive "
                    "proxy-gain geometric prefixes enter micro NLL; the best finalist "
                    "is committed only when independent convergence J decreases"
                ),
                "exchange_phase": (
                    "inside bit tolerance, freeze lambda and match cross-level down "
                    "donors with up receivers by actual log2(level) cost; commit only "
                    "when independent convergence NLL decreases"
                ),
                "prefix_sizes": self.args.fast_prefix_sizes,
                "prefix_trust_ratio": self.args.fast_prefix_trust_ratio,
                "convergence_finalists": self.args.fast_convergence_finalists,
                "exchange_pool_size_per_direction": self.args.exchange_pool_size,
                "exchange_max_per_side": self.args.exchange_max_per_side,
                "exchange_bundle_limit": self.args.exchange_bundle_limit,
                "exchange_micro_finalists": self.args.exchange_micro_finalists,
                "exchange_convergence_finalists": (
                    self.args.exchange_convergence_finalists
                ),
                "exchange_average_bit_balance_tolerance": (
                    self.args.exchange_balance_tolerance
                ),
                "exchange_no_change_rounds": self.args.exchange_no_change_rounds,
                "max_tokens": self.args.proxy_max_tokens,
                "rows": self.args.proxy_rows,
                "first_weight": self.args.proxy_first_weight,
                "fisher_weight": self.args.proxy_fisher_weight,
                "micro_indices": self.micro_indices,
            }
        if self.args.search_mode == "branch_bound":
            precision["branch_bound"] = {
                "split": self.args.bb_split,
                "leaf_lower_bound": self.args.bb_leaf_lower_bound,
                "interaction_slack": self.args.bb_interaction_slack,
                "interaction_slack_per_unit": self.args.bb_interaction_slack_per_unit,
                "measurement_margin": self.args.bb_measurement_margin,
                "anchor_prior": (
                    str(self.args.bb_anchor_prior.resolve())
                    if self.args.bb_anchor_prior else None
                ),
                "epsilon_anchors": self.args.bb_epsilon_anchors,
                "epsilon_groups_per_anchor": self.args.bb_epsilon_groups_per_anchor,
                "epsilon_safety_factor": self.args.bb_epsilon_safety_factor,
                "epsilon_denominator_floor": self.args.bb_epsilon_denominator_floor,
                "adaptive_epsilon": self.args.bb_adaptive_epsilon,
                "epsilon_lifetime": (
                    "calibrate once per direction, then retain the maximum "
                    "adaptive epsilon for later searches"
                ),
                "deferred_reactivation": True,
                "keep_ties": self.args.bb_keep_ties,
                "group_score_definition": (
                    "joint_group_loss_utility + max_singleton_bit_reward_in_group"
                ),
                "traversal": "upper_bound_ordered_depth_first",
            }
        atomic_json(self.args.out_dir / "precision_map.json", precision)
        self.state["status"] = "complete"
        self.state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(self.args.out_dir / "optimization_summary.json", self.state)
        atomic_json(self.args.work_dir / "state.json", self.state)
        (self.args.out_dir / ".quant_done").touch()
        print(
            f"[complete] lambda={self.args.lambda_value} cycles={self.state['cycles_completed']} "
            f"searches={self.state['attempts_completed']} "
            f"bits={bits:.8f} final_nll={final_nll:.8f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument(
        "--adaptive-lambda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="update lambda after every complete down/up cycle to enforce a bit target",
    )
    parser.add_argument("--target-average-bits", type=float, default=2.0)
    parser.add_argument("--dual-lr", type=float, default=0.25)
    parser.add_argument(
        "--stabilized-adaptive-lambda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "hold lambda fixed until a full down/up pair has no changes, then "
            "use a deadbanded distance-adaptive controller"
        ),
    )
    parser.add_argument(
        "--clipped-adaptive-lambda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after fixed-lambda inner convergence, use a deadbanded minimum/"
            "maximum clipped dual step instead of the near-target smooth step"
        ),
    )
    parser.add_argument("--dual-max-step", type=float, default=0.01)
    parser.add_argument("--dual-min-step", type=float, default=0.005)
    parser.add_argument("--dual-response-gain", type=float, default=0.5)
    parser.add_argument("--dual-min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--dual-full-lr-error", type=float, default=0.05)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--bit-tolerance", type=float, default=0.01)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument(
        "--initial-level",
        type=int,
        default=5,
        help=(
            "Uniform routed-expert level at optimization start. The immutable "
            "5-level checkpoint is loaded first, then all experts are restored "
            "from the level bank when this value differs from five."
        ),
    )
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-level", type=int, default=3)
    parser.add_argument("--max-level", type=int, default=16)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=50,
        help="maximum complete down-to-convergence/up-to-convergence cycles",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--search-mode",
        choices=(
            "legacy",
            "verified_beam",
            "branch_bound",
            "proxy_batch",
            "proxy_prefix",
            "fast_target_exchange",
            "tf_trust",
        ),
        default="branch_bound",
    )
    parser.add_argument("--screen-divisor", type=int, default=4)
    parser.add_argument("--top-layers", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--beam-seeds", default="17,73")
    parser.add_argument("--bb-split", choices=("screen", "probe"), default="probe")
    parser.add_argument("--bb-leaf-lower-bound", type=float, default=0.0)
    parser.add_argument("--bb-interaction-slack", type=float, default=0.0)
    parser.add_argument("--bb-interaction-slack-per-unit", type=float, default=0.0)
    parser.add_argument("--bb-measurement-margin", type=float, default=0.0)
    parser.add_argument("--bb-anchor-prior", type=Path, default=DEFAULT_ANCHOR_PRIOR)
    parser.add_argument("--bb-epsilon-anchors", type=int, default=10)
    parser.add_argument("--bb-epsilon-groups-per-anchor", type=int, default=2)
    parser.add_argument("--bb-epsilon-safety-factor", type=float, default=2.0)
    parser.add_argument("--bb-epsilon-denominator-floor", type=float, default=1e-12)
    parser.add_argument(
        "--bb-adaptive-epsilon",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bb-keep-ties", action="store_true")
    parser.add_argument("--proxy-shortlist", type=int, default=1000)
    parser.add_argument("--proxy-audit", type=int, default=64)
    parser.add_argument("--proxy-stage1-keep", type=int, default=256)
    parser.add_argument("--proxy-stage2-keep", type=int, default=64)
    parser.add_argument("--proxy-finalists", type=int, default=32)
    parser.add_argument("--proxy-accept-limit", type=int, default=32)
    parser.add_argument("--proxy-failure-limit", type=int, default=32)
    parser.add_argument("--proxy-max-tokens", type=int, default=512)
    parser.add_argument("--proxy-rows", type=int, default=1)
    parser.add_argument("--proxy-micro-divisor", type=int, default=16)
    parser.add_argument("--proxy-first-weight", type=float, default=1.0)
    parser.add_argument("--proxy-fisher-weight", type=float, default=1.0)
    parser.add_argument("--proxy-seed", type=int, default=20260910)
    parser.add_argument("--proxy-smoke-only", action="store_true")
    parser.add_argument("--proxy-batch-smoke-only", action="store_true")
    parser.add_argument("--prefix-sizes", default="32,64,128,256,512,1000")
    parser.add_argument(
        "--prefix-until-no-change",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--adaptive-prefix-sizes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prefix-near-max", type=int, default=16)
    parser.add_argument("--prefix-mid-max", type=int, default=128)
    parser.add_argument("--prefix-mid-threshold", type=float, default=0.05)
    parser.add_argument("--prefix-refine-rounds", type=int, default=2)
    parser.add_argument(
        "--prefix-scan-split", choices=("probe", "screen"), default="probe",
        help="validation split for the prefix-size scan and refinement (screen = stratified probe/4)",
    )
    parser.add_argument(
        "--prefix-finalist-split", choices=("convergence", "probe"), default="convergence",
        help="probe: re-score the scan finalists on probe and confirm only the best on convergence",
    )
    parser.add_argument(
        "--prefix-confirm-count", type=int, default=1,
        help="with --prefix-finalist-split probe: how many of the probe-ranked finalists are confirmed on convergence",
    )
    parser.add_argument(
        "--prefix-shrink-retries", type=int, default=0,
        help="after a rejected convergence confirmation, retry at half the prefix size this many times",
    )
    parser.add_argument("--prefix-convergence-finalists", type=int, default=3)
    parser.add_argument(
        "--prefix-positive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--proxy-prefix-smoke-only", action="store_true")
    parser.add_argument(
        "--fast-prefix-sizes",
        default="1,4,16,64,128,256,512,1000",
        help="geometric prefix sizes screened on the micro validation split",
    )
    parser.add_argument("--fast-prefix-trust-ratio", type=float, default=0.5)
    parser.add_argument("--fast-convergence-finalists", type=int, default=1)
    parser.add_argument("--exchange-pool-size", type=int, default=128)
    parser.add_argument("--exchange-max-per-side", type=int, default=16)
    parser.add_argument("--exchange-bundle-limit", type=int, default=32)
    parser.add_argument("--exchange-beam-width", type=int, default=256)
    parser.add_argument("--exchange-micro-finalists", type=int, default=8)
    parser.add_argument("--exchange-convergence-finalists", type=int, default=2)
    parser.add_argument(
        "--exchange-balance-tolerance",
        type=float,
        default=1e-5,
        help="maximum absolute average-bit change of a bit-neutral bundle",
    )
    parser.add_argument("--exchange-no-change-rounds", type=int, default=2)
    parser.add_argument(
        "--tf-trust-moves", type=int, default=512,
        help="tf_trust: cap on one-level moves taken from the top of one Taylor--Fisher ranking per phase",
    )
    parser.add_argument(
        "--tf-trust-retries", type=int, default=2,
        help="tf_trust: after a rejected batch, retry at half its size this many times",
    )
    parser.add_argument(
        "--tf-verify-split", choices=("convergence", "none"), default="convergence",
        help="tf_trust: independent split that accepts or rejects each batch (none commits it unchecked)",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--replay-all-levels", action="store_true",
        help="on load, install every expert from the bank, including level 5 (needed when the bank is "
             "not the one the K=5 physical base was materialized from)",
    )
    parser.add_argument("--no-change-pairs", type=int, default=1)
    args = parser.parse_args()
    if args.max_attempts is not None:
        args.max_cycles = args.max_attempts
    args.beam_seeds = [int(value) for value in args.beam_seeds.split(",") if value.strip()]
    args.prefix_sizes = sorted({
        int(value) for value in args.prefix_sizes.split(",") if value.strip()
    })
    args.fast_prefix_sizes = sorted({
        int(value) for value in args.fast_prefix_sizes.split(",") if value.strip()
    })
    torch.set_num_threads(args.num_threads)
    if args.lambda_value < 0:
        raise ValueError("lambda must be non-negative")
    if args.adaptive_lambda and (args.lambda_min < 0 or args.lambda_max < args.lambda_min):
        raise ValueError("lambda bounds must satisfy 0 <= min <= max")
    if args.adaptive_lambda and not args.lambda_min <= args.lambda_value <= args.lambda_max:
        raise ValueError("initial lambda must lie inside [lambda-min, lambda-max]")
    if args.adaptive_lambda and args.dual_lr <= 0:
        raise ValueError("dual-lr must be positive")
    if args.adaptive_lambda and args.bit_tolerance < 0:
        raise ValueError("bit-tolerance must be non-negative")
    if args.stabilized_adaptive_lambda and not args.adaptive_lambda:
        raise ValueError("stabilized adaptive lambda requires --adaptive-lambda")
    if args.clipped_adaptive_lambda and not args.stabilized_adaptive_lambda:
        raise ValueError(
            "clipped adaptive lambda requires --stabilized-adaptive-lambda"
        )
    if args.dual_max_step <= 0:
        raise ValueError("dual-max-step must be positive")
    if not 0 < args.dual_min_step <= args.dual_max_step:
        raise ValueError("dual-min-step must lie in (0, dual-max-step]")
    if args.dual_response_gain <= 0:
        raise ValueError("dual-response-gain must be positive")
    if not 0 < args.dual_min_lr_ratio <= 1:
        raise ValueError("dual-min-lr-ratio must lie in (0, 1]")
    if args.dual_full_lr_error <= args.bit_tolerance:
        raise ValueError("dual-full-lr-error must exceed bit-tolerance")
    if args.adaptive_lambda and not (
        math.log2(args.min_level)
        <= args.target_average_bits
        <= math.log2(args.max_level)
    ):
        raise ValueError("target-average-bits must be reachable inside the level bounds")
    if not args.min_level <= args.initial_level <= args.max_level:
        raise ValueError("initial-level must lie inside [min-level, max-level]")
    if args.no_change_pairs < 1:
        raise ValueError("no-change-pairs must be positive")
    if args.max_cycles < 1:
        raise ValueError("max-cycles must be positive")
    positive_proxy_arguments = (
        args.proxy_shortlist,
        args.proxy_stage1_keep,
        args.proxy_stage2_keep,
        args.proxy_finalists,
        args.proxy_accept_limit,
        args.proxy_failure_limit,
        args.proxy_max_tokens,
        args.proxy_rows,
        args.proxy_micro_divisor,
    )
    if min(positive_proxy_arguments) < 1:
        raise ValueError("proxy sizes, limits, rows, tokens, and divisor must be positive")
    if args.proxy_audit < 0:
        raise ValueError("proxy-audit must be non-negative")
    if min(args.proxy_first_weight, args.proxy_fisher_weight) < 0:
        raise ValueError("proxy Taylor/Fisher weights must be non-negative")
    if args.search_mode == "proxy_batch" and not (
        args.proxy_accept_limit <= args.proxy_finalists
        <= args.proxy_stage2_keep <= args.proxy_stage1_keep
        <= args.proxy_shortlist + args.proxy_audit
    ):
        raise ValueError("proxy funnel sizes must be monotonically non-increasing")
    if not args.prefix_sizes or min(args.prefix_sizes) < 1:
        raise ValueError("prefix-sizes must contain positive integers")
    if args.prefix_refine_rounds < 0:
        raise ValueError("prefix-refine-rounds must be non-negative")
    if args.prefix_shrink_retries < 0:
        raise ValueError("prefix-shrink-retries must be non-negative")
    if args.prefix_confirm_count < 1:
        raise ValueError("prefix-confirm-count must be positive")
    if args.prefix_convergence_finalists < 1:
        raise ValueError("prefix-convergence-finalists must be positive")
    if args.adaptive_prefix_sizes and not args.adaptive_lambda:
        raise ValueError("adaptive prefix sizes require --adaptive-lambda")
    if min(args.prefix_near_max, args.prefix_mid_max) < 1:
        raise ValueError("adaptive prefix caps must be positive")
    if args.prefix_near_max > args.prefix_mid_max:
        raise ValueError("prefix-near-max cannot exceed prefix-mid-max")
    if args.prefix_mid_threshold <= args.bit_tolerance:
        raise ValueError("prefix-mid-threshold must exceed bit-tolerance")
    if args.search_mode == "fast_target_exchange" and not args.adaptive_lambda:
        raise ValueError("fast_target_exchange requires --adaptive-lambda")
    if not args.fast_prefix_sizes or min(args.fast_prefix_sizes) < 1:
        raise ValueError("fast-prefix-sizes must contain positive integers")
    if not 0 < args.fast_prefix_trust_ratio <= 1:
        raise ValueError("fast-prefix-trust-ratio must lie in (0, 1]")
    if min(
        args.fast_convergence_finalists,
        args.exchange_pool_size,
        args.exchange_max_per_side,
        args.exchange_bundle_limit,
        args.exchange_beam_width,
        args.exchange_micro_finalists,
        args.exchange_convergence_finalists,
        args.exchange_no_change_rounds,
    ) < 1:
        raise ValueError("fast-target and exchange counts must be positive")
    if args.exchange_balance_tolerance < 0:
        raise ValueError("exchange-balance-tolerance must be non-negative")
    if args.tf_trust_moves < 1 or args.tf_trust_retries < 0:
        raise ValueError("tf-trust-moves must be positive and tf-trust-retries non-negative")
    if args.exchange_convergence_finalists > args.exchange_micro_finalists:
        raise ValueError(
            "exchange convergence finalists cannot exceed micro finalists"
        )
    if args.bb_epsilon_anchors < 0:
        raise ValueError("bb-epsilon-anchors must be non-negative")
    if args.bb_epsilon_groups_per_anchor < 0:
        raise ValueError("bb-epsilon-groups-per-anchor must be non-negative")
    if args.bb_epsilon_safety_factor < 1:
        raise ValueError("bb-epsilon-safety-factor must be at least one")
    if args.bb_epsilon_denominator_floor < 0:
        raise ValueError("bb-epsilon-denominator-floor must be non-negative")
    if min(
        args.bb_interaction_slack,
        args.bb_interaction_slack_per_unit,
        args.bb_measurement_margin,
    ) < 0:
        raise ValueError("branch-bound safety margins must be non-negative")
    if not (args.bank_dir / ".bank_done").exists():
        raise FileNotFoundError(f"TWLA level bank is incomplete: {args.bank_dir}")
    if not (args.initial_checkpoint / ".quant_done").exists():
        raise FileNotFoundError(
            f"physical base checkpoint is incomplete: {args.initial_checkpoint}"
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "lambda": args.lambda_value,
        "initial_level": args.initial_level,
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "bank_dir": str(args.bank_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "min_level": args.min_level,
        "max_level": args.max_level,
        "max_cycles": args.max_cycles,
        "min_improvement": args.min_improvement,
        "search_mode": args.search_mode,
        "screen_divisor": args.screen_divisor,
        "top_layers": args.top_layers,
        "beam_width": args.beam_width,
        "beam_seeds": args.beam_seeds,
        "audit_only": args.audit_only,
        "no_change_pairs": args.no_change_pairs,
        "proxy_smoke_only": args.proxy_smoke_only,
        "proxy_batch_smoke_only": args.proxy_batch_smoke_only,
        "proxy_prefix_smoke_only": args.proxy_prefix_smoke_only,
    }
    if args.adaptive_lambda:
        run_config.update({
            "adaptive_lambda": True,
            "target_average_bits": args.target_average_bits,
            "dual_lr": args.dual_lr,
            "stabilized_adaptive_lambda": args.stabilized_adaptive_lambda,
            "clipped_adaptive_lambda": args.clipped_adaptive_lambda,
            "dual_max_step": args.dual_max_step,
            "dual_min_step": args.dual_min_step,
            "dual_response_gain": args.dual_response_gain,
            "dual_min_lr_ratio": args.dual_min_lr_ratio,
            "dual_full_lr_error": args.dual_full_lr_error,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "bit_tolerance": args.bit_tolerance,
        })
    if args.search_mode in {"proxy_batch", "proxy_prefix", "fast_target_exchange", "tf_trust"}:
        run_config.update({
            "proxy_shortlist": args.proxy_shortlist,
            "proxy_audit": args.proxy_audit,
            "proxy_stage1_keep": args.proxy_stage1_keep,
            "proxy_stage2_keep": args.proxy_stage2_keep,
            "proxy_finalists": args.proxy_finalists,
            "proxy_accept_limit": args.proxy_accept_limit,
            "proxy_failure_limit": args.proxy_failure_limit,
            "proxy_max_tokens": args.proxy_max_tokens,
            "proxy_rows": args.proxy_rows,
            "proxy_micro_divisor": args.proxy_micro_divisor,
            "proxy_first_weight": args.proxy_first_weight,
            "proxy_fisher_weight": args.proxy_fisher_weight,
            "proxy_seed": args.proxy_seed,
        })
    if args.search_mode == "fast_target_exchange":
        run_config.update({
            "fast_prefix_sizes": args.fast_prefix_sizes,
            "fast_prefix_trust_ratio": args.fast_prefix_trust_ratio,
            "fast_convergence_finalists": args.fast_convergence_finalists,
            "exchange_pool_size": args.exchange_pool_size,
            "exchange_max_per_side": args.exchange_max_per_side,
            "exchange_bundle_limit": args.exchange_bundle_limit,
            "exchange_beam_width": args.exchange_beam_width,
            "exchange_micro_finalists": args.exchange_micro_finalists,
            "exchange_convergence_finalists": args.exchange_convergence_finalists,
            "exchange_balance_tolerance": args.exchange_balance_tolerance,
            "exchange_no_change_rounds": args.exchange_no_change_rounds,
        })
    if args.search_mode == "tf_trust":
        run_config.update({
            "tf_trust_moves": args.tf_trust_moves,
            "tf_trust_retries": args.tf_trust_retries,
            "tf_verify_split": args.tf_verify_split,
        })
    if args.search_mode == "proxy_prefix":
        run_config.update({
            "prefix_sizes": args.prefix_sizes,
            "prefix_refine_rounds": args.prefix_refine_rounds,
            "prefix_scan_split": args.prefix_scan_split,
            "prefix_finalist_split": args.prefix_finalist_split,
            "prefix_shrink_retries": args.prefix_shrink_retries,
            "prefix_confirm_count": args.prefix_confirm_count,
            "prefix_convergence_finalists": args.prefix_convergence_finalists,
            "prefix_positive_only": args.prefix_positive_only,
            "prefix_until_no_change": args.prefix_until_no_change,
            "adaptive_prefix_sizes": args.adaptive_prefix_sizes,
            "prefix_near_max": args.prefix_near_max,
            "prefix_mid_max": args.prefix_mid_max,
            "prefix_mid_threshold": args.prefix_mid_threshold,
        })
    if args.search_mode == "branch_bound":
        run_config.update({
            "bb_split": args.bb_split,
            "bb_leaf_lower_bound": args.bb_leaf_lower_bound,
            "bb_interaction_slack": args.bb_interaction_slack,
            "bb_interaction_slack_per_unit": args.bb_interaction_slack_per_unit,
            "bb_measurement_margin": args.bb_measurement_margin,
            "bb_anchor_prior": (
                str(args.bb_anchor_prior.resolve()) if args.bb_anchor_prior else None
            ),
            "bb_epsilon_anchors": args.bb_epsilon_anchors,
            "bb_epsilon_groups_per_anchor": args.bb_epsilon_groups_per_anchor,
            "bb_epsilon_safety_factor": args.bb_epsilon_safety_factor,
            "bb_epsilon_denominator_floor": args.bb_epsilon_denominator_floor,
            "bb_adaptive_epsilon": args.bb_adaptive_epsilon,
            "bb_keep_ties": args.bb_keep_ties,
        })
    state_path = args.work_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("run_config") != run_config:
            raise ValueError("work directory contains a different configuration")
        if state.get("status") == "complete" and (args.out_dir / ".quant_done").exists():
            print("[resume] already complete", flush=True)
            return
    else:
        state = {
            "status": "initializing",
            "run_config": run_config,
            "levels": [
                [args.initial_level for _ in range(256)] for _ in range(40)
            ],
            "attempts_completed": 0,
            "cycles_completed": 0,
            "next_direction": "down",
            "phase_updates": 0,
            "cycle_had_change": False,
            "consecutive_no_change_cycles": 0,
            "probe_nll": None,
            "screen_nll": None,
            "convergence_nll": None,
            "micro_nll": None,
            "history": [],
            "phase_history": [],
            "cycle_history": [],
            "branch_bound_epsilon": {},
            "proxy_rounds_completed": 0,
            "proxy_batch_history": [],
            "proxy_prefix_history": [],
            "fast_target_history": [],
            "exchange_history": [],
            "fast_stage": "target",
            "exchange_no_change_rounds": 0,
            "initial_lambda": args.lambda_value,
            "current_lambda": args.lambda_value,
            "lambda_history": [],
            "previous_dual_residual": None,
        }
        atomic_json(state_path, state)

    # Explicit fields make an interrupted future V3 run resumable at the same
    # direction phase. They are harmless defaults for a state written before
    # these counters were introduced.
    state.setdefault("cycles_completed", 0)
    state.setdefault("phase_updates", 0)
    state.setdefault("cycle_had_change", False)
    state.setdefault("consecutive_no_change_cycles", 0)
    state.setdefault("phase_history", [])
    state.setdefault("cycle_history", [])
    state.setdefault("branch_bound_epsilon", {})
    state.setdefault("proxy_rounds_completed", 0)
    state.setdefault("proxy_batch_history", [])
    state.setdefault("proxy_prefix_history", [])
    state.setdefault("fast_target_history", [])
    state.setdefault("exchange_history", [])
    state.setdefault("fast_stage", "target")
    state.setdefault("exchange_no_change_rounds", 0)
    state.setdefault("micro_nll", None)
    state.setdefault("initial_lambda", state["run_config"]["lambda"])
    state.setdefault("current_lambda", state["initial_lambda"])
    state.setdefault("lambda_history", [])
    state.setdefault("previous_dual_residual", None)
    if args.adaptive_lambda:
        args.lambda_value = float(state["current_lambda"])

    print(f"[load] checkpoint={args.initial_checkpoint} device={args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.initial_checkpoint)
    model = load_pretrained_streaming(args.initial_checkpoint, args.device, dtype=torch.bfloat16)
    model.eval()
    model.model.language_model.config.use_cache = False

    # Reapply every assignment that differs from the immutable physical
    # all-5-level checkpoint. This both creates a uniform 3/4-level logical
    # initialization and restores interrupted mixed-level states exactly.
    replay = {
        (layer, expert): int(level)
        for layer, row in enumerate(state["levels"])
        for expert, level in enumerate(row)
        if args.replay_all_levels or int(level) != 5
    }
    if replay:
        print(f"[resume] replaying {len(replay)} committed expert assignments", flush=True)
        install_assignments(model.model.language_model, replay, args.bank_dir, args.device)

    optimizer = Optimizer(args, model, tokenizer, state)
    if args.proxy_smoke_only:
        if args.search_mode not in {"proxy_batch", "proxy_prefix", "fast_target_exchange"}:
            raise ValueError(
                "--proxy-smoke-only requires --search-mode proxy_batch or proxy_prefix"
            )
        scores = optimizer.taylor_fisher_shortlist(state["next_direction"])
        print(
            f"[proxy-smoke] candidates={len(scores)} "
            f"top={scores[0] if scores else None}",
            flush=True,
        )
        return
    if (
        state.get("screen_nll") is None
        or state["probe_nll"] is None
        or state["convergence_nll"] is None
        or (
            args.search_mode == "fast_target_exchange"
            and state.get("micro_nll") is None
        )
    ):
        if args.search_mode == "fast_target_exchange":
            print("[baseline] evaluating micro split", flush=True)
            state["micro_nll"], micro_tokens = validation_nll(
                model, optimizer.micro, args.device
            )
            state["micro_loss_tokens"] = micro_tokens
            state["micro_indices"] = optimizer.micro_indices
        print("[baseline] evaluating stratified screen split", flush=True)
        state["screen_nll"], screen_tokens = validation_nll(
            model, optimizer.screen, args.device
        )
        print("[baseline] evaluating probe split", flush=True)
        state["probe_nll"], probe_tokens = validation_nll(model, optimizer.probe, args.device)
        print("[baseline] evaluating convergence split", flush=True)
        state["convergence_nll"], convergence_tokens = validation_nll(
            model, optimizer.convergence, args.device
        )
        state["probe_loss_tokens"] = probe_tokens
        state["screen_loss_tokens"] = screen_tokens
        state["screen_indices"] = optimizer.screen_indices
        state["convergence_loss_tokens"] = convergence_tokens
        state["status"] = "optimizing"
        atomic_json(state_path, state)
    if args.proxy_batch_smoke_only:
        if args.search_mode != "proxy_batch":
            raise ValueError("--proxy-batch-smoke-only requires --search-mode proxy_batch")
        accepted = optimizer.proxy_batch_attempt(state["next_direction"])
        print(f"[proxy-batch-smoke] accepted={accepted}", flush=True)
        return
    if args.proxy_prefix_smoke_only:
        if args.search_mode != "proxy_prefix":
            raise ValueError("--proxy-prefix-smoke-only requires --search-mode proxy_prefix")
        accepted = optimizer.proxy_prefix_attempt(state["next_direction"])
        print(f"[proxy-prefix-smoke] accepted={accepted}", flush=True)
        return
    optimizer.run()
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

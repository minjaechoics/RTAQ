"""Activation-space Taylor--Fisher proxy for routed-expert level moves.

The proxy is deliberately used only to rank candidates.  A caller must
validate every committed move with the exact validation-NLL objective.

For a routed expert output perturbation ``delta`` and the validation-loss
gradient ``g`` at the current allocation, the estimated loss change is

    g^T delta + 0.5 * delta^T diag(F) delta,

where the empirical diagonal Fisher is approximated by squared activation
gradients.  The implementation evaluates the *actual* target TWLA weights
from the level bank, so the perturbation is level- and state-specific.
"""

from __future__ import annotations

from contextlib import ExitStack
import math
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from quantize.E2M_ATQ_asymmetric_codebook import decode_asymmetric_representation


def _bank_path(bank_dir: Path, layer: int, level: int) -> Path:
    return bank_dir / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"


@torch.no_grad()
def _decode_expert_weights(handle, expert: int, level: int, device, dtype):
    """Decode one expert exactly as the checkpoint installer does."""

    center = (level - 1) / 2.0

    if "gate_up_codebook" in handle.keys():
        gate_up = decode_asymmetric_representation(
            handle.get_slice("gate_up_codes")[expert : expert + 1].to(device),
            handle.get_slice("gate_up_codebook")[expert : expert + 1].to(device).float(),
            handle.get_slice("gate_up_mu")[expert : expert + 1].to(device).float(),
            handle.get_slice("gate_up_alpha")[expert : expert + 1].to(device).float(),
            handle.get_slice("gate_up_rotation_left")[expert : expert + 1].to(device).float(),
            handle.get_slice("gate_up_rotation_right")[expert : expert + 1].to(device).float(),
        )[0].to(dtype)
        down = decode_asymmetric_representation(
            handle.get_slice("down_codes")[expert : expert + 1].to(device),
            handle.get_slice("down_codebook")[expert : expert + 1].to(device).float(),
            handle.get_slice("down_mu")[expert : expert + 1].to(device).float(),
            handle.get_slice("down_alpha")[expert : expert + 1].to(device).float(),
            handle.get_slice("down_rotation_left")[expert : expert + 1].to(device).float(),
            handle.get_slice("down_rotation_right")[expert : expert + 1].to(device).float(),
        )[0].to(dtype)
        return gate_up, down

    gate_up_codes = handle.get_slice("gate_up_codes")[expert : expert + 1]
    gate_up_mu = handle.get_slice("gate_up_mu")[expert : expert + 1]
    gate_up_alpha = handle.get_slice("gate_up_alpha")[expert : expert + 1]
    gate_up = gate_up_codes.to(device).float().sub_(center)
    gate_up.mul_(gate_up_alpha.to(device).float()[:, :, None])
    gate_up.add_(gate_up_mu.to(device).float()[:, :, None])
    gate_up = gate_up[0].to(dtype)

    down_codes = handle.get_slice("down_codes")[expert : expert + 1]
    down_mu = handle.get_slice("down_mu")[expert : expert + 1]
    down_alpha = handle.get_slice("down_alpha")[expert : expert + 1]
    down = down_codes.to(device).float().sub_(center)
    down.mul_(down_alpha.to(device).float()[:, :, None])
    down.add_(down_mu.to(device).float()[:, :, None])
    down = down[0].to(dtype)
    return gate_up, down


class RoutedExpertTaylorFisher:
    """Temporarily instruments every fused routed-expert module."""

    def __init__(
        self,
        text_model,
        levels: list[list[int]],
        direction: str,
        bank_dir: Path,
        min_level: int,
        max_level: int,
        fisher_token_scale: float,
    ):
        if direction not in {"down", "up"}:
            raise ValueError(f"invalid direction: {direction}")
        self.text_model = text_model
        self.levels = levels
        self.direction = direction
        self.bank_dir = bank_dir
        self.min_level = min_level
        self.max_level = max_level
        self.fisher_token_scale = float(fisher_token_scale)
        self.stats: dict[tuple[int, int], dict[str, float | int]] = {}
        self._originals: list[tuple[object, object]] = []

    def _target_level(self, layer: int, expert: int) -> int | None:
        step = -1 if self.direction == "down" else 1
        target = int(self.levels[layer][expert]) + step
        if self.min_level <= target <= self.max_level:
            return target
        return None

    def _make_forward(self, layer: int):
        owner = self

        def instrumented_forward(module, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                target_levels = {
                    int(index[0]): owner._target_level(layer, int(index[0]))
                    for index in expert_hit
                    if int(index[0]) < module.num_experts
                }
                needed_levels = sorted({
                    level for level in target_levels.values() if level is not None
                })

            with ExitStack() as stack:
                handles = {
                    level: stack.enter_context(
                        safe_open(
                            str(_bank_path(owner.bank_dir, layer, level)),
                            framework="pt",
                            device="cpu",
                        )
                    )
                    for level in needed_levels
                }

                for expert_idx_tensor in expert_hit:
                    expert = int(expert_idx_tensor[0])
                    if expert == module.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert])
                    current_input = hidden_states[token_idx]
                    gate_up = F.linear(current_input, module.gate_up_proj[expert])
                    gate, up = gate_up.chunk(2, dim=-1)
                    intermediate = module.act_fn(gate) * up
                    down_output = F.linear(intermediate, module.down_proj[expert])
                    route_weight = top_k_weights[token_idx, top_k_pos, None]
                    weighted_output = down_output * route_weight

                    target_level = target_levels.get(expert)
                    if target_level is not None and weighted_output.requires_grad:
                        with torch.no_grad():
                            target_gate_up, target_down = _decode_expert_weights(
                                handles[target_level],
                                expert,
                                target_level,
                                current_input.device,
                                current_input.dtype,
                            )
                            target_gate, target_up = F.linear(
                                current_input.detach(), target_gate_up
                            ).chunk(2, dim=-1)
                            target_intermediate = module.act_fn(target_gate) * target_up
                            target_output = F.linear(target_intermediate, target_down)
                            delta = (
                                target_output - down_output.detach()
                            ) * route_weight.detach()
                            unit = (layer, expert)
                            entry = owner.stats.setdefault(
                                unit,
                                {
                                    "first_order": 0.0,
                                    "fisher_second_order": 0.0,
                                    "routing_count": 0,
                                    "gate_score_sum": 0.0,
                                },
                            )
                            entry["routing_count"] += int(token_idx.numel())
                            entry["gate_score_sum"] += float(route_weight.float().sum().cpu())

                        def accumulate(gradient, delta=delta, unit=unit):
                            product = gradient.detach().float() * delta.float()
                            first = float(product.double().sum().cpu())
                            fisher = 0.5 * owner.fisher_token_scale * float(
                                product.double().square().sum().cpu()
                            )
                            target = owner.stats[unit]
                            target["first_order"] += first
                            target["fisher_second_order"] += fisher

                        weighted_output.register_hook(accumulate)

                    final_hidden_states.index_add_(
                        0,
                        token_idx,
                        weighted_output.to(final_hidden_states.dtype),
                    )
            return final_hidden_states

        return instrumented_forward

    def install(self) -> None:
        if self._originals:
            raise RuntimeError("Taylor--Fisher instrumentation is already installed")
        for layer, decoder in enumerate(self.text_model.layers):
            experts = decoder.mlp.experts
            self._originals.append((experts, experts.forward))
            experts.forward = types.MethodType(self._make_forward(layer), experts)

    def restore(self) -> None:
        for experts, original in self._originals:
            experts.forward = original
        self._originals.clear()


def _selected_proxy_rows(
    payload: dict[str, torch.Tensor],
    max_tokens: int,
    rows: int,
    offset: int,
) -> list[int]:
    """Choose loss-dense rows, rotating ties/near-ties across proxy rounds."""

    scored = []
    for row in range(int(payload["input_ids"].shape[0])):
        length = min(int(payload["attention_mask"][row].sum()), max_tokens)
        loss_tokens = int(payload["loss_mask"][row, 1:length].sum())
        scored.append((loss_tokens, length, -row, row))
    scored.sort(reverse=True)
    pool = [item[-1] for item in scored[: max(rows * 8, rows)]]
    if not pool:
        raise RuntimeError("proxy dataset has no rows")
    start = offset % len(pool)
    return [pool[(start + index) % len(pool)] for index in range(min(rows, len(pool)))]


def collect_taylor_fisher_scores(
    model,
    text_model,
    payload: dict[str, torch.Tensor],
    levels: list[list[int]],
    direction: str,
    bank_dir: Path,
    device: str,
    min_level: int,
    max_level: int,
    max_tokens: int = 512,
    rows: int = 1,
    row_offset: int = 0,
) -> tuple[dict[tuple[int, int], dict], dict]:
    """Collect current-state Taylor--Fisher terms for all eligible experts."""

    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    if rows < 1:
        raise ValueError("rows must be positive")

    selected_rows = _selected_proxy_rows(payload, max_tokens, rows, row_offset)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.enable_input_require_grads()
    aggregate: dict[tuple[int, int], dict] = {}
    row_summaries = []
    started = time.time()

    try:
        for row in selected_rows:
            length = min(int(payload["attention_mask"][row].sum()), max_tokens)
            ids = payload["input_ids"][row, :length].to(device).unsqueeze(0)
            loss_mask = payload["loss_mask"][row, 1:length].to(device).bool()
            positions = loss_mask.nonzero(as_tuple=False).flatten()
            token_count = int(positions.numel())
            if token_count == 0:
                raise RuntimeError(f"proxy row {row} has no selected loss tokens")

            collector = RoutedExpertTaylorFisher(
                text_model=text_model,
                levels=levels,
                direction=direction,
                bank_dir=bank_dir,
                min_level=min_level,
                max_level=max_level,
                fisher_token_scale=float(token_count),
            )
            collector.install()
            model.zero_grad(set_to_none=True)
            row_started = time.time()
            try:
                output = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits[0, :-1]
                labels = ids[0, 1:]
                loss_sum = None
                for start in range(0, token_count, 64):
                    chosen = positions[start : start + 64]
                    scores = logits.index_select(0, chosen).float()
                    targets = labels.index_select(0, chosen)
                    term = (
                        torch.logsumexp(scores, dim=1)
                        - scores.gather(1, targets[:, None]).squeeze(1)
                    ).sum()
                    loss_sum = term if loss_sum is None else loss_sum + term
                assert loss_sum is not None
                loss = loss_sum / token_count
                loss.backward()
                loss_value = float(loss.detach().cpu())
            finally:
                collector.restore()

            for unit, values in collector.stats.items():
                target = aggregate.setdefault(
                    unit,
                    {
                        "first_order": 0.0,
                        "fisher_second_order": 0.0,
                        "routing_count": 0,
                        "gate_score_sum": 0.0,
                    },
                )
                for key in ("first_order", "fisher_second_order", "gate_score_sum"):
                    target[key] += float(values[key])
                target["routing_count"] += int(values["routing_count"])
            row_summaries.append({
                "row": row,
                "sequence_tokens": length,
                "loss_tokens": token_count,
                "nll": loss_value,
                "routed_experts": len(collector.stats),
                "elapsed_seconds": time.time() - row_started,
            })
            del ids, loss_mask, positions, output, logits, labels, loss_sum, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    finally:
        model.disable_input_require_grads()

    divisor = float(len(selected_rows))
    for values in aggregate.values():
        values["first_order"] /= divisor
        values["fisher_second_order"] /= divisor
        values["gate_score_sum"] /= divisor
        values["routing_count"] = int(round(values["routing_count"] / divisor))

    summary = {
        "direction": direction,
        "rows": selected_rows,
        "max_tokens": max_tokens,
        "row_summaries": row_summaries,
        "covered_experts": len(aggregate),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    return aggregate, summary


def objective_gain_from_proxy(
    first_order: float,
    fisher_second_order: float,
    current_level: int,
    target_level: int,
    lambda_value: float,
    total_units: int,
    first_weight: float = 1.0,
    fisher_weight: float = 1.0,
) -> float:
    """Predicted ``J(current) - J(candidate)`` for one level move."""

    loss_delta = first_weight * first_order + fisher_weight * fisher_second_order
    bit_reward = lambda_value * (
        math.log2(current_level) - math.log2(target_level)
    ) / total_units
    return -loss_delta + bit_reward

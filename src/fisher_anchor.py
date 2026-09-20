#!/usr/bin/env python3
"""BF16-anchored Fisher-weighted routed-expert costs and knapsack allocation (qwen_fisher_anchor_v1).

For routed expert e of layer l stored at TWLA level K, the loss increase is estimated around the BF16
model, where the expected loss gradient vanishes, as

    C[l, e, K] = mean over rows r of 0.5 * T_r * sum_{t in r} sum_d (g_{t,d} * delta_{t,d})^2,
    delta_t    = w_{t,e} * (f_e^K(x_t) - f_e^BF16(x_t)),

with g_t = dL_r/dh_t the gradient of row r's mean next-token loss at the output of the sparse MoE
block, x_t the routed input, w_{t,e} the router weight and T_r the row's loss tokens.  This is the
second-order term of taylor_fisher_proxy.py evaluated at the BF16 weights for every level at once, so
no first-order term (pure sampling noise at the optimum) enters the ranking and moves in both
directions are costed on one scale.

Stages (each resumable from its outputs):
  collect   one BF16 forward+backward per calibration row; per layer stores x, g*T_r, top-k index/weight
  costs     per bank and layer: decode every level once and accumulate C, the plain output error and
            the BF16 first-order term (diagnostic only)
  allocate  multiple-choice knapsack at the target average log2(K) with a dual lambda; candidate variants
            and their agreement with reference (validation-NLL) allocations
  select    install each candidate from the bank, measure convergence-split NLL, keep the best selectable
            candidate and save it as a checkpoint with final-split NLL
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

TWLA = Path(__file__).resolve().parent
sys.path.insert(0, str(TWLA))


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------------------------- collect

def cmd_collect(args) -> None:
    import torch
    from safetensors.torch import save_file
    from streaming_load import load_pretrained_streaming

    dtype = getattr(torch, args.dtype)
    store = getattr(torch, args.store_dtype)
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rows = list(range(args.row_start, min(args.row_start + args.rows, int(payload["input_ids"].shape[0]))))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (args.out_dir / f"row_{row:04d}.safetensors").exists()]
    log(f"collect rows={rows[0]}..{rows[-1]} pending={len(pending)} max_tokens={args.max_tokens}")
    input_device = args.device
    if pending:
        if getattr(args, "device_map", None):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                str(args.model_dir), dtype=dtype, device_map=args.device_map, low_cpu_mem_usage=True)
            input_device = model.get_input_embeddings().weight.device
            log(f"sharded master across devices with device_map={args.device_map}; inputs on {input_device}")
        else:
            model = load_pretrained_streaming(args.model_dir, args.device, dtype=dtype)
        model.eval()
        text = model.model.language_model
        text.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.enable_input_require_grads()
        if getattr(args, "grad_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            log("gradient checkpointing enabled")

    for row in pending:
        started = time.time()
        length = min(int(payload["attention_mask"][row].sum()), args.max_tokens)
        ids = payload["input_ids"][row, :length].to(input_device).unsqueeze(0)
        loss_mask = payload["loss_mask"][row, 1:length].to(input_device).bool()
        positions = loss_mask.nonzero(as_tuple=False).flatten()
        token_count = int(positions.numel())
        if token_count == 0:
            # Truncating to --max-tokens can cut a row's whole assistant span; skip it instead of failing the run.
            log(f"row {row}: no loss tokens within {length} tokens, skipped")
            continue
        captured: dict[tuple[int, str], torch.Tensor] = {}
        handles = []

        for layer, decoder in enumerate(text.layers):
            def gate_hook(_module, _inputs, output, layer=layer):
                _, weights, indices = output
                captured[(layer, "w")] = weights.detach().to(store)
                captured[(layer, "idx")] = indices.detach().to(torch.int32)

            def block_hook(_module, hook_args, hook_kwargs, output, layer=layer):
                hidden = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
                captured[(layer, "x")] = hidden.detach().reshape(-1, hidden.shape[-1]).to(store)
                if not output.requires_grad:
                    raise RuntimeError("MoE block output does not require grad; input grads are not enabled")

                def grad_hook(grad, layer=layer):
                    captured[(layer, "g")] = (grad.detach().float().reshape(-1, grad.shape[-1]) * token_count).to(store)

                output.register_hook(grad_hook)

            handles.append(decoder.mlp.gate.register_forward_hook(gate_hook))
            handles.append(decoder.mlp.register_forward_hook(block_hook, with_kwargs=True))
        try:
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            logits = output.logits[0, :-1]
            labels = ids[0, 1:]
            loss_sum = None
            for start in range(0, token_count, 64):
                chosen = positions[start:start + 64]
                scores = logits.index_select(0, chosen).float()
                targets = labels.index_select(0, chosen)
                term = (torch.logsumexp(scores, dim=1) - scores.gather(1, targets[:, None]).squeeze(1)).sum()
                loss_sum = term if loss_sum is None else loss_sum + term
            loss = loss_sum / token_count
            loss.backward()
            loss_value = float(loss.detach().cpu())
        finally:
            for handle in handles:
                handle.remove()
        tensors = {}
        for layer in range(len(text.layers)):
            for name in ("x", "g", "idx", "w"):
                tensors[f"layer{layer}.{name}"] = captured[(layer, name)].cpu().contiguous()
        temporary = args.out_dir / f"row_{row:04d}.safetensors.tmp"
        save_file(tensors, str(temporary), metadata={
            "row": str(row), "tokens": str(length), "loss_tokens": str(token_count), "nll": repr(loss_value),
            "gradient_scale": "g is dL/dh multiplied by loss_tokens", "store_dtype": args.store_dtype,
        })
        os.replace(temporary, args.out_dir / f"row_{row:04d}.safetensors")
        del captured, tensors, output, logits, labels, loss_sum, loss, ids, loss_mask, positions
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        log(f"row {row}: tokens={length} loss_tokens={token_count} nll={loss_value:.4f} "
            f"elapsed={time.time() - started:.1f}s")
    (args.out_dir / ".done").touch()


# ----------------------------------------------------------------------------------------------- costs

def cmd_costs(args) -> None:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.activations import ACT2FN

    from optimize_twla_hierarchical_nll_packedbank import bank_path_resolved
    from qwen_glmstyle_proxy_v1_bank_io import decode_range

    config = AutoConfig.from_pretrained(args.model_dir)
    text_config = getattr(config, "text_config", config)
    act = ACT2FN[text_config.hidden_act]
    num_layers, num_experts = int(text_config.num_hidden_layers), int(text_config.num_experts)
    levels = [int(value) for value in args.levels.split(",")]
    banks = dict(item.split("=", 1) for item in args.bank)
    row_files = sorted(glob.glob(str(args.moments / "row_*.safetensors")))[: args.max_rows or None]
    if not row_files:
        raise FileNotFoundError(f"no moment rows in {args.moments}")
    weight_map = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"] \
        if (args.model_dir / "model.safetensors.index.json").exists() else None
    prefix = "model.language_model.layers"
    compute = torch.float32
    log(f"costs banks={list(banks)} levels={levels} rows={len(row_files)} layers={num_layers} experts={num_experts}")

    def bf16_expert_weights(layer: int):
        tensors = {}
        for name in ("gate_up_proj", "down_proj"):
            key = f"{prefix}.{layer}.mlp.experts.{name}"
            shard = args.model_dir / weight_map[key] if weight_map else next(args.model_dir.glob("*.safetensors"))
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                tensors[name] = handle.get_tensor(key).to(args.device)
        return tensors["gate_up_proj"], tensors["down_proj"]

    def expert_forward(inputs, gate_up, down):
        gate, up = F.linear(inputs, gate_up.to(compute)).chunk(2, dim=-1)
        return F.linear(act(gate) * up, down.to(compute))

    for bank_name, bank_dir in banks.items():
        (args.out_dir / bank_name).mkdir(parents=True, exist_ok=True)

    for layer in range(num_layers):
        pending = [name for name in banks if not (args.out_dir / name / f"layer_{layer:02d}.safetensors").exists()]
        if not pending:
            continue
        started = time.time()
        xs, gs, idxs, ws, scales = [], [], [], [], []
        for row_file in row_files:
            with safe_open(row_file, framework="pt", device="cpu") as handle:
                loss_tokens = int((handle.metadata() or {})["loss_tokens"])
                x = handle.get_tensor(f"layer{layer}.x")
                xs.append(x)
                gs.append(handle.get_tensor(f"layer{layer}.g"))
                idxs.append(handle.get_tensor(f"layer{layer}.idx"))
                ws.append(handle.get_tensor(f"layer{layer}.w"))
                # Per-token factor that turns sum_d (g*T_r * delta)^2 into 0.5 * T_r * sum_d (dL/dh * delta)^2 / rows.
                scales.append(torch.full((x.shape[0],), 0.5 / (loss_tokens * len(row_files)), dtype=torch.float64))
        x = torch.cat(xs).to(args.device, compute)
        g = torch.cat(gs).to(args.device, compute)
        idx = torch.cat(idxs).to(args.device, torch.long)
        w = torch.cat(ws).to(args.device, compute)
        token_scale = torch.cat(scales).to(args.device)
        row_of_token = torch.cat([torch.full((xs[i].shape[0],), i, dtype=torch.long) for i in range(len(xs))]).to(args.device)
        loss_tokens_of_row = []
        for row_file in row_files:
            with safe_open(row_file, framework="pt", device="cpu") as handle:
                loss_tokens_of_row.append(int((handle.metadata() or {})["loss_tokens"]))
        first_scale = (1.0 / (torch.tensor(loss_tokens_of_row, dtype=torch.float64, device=args.device)
                              * len(row_files)))[row_of_token]
        del xs, gs, idxs, ws, scales
        top_k = idx.shape[1]
        token_of_route = torch.arange(x.shape[0], device=args.device).repeat_interleave(top_k)
        expert_of_route = idx.reshape(-1)
        weight_of_route = w.reshape(-1)
        order = torch.argsort(expert_of_route, stable=True)
        token_of_route, expert_of_route, weight_of_route = token_of_route[order], expert_of_route[order], weight_of_route[order]
        counts = torch.bincount(expert_of_route, minlength=num_experts)
        bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=args.device), counts.cumsum(0)]).tolist()
        gate_sum = torch.zeros(num_experts, dtype=torch.float64, device=args.device).index_add_(
            0, expert_of_route, weight_of_route.double())

        gate_up_ref, down_ref = bf16_expert_weights(layer)
        reference = torch.empty(token_of_route.numel(), x.shape[1], dtype=compute, device=args.device)
        for expert in range(num_experts):
            start, end = bounds[expert], bounds[expert + 1]
            if start < end:
                reference[start:end] = expert_forward(x[token_of_route[start:end]], gate_up_ref[expert], down_ref[expert])
        del gate_up_ref, down_ref

        for bank_name in pending:
            fisher = torch.zeros(num_experts, len(levels), dtype=torch.float64)
            first = torch.zeros(num_experts, len(levels), dtype=torch.float64)
            output_error = torch.zeros(num_experts, len(levels), dtype=torch.float64)
            gamma = torch.ones(num_experts, len(levels), dtype=torch.float64)
            for level_index, level in enumerate(levels):
                gate_up_k, down_k = decode_range(bank_path_resolved(Path(banks[bank_name]), layer, level), level,
                                                 0, num_experts, args.device)
                for expert in range(num_experts):
                    start, end = bounds[expert], bounds[expert + 1]
                    if start == end:
                        continue
                    tokens = token_of_route[start:end]
                    weighted = expert_forward(x[tokens], gate_up_k[expert], down_k[expert]) \
                        * weight_of_route[start:end, None]
                    weighted_reference = reference[start:end] * weight_of_route[start:end, None]
                    if args.gamma:
                        # Least-squares output scalar: the quantizer projects away what it cannot represent,
                        # so the surviving output is systematically short. One scalar per (expert, level)
                        # restores the magnitude the router already assumes.
                        denominator = float(weighted.double().square().sum())
                        scale = float((weighted.double() * weighted_reference.double()).sum()) / max(denominator, 1e-30)
                        gamma[expert, level_index] = scale
                        weighted = weighted * scale
                    delta = weighted - weighted_reference
                    product = g[tokens] * delta
                    fisher[expert, level_index] = float((product.double().square().sum(1) * token_scale[tokens]).sum())
                    first[expert, level_index] = float((product.double().sum(1) * first_scale[tokens]).sum())
                    output_error[expert, level_index] = float(delta.double().square().sum()) / x.shape[0]
                del gate_up_k, down_k
            temporary = args.out_dir / bank_name / f"layer_{layer:02d}.safetensors.tmp"
            save_file({
                "fisher": fisher, "first_order": first, "output_error": output_error,
                "routing_count": counts.cpu(), "gate_sum": gate_sum.cpu(), "gamma": gamma,
            }, str(temporary), metadata={"levels": ",".join(map(str, levels)), "rows": str(len(row_files)),
                                          "tokens": str(int(x.shape[0])), "bank": str(banks[bank_name])})
            os.replace(temporary, args.out_dir / bank_name / f"layer_{layer:02d}.safetensors")
        del x, g, idx, w, reference, token_scale, first_scale
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        log(f"layer {layer}: banks={pending} routes={int(counts.sum())} elapsed={time.time() - started:.1f}s")

    for bank_name, bank_dir in banks.items():
        atomic_json(args.out_dir / bank_name / "costs_meta.json", {
            "bank": str(bank_dir), "levels": levels, "rows": len(row_files), "moments": str(args.moments),
            "layers": num_layers, "experts": num_experts,
            "definition": "fisher[l,e,K] = mean_rows 0.5*T_r*sum_t sum_d (dL/dh * w*(f_K(x)-f_BF16(x)))^2",
            "gamma_corrected": bool(args.gamma),
        })
        (args.out_dir / bank_name / ".done").touch()


# -------------------------------------------------------------------------------------------- allocate

def load_cost_table(directory: Path):
    import numpy as np
    from safetensors.numpy import load_file

    meta = json.loads((directory / "costs_meta.json").read_text())
    tables = {"fisher": [], "output_error": [], "first_order": [], "routing_count": []}
    for layer in range(meta["layers"]):
        data = load_file(str(directory / f"layer_{layer:02d}.safetensors"))
        for key in tables:
            tables[key].append(data[key])
    return meta, {key: np.stack(value) for key, value in tables.items()}


def knapsack(cost, bits, allowed, target: float):
    """Per-expert argmin of cost + lambda*bits hitting the average-bit target, then greedy exact fill."""
    import numpy as np

    masked = np.where(allowed, cost, np.inf)

    def assign(lam):
        return np.argmin(masked + lam * bits[None, None, :], axis=2)

    def average(index):
        return float(bits[index].mean())

    lo, hi = 0.0, 1.0
    while average(assign(hi)) > target and hi < 1e12:
        hi *= 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if average(assign(mid)) > target:
            lo = mid
        else:
            hi = mid
    index = assign(hi)
    flat = index.reshape(-1)
    cost_flat = masked.reshape(-1, len(bits))
    total_bits_budget = target * flat.size
    used = float(bits[flat].sum())
    # Greedy fill: raise the experts with the best cost reduction per added bit while the budget allows.
    for _ in range(flat.size):
        upper = flat + 1
        valid = upper < len(bits)
        gain = np.full(flat.size, -np.inf)
        rows = np.nonzero(valid)[0]
        extra = bits[upper[rows]] - bits[flat[rows]]
        reduction = cost_flat[rows, flat[rows]] - cost_flat[rows, upper[rows]]
        ok = np.isfinite(reduction) & (used + extra <= total_bits_budget + 1e-9)
        gain[rows[ok]] = reduction[ok] / extra[ok]
        best = int(np.argmax(gain))
        if not np.isfinite(gain[best]) or gain[best] <= 0:
            break
        used += float(bits[flat[best] + 1] - bits[flat[best]])
        flat[best] += 1
    return flat.reshape(index.shape), hi


def exchange_from_initial(cost, bits, init: int, target: float, max_moves: int | None = None):
    """Start from the uniform initial level and apply one-level up/down moves only where they pay.

    Ups are ranked by cost reduction per added bit, downs by cost increase per freed bit.  For every
    number of ups the cheapest downs that keep the average at the target are taken; the split with the
    lowest total cost wins.  Unlike the global knapsack this never moves an expert whose move does not
    pay, so it cannot polarise a non-convex cost curve away from the initial level.  ``max_moves`` caps the
    number of experts that leave the initial level (a trust region on the additive cost model).
    """
    import numpy as np

    num_layers, num_experts, _ = cost.shape
    flat = cost.reshape(-1, len(bits))
    total = flat.shape[0]
    base_bits = bits[init] * total
    budget = target * total - base_bits  # bits that ups may add net of downs (negative: net downs required)
    up_ok = init + 1 < len(bits)
    down_ok = init - 1 >= 0
    up_gain = flat[:, init] - flat[:, init + 1] if up_ok else np.full(total, -np.inf)
    up_bits = bits[init + 1] - bits[init] if up_ok else 1.0
    down_cost = flat[:, init - 1] - flat[:, init] if down_ok else np.full(total, np.inf)
    down_bits = bits[init] - bits[init - 1] if down_ok else 1.0
    up_order = np.argsort(-up_gain, kind="stable")
    down_order = np.argsort(down_cost, kind="stable")
    up_cum_gain = np.concatenate([[0.0], np.cumsum(up_gain[up_order])])
    down_cum_cost = np.concatenate([[0.0], np.cumsum(down_cost[down_order])])
    best = (np.inf, 0, 0)
    max_ups = int(np.sum(up_gain > 0)) if up_ok else 0
    for n_up in range(0, max_ups + 1):
        needed = n_up * up_bits - budget  # bits the downs must free
        n_down = max(0, int(math.ceil(needed / down_bits - 1e-9))) if needed > 0 else 0
        if n_down > total - n_up or (n_down > 0 and not down_ok):
            continue
        if max_moves is not None and n_up + n_down > max_moves:
            if n_up == 0:
                max_moves = None  # the target alone needs more moves than the cap allows
            else:
                break
        change = -up_cum_gain[n_up] + down_cum_cost[n_down]
        if change < best[0]:
            best = (change, n_up, n_down)
    _, n_up, n_down = best
    index = np.full(total, init, dtype=np.int64)
    chosen_up = up_order[:n_up]
    index[chosen_up] = init + 1
    taken = set(chosen_up.tolist())
    downs = [unit for unit in down_order.tolist() if unit not in taken][:n_down]
    index[downs] = init - 1
    # If overlap left the average above the target, add the next cheapest downs.
    extra = [unit for unit in down_order.tolist() if unit not in taken and index[unit] == init]
    while bits[index].mean() > target + 1e-9 and extra:
        index[extra.pop(0)] = init - 1
    return index.reshape(num_layers, num_experts)


def agreement(index, reference, bits, levels):
    import numpy as np

    def pr(mask_ref, mask):
        tp = int((mask_ref & mask).sum())
        return tp / max(1, int(mask.sum())), tp / max(1, int(mask_ref.sum()))

    init = int(np.bincount(reference.ravel()).argmax())
    p_down, r_down = pr(reference < init, index < init)
    p_up, r_up = pr(reference > init, index > init)
    layer_ref, layer_new = bits[reference].mean(1), bits[index].mean(1)
    corr = float(np.corrcoef(layer_ref, layer_new)[0, 1]) if layer_ref.std() > 0 and layer_new.std() > 0 else None
    return {
        "exact": float((index == reference).mean()),
        "level_mae": float(np.abs(np.array(levels)[index] - np.array(levels)[reference]).mean()),
        "down_precision": p_down, "down_recall": r_down, "up_precision": p_up, "up_recall": r_up,
        "random_down_precision": float((reference < init).mean()),
        "random_up_precision": float((reference > init).mean()),
        "layer_bits_corr": corr,
    }


def cmd_allocate(args) -> None:
    import numpy as np

    meta, table = load_cost_table(args.costs)
    levels = meta["levels"]
    pinned = None
    if getattr(args, "pin", None):
        pin_list = json.loads(Path(args.pin).read_text())
        pinned = np.zeros(table["fisher"].shape[:2], dtype=bool)
        for layer, expert in pin_list:
            pinned[int(layer), int(expert)] = True
        log(f"pinning {int(pinned.sum())} experts at level K={args.pin_level} ({math.log2(args.pin_level):.2f} bits)")
    extra_pin_index = None
    if pinned is not None and args.pin_level not in levels:
        # A pin level the cost table never measured (e.g. K=10 built into the bank only for pinning). Pinned
        # experts never compete in the knapsack, so their cost column is irrelevant: append the level with zero
        # cost, and below forbid it for every expert that is not pinned.
        levels = list(levels) + [args.pin_level]
        extra_pin_index = len(levels) - 1
        for key, value in list(table.items()):
            if getattr(value, "ndim", 0) == 3:
                table[key] = np.concatenate([value, np.zeros(value.shape[:2] + (1,), dtype=value.dtype)], axis=2)
        log(f"pin level K={args.pin_level} is not in the cost table; added for pinned experts only")
    bits = np.log2(np.array(levels, dtype=np.float64))
    level_index = {level: i for i, level in enumerate(levels)}
    init = level_index[args.initial_level]
    shape = table["fisher"].shape
    variants = {
        "fisher_full": ("fisher", 0, len(levels) - 1),
        "fisher_trust1": ("fisher", max(0, init - 1), min(len(levels) - 1, init + 1)),
        "fisher_trust2": ("fisher", max(0, init - 1), min(len(levels) - 1, init + 2)),
        "outerr_trust1": ("output_error", max(0, init - 1), min(len(levels) - 1, init + 1)),
    }
    effective_target = args.target_bits
    if pinned is not None and not args.pin_in_budget:
        n_total = int(pinned.size)
        n_pin = int(pinned.sum())
        effective_target = (args.target_bits * (n_total - n_pin) + math.log2(args.pin_level) * n_pin) / n_total
        log(f"pinned experts are excluded from the budget: the other {n_total - n_pin} experts still average "
            f"{args.target_bits:.4f} bits, so the whole-model target becomes {effective_target:.4f}")
    references = {}
    for item in args.reference or []:
        name, path = item.split("=", 1)
        payload = json.loads(Path(path).read_text())
        raw = payload["levels"] if isinstance(payload, dict) and "levels" in payload else payload
        references[name] = np.vectorize(level_index.get)(np.array(raw))
    candidates = []
    variants["fisher_exchange1"] = ("fisher", None, None)
    # Trust-region caps matching how far the validation-NLL searches moved from uniform K=4 (12% and 27%).
    caps = {"fisher_exchange_cap12": 0.12, "fisher_exchange_cap27": 0.27}
    for name in caps:
        variants[name] = ("fisher", None, None)
    for name, (metric, low, high) in variants.items():
        if low is None:
            if pinned is not None:
                log(f"skipping {name}: the exchange search starts from uniform K and cannot honour pinned experts")
                continue
            cap = int(caps[name] * shape[0] * shape[1]) if name in caps else None
            index, lam = exchange_from_initial(table[metric], bits, init, effective_target, max_moves=cap), None
            low, high = max(0, init - 1), min(len(levels) - 1, init + 1)
        else:
            allowed = np.zeros(shape, dtype=bool)
            allowed[:, :, low:high + 1] = True
            if pinned is not None:
                allowed[pinned, :] = False
                allowed[pinned, level_index[args.pin_level]] = True
                if extra_pin_index is not None:
                    allowed[~pinned, extra_pin_index] = False
            index, lam = knapsack(table[metric], bits, allowed, effective_target)
        levels_out = np.array(levels)[index]
        depth = np.arange(shape[0])
        layer_bits = bits[index].mean(1)
        free_bits = float(bits[index][~pinned].mean()) if pinned is not None else float(bits[index].mean())
        row = {
            "name": name, "selectable": True, "metric": metric, "allowed_levels": levels[low:high + 1],
            "average_bits_excluding_pinned": free_bits,
            "lambda": lam, "average_bits": float(bits[index].mean()),
            "counts": {str(k): int((levels_out == k).sum()) for k in levels},
            "depth_bits_pearson": float(np.corrcoef(depth, layer_bits)[0, 1]) if layer_bits.std() > 0 else None,
            "predicted_fisher_cost": float(np.take_along_axis(table["fisher"], index[:, :, None], 2).sum()),
            "levels": levels_out.tolist(),
            "agreement": {ref: agreement(index, ref_index, bits, levels) for ref, ref_index in references.items()},
        }
        candidates.append(row)
        log(f"{name}: bits={row['average_bits']:.4f} counts={row['counts']} depth_r={row['depth_bits_pearson']} "
            + " ".join(f"{ref}:exact={a['exact']:.3f},down_p={a['down_precision']:.2f}(rand {a['random_down_precision']:.2f}),"
                       f"up_p={a['up_precision']:.2f}(rand {a['random_up_precision']:.2f})" for ref, a in row["agreement"].items()))
    for name, ref_index in references.items():
        candidates.append({"name": f"reference_{name}", "selectable": False, "average_bits": float(bits[ref_index].mean()),
                           "levels": np.array(levels)[ref_index].tolist()})
    # --include is --reference that select is allowed to pick: it carries a previously solved allocation
    # into this run so "same allocation, new correction" competes on equal footing with a fresh knapsack.
    for item in args.include or []:
        name, path = item.split("=", 1)
        payload = json.loads(Path(path).read_text())
        raw = payload["levels"] if isinstance(payload, dict) and "levels" in payload else payload
        inc_index = np.vectorize(level_index.get)(np.array(raw))
        candidates.append({"name": f"included_{name}", "selectable": True,
                           "average_bits": float(bits[inc_index].mean()), "levels": np.array(levels)[inc_index].tolist()})
        log(f"included_{name}: bits={float(bits[inc_index].mean()):.4f} (selectable)")
    uniform = np.full(shape[:2], args.initial_level)
    candidates.append({"name": f"uniform_K{args.initial_level}", "selectable": False,
                       "average_bits": float(math.log2(args.initial_level)), "levels": uniform.tolist()})
    # Diagnostics: size of the BF16-anchored first-order term relative to the second-order cost, and how
    # often the Fisher cost of the initial level lies above the chord of its neighbours in log2(K).
    ratio = np.abs(table["first_order"]).sum() / max(table["fisher"].sum(), 1e-30)
    nonconvex = None
    if 0 < init < len(levels) - 1:
        f = table["fisher"]
        chord = f[:, :, init - 1] + (f[:, :, init + 1] - f[:, :, init - 1]) * (
            (bits[init] - bits[init - 1]) / (bits[init + 1] - bits[init - 1]))
        nonconvex = float((f[:, :, init] > chord).mean())
    atomic_json(args.out, {"costs": str(args.costs), "target_bits": args.target_bits, "initial_level": args.initial_level,
                           "pinned_experts": int(pinned.sum()) if pinned is not None else 0,
                           "pinned_in_budget": bool(getattr(args, "pin_in_budget", False)),
                           "effective_target_bits": effective_target,
                           "pin_level": args.pin_level if pinned is not None else None,
                           "levels": levels, "first_order_abs_to_fisher_ratio": float(ratio),
                           "initial_level_nonconvex_fraction": nonconvex, "candidates": candidates})
    log(f"wrote {len(candidates)} candidates to {args.out}; |first|/fisher={ratio:.3f} "
        f"initial-level non-convex fraction={nonconvex}")


# ---------------------------------------------------------------------------------------------- select

def cmd_select(args) -> None:
    import torch
    from transformers import AutoTokenizer

    import optimize_twla_hierarchical_nll_packedbank  # noqa: F401  (patches bank accessors and save_pretrained)
    import optimize_twla_hierarchical_nll as base

    plan = json.loads(args.candidates.read_text())
    results_path = args.work_dir / "select_evaluations.jsonl"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["name"]] = record
    convergence = base.load_validation(args.data_dir / "convergence_validation.pt")
    final = base.load_validation(args.data_dir / "final_validation.pt")
    log(f"select: {len(plan['candidates'])} candidates, {len(done)} already measured")
    model = base.load_pretrained_streaming(args.physical_base, args.device, dtype=torch.bfloat16)
    model.eval()
    text = model.model.language_model
    text.config.use_cache = False

    gamma_levels, gamma_tables = None, {}
    if args.gamma_dir:
        from safetensors.torch import load_file as load_safetensors
        gamma_levels = json.loads((args.gamma_dir / "costs_meta.json").read_text())["levels"]
        for layer in range(len(text.layers)):
            gamma_tables[layer] = load_safetensors(str(args.gamma_dir / f"layer_{layer:02d}.safetensors"))["gamma"]
        log(f"gamma: loaded {len(gamma_tables)} layer tables from {args.gamma_dir} for levels {gamma_levels}")

    def install(levels):
        assignments = {(layer, expert): int(level) for layer, row in enumerate(levels) for expert, level in enumerate(row)}
        base.install_assignments(text, assignments, args.bank_dir, args.device)
        if not args.gamma_dir:
            return
        # install_assignments always rewrites down_proj from the bank, so gamma is never applied twice.
        with torch.no_grad():
            for layer, row in enumerate(levels):
                table = gamma_tables[layer]
                factors = torch.tensor([float(table[expert, gamma_levels.index(int(level))])
                                        for expert, level in enumerate(row)],
                                       device=args.device, dtype=torch.float32)
                data = text.layers[layer].mlp.experts.down_proj.data
                for start in range(0, data.shape[0], 32):
                    end = min(start + 32, data.shape[0])
                    # Multiply in float32 and round back once; a bf16 multiply loses ~2e-4 per element.
                    data[start:end] = (data[start:end].float() * factors[start:end, None, None]).to(data.dtype)

    for candidate in plan["candidates"]:
        if candidate["name"] in done:
            continue
        started = time.time()
        install(candidate["levels"])
        nll, tokens = base.validation_nll(model, convergence, args.device)
        record = {"name": candidate["name"], "selectable": candidate["selectable"],
                  "average_bits": candidate["average_bits"], "convergence_nll": nll, "convergence_tokens": tokens,
                  "elapsed_seconds": time.time() - started}
        with results_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        done[record["name"]] = record
        log(f"candidate {record['name']}: bits={record['average_bits']:.4f} convergence_nll={nll:.6f}")

    selectable = [done[c["name"]] for c in plan["candidates"] if c["selectable"]]
    best = min(selectable, key=lambda record: record["convergence_nll"])
    chosen = next(c for c in plan["candidates"] if c["name"] == best["name"])
    install(chosen["levels"])
    final_nll, final_tokens = base.validation_nll(model, final, args.device)
    log(f"selected {best['name']}: convergence_nll={best['convergence_nll']:.6f} final_nll={final_nll:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    AutoTokenizer.from_pretrained(args.physical_base).save_pretrained(args.out_dir)
    levels = chosen["levels"]
    summary = {
        "status": "complete", "method": "bf16_anchored_fisher_knapsack", "selected_candidate": best["name"],
        "levels": levels, "cycles_completed": 0, "convergence_nll": best["convergence_nll"],
        "final_validation_nll": final_nll, "final_validation_loss_tokens": final_tokens,
        "final_average_logical_bits": chosen["average_bits"], "target_average_bits": plan["target_bits"],
        "initial_level": plan["initial_level"], "bank_dir": str(args.bank_dir), "costs": plan["costs"],
        "gamma_dir": str(args.gamma_dir) if args.gamma_dir else None,
        "candidates": list(done.values()), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(args.out_dir / "optimization_summary.json", summary)
    atomic_json(args.out_dir / "precision_map.json", {
        "scheme": "bf16_anchored_fisher_knapsack_twla", "levels": levels, "bank_dir": str(args.bank_dir),
        "selection": "lowest convergence-split NLL among selectable knapsack candidates at the target bits",
        "routed_experts_only": True, "activation_bits": 16, "packed": False,
    })
    (args.out_dir / ".quant_done").touch()
    log(f"saved {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--model-dir", type=Path, required=True)
    collect.add_argument("--data", type=Path, required=True)
    collect.add_argument("--out-dir", type=Path, required=True)
    collect.add_argument("--rows", type=int, default=16)
    collect.add_argument("--row-start", type=int, default=0)
    collect.add_argument("--max-tokens", type=int, default=2048)
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--dtype", default="bfloat16")
    collect.add_argument("--store-dtype", default="bfloat16")
    collect.add_argument("--device-map", default=None,
                         help="shard the master across every visible GPU (e.g. balanced) instead of one card")
    collect.add_argument("--grad-checkpointing", action="store_true",
                         help="recompute activations in the backward pass; needed when the master does not leave "
                              "room for 40 layers of stored activations on one card")

    costs = sub.add_parser("costs")
    costs.add_argument("--model-dir", type=Path, required=True)
    costs.add_argument("--moments", type=Path, required=True)
    costs.add_argument("--bank", action="append", required=True, help="name=bank_dir (repeatable)")
    costs.add_argument("--levels", default="3,4,5,6,7,8,9")
    costs.add_argument("--out-dir", type=Path, required=True)
    costs.add_argument("--max-rows", type=int, default=0)
    costs.add_argument("--device", default="cuda:0")
    costs.add_argument("--gamma", action="store_true",
                       help="fit a least-squares output scalar per (expert, level) and measure the cost with it applied")

    allocate = sub.add_parser("allocate")
    allocate.add_argument("--costs", type=Path, required=True, help="per-bank cost directory")
    allocate.add_argument("--target-bits", type=float, required=True)
    allocate.add_argument("--initial-level", type=int, required=True)
    allocate.add_argument("--reference", action="append", help="name=json with 40x256 levels (repeatable)")
    allocate.add_argument("--include", action="append",
                          help="name=json with 40x256 levels, added as a SELECTABLE candidate (repeatable)")
    allocate.add_argument("--out", type=Path, required=True)
    allocate.add_argument("--pin", type=Path, help="json list of [layer, expert] to hold at --pin-level")
    allocate.add_argument("--pin-level", type=int, default=8, help="level the pinned experts are held at")
    allocate.add_argument("--pin-in-budget", action="store_true",
                          help="charge the pinned experts to the average-bit target (default: exclude them, so the "
                               "remaining experts still average --target-bits and nothing is downgraded to pay)")

    select = sub.add_parser("select")
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--bank-dir", type=Path, required=True)
    select.add_argument("--physical-base", type=Path, required=True)
    select.add_argument("--data-dir", type=Path, required=True)
    select.add_argument("--work-dir", type=Path, required=True)
    select.add_argument("--out-dir", type=Path, required=True)
    select.add_argument("--device", default="cuda:0")
    select.add_argument("--gamma-dir", type=Path, default=None,
                        help="cost directory built with --gamma; its scalars are folded into down_proj")

    args = parser.parse_args()
    {"collect": cmd_collect, "costs": cmd_costs, "allocate": cmd_allocate, "select": cmd_select}[args.command](args)


if __name__ == "__main__":
    main()

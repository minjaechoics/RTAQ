#!/usr/bin/env python3
"""Output-error code reassignment followed by a per-expert output scalar, for a fixed per-expert level allocation.

For every routed expert at its allocated level K, the bank's codebook is kept, so the bit width and storage format do
not change; only the code each weight takes is re-chosen. Columns are rounded left to right to the nearest codebook
value and each rounding error is pushed into the not-yet-rounded columns through the inverse input covariance
(GPTQ), so errors cancel in the expert's output instead of adding up:

  gate_up: H = sum_t w_t^2 x_t x_t^T over the calibration tokens routed to the expert (BF16 hidden states)
  down   : H = sum_t w_t^2 h_t h_t^T with h = act(gate) * up computed from the REASSIGNED gate_up, so down also
           absorbs the gate_up error

H is damped by damp * mean(diag H). With --damp auto, damping is chosen on calibration rows held out from fitting
(never on an evaluation benchmark): each value of --damp-grid is scored by held-out expert output error on a few
layers, and the best one is then used with every row.

--refit-affine (v2): after reassignment, each row's offset and scale (mu, alpha) are re-solved by H-weighted least
squares for the new codes, stored as bf16 like the bank's, and written to --affine-dir so the grid stays checkable.
Rows whose refit is degenerate or has alpha <= 0 keep the bank's values.

Last, gamma = <q, r> / <q, q> (q the recalibrated expert's weighted output, r the BF16 one) is folded into down_proj.
Experts with fewer than --min-tokens routed tokens keep gamma = 1; experts with none keep the bank reconstruction.

Runs on CPU (--workers processes) or GPUs (--devices cuda:0,cuda:1: one process per device). Per-layer results are
written to --work-dir as they finish (resumable), then assembled into a checkpoint whose non-expert tensors are
copied from --template.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import torch

TENSOR = "model.language_model.layers.{layer}.mlp.experts.{name}"


def gptq(W, H, codebook, mu, alpha, damp: float, block: int = 128):
    W = W.clone()
    cols = W.shape[1]
    Hd = H.double() + damp * torch.diag(H).double().mean() * torch.eye(cols, dtype=torch.float64, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True).float()
    Q = torch.zeros_like(W)
    codes = torch.zeros(W.shape, dtype=torch.long, device=W.device)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, H1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for i in range(i2 - i1):
            w = W1[:, i]
            code = (((w - mu) / alpha)[:, None] - codebook[None, :]).abs().argmin(1)
            q = mu + alpha * codebook[code]
            Q1[:, i] = q
            codes[:, i1 + i] = code
            err = (w - q) / H1[i, i]
            W1[:, i:] -= err[:, None] * H1[i, i:][None, :]
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes


def refit_affine(W, H, damp: float, codebook, codes, mu_old, alpha_old):
    """Per row, minimize (w - mu - alpha * l)^T Hd (w - mu - alpha * l) for fixed codes l; bf16-rounded like the bank."""
    cols = W.shape[1]
    Hd = H.double() + damp * torch.diag(H).double().mean() * torch.eye(cols, dtype=torch.float64, device=W.device)
    L = codebook.double()[codes]
    Wd = W.double()
    H1 = Hd.sum(1)
    HL = L @ Hd
    a11 = H1.sum()
    a12 = L @ H1
    a22 = (HL * L).sum(1)
    b1 = Wd @ H1
    b2 = (HL * Wd).sum(1)
    det = a11 * a22 - a12 * a12
    mu = (a22 * b1 - a12 * b2) / det
    alpha = (a11 * b2 - a12 * b1) / det
    ok = torch.isfinite(mu) & torch.isfinite(alpha) & (det.abs() > 1e-12 * a11.abs() * a22.abs()) & (alpha > 0)
    mu = torch.where(ok, mu, mu_old.double()).to(torch.bfloat16).float()
    alpha = torch.where(ok, alpha, alpha_old.double()).to(torch.bfloat16).float()
    Q = mu[:, None] + alpha[:, None] * codebook[codes]
    return Q, mu, alpha, int((~ok).sum())


def load_inputs(rows, layer, device):
    from safetensors import safe_open

    xs, idxs, ws = [], [], []
    for row in rows:
        with safe_open(str(row), framework="pt") as handle:
            xs.append(handle.get_tensor(f"layer{layer}.x").float())
            idxs.append(handle.get_tensor(f"layer{layer}.idx").long())
            ws.append(handle.get_tensor(f"layer{layer}.w").float())
    x, idx, w = torch.cat(xs).to(device), torch.cat(idxs).to(device), torch.cat(ws).to(device)
    token = torch.arange(x.shape[0], device=device).repeat_interleave(idx.shape[1])
    expert, weight = idx.reshape(-1), w.reshape(-1)
    order = torch.argsort(expert, stable=True)
    token, weight = token[order], weight[order]
    counts = torch.bincount(expert, minlength=256)
    bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]).tolist()
    return x, token, weight, bounds, counts


def process_layer(task: dict) -> dict:
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.activations import ACT2FN

    from optimize_twla_hierarchical_nll_packedbank import bank_path_resolved
    from qwen_glmstyle_proxy_v1_bank_io import unpack_u4

    device = task["device"]
    if device == "cpu":
        torch.set_num_threads(task["threads"])
    layer, work = task["layer"], Path(task["work_dir"])
    write = task["write"]
    out = work / f"layer_{layer:02d}.safetensors"
    result_path = Path(task["result_path"])
    if result_path.exists() and (not write or out.exists()):
        return json.loads(result_path.read_text())
    started = time.time()
    model_dir = Path(task["model_dir"])
    config = AutoConfig.from_pretrained(model_dir)
    act = ACT2FN[getattr(config, "text_config", config).hidden_act]
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def master(name):
        key = TENSOR.format(layer=layer, name=name)
        with safe_open(str(model_dir / weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key).float().to(device)

    G, D = master("gate_up_proj"), master("down_proj")
    E = G.shape[0]
    x, token, weight, bounds, counts = load_inputs(task["rows"], layer, device)
    held = load_inputs(task["eval_rows"], layer, device) if task["eval_rows"] else None
    if task["experts"]:
        experts = counts.argsort(descending=True)[: task["experts"]].tolist()
    else:
        experts = list(range(E))

    levels = task["levels"]
    handles = {level: safe_open(str(bank_path_resolved(Path(task["bank_dir"]), layer, level)), framework="pt")
               for level in sorted(set(levels))}

    def bank_expert(level, expert, prefix, width):
        handle = handles[level]
        keys = set(handle.keys())
        codes = unpack_u4(handle.get_slice(f"{prefix}_codes_u4")[expert:expert + 1])[0].long()[:, :width].to(device)
        if f"{prefix}_codebook" in keys:
            codebook = handle.get_slice(f"{prefix}_codebook")[expert].float().to(device)
        else:  # symmetric ternary: code - (K - 1) / 2
            codebook = torch.arange(level, device=device).float() - (level - 1) / 2
        mu = handle.get_slice(f"{prefix}_mu")[expert].float().to(device)
        alpha = handle.get_slice(f"{prefix}_alpha")[expert].float().to(device)
        return mu[:, None] + alpha[:, None] * codebook[codes], codebook, mu, alpha

    def expert_out(inputs, gate_up, down):
        gate, up = F.linear(inputs, gate_up).chunk(2, dim=-1)
        return F.linear(act(gate) * up, down)

    if write:
        new_G = torch.empty(G.shape, dtype=torch.bfloat16)
        new_D = torch.empty(D.shape, dtype=torch.bfloat16)
        aff = {k: torch.empty((E, G.shape[1] if k.startswith("gate_up") else D.shape[1]), dtype=torch.bfloat16)
               for k in ("gate_up_mu", "gate_up_alpha", "down_mu", "down_alpha")}
    sums = {k: 0.0 for k in ("fit_ref", "fit_bank", "fit_recal", "fit_final", "eval_ref", "eval_bank", "eval_recal", "eval_final")}
    gamma_per_expert = [1.0] * E
    gammas, fallback, degenerate_rows = [], 0, 0
    for expert in (range(E) if write else experts):
        level = int(levels[expert])
        G_bank, cb_g, mu_g, al_g = bank_expert(level, expert, "gate_up", G.shape[2])
        D_bank, cb_d, mu_d, al_d = bank_expert(level, expert, "down", D.shape[2])
        start, end = bounds[expert], bounds[expert + 1]
        if start == end:
            if write:
                new_G[expert], new_D[expert] = G_bank.cpu().to(torch.bfloat16), D_bank.cpu().to(torch.bfloat16)
                aff["gate_up_mu"][expert], aff["gate_up_alpha"][expert] = mu_g.cpu().to(torch.bfloat16), al_g.cpu().to(torch.bfloat16)
                aff["down_mu"][expert], aff["down_alpha"][expert] = mu_d.cpu().to(torch.bfloat16), al_d.cpu().to(torch.bfloat16)
            fallback += 1
            continue
        xe = x[token[start:end]]
        we = weight[start:end][:, None]
        Hg = (xe * we.square()).T @ xe
        Gq, codes_g = gptq(G[expert], Hg, cb_g, mu_g, al_g, task["damp"])
        mu_g2, al_g2 = mu_g, al_g
        if task["refit"]:
            Gq, mu_g2, al_g2, bad = refit_affine(G[expert], Hg, task["damp"], cb_g, codes_g, mu_g, al_g)
            degenerate_rows += bad
        gate, up = F.linear(xe, Gq).chunk(2, dim=-1)
        hq = act(gate) * up
        Hdn = (hq * we.square()).T @ hq
        Dq, codes_d = gptq(D[expert], Hdn, cb_d, mu_d, al_d, task["damp"])
        mu_d2, al_d2 = mu_d, al_d
        if task["refit"]:
            Dq, mu_d2, al_d2, bad = refit_affine(D[expert], Hdn, task["damp"], cb_d, codes_d, mu_d, al_d)
            degenerate_rows += bad
        r = (expert_out(xe, G[expert], D[expert]) * we).double()
        q = (expert_out(xe, Gq, Dq) * we).double()
        qb = (expert_out(xe, G_bank, D_bank) * we).double()
        gamma = float((q * r).sum() / q.square().sum().clamp_min(1e-30)) if end - start >= task["min_tokens"] else 1.0
        sums["fit_ref"] += float(r.square().sum())
        sums["fit_bank"] += float((qb - r).square().sum())
        sums["fit_recal"] += float((q - r).square().sum())
        sums["fit_final"] += float((gamma * q - r).square().sum())
        if held is not None:
            hx, htok, hw, hb, _ = held
            s2, e2 = hb[expert], hb[expert + 1]
            if s2 < e2:
                xh, wh = hx[htok[s2:e2]], hw[s2:e2][:, None]
                rh = (expert_out(xh, G[expert], D[expert]) * wh).double()
                qh = (expert_out(xh, Gq, Dq) * wh).double()
                qbh = (expert_out(xh, G_bank, D_bank) * wh).double()
                sums["eval_ref"] += float(rh.square().sum())
                sums["eval_bank"] += float((qbh - rh).square().sum())
                sums["eval_recal"] += float((qh - rh).square().sum())
                sums["eval_final"] += float((gamma * qh - rh).square().sum())
        gammas.append(gamma)
        gamma_per_expert[expert] = gamma
        if write:
            new_G[expert] = Gq.cpu().to(torch.bfloat16)
            new_D[expert] = (Dq * gamma).cpu().to(torch.bfloat16)   # multiply in float32, round to bf16 once
            aff["gate_up_mu"][expert], aff["gate_up_alpha"][expert] = mu_g2.cpu().to(torch.bfloat16), al_g2.cpu().to(torch.bfloat16)
            aff["down_mu"][expert], aff["down_alpha"][expert] = mu_d2.cpu().to(torch.bfloat16), al_d2.cpu().to(torch.bfloat16)

    def ratio(a, b):
        return (sums[a] / sums[b]) ** 0.5 if sums[b] > 0 else None
    stats = {"layer": layer, "damp": task["damp"], "refit": task["refit"], "seconds": time.time() - started,
             "experts_without_tokens": fallback, "degenerate_refit_rows": degenerate_rows,
             "gamma_mean": sum(gammas) / max(len(gammas), 1),
             "output_error_bank": ratio("fit_bank", "fit_ref"), "output_error_gptq": ratio("fit_recal", "fit_ref"),
             "output_error_final": ratio("fit_final", "fit_ref"),
             "heldout_error_bank": ratio("eval_bank", "eval_ref"), "heldout_error_recal": ratio("eval_recal", "eval_ref"),
             "heldout_error_final": ratio("eval_final", "eval_ref"), "sums": sums, "gammas": gamma_per_expert}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if write:
        work.mkdir(parents=True, exist_ok=True)
        if task["affine_dir"]:
            affine_dir = Path(task["affine_dir"]); affine_dir.mkdir(parents=True, exist_ok=True)
            tmp_aff = affine_dir / f"layer_{layer:02d}.safetensors.tmp"
            save_file(aff, str(tmp_aff))
            os.replace(tmp_aff, affine_dir / f"layer_{layer:02d}.safetensors")
        tmp = out.with_suffix(".tmp")
        save_file({"gate_up_proj": new_G, "down_proj": new_D}, str(tmp))
        os.replace(tmp, out)
    result_path.write_text(json.dumps(stats))
    tag = "build" if write else f"select damp={task['damp']}"
    print(f"[gptq] {tag} layer {layer:02d}: fit error bank {stats['output_error_bank']:.3f} -> recal "
          f"{stats['output_error_gptq']:.3f} -> +gamma {stats['output_error_final']:.3f}"
          + (f" | held-out bank {stats['heldout_error_bank']:.3f} -> +gamma {stats['heldout_error_final']:.3f}"
             if stats["heldout_error_final"] is not None else "")
          + f"  degenerate rows {degenerate_rows}  {stats['seconds']:.0f}s", flush=True)
    if device != "cpu":
        torch.cuda.empty_cache()
    return stats


def _device_worker(tasks):
    for task in tasks:
        process_layer(task)


def run_tasks(tasks, devices, workers):
    if devices == ["cpu"]:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(process_layer, tasks, chunksize=1)
    ctx = mp.get_context("spawn")
    buckets = [[] for _ in devices]
    for i, task in enumerate(tasks):
        task["device"] = devices[i % len(devices)]
        buckets[i % len(devices)].append(task)
    procs = [ctx.Process(target=_device_worker, args=(bucket,)) for bucket in buckets if bucket]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"device worker exited with {p.exitcode}")
    return [json.loads(Path(t["result_path"]).read_text()) for t in tasks]


def assemble(args, levels, layer_stats, damp) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    template, out, work = args.template, args.out_dir, args.work_dir
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((template / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    # layer -> expert tensor keys still to be written; a layer's work file is deleted once all of them are on disk
    remaining = {}
    for key in index["weight_map"]:
        if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            parts = key.split(".")
            remaining.setdefault(int(parts[parts.index("layers") + 1]), set()).add(key)
    for number, shard in enumerate(shards, 1):
        target = out / shard
        if target.exists():
            continue
        with safe_open(str(template / shard), framework="pt") as handle:
            metadata = handle.metadata() or {}
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key in list(tensors):
            parts = key.split(".")
            if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")) and "layers" in parts:
                layer = int(parts[parts.index("layers") + 1])
                with safe_open(str(work / f"layer_{layer:02d}.safetensors"), framework="pt") as handle:
                    tensors[key] = handle.get_tensor(parts[-1])
        tmp = target.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=metadata)
        os.replace(tmp, target)
        print(f"[save] shard {number}/{len(shards)} written", flush=True)
        if args.delete_work_as_assembled:
            for key in tensors:
                for layer, keys in remaining.items():
                    keys.discard(key)
            for layer, keys in list(remaining.items()):
                if not keys:
                    (work / f"layer_{layer:02d}.safetensors").unlink(missing_ok=True)
                    del remaining[layer]
    for item in template.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name not in (
                "optimization_summary.json", "precision_map.json", ".quant_done"):
            shutil.copy2(item, out / item.name)
    bits = [math.log2(level) for row in levels for level in row]
    summary = {
        "status": "complete",
        "method": "fixed_allocation_gptq_code_reassignment" + ("_affine_refit" if args.refit_affine else "") + "_then_output_gamma",
        "levels": levels, "final_average_logical_bits": sum(bits) / len(bits), "bank_dir": str(args.bank_dir),
        "levels_source": str(args.levels), "moments": str(args.moments), "damp": damp, "refit_affine": args.refit_affine,
        "affine_dir": str(args.affine_dir) if args.affine_dir else None, "min_tokens": args.min_tokens,
        "template": str(template),
        "per_layer": [{k: v for k, v in s.items() if k not in ("gammas", "sums")} for s in layer_stats],
        "gammas": {str(s["layer"]): s["gammas"] for s in layer_stats},
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "optimization_summary.json").write_text(json.dumps(summary))
    (out / ".quant_done").touch()
    print(f"[save] {out} complete, average logical bits {summary['final_average_logical_bits']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=Path, required=True, help="json 40x256 levels, or a summary with 'levels'")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="BF16 master")
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True, help="checkpoint supplying non-expert tensors")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--damp", default="1.0", help="float, or 'auto' to select on held-out calibration rows")
    parser.add_argument("--damp-grid", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--damp-layers", default="0,13,26,39")
    parser.add_argument("--damp-experts", type=int, default=64, help="most-routed experts per layer used for selection")
    parser.add_argument("--holdout-rows", type=int, default=15)
    parser.add_argument("--refit-affine", action="store_true")
    parser.add_argument("--affine-dir", type=Path, default=None, help="where refit mu/alpha are written (not the checkpoint)")
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--devices", default="cpu", help="'cpu' or comma-separated cuda devices, one process each")
    parser.add_argument("--workers", type=int, default=4, help="CPU processes")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--layers", default=None, help="comma-separated subset (tests); skips assembly")
    parser.add_argument("--delete-work-as-assembled", action="store_true",
                        help="delete each layer's work file once its tensors are written into the checkpoint")
    args = parser.parse_args()

    payload = json.loads(args.levels.read_text())
    levels = payload["levels"] if isinstance(payload, dict) else payload
    devices = [d.strip() for d in args.devices.split(",")]
    rows = sorted(args.moments.glob("row_*.safetensors"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    base = {"work_dir": str(args.work_dir), "model_dir": str(args.model_dir), "bank_dir": str(args.bank_dir),
            "min_tokens": args.min_tokens, "threads": args.threads, "refit": args.refit_affine,
            "affine_dir": str(args.affine_dir) if args.affine_dir else None, "device": devices[0]}

    damp = args.damp
    if damp == "auto":
        fit_rows, eval_rows = rows[: -args.holdout_rows], rows[-args.holdout_rows:]
        grid = [float(v) for v in args.damp_grid.split(",")]
        select_layers = [int(v) for v in args.damp_layers.split(",")]
        tasks = [dict(base, layer=layer, levels=levels[layer], damp=value, rows=[str(r) for r in fit_rows],
                      eval_rows=[str(r) for r in eval_rows], experts=args.damp_experts, write=False,
                      result_path=str(args.work_dir / "selection" / f"layer{layer:02d}_damp{value}.json"))
                 for value in grid for layer in select_layers]
        results = run_tasks(tasks, devices, args.workers)
        table = {}
        for value in grid:
            chosen = [s for s in results if s["damp"] == value]
            err = sum(s["sums"]["eval_final"] for s in chosen) / sum(s["sums"]["eval_ref"] for s in chosen)
            bank = sum(s["sums"]["eval_bank"] for s in chosen) / sum(s["sums"]["eval_ref"] for s in chosen)
            table[value] = {"heldout_error_final": err ** 0.5, "heldout_error_bank": bank ** 0.5}
        damp = min(table, key=lambda v: table[v]["heldout_error_final"])
        (args.work_dir / "damp_selection.json").write_text(json.dumps(
            {"grid": table, "selected": damp, "fit_rows": len(fit_rows), "holdout_rows": len(eval_rows),
             "layers": select_layers, "experts_per_layer": args.damp_experts, "refit_affine": args.refit_affine}, indent=2))
        for value, row in table.items():
            print(f"[damp] {value:<6} held-out output error {row['heldout_error_final']:.4f} (bank {row['heldout_error_bank']:.4f})"
                  + ("  <- selected" if value == damp else ""), flush=True)
    damp = float(damp)

    layers = [int(v) for v in args.layers.split(",")] if args.layers else list(range(len(levels)))
    tasks = [dict(base, layer=layer, levels=levels[layer], damp=damp, rows=[str(r) for r in rows], eval_rows=[],
                  experts=0, write=True, result_path=str(args.work_dir / f"layer_{layer:02d}.json")) for layer in layers]
    started = time.time()
    stats = sorted(run_tasks(tasks, devices, args.workers), key=lambda s: s["layer"])
    n = len(stats)
    print(f"[gptq] {n} layers in {time.time() - started:.0f}s at damp {damp}; mean fit output error bank "
          f"{sum(s['output_error_bank'] for s in stats) / n:.3f} -> recal {sum(s['output_error_gptq'] for s in stats) / n:.3f}"
          f" -> +gamma {sum(s['output_error_final'] for s in stats) / n:.3f}", flush=True)
    if args.layers:
        return
    assemble(args, levels, stats, damp)


if __name__ == "__main__":
    main()

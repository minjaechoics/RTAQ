#!/usr/bin/env python3
"""Termination-expert set P from routing-probe outputs.

P = union over sources s of { (l, e) : rho_close^s(l, e) >= 0.8 and rho_reason^s(l, e) < 0.3 }, where rho_close is an
expert's selection rate at the closing positions (the position that predicts </think>) and rho_reason its rate on
positions sampled inside the reasoning traces. Each source is one probe_termination_routing.py .npz, and the rule is
evaluated per source on that source's own closing positions and reasoning baseline. The release pipeline uses
S = {MATH-500} x {BF16, unpinned 2-bit} (scripts/05_find_termination_experts.sh), so GPQA-Diamond is never seen by the
pin selection.

  usage: build_pins.py --out pin.json name=path.npz [name=path.npz ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rule(path: Path, close: float, reason: float):
    z = np.load(path, allow_pickle=True)
    n_close, n_reason = int(z["totals"][0]), int(z["totals"][3])
    rc, rr = z["counts_close_think"] / max(n_close, 1), z["counts_reasoning"] / max(n_reason, 1)
    return (rc >= close) & (rr < reason), z["raw_close_think"], n_close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="name=path.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rho_close", type=float, default=0.8)
    parser.add_argument("--rho_reason", type=float, default=0.3)
    parser.add_argument("--min_close", type=int, default=50, help="a source with fewer closing positions is refused")
    args = parser.parse_args()
    masks, raws, report = {}, {}, {}
    for item in args.sources:
        name, path = item.split("=", 1)
        masks[name], raws[name], n_close = rule(Path(path), args.rho_close, args.rho_reason)
        if n_close < args.min_close:
            raise SystemExit(f"{name}: only {n_close} closing positions; refusing to derive pins from it")
        report[name] = {"npz": path, "n_close": n_close, "n_rule": int(masks[name].sum())}
    union = np.zeros_like(next(iter(masks.values())))
    for name, mask in masks.items():
        report[name]["added_to_union"] = int((mask & ~union).sum())
        union |= mask
    L = union.shape[0]
    for name, raw in raws.items():  # share of the closing-position routing slots that land on pinned experts
        hit = lambda m: float(m[np.arange(L)[None, :, None], raw].mean())
        report[name].update(slot_coverage_own_set=hit(masks[name]), slot_coverage_union=hit(union))
    pins = [[int(l), int(e)] for l, e in zip(*np.where(union))]
    report_out = {"rule": {"rho_close": args.rho_close, "rho_reason": args.rho_reason}, "sources": report,
                  "n_pin": int(union.sum()), "share_of_experts": float(union.mean()),
                  "per_block": np.bincount(np.where(union)[0], minlength=L).tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # --out is the plain [[layer, expert], ...] list that fisher_anchor.py allocate --pin and the verifiers read;
    # the per-source statistics go next to it.
    args.out.write_text(json.dumps(pins))
    args.out.with_name(args.out.stem + ".report.json").write_text(json.dumps(report_out, indent=1))
    print(json.dumps(report_out, indent=1))
    print(f"{len(pins)} pinned experts -> {args.out}")


if __name__ == "__main__":
    main()

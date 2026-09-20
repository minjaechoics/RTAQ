#!/usr/bin/env python3
"""Heatmaps and per-layer summaries for probe_termination_routing.py dumps.

Reads the .npz counts written by the probe (layer x expert selection counts per position class) and draws, per
model: a layer x expert heatmap of the selection-frequency difference between each termination class
(</think>, pre-EOS, loop onset) and the sampled reasoning baseline, plus a per-layer total-variation bar chart.
When both the quantized and the original dumps are present it also plots their per-layer shift side by side, which
is the ranking used to decide where extra bits should go.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ("close_think", "pre_eos", "loop_onset")


def load(path: Path):
    data = np.load(path.with_suffix(".npz"))
    report = json.loads(path.read_text())
    totals = {cls: report["totals"].get(cls, 0) for cls in ("close_think", "pre_eos", "loop_onset", "reasoning")}
    freqs = {}
    for cls in ("close_think", "pre_eos", "loop_onset", "reasoning"):
        key = f"counts_{cls}"
        if key in data and totals[cls]:
            freqs[cls] = data[key] / totals[cls]
    return report, freqs, totals


def heatmaps(name: str, freqs: dict, totals: dict, out_dir: Path) -> dict:
    base = freqs.get("reasoning")
    shifts = {}
    present = [cls for cls in CLASSES if cls in freqs]
    if base is None or not present:
        print(f"[{name}] nothing to plot (classes present: {list(freqs)})")
        return shifts
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 3.2 * len(present)), squeeze=False)
    for ax, cls in zip(axes[:, 0], present):
        diff = freqs[cls] - base
        limit = float(np.abs(diff).max()) or 1.0
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(f"{name}: {cls} minus reasoning baseline (n={totals[cls]} positions)")
        ax.set_xlabel("expert (0-255)")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="selection frequency difference")
        shifts[cls] = np.abs(diff).sum(axis=1) / 2
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_expert_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    layers = np.arange(next(iter(shifts.values())).shape[0])
    width = 0.8 / len(shifts)
    for i, (cls, shift) in enumerate(shifts.items()):
        ax.bar(layers + i * width - 0.4, shift, width=width, label=cls)
    ax.set_xlabel("layer")
    ax.set_ylabel("routing shift vs reasoning (total variation)")
    ax.set_title(f"{name}: per-layer routing shift at termination positions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_layer_shift.png", dpi=130)
    plt.close(fig)
    return shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="directory holding the probe .npz files")
    args = parser.parse_args()
    out_dir = args.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_shifts = {}
    summary = {}
    for name in ("quantized", "original"):
        path = out_dir / f"{name}.json"
        if not path.exists():
            print(f"[{name}] no dump at {path}, skipped")
            continue
        report, freqs, totals = load(path)
        all_shifts[name] = heatmaps(name, freqs, totals, out_dir)
        summary[name] = {
            "totals": totals,
            "top_layers": {cls: [int(i) for i in np.argsort(shift)[::-1][:8]] for cls, shift in all_shifts[name].items()},
            "top_experts": {cls: report.get(f"top_{cls}", [])[:15] for cls in CLASSES if f"top_{cls}" in report},
        }
        for cls, shift in all_shifts[name].items():
            order = np.argsort(shift)[::-1][:8]
            print(f"[{name}] {cls}: layers with the largest routing shift " +
                  ", ".join(f"L{int(i)}={shift[int(i)]:.2f}" for i in order))

    if len(all_shifts) == 2:
        for cls in CLASSES:
            if cls in all_shifts["quantized"] and cls in all_shifts["original"]:
                fig, ax = plt.subplots(figsize=(12, 4))
                layers = np.arange(all_shifts["quantized"][cls].shape[0])
                ax.bar(layers - 0.2, all_shifts["quantized"][cls], width=0.4, label="quantized")
                ax.bar(layers + 0.2, all_shifts["original"][cls], width=0.4, label="original BF16")
                ax.set_xlabel("layer"); ax.set_ylabel("routing shift vs reasoning")
                ax.set_title(f"{cls}: per-layer routing shift, quantized vs original")
                ax.legend(); fig.tight_layout()
                fig.savefig(out_dir / f"compare_{cls}_layer_shift.png", dpi=130)
                plt.close(fig)
    (out_dir / "routing_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote plots and routing_summary.json to {out_dir}")


if __name__ == "__main__":
    main()

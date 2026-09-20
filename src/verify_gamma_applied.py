#!/usr/bin/env python
"""Check that a gamma-corrected checkpoint stores exactly `bank decode x gamma` in down_proj.

This is check (2) from the gamma design note: a bf16 multiply loses ~2e-4 per element, so the
scalars must be applied in float32 and rounded back once. Sampling a few layers is enough to
catch a dtype regression.
"""
import argparse, json, random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qwen_glmstyle_proxy_v1_bank_io import decode_range
from optimize_twla_hierarchical_nll_packedbank import bank_path_resolved

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--bank-dir", type=Path, required=True)
parser.add_argument("--gamma-dir", type=Path, required=True)
parser.add_argument("--layers", type=int, default=4)
args = parser.parse_args()

summary = json.loads((args.checkpoint / "optimization_summary.json").read_text())
levels_grid = summary["levels"]
gamma_levels = json.loads((args.gamma_dir / "costs_meta.json").read_text())["levels"]
print(f"checkpoint gamma_dir={summary.get('gamma_dir')}  selected={summary['selected_candidate']} "
      f"bits={summary['final_average_logical_bits']:.4f}")
assert summary.get("gamma_dir"), "optimization_summary.json has no gamma_dir: gamma was not applied"

weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
prefix = "model.language_model.layers"
random.seed(0)
worst = 0.0
for layer in sorted(random.sample(range(len(levels_grid)), args.layers)):
    key = f"{prefix}.{layer}.mlp.experts.down_proj"
    with safe_open(str(args.checkpoint / weight_map[key]), framework="pt", device="cpu") as handle:
        saved = handle.get_tensor(key)
    table = load_file(str(args.gamma_dir / f"layer_{layer:02d}.safetensors"))["gamma"]
    row = levels_grid[layer]
    for expert in random.sample(range(len(row)), 8):
        level = int(row[expert])
        _, down = decode_range(bank_path_resolved(args.bank_dir, layer, level), level, expert, expert + 1, "cpu")
        factor = float(table[expert, gamma_levels.index(level)])
        expected = (down[0].to(torch.float32) * factor).to(torch.bfloat16)
        delta = float((saved[expert].float() - expected.float()).abs().max())
        worst = max(worst, delta)
    print(f"  layer {layer:2d}: max |saved - bank*gamma| = {worst:.3e}")
print(f"\n[검증2] worst deviation {worst:.3e} -> {'PASS' if worst == 0.0 else 'FAIL'}")
raise SystemExit(0 if worst == 0.0 else 1)

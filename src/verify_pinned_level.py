#!/usr/bin/env python3
"""Check that a materialized checkpoint holds exactly the bank decode of its planned level for sampled experts.

Samples pinned experts (which must sit at --pin-level) and non-pinned experts, decodes each from the bank
at the level optimization_summary.json assigned, and requires bit-exact equality with the saved tensors.
"""
import argparse, json, random
from pathlib import Path

import torch
from safetensors import safe_open

from optimize_twla_hierarchical_nll_packedbank import bank_path_resolved
from qwen_glmstyle_proxy_v1_bank_io import decode_range

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--bank-dir", type=Path, required=True)
parser.add_argument("--pin", type=Path, required=True)
parser.add_argument("--pin-level", type=int, required=True)
parser.add_argument("--samples", type=int, default=12)
args = parser.parse_args()

levels = json.loads((args.checkpoint / "optimization_summary.json").read_text())["levels"]
pins = [tuple(map(int, p)) for p in json.loads(args.pin.read_text())]
wrong = [p for p in pins if int(levels[p[0]][p[1]]) != args.pin_level]
assert not wrong, f"{len(wrong)} pinned experts are not at K={args.pin_level}, e.g. {wrong[:3]}"
pin_set = set(pins)
random.seed(0)
others = [(l, e) for l in range(len(levels)) for e in range(len(levels[l])) if (l, e) not in pin_set]
picks = random.sample(pins, args.samples) + random.sample(others, args.samples)
weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
worst = 0.0
for layer, expert in picks:
    level = int(levels[layer][expert])
    gate_up, down = decode_range(bank_path_resolved(args.bank_dir, layer, level), level, expert, expert + 1, "cpu")
    for name, expected in (("gate_up_proj", gate_up[0]), ("down_proj", down[0])):
        key = f"model.language_model.layers.{layer}.mlp.experts.{name}"
        with safe_open(str(args.checkpoint / weight_map[key]), framework="pt") as handle:
            saved = handle.get_slice(key)[expert]
        worst = max(worst, float((saved.float() - expected.float()).abs().max()))
print(f"all {len(pins)} pinned experts at K={args.pin_level}; {len(picks)} sampled experts, "
      f"max |saved - bank decode| = {worst:.3e}")
raise SystemExit(0 if worst == 0.0 else 1)

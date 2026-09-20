#!/usr/bin/env python3
"""Check a GPTQ+gamma checkpoint: routed expert weights still lie exactly on their level's grid, and non-expert
tensors are untouched.

gate_up must equal bf16(mu_i + alpha_i * c_k) element-wise for some code k of its row; down must equal
bf16((mu_i + alpha_i * c_k) * gamma). Exact membership proves the bit width and storage format are unchanged: only
codes moved. Also reports the fraction of codes that differ from the bank's original assignment.
"""
import argparse, json, random
from pathlib import Path

import torch
from safetensors import safe_open

from optimize_twla_hierarchical_nll_packedbank import bank_path_resolved
from qwen_glmstyle_proxy_v1_bank_io import unpack_u4

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--bank-dir", type=Path, required=True)
parser.add_argument("--template", type=Path, required=True)
parser.add_argument("--pin", type=Path, default=None)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--affine-dir", type=Path, default=None, help="refit mu/alpha written by apply_gptq_gamma --refit-affine")
args = parser.parse_args()

summary = json.loads((args.checkpoint / "optimization_summary.json").read_text())
levels, gammas = summary["levels"], summary["gammas"]
wmap = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
random.seed(0)
picks = [(random.randrange(len(levels)), random.randrange(len(levels[0]))) for _ in range(args.samples)]
if args.pin:
    picks += random.sample([tuple(p) for p in json.loads(args.pin.read_text())], args.samples)
changed = total = 0
for layer, expert in picks:
    level = int(levels[layer][expert])
    gamma = float(gammas[str(layer)][expert])
    with safe_open(str(bank_path_resolved(args.bank_dir, layer, level)), framework="pt") as bank:
        keys = set(bank.keys())
        for prefix, name in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            key = f"model.language_model.layers.{layer}.mlp.experts.{name}"
            with safe_open(str(args.checkpoint / wmap[key]), framework="pt") as handle:
                saved = handle.get_slice(key)[expert]
            width = saved.shape[1]
            codes = unpack_u4(bank.get_slice(f"{prefix}_codes_u4")[expert:expert + 1])[0].long()[:, :width]
            cb = (bank.get_slice(f"{prefix}_codebook")[expert].float() if f"{prefix}_codebook" in keys
                  else torch.arange(level).float() - (level - 1) / 2)
            if args.affine_dir:
                with safe_open(str(args.affine_dir / f"layer_{layer:02d}.safetensors"), framework="pt") as aff:
                    mu, alpha = aff.get_slice(f"{prefix}_mu")[expert].float(), aff.get_slice(f"{prefix}_alpha")[expert].float()
            else:
                mu, alpha = bank.get_slice(f"{prefix}_mu")[expert].float(), bank.get_slice(f"{prefix}_alpha")[expert].float()
            grid = mu[:, None] + alpha[:, None] * cb[None, :]                       # [rows, K] float32
            if prefix == "down":
                grid = grid * gamma
            grid = grid.to(torch.bfloat16)
            match = saved[:, :, None] == grid[:, None, :]                          # [rows, cols, K]
            assert bool(match.any(-1).all()), f"layer {layer} expert {expert} {name}: weights off the K={level} grid"
            new_codes = match.float().argmax(-1)
            changed += int((new_codes != codes).sum()); total += codes.numel()
print(f"{len(picks)} experts x 2 matrices on their level grid exactly; codes changed vs bank: {100 * changed / total:.1f}%")
# non-expert tensors untouched
shard = wmap["model.language_model.embed_tokens.weight"]
with safe_open(str(args.checkpoint / shard), framework="pt") as a, safe_open(str(args.template / shard), framework="pt") as b:
    for key in a.keys():
        if ".mlp.experts." not in key:
            assert torch.equal(a.get_tensor(key), b.get_tensor(key)), f"{key} differs from template"
print(f"non-expert tensors in {shard} identical to template")

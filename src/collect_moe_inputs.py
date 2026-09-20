#!/usr/bin/env python3
"""Forward-only capture of routed-expert inputs for code reassignment and output scaling.

For every packed calibration row (default: all 115 rows of calibration_packed2048.jsonl, the bank's calibration set),
the BF16 master is run once under inference mode and, per layer, the MoE block input hidden states and the router's
top-k indices and weights are stored:

  layer{l}.x   [T, H]      MoE block input (store dtype, bf16 by default)
  layer{l}.idx [T, top_k]  selected experts (int32)
  layer{l}.w   [T, top_k]  normalized router weights

This is the moments format `apply_gptq_gamma.py` reads, without the per-token output gradients `fisher_anchor.py
collect` also stores, so no backward pass and a single GPU suffice. Rows already on disk are skipped (resumable).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from streaming_load import load_pretrained_streaming


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="packed jsonl with input_ids per row")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--store-dtype", default="bfloat16")
    args = parser.parse_args()

    store = getattr(torch, args.store_dtype)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.rows:
        rows = rows[: args.rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [i for i in range(len(rows)) if not (args.out_dir / f"row_{i:04d}.safetensors").exists()]
    log(f"{len(rows)} rows, {len(pending)} pending -> {args.out_dir}")
    if not pending:
        return
    model = load_pretrained_streaming(args.model_dir, args.device, dtype=torch.bfloat16)
    model.eval()
    text = model.model.language_model
    text.config.use_cache = False

    captured: dict[tuple[int, str], torch.Tensor] = {}
    for layer, decoder in enumerate(text.layers):
        def gate_hook(_module, _inputs, output, layer=layer):
            _, weights, indices = output
            captured[(layer, "w")] = weights.detach().to(store).cpu()
            captured[(layer, "idx")] = indices.detach().to(torch.int32).cpu()

        def block_hook(_module, hook_args, hook_kwargs, _output, layer=layer):
            hidden = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
            captured[(layer, "x")] = hidden.detach().reshape(-1, hidden.shape[-1]).to(store).cpu()

        decoder.mlp.gate.register_forward_hook(gate_hook)
        decoder.mlp.register_forward_hook(block_hook, with_kwargs=True)

    for index in pending:
        started = time.time()
        ids = torch.tensor(rows[index]["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        captured.clear()
        with torch.inference_mode():
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        tensors = {}
        for layer in range(len(text.layers)):
            for kind in ("x", "idx", "w"):
                tensors[f"layer{layer}.{kind}"] = captured[(layer, kind)].contiguous()
        target = args.out_dir / f"row_{index:04d}.safetensors"
        tmp = target.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata={"row": str(index), "tokens": str(ids.shape[1]),
                                              "source": str(rows[index].get("source", "")), "store_dtype": args.store_dtype})
        tmp.rename(target)
        log(f"row {index + 1}/{len(rows)}: {ids.shape[1]} tokens, {time.time() - started:.1f}s")
    log("complete")


if __name__ == "__main__":
    main()

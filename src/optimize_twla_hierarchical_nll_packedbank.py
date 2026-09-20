#!/usr/bin/env python3
"""optimize_twla_hierarchical_nll.py on a qwen_glmstyle_proxy_v1 level bank.

The NLL optimizer and taylor_fisher_proxy.py are left untouched.  This wrapper swaps the three
bank accessors they use for versions that understand qwen_glmstyle_proxy_v1 banks:

* bank paths are resolved through the bank's ``bank_sources.json`` (levels reused from the
  legacy bank live in that bank's directory);
* variants may carry 4-bit packed codes (``*_codes_u4``) and identity or trained rotations;
* legacy uint8 files decode exactly as before (verified bit-identical).

All arguments are the optimizer's own.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimize_twla_hierarchical_nll as base  # noqa: E402
import taylor_fisher_proxy as proxy  # noqa: E402
from qwen_glmstyle_proxy_v1_bank_io import PACKED_FORMAT, decode_range, unpack_u4  # noqa: E402
from quantize.E2M_ATQ_asymmetric_codebook import decode_asymmetric_representation  # noqa: E402


@lru_cache(maxsize=8)
def _sources(bank_dir: str) -> dict:
    path = Path(bank_dir) / "bank_sources.json"
    if not path.exists():
        return {}
    layers = json.loads(path.read_text())["layers"]
    return {(int(layer), int(level)): Path(source) for layer, levels in layers.items() for level, source in levels.items()}


def bank_path_resolved(bank_dir: Path, layer: int, level: int) -> Path:
    return _sources(str(bank_dir)).get((int(layer), int(level))) or (
        Path(bank_dir) / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"
    )


@torch.no_grad()
def install_range_packed(experts, path: Path, level: int, start: int, end: int, device: str) -> None:
    for chunk_start in range(start, end, 32):
        chunk_end = min(chunk_start + 32, end)
        gate_up, down = decode_range(Path(path), level, chunk_start, chunk_end, device)
        experts.gate_up_proj.data[chunk_start:chunk_end].copy_(gate_up.to(experts.gate_up_proj.dtype))
        experts.down_proj.data[chunk_start:chunk_end].copy_(down.to(experts.down_proj.dtype))
        del gate_up, down


@torch.no_grad()
def decode_expert_weights_packed(handle, expert: int, level: int, device, dtype):
    """taylor_fisher_proxy._decode_expert_weights for legacy and packed variants."""
    packed = (handle.metadata() or {}).get("format") == PACKED_FORMAT
    asymmetric = "gate_up_codebook" in handle.keys()
    center = (level - 1) / 2.0
    decoded = []
    for prefix in ("gate_up", "down"):
        codes = handle.get_slice(f"{prefix}_codes_u4" if packed else f"{prefix}_codes")[expert:expert + 1].to(device)
        if packed:
            codes = unpack_u4(codes)
        mu = handle.get_slice(f"{prefix}_mu")[expert:expert + 1].to(device).float()
        alpha = handle.get_slice(f"{prefix}_alpha")[expert:expert + 1].to(device).float()
        if asymmetric:
            weight = decode_asymmetric_representation(
                codes,
                handle.get_slice(f"{prefix}_codebook")[expert:expert + 1].to(device).float(),
                mu, alpha,
                handle.get_slice(f"{prefix}_rotation_left")[expert:expert + 1].to(device).float(),
                handle.get_slice(f"{prefix}_rotation_right")[expert:expert + 1].to(device).float(),
            )
        else:
            weight = codes.float().sub_(center)
            weight.mul_(alpha[:, :, None])
            weight.add_(mu[:, :, None])
        decoded.append(weight[0].to(dtype))
    return decoded[0], decoded[1]


SHARD_BYTES = 4 * 2**30


@torch.no_grad()
def streaming_save_pretrained(model, save_directory, **_kwargs) -> None:
    """Write the checkpoint shard by shard straight from the resident tensors.

    transformers' save_pretrained materializes a host copy of the whole state dict, which on Thor's
    unified memory (model already resident, 67 GB) exceeded RAM and was OOM-killed.  This keeps the
    peak at one shard.  Keys and dtypes are the model's own state_dict, so the layout matches the
    fused-expert checkpoints this pipeline reads and vLLM loads.
    """
    import json as _json
    import os as _os
    from safetensors.torch import save_file

    out = Path(save_directory)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(out)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out)
    state = model.state_dict()
    seen_storage: dict[tuple[int, int], str] = {}
    names = []
    for name, tensor in state.items():
        key = (tensor.untyped_storage().data_ptr(), int(tensor.storage_offset()))
        if key in seen_storage:
            continue  # tied weight; HF also stores it once
        seen_storage[key] = name
        names.append(name)
    shards: list[list[str]] = [[]]
    size = 0
    for name in names:
        nbytes = state[name].numel() * state[name].element_size()
        if shards[-1] and size + nbytes > SHARD_BYTES:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += nbytes
    weight_map = {}
    total = 0
    for index, shard in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        tensors = {name: state[name].detach().to("cpu", copy=True).contiguous() for name in shard}
        total += sum(v.numel() * v.element_size() for v in tensors.values())
        temporary = out / (filename + ".tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        _os.replace(temporary, out / filename)
        weight_map.update({name: filename for name in shard})
        del tensors
        print(f"[save] shard {index}/{len(shards)} written", flush=True)
    (out / "model.safetensors.index.json").write_text(
        _json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )


base.bank_path = bank_path_resolved
base.install_range = install_range_packed
proxy._bank_path = bank_path_resolved
proxy._decode_expert_weights = decode_expert_weights_packed
from transformers import PreTrainedModel  # noqa: E402

PreTrainedModel.save_pretrained = streaming_save_pretrained

if __name__ == "__main__":
    base.main()

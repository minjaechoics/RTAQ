#!/usr/bin/env python3
"""Level-bank files for the qwen_glmstyle_proxy_v1 experiment.

A variant file holds one decoder layer's 256 routed experts at one TWLA level: codes plus
row-wise mu/alpha, and for asymmetric levels each expert's codebook and Kronecker rotation.
Variants written for this run pack their codes into 4-bit nibbles (K <= 15), which halves the
0.8 GB per file so both new banks fit on disk beside their checkpoints.  Files reused from
bipea_asymmetric_dp_codebook_level_bank_v1 keep uint8 codes and are read in place.  Decoding is
optimize_twla_hierarchical_nll.install_range's arithmetic, so a materialized expert equals the
one the NLL optimizer hot-swaps from the same file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantize.E2M_ATQ_asymmetric_codebook import decode_asymmetric_representation

RUN_ID = "qwen_glmstyle_proxy_v1"
NUM_LAYERS = 40
NUM_EXPERTS = 256
LEVELS = tuple(range(3, 10))
PROJECTIONS = ("gate_up", "down")
PACKED_FORMAT = "qwen_glmstyle_proxy_v1_u4_codes"
SYMMETRIC_SCHEME = "symmetric_twla"
ASYMMETRIC_VARIANT = "asymmetric_dp_codebook"
HYBRID_SCHEME = "hybrid_symmetric_ternary_asymmetric_dp_codebook"


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bank_path(bank_dir: Path, layer: int, level: int) -> Path:
    return Path(bank_dir) / f"layer_{layer:02d}" / f"level_{level:02d}.safetensors"


def pack_u4(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.shape[-1] % 2:
        raise ValueError(f"cannot nibble-pack {codes.dtype} codes of shape {tuple(codes.shape)}")
    if int(codes.max()) > 15:
        raise ValueError("codes do not fit in 4 bits")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_u4(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 15, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def save_variant(path: Path, buffers: dict[str, torch.Tensor], level: int, variant: str) -> None:
    tensors = {}
    for key, value in buffers.items():
        if key.endswith("_codes"):
            tensors[f"{key}_u4"] = pack_u4(value)
        else:
            tensors[key] = value.contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata={
        "format": PACKED_FORMAT, "level": str(level), "variant": variant, "run_id": RUN_ID,
    })
    os.replace(temporary, path)


def describe_variant(path: Path, level: int) -> dict:
    """Read a variant header and confirm both projections store ``level``."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
        packed = metadata.get("format") == PACKED_FORMAT
        asymmetric = "gate_up_codebook" in keys
        projection_levels = {}
        for prefix in PROJECTIONS:
            codes_key = f"{prefix}_codes_u4" if packed else f"{prefix}_codes"
            missing = {codes_key, f"{prefix}_mu", f"{prefix}_alpha"} - keys
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}")
            experts = int(handle.get_slice(codes_key).get_shape()[0])
            if experts != NUM_EXPERTS:
                raise ValueError(f"{path} holds {experts} experts, expected {NUM_EXPERTS}")
            if asymmetric:
                projection_levels[prefix] = int(handle.get_slice(f"{prefix}_codebook").get_shape()[1])
            elif packed:
                projection_levels[prefix] = int(metadata["level"])
            else:
                projection_levels[prefix] = int(Path(path).stem.rsplit("_", 1)[-1])
        if packed and int(metadata["level"]) != level:
            raise ValueError(f"{path} records level {metadata['level']}, expected {level}")
    if any(value != level for value in projection_levels.values()):
        raise ValueError(f"{path} stores levels {projection_levels}, expected {level}")
    return {
        "path": str(path),
        "level": level,
        "packed": packed,
        "asymmetric": asymmetric,
        "projection_levels": projection_levels,
    }


@torch.no_grad()
def decode_range(path: Path, level: int, start: int, end: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode experts [start, end) of one variant to bf16 gate_up and down weights on ``device``."""
    center = (level - 1) / 2.0
    decoded = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        packed = (handle.metadata() or {}).get("format") == PACKED_FORMAT
        asymmetric = "gate_up_codebook" in handle.keys()
        for prefix in PROJECTIONS:
            codes = handle.get_slice(f"{prefix}_codes_u4" if packed else f"{prefix}_codes")[start:end].to(device)
            if packed:
                codes = unpack_u4(codes)
            mu = handle.get_slice(f"{prefix}_mu")[start:end].to(device).float()
            alpha = handle.get_slice(f"{prefix}_alpha")[start:end].to(device).float()
            if asymmetric:
                reconstructed = decode_asymmetric_representation(
                    codes,
                    handle.get_slice(f"{prefix}_codebook")[start:end].to(device).float(),
                    mu,
                    alpha,
                    handle.get_slice(f"{prefix}_rotation_left")[start:end].to(device).float(),
                    handle.get_slice(f"{prefix}_rotation_right")[start:end].to(device).float(),
                )
            else:
                reconstructed = mu[:, :, None] + alpha[:, :, None] * (codes.float() - center)
            decoded.append(reconstructed.to(torch.bfloat16))
            del codes, mu, alpha, reconstructed
    return decoded[0], decoded[1]

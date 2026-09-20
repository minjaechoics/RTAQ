"""Load a Hugging Face checkpoint onto one device without holding it in memory twice.

``Model.from_pretrained(path).to(device)`` keeps the mmapped checkpoint alive while the
device copy is built, so peak memory is about twice the checkpoint. On a unified-memory
device such as Jetson Thor (122 GiB shared by CPU and GPU) that kills a 67 GiB bf16
Qwen3.6-35B-A3B load. This builds the model on the meta device and reads the weights
straight onto ``device`` one safetensors shard at a time, so the peak is one copy plus one
shard. The returned model matches ``from_pretrained``: eval mode, gradients left enabled,
and the checkpoint's generation config attached.
"""
from __future__ import annotations

import glob
import os
import re

import torch


def load_pretrained_streaming(model_dir, device="cuda:0", dtype=torch.bfloat16, model_cls=None):
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

    model_cls = model_cls or AutoModelForImageTextToText
    device = str(torch.device(device))
    model_dir = str(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if getattr(config, "quantization_config", None) is not None or not shards:
        # quantized exports and non-safetensors checkpoints need transformers' own loader
        return model_cls.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    from accelerate import init_empty_weights
    from safetensors import safe_open

    with init_empty_weights(include_buffers=False):
        model = model_cls.from_config(config, dtype=dtype)
    expected = set(model.state_dict())
    ignored = [re.compile(p) for p in (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])]
    unexpected = []
    for path in shards:
        shard = {}
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                if key in expected:
                    t = f.get_tensor(key)
                    shard[key] = t.to(dtype) if t.is_floating_point() else t
                elif not key.startswith("mtp.") and not any(p.search(key) for p in ignored):
                    unexpected.append(key)
        model.load_state_dict(shard, strict=False, assign=True)
        del shard
    if unexpected:
        raise RuntimeError(f"{model_dir}: {len(unexpected)} checkpoint tensors have no slot in "
                           f"{model_cls.__name__}, e.g. {unexpected[:3]}")

    model.tie_weights()
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"{model_dir}: {len(missing)} weights missing from the checkpoint, "
                           f"e.g. {missing[:3]}")

    model.to(device)
    # from_pretrained records the key conversions it applied; none were needed here. Without the
    # record, save_pretrained reverses every registered Qwen3.5-MoE conversion and splits each fused
    # expert tensor per expert, which stalls for hours on this model.
    model._weight_conversions = []
    if os.path.exists(os.path.join(model_dir, "generation_config.json")):
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    return model.eval()

#!/usr/bin/env python3
"""Prepare leakage-safe, document-level data for BIPEA V3.

The preparation has three stages:

1. ``prepare`` makes document-disjoint validation splits and a stratified
   calibration candidate pool.  It also emits assistant-only loss masks.
2. ``route-scan`` records BF16 top-k routed-expert counts for one candidate
   shard.  Four independent processes can scan the pool on four GPUs.
3. ``route-select`` greedily chooses a source-stratified calibration set that
   improves routed-expert coverage, then writes four balanced calibration
   shards.

GPQA is not read by this program at any stage.
"""

from __future__ import annotations

from streaming_load import load_pretrained_streaming

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(os.environ.get("RTAQ_ROOT") or Path(__file__).resolve().parent)
DEFAULT_SOURCE = ROOT / "data/corpus"          # written by scripts/01_prepare_corpus.sh
DEFAULT_OUTPUT = ROOT / "data/calibration"
DEFAULT_MODEL = Path(os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B"))

IM_START = 248045
IM_END = 248046
ASSISTANT = 74455
NEWLINE = 198
PAD = 248044

# These are exactly proportional to recovery_nogpqa_v1's source mixture.
CALIBRATION_QUOTAS = {
    "openr1_math_verified": 32,
    "nemotron_math_v3_verified": 16,
    "nemotron_ptd_math": 16,
    "nemotron_ptd_code": 16,
    "nemotron_ptd_stem": 16,
    "nemotron_science": 8,
    "nemotron_ptd_chat": 8,
    "nemotron_chat_reasoning_off": 8,
    "cnn_dailymail_anchor": 8,
}

# Exact document-level split of the 512-document held-out pool.
VALIDATION_QUOTAS = {
    "probe": {source: count // 2 for source, count in CALIBRATION_QUOTAS.items()},
    "convergence": dict(CALIBRATION_QUOTAS),
    "final": {source: count * 5 // 2 for source, count in CALIBRATION_QUOTAS.items()},
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def assistant_loss_mask(ids: list[int]) -> list[int]:
    """Mark assistant content and its terminating IM_END token."""
    mask = [0] * len(ids)
    active = False
    index = 0
    while index < len(ids):
        if ids[index : index + 3] == [IM_START, ASSISTANT, NEWLINE]:
            active = True
            index += 3
            continue
        if active:
            mask[index] = 1
            if ids[index] == IM_END:
                active = False
        index += 1
    return mask


def materialize(row: dict, max_length: int) -> dict:
    ids = list(map(int, row["input_ids"][:max_length]))
    mask = assistant_loss_mask(ids)
    return {
        "source": row["source"],
        "source_id": str(row.get("source_id", row["_line_no"])),
        "source_line_no": int(row["_line_no"]),
        "prompt_sha256": row.get("prompt_sha256"),
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "loss_mask": mask,
        "sequence_tokens": len(ids),
        "assistant_loss_tokens": sum(mask),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def pack_calibration_rows(rows: list[dict], seqlen: int) -> tuple[list[dict], int]:
    """Concatenate selected calibration documents into fixed-length blocks.

    The quantization driver consumes exact-length sequences and does not use
    padding masks.  Every selected document already terminates with its chat
    end token, so concatenation preserves an explicit boundary while avoiding
    thousands of padding activations.  Only the final incomplete tail is
    omitted and its exact size is recorded in the manifest.
    """
    stream: list[int] = []
    for row in rows:
        stream.extend(map(int, row["input_ids"]))
    usable = len(stream) // int(seqlen) * int(seqlen)
    packed = []
    for start in range(0, usable, int(seqlen)):
        ids = stream[start : start + int(seqlen)]
        packed.append({
            "source": "route_balanced_document_stream",
            "seq_len": int(seqlen),
            "input_ids": ids,
            "packed_block_index": len(packed),
        })
    return packed, len(stream) - usable


def write_padded_pt(path: Path, rows: list[dict], max_length: int) -> None:
    input_ids = torch.full((len(rows), max_length), PAD, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
    loss_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
    metadata = []
    for index, row in enumerate(rows):
        length = min(len(row["input_ids"]), max_length)
        input_ids[index, :length] = torch.tensor(row["input_ids"][:length], dtype=torch.long)
        attention_mask[index, :length] = True
        loss_mask[index, :length] = torch.tensor(row["loss_mask"][:length], dtype=torch.bool)
        metadata.append({key: value for key, value in row.items() if key not in {
            "input_ids", "attention_mask", "loss_mask"
        }})
    atomic_torch_save(path, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "metadata": metadata,
    })


def source_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "documents": len(rows),
        "sequence_tokens": sum(row["sequence_tokens"] for row in rows),
        "assistant_loss_tokens": sum(row["assistant_loss_tokens"] for row in rows),
        "source_counts": source_counts(rows),
        "prompt_hashes_present": sum(bool(row.get("prompt_sha256")) for row in rows),
    }


def stratified_take(
    groups: dict[str, list[dict]], quotas: dict[str, int], rng: random.Random
) -> list[dict]:
    selected = []
    for source, count in quotas.items():
        available = groups[source]
        if len(available) < count:
            raise RuntimeError(f"{source}: need {count} documents, found {len(available)}")
        rng.shuffle(available)
        selected.extend(available[:count])
        groups[source] = available[count:]
    rng.shuffle(selected)
    return selected


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    seen_lines: dict[tuple[str, int], str] = {}
    seen_hashes: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            key = (row["source"], int(row["source_line_no"]))
            if key in seen_lines:
                raise RuntimeError(f"document overlap: {key} in {seen_lines[key]} and {split}")
            seen_lines[key] = split
            fingerprint = row.get("prompt_sha256")
            if fingerprint:
                if fingerprint in seen_hashes:
                    raise RuntimeError(
                        f"prompt overlap: {fingerprint} in {seen_hashes[fingerprint]} and {split}"
                    )
                seen_hashes[fingerprint] = split


def cmd_prepare(args) -> None:
    rng = random.Random(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(args.source / "train.jsonl")
    validation = read_jsonl(args.source / "validation.jsonl")

    validation_groups = defaultdict(list)
    for row in validation:
        validation_groups[row["source"]].append(row)
    validation_splits = {}
    for split in ("probe", "convergence", "final"):
        raw = stratified_take(validation_groups, VALIDATION_QUOTAS[split], rng)
        validation_splits[split] = [materialize(row, args.validation_max_length) for row in raw]
    if any(validation_groups.values()):
        leftovers = {key: len(value) for key, value in validation_groups.items() if value}
        raise RuntimeError(f"validation allocation left documents behind: {leftovers}")
    assert_disjoint(validation_splits)

    train_groups = defaultdict(list)
    for row in train:
        if len(row["input_ids"]) >= args.calibration_min_length:
            train_groups[row["source"]].append(row)
    candidate_quotas = {
        source: count * args.candidate_multiplier
        for source, count in CALIBRATION_QUOTAS.items()
    }
    raw_candidates = stratified_take(train_groups, candidate_quotas, rng)
    candidates = [materialize(row, args.calibration_max_length) for row in raw_candidates]
    for index, row in enumerate(candidates):
        row["candidate_index"] = index

    # A fully usable stratified fallback is emitted immediately.  route-select
    # later replaces it with the coverage-balanced selection without touching
    # any validation split.
    candidate_groups = defaultdict(list)
    for row in candidates:
        candidate_groups[row["source"]].append(row)
    fallback = stratified_take(candidate_groups, CALIBRATION_QUOTAS, rng)

    write_jsonl(output / "calibration_candidates.jsonl", candidates)
    write_jsonl(output / "calibration_stratified.jsonl", fallback)
    write_padded_pt(
        output / "calibration_stratified.pt", fallback, args.calibration_max_length
    )
    for split, rows in validation_splits.items():
        write_jsonl(output / f"{split}_validation.jsonl", rows)
        write_padded_pt(
            output / f"{split}_validation.pt", rows, args.validation_max_length
        )

    manifest = {
        "status": "prepared_waiting_for_route_scan",
        "format": "bipea_expert_nogpqa_v3",
        "seed": args.seed,
        "gpqa_records": 0,
        "source_train": str((args.source / "train.jsonl").resolve()),
        "source_validation": str((args.source / "validation.jsonl").resolve()),
        "split_unit": "original_document",
        "loss_contract": "full-context forward; assistant/reasoning tokens plus assistant IM_END only",
        "calibration_contract": (
            "source-stratified 8x candidate pool; final 128 selected by BF16 routed-expert coverage"
        ),
        "calibration_max_length": args.calibration_max_length,
        "validation_max_length": args.validation_max_length,
        "candidate_multiplier": args.candidate_multiplier,
        "calibration_candidates": dataset_summary(candidates),
        "calibration_stratified_fallback": dataset_summary(fallback),
        "validation": {
            split: dataset_summary(rows) for split, rows in validation_splits.items()
        },
        "selection_usage": {
            "probe": "expert-level promotion/demotion verification only",
            "convergence": "whole-sweep acceptance, rollback, and convergence only",
            "final": "frozen-checkpoint evaluation only; never allocation or stopping",
        },
    }
    atomic_json(output / "dataset_manifest.json", manifest)
    (output / ".prepare_done").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


@torch.inference_mode()
def cmd_route_scan(args) -> None:
    from transformers import AutoModelForImageTextToText

    candidates = read_jsonl(args.output / "calibration_candidates.jsonl")
    assigned = [
        row for row in candidates
        if int(row["candidate_index"]) % args.num_shards == args.shard_idx
    ]
    route_dir = args.output / "route_scan"
    route_dir.mkdir(parents=True, exist_ok=True)
    output_path = route_dir / f"shard{args.shard_idx}.pt"
    if output_path.exists():
        print(f"route shard {args.shard_idx} already exists: {output_path}", flush=True)
        return

    model = load_pretrained_streaming(args.model, args.device, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    text_model.config.use_cache = False
    num_layers = int(text_model.config.num_hidden_layers)
    num_experts = int(text_model.config.num_experts)

    indices = []
    counts = []
    for ordinal, row in enumerate(assigned):
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        outputs = text_model(
            input_ids=ids,
            use_cache=False,
            output_router_logits=True,
            return_dict=True,
        )
        router_logits = outputs.router_logits
        if router_logits is None or len(router_logits) != num_layers:
            raise RuntimeError(
                f"expected {num_layers} router-logit tensors, got "
                f"{None if router_logits is None else len(router_logits)}"
            )
        document_counts = torch.zeros((num_layers, num_experts), dtype=torch.int32)
        for layer_idx, logits in enumerate(router_logits):
            topk = logits.float().topk(int(text_model.config.num_experts_per_tok), dim=-1).indices
            bincount = torch.bincount(topk.reshape(-1), minlength=num_experts)
            document_counts[layer_idx] = bincount.to(device="cpu", dtype=torch.int32)
        indices.append(int(row["candidate_index"]))
        counts.append(document_counts)
        if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(assigned):
            print(
                f"route shard={args.shard_idx} {ordinal + 1}/{len(assigned)} "
                f"candidate={row['candidate_index']} tokens={len(row['input_ids'])}",
                flush=True,
            )
        del ids, outputs, router_logits

    payload = {
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "candidate_indices": torch.tensor(indices, dtype=torch.long),
        "route_counts": torch.stack(counts),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "model": str(args.model.resolve()),
    }
    atomic_torch_save(output_path, payload)
    (route_dir / f".shard{args.shard_idx}_done").touch()
    print(f"completed route shard {args.shard_idx}: {output_path}", flush=True)


def greedy_route_select(candidates: list[dict], counts: torch.Tensor) -> list[int]:
    groups = defaultdict(list)
    for row in candidates:
        groups[row["source"]].append(int(row["candidate_index"]))
    flattened = counts.reshape(counts.shape[0], -1).float()
    total = torch.zeros(flattened.shape[1], dtype=torch.float32)
    remaining = dict(CALIBRATION_QUOTAS)
    selected = []
    selected_mask = torch.zeros(len(candidates), dtype=torch.bool)
    sources = list(CALIBRATION_QUOTAS)

    while sum(remaining.values()):
        for source in sources:
            if remaining[source] <= 0:
                continue
            available = [index for index in groups[source] if not selected_mask[index]]
            if not available:
                raise RuntimeError(f"no route candidate remains for {source}")
            candidate_tensor = torch.tensor(available, dtype=torch.long)
            # Concave coverage utility: experts already well covered receive
            # less value, while unseen and rare routed experts receive more.
            weights = torch.rsqrt(total + 1.0)
            scores = flattened.index_select(0, candidate_tensor).matmul(weights)
            best = available[int(scores.argmax())]
            selected.append(best)
            selected_mask[best] = True
            total += flattened[best]
            remaining[source] -= 1
    return selected


def route_coverage(counts: torch.Tensor) -> dict:
    values = counts.reshape(-1).to(torch.float32)
    sorted_values = values.sort().values
    quantiles = {}
    for label, fraction in (("p00", 0.0), ("p01", 0.01), ("p05", 0.05),
                            ("p50", 0.50), ("p95", 0.95), ("p99", 0.99), ("p100", 1.0)):
        index = min(int(round(fraction * (len(sorted_values) - 1))), len(sorted_values) - 1)
        quantiles[label] = int(sorted_values[index].item())
    return {
        "layer_expert_units": int(values.numel()),
        "zero_hit_units": int((values == 0).sum().item()),
        "units_below_64_hits": int((values < 64).sum().item()),
        "units_below_256_hits": int((values < 256).sum().item()),
        "hits_quantiles": quantiles,
        "total_route_assignments": int(values.sum().item()),
    }


def cmd_route_select(args) -> None:
    candidates = read_jsonl(args.output / "calibration_candidates.jsonl")
    merged = torch.zeros((len(candidates), 40, 256), dtype=torch.int32)
    present = torch.zeros(len(candidates), dtype=torch.bool)
    for shard_idx in range(args.num_shards):
        path = args.output / "route_scan" / f"shard{shard_idx}.pt"
        payload = torch.load(path, map_location="cpu")
        indices = payload["candidate_indices"].long()
        merged[indices] = payload["route_counts"].to(torch.int32)
        present[indices] = True
    if not bool(present.all()):
        missing = (~present).nonzero().flatten().tolist()[:20]
        raise RuntimeError(f"route scan missing candidates: {missing}")

    selected_indices = greedy_route_select(candidates, merged)
    selected = [candidates[index] for index in selected_indices]
    if source_counts(selected) != dict(sorted(CALIBRATION_QUOTAS.items())):
        raise RuntimeError(f"bad selected source counts: {source_counts(selected)}")

    write_jsonl(args.output / "calibration.jsonl", selected)
    write_padded_pt(args.output / "calibration.pt", selected, args.calibration_max_length)
    packed, dropped_tail_tokens = pack_calibration_rows(
        selected, args.calibration_max_length
    )
    write_jsonl(args.output / "calibration_packed2048.jsonl", packed)
    selected_counts = merged[selected_indices].sum(dim=0)

    # Round-robin assignment preserves route-selection order and gives each
    # GPU exactly 32 documents.
    for shard_idx in range(4):
        shard_rows = selected[shard_idx::4]
        write_jsonl(args.output / f"calibration_shard{shard_idx}.jsonl", shard_rows)
        write_padded_pt(
            args.output / f"calibration_shard{shard_idx}.pt",
            shard_rows,
            args.calibration_max_length,
        )
        packed_shard = packed[shard_idx::4]
        write_jsonl(
            args.output / f"calibration_packed2048_shard{shard_idx}.jsonl",
            packed_shard,
        )

    manifest_path = args.output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["calibration"] = dataset_summary(selected)
    manifest["calibration"]["selection"] = "BF16 top-8 routed-expert coverage greedy"
    manifest["calibration"]["packed_twla"] = {
        "path": str((args.output / "calibration_packed2048.jsonl").resolve()),
        "sequences": len(packed),
        "sequence_length": int(args.calibration_max_length),
        "used_tokens": len(packed) * int(args.calibration_max_length),
        "dropped_final_tail_tokens": dropped_tail_tokens,
        "document_boundary": "each source document already ends with its chat IM_END token",
    }
    manifest["calibration"]["coverage"] = route_coverage(selected_counts)
    manifest["calibration"]["selected_candidate_indices_sha256"] = hashlib.sha256(
        json.dumps(selected_indices).encode("utf-8")
    ).hexdigest()
    atomic_json(manifest_path, manifest)
    (args.output / ".route_select_done").touch()
    print(json.dumps(manifest["calibration"], ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--seed", type=int, default=260910)
    prepare.add_argument("--candidate-multiplier", type=int, default=8)
    prepare.add_argument("--calibration-min-length", type=int, default=512)
    prepare.add_argument("--calibration-max-length", type=int, default=2048)
    prepare.add_argument("--validation-max-length", type=int, default=4096)
    prepare.set_defaults(func=cmd_prepare)

    scan = subparsers.add_parser("route-scan")
    scan.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    scan.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    scan.add_argument("--device", default="cuda:0")
    scan.add_argument("--shard-idx", type=int, required=True)
    scan.add_argument("--num-shards", type=int, default=4)
    scan.set_defaults(func=cmd_route_scan)

    select = subparsers.add_parser("route-select")
    select.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    select.add_argument("--num-shards", type=int, default=4)
    select.add_argument("--calibration-max-length", type=int, default=2048)
    select.set_defaults(func=cmd_route_select)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)

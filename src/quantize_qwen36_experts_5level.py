"""Ternary-quantizes (W1.58A16, E2M-ATQ / TWLA arXiv:2606.13054v2 Algorithm 3)
EVERY routed expert of Qwen3.6-35B-A3B (Qwen3_5MoeForConditionalGeneration,
model_type=qwen3_5_moe) -- the direct Qwen3.6 counterpart of what this repo
already did for GLM-5.2, but simpler in three ways that matter for how this
script is structured:

1. Routed experts here are NOT per-expert `nn.Linear` submodules (GLM-5.2's
   layout, which `find_qlayers(module, layers=[nn.Linear])` targets) -- they
   are two batched 3D `nn.Parameter` tensors on `Qwen3_5MoeExperts`:
   `gate_up_proj` (num_experts, 2*moe_intermediate_size, hidden_size) and
   `down_proj` (num_experts, hidden_size, moe_intermediate_size), indexed
   and multiplied via `nn.functional.linear` inline inside
   `Qwen3_5MoeExperts.forward`'s per-expert loop -- there is no submodule to
   monkey-patch/replace the way `find_qlayers` + `TernaryQLinear` did.
   `_CalibratingExperts.forward` below is therefore a near-verbatim copy of
   the real `Qwen3_5MoeExperts.forward` (transformers/models/qwen3_5_moe/
   modeling_qwen3_5_moe.py) with two extra lines that accumulate each
   expert's calibration second moment (Eq. 7, S = sum_b X_b^T X_b) for its
   TWO weight matrices independently -- `gate_up_proj`'s input X (the
   router-selected hidden_states before the expert MLP) and `down_proj`'s
   input Y (act_fn(gate)*up, after the first matmul) -- while still
   producing byte-identical output to the unmodified module, so calibration
   never perturbs the actual forward pass it's riding on.

2. FAKE-QUANT, not packed/kernel-accelerated: the E2M-ATQ reconstruction
   (mu* + alpha*·T) is written back into `gate_up_proj`/`down_proj` IN
   PLACE, still as dense bf16 -- unlike GLM-5.2's `pack_ternary_checkpoint.py`
   + Triton `TernaryRotatedLinear` path (built because GLM-5.2 is 1.4TB and
   needed real compression to fit in GPU memory at all). Qwen3.6-35B-A3B is
   67GB dense -- it already fits on ONE GPU without compression, and the
   user's own follow-up plan is 8-way single-GPU-per-shard parallelism, not
   a memory-constrained deploy -- so there is no packing/kernel work to do
   here; the measurement that matters is what W1.58A16 does to GPQA
   accuracy, which the dense-but-ternarized-VALUES reconstruction captures
   exactly as faithfully as a packed format would, just without the storage
   win. Whole model is saved via `save_pretrained` afterward -- same size,
   ternarized values.

3. No KOTMS activation rotation: GLM-5.2's pipeline layers a rotation trick
   (this repo's KOTMS.py-descended code) on top of E2M-ATQ for extra
   accuracy. That rotation is architecture-specific engineering (has to
   commute correctly through GLM-5.2's exact module boundaries) that would
   need its own from-scratch derivation for Qwen3.6's very different
   decoder layer (hybrid linear-attention/full-attention, batched experts,
   mrope). Skipped here to keep scope bounded -- this script applies E2M-ATQ
   (the paper's actual algorithm) directly to the un-rotated weights, which
   is still a faithful "TWLA" ternary quantization, just without that one
   GLM-5.2-specific enhancement layered on top.

4. Optional reasoning-length/overthinking-aware calibration weighting
   (--hesitation_boost, default 1.0 = OFF, byte-identical to plain
   E2M-ATQ): grounded in two papers (fetched and read, not guessed at) --
   AYOT/ScaleQ-1.58 (arXiv:2608.01078) finds that feeding a reasoning
   model's OWN self-generated reasoning traces (not bare questions) as
   calibration CONTEXT before ternarizing avoids the collapse plain
   calibration causes on reasoning-heavy tasks; see
   build_qwen36_reasoning_calibset.py, which builds exactly that kind of
   calibset for this script to consume via --calibset (no loader changes
   needed -- same {"source","seq_len","input_ids"} contract). AYOT itself
   is calibration-SET-only (confirmed from the paper's own text: "AYOT is
   formed by simply integrating ... no loss weighting or token-level
   masking is introduced") -- it does not touch the per-token calibration
   OBJECTIVE at all.

   --hesitation_boost is this script's OWN extension on top of that,
   motivated by (not copied from) a separate paper, arXiv:2606.00206
   ("Quantized Reasoning Models Think They Need to Think Longer, but They
   Do Not"), already used training-free/decode-time in
   overthinking_penalty.py: PTQ's quantization noise is empirically
   concentrated at high-entropy positions, and those are disproportionately
   at/near hesitation-marker tokens ("Wait", "However", "Alternatively",
   ...). If a calibration-time mechanism can be made to spend more of its
   fidelity budget exactly there, that is a strict generalization of E2M-ATQ
   worth trying -- and, mechanically, it drops in cleanly: E2M_ATQ.py's
   Stage II (manifold_relocation) already treats S purely as a symmetric
   PSD "calibration-induced metric" (Eq. 7-9) and never assumes it is an
   UNWEIGHTED sum_b x_b x_b^T, so reweighting which calibration tokens
   contribute more to S -- S = sum_b w_b x_b x_b^T -- needs no change to
   E2M_ATQ.py itself, only to how _CalibratingExperts accumulates S below.
   Neither AYOT nor arXiv:2606.00206 describes this weighting; it is this
   script's own combination of the two ideas, on by request via
   --hesitation_boost > 1.0, and OFF (byte-identical to upstream E2M-ATQ)
   by default.

Calibration data: wikitext2 (same source datautils.py already knows how to
fetch), tokenized with Qwen3.6's OWN tokenizer (the existing
calibset/zai-org__GLM-5.2/*.jsonl sets are pre-tokenized for GLM-5.2's
vocab and can't be reused here).

Memory shape: holding all 40 layers' calibration second moments in fp32
simultaneously would be ~183GB (256 experts/layer x (2048^2 + 512^2) x
4 bytes x 40 layers) -- as much as the GPU itself. Every layer's S is
therefore merged into a CPU-resident fp32 accumulator dict immediately
after each calibration sample's forward pass and freed from GPU, so GPU
memory only ever holds one sample's transient per-expert second moments at
a time; only the much cheaper embed_tokens output + this accumulator dict
live for the calibration set's full duration.
"""

import os
RTAQ_ROOT = os.environ.get("RTAQ_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from streaming_load import load_pretrained_streaming

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

# datautils is only needed by the legacy wikitext/PTB/C4 loaders, which this pipeline never uses
from quantize.E2M_ATQ_5level import e2m_atq_quantize

QUANT_LEVELS = 5

MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen/Qwen3.6-35B-A3B")
DEFAULT_CALIBSET = None


def load_qwen36_calibset(path: str, nsamples, seed: int, seqlen: int):
    """Same chunking contract as datautils.get_calibset (that function's
    CALIBSET_DIR is hardcoded to GLM-5.2's calibset dir, so this is a small
    standalone copy rather than a shared-file edit): items whose seq_len is
    a multiple of `seqlen` get split into that many non-overlapping
    seqlen-length chunks; items shorter than `seqlen` are an error (can't
    chunk them); random.shuffle + optional truncation to `nsamples` after.
    See build_qwen36_calibset.py -- this reads ITS output (GLM-5.2's
    robust_mixed+baseline_c4 composition, re-tokenized for Qwen3.6).
    """
    import random

    import torch as _torch

    trainloader = []
    n_skipped = 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            ids = _torch.tensor(item["input_ids"], dtype=_torch.long).unsqueeze(0)
            length = ids.shape[1]
            if length < seqlen:
                # Re-tokenizing GLM-5.2's calibset text with Qwen3.6's
                # tokenizer doesn't preserve token counts (different vocab,
                # different subword boundaries) -- a real run found a c4
                # item that was exactly 2048 GLM-5.2 tokens come out as
                # 1976 Qwen tokens. Skipping (not erroring) loses a
                # fraction of a percent of the calibset rather than
                # crashing the whole run over one short item.
                n_skipped += 1
                continue
            n_chunks = length // seqlen
            for c in range(n_chunks):
                inp = ids[:, c * seqlen : (c + 1) * seqlen]
                trainloader.append((inp, None))
    if n_skipped:
        print(f"[load_qwen36_calibset] skipped {n_skipped} item(s) shorter than seqlen={seqlen} "
              f"after re-tokenization.", flush=True)

    random.seed(seed)
    random.shuffle(trainloader)
    if nsamples is not None and 0 < nsamples < len(trainloader):
        trainloader = trainloader[:nsamples]
    return trainloader


def compute_hesitation_weights(
    input_ids_1d: torch.Tensor, marker_id_set: set, boost: float, window: int,
) -> torch.Tensor:
    """Per-token calibration weight for ONE sample's fixed (teacher-forced)
    token sequence: `boost` at any position within `window` tokens of a
    hesitation-marker occurrence (inclusive, both directions -- a marker
    token itself always counts as its own window-0 hit), else 1.0.
    `input_ids_1d`: 1D LongTensor (seqlen,), any device. Returns a 1D
    float32 CPU tensor (seqlen,) -- deliberately CPU-resident like the rest
    of this file's cross-layer accumulator state (module docstring's
    "memory shape" note), since it is looked up with a small `.cpu()`
    gather per expert-hit rather than kept GPU-resident for the whole run.
    """
    input_ids_1d = input_ids_1d.reshape(-1).cpu()
    seqlen = input_ids_1d.shape[0]
    hit = torch.zeros(seqlen, dtype=torch.bool)
    if marker_id_set:
        marker_tensor = torch.tensor(sorted(marker_id_set), dtype=input_ids_1d.dtype)
        hit = torch.isin(input_ids_1d, marker_tensor)
    if window > 0 and bool(hit.any()):
        near = hit.clone()
        for w in range(1, int(window) + 1):
            near[:-w] |= hit[w:]
            near[w:] |= hit[:-w]
        hit = near
    weights = torch.ones(seqlen, dtype=torch.float32)
    weights[hit] = float(boost)
    return weights


class _CalibratingExperts:
    """Bound-method replacement for `Qwen3_5MoeExperts.forward`, installed
    per-layer only during calibration. `accum` is this layer's
    {expert_idx: [S_gate_up (hidden,hidden) fp32 GPU, S_down (inter,inter)
    fp32 GPU, hit_count]} dict, mutated in place -- shared with the caller
    so results survive after the monkey-patch is reverted.

    `current_weights` (set per calibration sample via `set_sample_weights`,
    see module docstring's item 4) is either None (the default: every
    position implicitly weight 1.0, S accumulates as plain
    sum_b x_b x_b^T -- byte-identical to upstream E2M-ATQ) or a 1D fp32
    CPU tensor of length seqlen giving THIS sample's per-position
    hesitation-aware weight, in which case S accumulates as the weighted
    second moment sum_b w_b x_b x_b^T instead.
    """

    def __init__(self, real_module, accum: dict):
        self.real = real_module
        self.accum = accum
        self.current_weights = None

    def set_sample_weights(self, w) -> None:
        self.current_weights = w

    @torch.no_grad()
    def __call__(self, hidden_states, top_k_index, top_k_weights):
        m = self.real
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=m.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            if expert_idx == m.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]  # X: input to gate_up_proj
            gate, up = F.linear(current_state, m.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = m.act_fn(gate) * up  # Y: input to down_proj
            out = F.linear(current_hidden_states, m.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))

            Xf = current_state.reshape(-1, current_state.shape[-1]).float()
            Yf = current_hidden_states.reshape(-1, current_hidden_states.shape[-1]).float()
            if expert_idx not in self.accum:
                # Keep the second-moment accumulators on the SAME device as the
                # activations.  Allocating them on CPU forces `(Xf.t() @ Xf).cpu()`
                # below, and that .cpu() is a SYNCHRONIZING device-to-host copy --
                # ~32k of them per layer (128 calib samples x 256 experts x 2
                # tensors), which serializes the whole pass and leaves the GPU
                # essentially idle.  On GPU the accumulators cost
                # 256 x (2048^2 + 512^2) x 4B = 4.6 GB, trivial next to the
                # ~68 GB the resident bf16 model already holds.
                self.accum[expert_idx] = [
                    torch.zeros(Xf.shape[1], Xf.shape[1], dtype=torch.float32, device=Xf.device),
                    torch.zeros(Yf.shape[1], Yf.shape[1], dtype=torch.float32, device=Yf.device),
                    0,
                ]
            entry = self.accum[expert_idx]
            if self.current_weights is not None:
                # Hesitation-marker-weighted second moment (module
                # docstring item 4): S += X^T diag(w) X = (w*X)^T X, a
                # congruence transform of diag(w>=0) so S stays symmetric
                # PSD, exactly what manifold_relocation's Eq. 46-51 closed
                # form requires -- it never assumed w==1.
                wt = self.current_weights[token_idx.cpu()].to(device=Xf.device, dtype=torch.float32)
                entry[0] += (Xf * wt[:, None]).t() @ Xf
                entry[1] += (Yf * wt[:, None]).t() @ Yf
            else:
                entry[0] += Xf.t() @ Xf
                entry[1] += Yf.t() @ Yf
            entry[2] += int(Xf.shape[0])

        return final_hidden_states




@torch.no_grad()
def quantize_layer_experts(
    layer, layer_idx: int, calib_hidden_states, common_kwargs, args, apply_quant: bool = True,
    token_weights=None,
) -> dict:
    """Runs every calibration sample through this ONE decoder layer (real
    forward, full model correctness -- position embeddings/masks come from
    `common_kwargs`, computed once up front by the model's own
    `Qwen3_5MoeTextModel.forward` logic, not hand-rolled here) with
    `mlp.experts.forward` swapped to `_CalibratingExperts` for the
    duration, then applies `e2m_atq_quantize` to every expert that was hit
    at least once. Returns {"layer": layer_idx, "n_experts_hit": ...,
    "n_experts_dead": ..., "mean_sq_err": ...} and the (unmodified-weight)
    per-sample layer outputs, so the caller can feed them to the next
    layer -- see the module docstring's memory-shape note for why this
    single-pass-per-sample approach (not a second post-quantization
    forward) is what keeps this tractable.

    `apply_quant=False` (for the layer-range ablation checkpoints, e.g.
    "only ternarize layers 10-19") skips the calibration hook AND the
    quantize-in-place step entirely -- this layer just runs a plain forward
    to propagate hidden_states to the next layer, staying dense/untouched.
    Deeper in-range layers still calibrate on the REAL activations that
    result from earlier layers being dense (not a hypothetical
    all-quantized chain), which is the faithful thing to do for measuring
    "what does ternarizing ONLY this layer range do".

    `token_weights`, if given, is a list of 1D fp32 CPU tensors aligned
    index-for-index with `calib_hidden_states` (one per calibration
    sample) -- see module docstring item 4 and `compute_hesitation_weights`.
    None (default) leaves every sample's weighting at None too, i.e. off.
    """
    experts_module = layer.mlp.experts
    layer_type = (getattr(layer, "block_type", None) or layer.layer_type)
    mask_key = "linear_attn_mask" if layer_type == "linear_attention" else "causal_mask"

    if not apply_quant:
        outputs = []
        for hs in calib_hidden_states:
            hs = hs.to(args.device)
            out = layer(
                hs,
                position_embeddings=common_kwargs["position_embeddings"],
                attention_mask=common_kwargs[mask_key],
                position_ids=common_kwargs["text_position_ids"],
                past_key_values=None,
                use_cache=False,
            )
            outputs.append(out.to("cpu"))
        return {"layer": layer_idx, "n_experts_hit": 0, "n_experts_dead": 0,
                "sum_sq_err": 0.0, "skipped": True}, outputs

    accum: dict = {}
    calibrating = _CalibratingExperts(experts_module, accum)
    original_forward = experts_module.forward
    experts_module.forward = calibrating

    outputs = []
    for sample_idx, hs in enumerate(calib_hidden_states):
        hs = hs.to(args.device)
        if token_weights is not None:
            calibrating.set_sample_weights(token_weights[sample_idx])
        out = layer(
            hs,
            position_embeddings=common_kwargs["position_embeddings"],
            attention_mask=common_kwargs[mask_key],
            position_ids=common_kwargs["text_position_ids"],
            past_key_values=None,
            use_cache=False,
        )
        outputs.append(out.to("cpu"))

    experts_module.forward = original_forward

    n_hit = len(accum)
    n_dead = experts_module.num_experts - n_hit
    total_err = 0.0
    for expert_idx, (S_gu, S_dn, hits) in accum.items():
        gu = experts_module.gate_up_proj.data[expert_idx].float().to(args.device)
        S_gu = S_gu.to(args.device)
        W_bar = e2m_atq_quantize(gu, S_gu, euclidean_iters=args.euclidean_iters)
        total_err += float(((gu - W_bar) ** 2).sum())
        experts_module.gate_up_proj.data[expert_idx] = W_bar.to(experts_module.gate_up_proj.dtype)
        del gu, S_gu, W_bar

        dn = experts_module.down_proj.data[expert_idx].float().to(args.device)
        S_dn = S_dn.to(args.device)
        W_bar2 = e2m_atq_quantize(dn, S_dn, euclidean_iters=args.euclidean_iters)
        total_err += float(((dn - W_bar2) ** 2).sum())
        experts_module.down_proj.data[expert_idx] = W_bar2.to(experts_module.down_proj.dtype)
        del dn, S_dn, W_bar2
        torch.cuda.empty_cache()

    if n_dead:
        print(f"[layer {layer_idx}] WARNING: {n_dead}/{experts_module.num_experts} experts got 0 "
              f"calibration hits, left unquantized (dense bf16, no ternary reconstruction).", flush=True)

    stats = {"layer": layer_idx, "n_experts_hit": n_hit, "n_experts_dead": n_dead,
             "sum_sq_err": total_err}
    return stats, outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--calibset", type=str, default=DEFAULT_CALIBSET,
        help="path to a build_qwen36_calibset.py-format jsonl (source/seq_len/input_ids per line). "
        "Pass 'wikitext2' to use plain WikiText2 instead (this script's original default, before the "
        "GLM-5.2-composition-matched calibset existed).",
    )
    p.add_argument("--euclidean_iters", type=int, default=15)
    p.add_argument(
        "--hesitation_boost", type=float, default=1.0,
        help="Reasoning-length-aware E2M-ATQ extension (module docstring item 4, motivated by "
        "arXiv:2606.00206): multiply a calibration token's contribution to the second moment S "
        "(Eq. 7) by this factor at token positions at/near one of overthinking_penalty.py's 50 "
        "hesitation markers, leaving every other position at weight 1.0. 1.0 (default) disables this "
        "-- token_weights stays None and accumulation is byte-identical to plain unweighted E2M-ATQ.",
    )
    p.add_argument(
        "--hesitation_window", type=int, default=3,
        help="Also boost tokens within this many positions of a hesitation-marker token (both "
        "directions), not just the marker token itself. Only matters if --hesitation_boost != 1.0.",
    )
    p.add_argument(
        "--hesitation_marker_allow_multi_token", action="store_true",
        help="see overthinking_penalty.resolve_marker_ids -- off by default, same reasoning (a shared "
        "first token piece would boost unrelated words too). Only matters if --hesitation_boost != 1.0.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_layers", type=int, default=None, help="debug: quantize only the first N layers")
    p.add_argument("--skip_save", action="store_true", default=False, help="debug: don't write out_dir")
    p.add_argument(
        "--layer_start", type=int, default=None,
        help="Only ternarize experts in layers [layer_start, layer_end] (inclusive, 0-indexed); layers "
        "outside the range still run forward normally (needed to propagate correct calibration hidden "
        "states to in-range layers deeper in the stack) but are left dense/unquantized. Default: quantize "
        "every layer.",
    )
    p.add_argument("--layer_end", type=int, default=None, help="see --layer_start; required together.")
    p.add_argument(
        "--skip_layers", type=str, default="",
        help="Comma-separated decoder-layer indices whose routed experts stay dense BF16 while all "
        "other selected layers are ternarized, e.g. '35,38,39'. This only changes layer selection; "
        "calibration and E2M-ATQ are otherwise identical.",
    )
    p.add_argument(
        "--num_threads", type=int, default=None,
        help="torch.set_num_threads() -- PyTorch defaults to ALL physical cores per process; running "
        "several of these concurrently (e.g. one per layer-range ablation checkpoint) then fights over "
        "the same cores instead of actually parallelizing, since each expert-hit in the calibration hook "
        "is a small Python-level op with real CUDA-launch/interpreter overhead. A real run on a 72-core "
        "box went from ~65-120s/layer to 3000-9000s/layer once 5 of these were running unpinned "
        "simultaneously. Pass e.g. 14 when running 5-way concurrent (72/5); default (unset) leaves torch's "
        "own default in place, correct only for a single solo run.",
    )
    args = p.parse_args()
    if (args.layer_start is None) != (args.layer_end is None):
        raise ValueError("--layer_start and --layer_end must be given together.")
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))

    print(f"Loading Qwen3.6-35B-A3B onto {args.device} (bf16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_pretrained_streaming(MODEL_DIR, args.device, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    config = text_model.config

    if args.calibset == "wikitext2":
        print(f"Loading calibration data: wikitext2, nsamples={args.nsamples}, seqlen={args.seqlen}...", flush=True)
        trainloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                      seqlen=args.seqlen, model=MODEL_DIR)
    else:
        print(f"Loading calibration data: {args.calibset}, nsamples={args.nsamples}, seqlen={args.seqlen}...",
              flush=True)
        trainloader = load_qwen36_calibset(args.calibset, nsamples=args.nsamples, seed=args.seed,
                                            seqlen=args.seqlen)

    token_weights = None
    if args.hesitation_boost != 1.0:
        from overthinking_penalty import OVERTHINKING_MARKERS, resolve_marker_ids

        marker_ids, _kept, _dropped = resolve_marker_ids(
            tokenizer, OVERTHINKING_MARKERS, allow_multi_token=args.hesitation_marker_allow_multi_token,
        )
        marker_id_set = set(marker_ids)
        print(f"[hesitation-weighting] boost={args.hesitation_boost} window={args.hesitation_window} "
              f"on {len(marker_id_set)} marker token ids", flush=True)
        token_weights = []
        total_tok = total_boosted = 0
        for inp, _ in trainloader:
            w = compute_hesitation_weights(inp[0], marker_id_set, args.hesitation_boost, args.hesitation_window)
            token_weights.append(w)
            total_tok += w.numel()
            total_boosted += int((w > 1.0).sum())
        pct = 100.0 * total_boosted / max(total_tok, 1)
        print(f"[hesitation-weighting] {total_boosted}/{total_tok} tokens ({pct:.1f}%) boosted to "
              f"{args.hesitation_boost}x across the calibset (window=+/-{args.hesitation_window}).", flush=True)

    from transformers.masking_utils import create_causal_mask

    with torch.no_grad():
        calib_hidden_states = []
        for inp, _ in trainloader:
            inp = inp.to(args.device)
            emb = text_model.embed_tokens(inp)
            calib_hidden_states.append(emb.to("cpu"))

        sample_inputs_embeds = calib_hidden_states[0].to(args.device)
        seqlen = sample_inputs_embeds.shape[1]
        position_ids = torch.arange(seqlen, device=args.device).view(1, 1, -1).expand(4, 1, -1)
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]
        causal_mask = create_causal_mask(
            config=config, inputs_embeds=sample_inputs_embeds, attention_mask=None,
            past_key_values=None, position_ids=text_position_ids,
        )
        linear_attn_mask = None  # full attention_mask==None case -> no left-padding, matches
        # `Qwen3_5MoeTextModel._update_linear_attn_mask`'s `attention_mask is None` -> None path
        position_embeddings = text_model.rotary_emb(sample_inputs_embeds, mrope_position_ids)

    common_kwargs = {
        "causal_mask": causal_mask,
        "linear_attn_mask": linear_attn_mask,
        "text_position_ids": text_position_ids,
        "position_embeddings": position_embeddings,
    }

    num_layers = config.num_hidden_layers
    if args.max_layers is not None:
        num_layers = min(num_layers, int(args.max_layers))
    skip_layers = {int(x) for x in args.skip_layers.split(",") if x.strip()}
    invalid_skip_layers = sorted(i for i in skip_layers if i < 0 or i >= num_layers)
    if invalid_skip_layers:
        raise ValueError(f"--skip_layers contains indices outside [0,{num_layers}): {invalid_skip_layers}")
    if skip_layers:
        print(f"Keeping routed experts dense BF16 in layers: {sorted(skip_layers)}", flush=True)
    print(f"Quantizing {num_layers} layers x {config.num_experts} experts each "
          f"(gate_up_proj + down_proj, {QUANT_LEVELS}-level E2M-ATQ)...", flush=True)

    all_stats = []
    t0 = time.time()
    for i in range(num_layers):
        layer = text_model.layers[i]
        apply_quant = (
            (args.layer_start is None or (args.layer_start <= i <= args.layer_end))
            and i not in skip_layers
        )
        stats, calib_hidden_states = quantize_layer_experts(
            layer, i, calib_hidden_states, common_kwargs, args, apply_quant=apply_quant,
            token_weights=token_weights,
        )
        all_stats.append(stats)
        elapsed = time.time() - t0
        tag = "" if apply_quant else " (SKIPPED, dense)"
        print(f"[layer {i + 1}/{num_layers}]{tag} hit={stats['n_experts_hit']} dead={stats['n_experts_dead']} "
              f"sum_sq_err={stats['sum_sq_err']:.2f} elapsed={elapsed:.1f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "twla_quant_stats.json"), "w") as f:
        json.dump({
            "nsamples": args.nsamples, "seqlen": args.seqlen, "calibset": args.calibset,
            "hesitation_boost": args.hesitation_boost, "hesitation_window": args.hesitation_window,
            "skip_layers": sorted(skip_layers), "num_levels": QUANT_LEVELS,
            "layers": all_stats,
        }, f, indent=2)

    if not args.skip_save:
        print(f"Saving quantized model -> {args.out_dir} ...", flush=True)
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

    total_dead = sum(s["n_experts_dead"] for s in all_stats)
    if not args.skip_save:
        open(os.path.join(args.out_dir, ".quant_done"), "a").close()
    print(f"\nDone. {total_dead} total dead-expert (0-hit) instances across {num_layers} layers "
          f"(left unquantized). Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Ablation - the output gain alone. The Fisher costs are re-measured with a least-squares output scalar per (expert,
# level), the pinned allocation is re-solved on them (the step-6 allocation is included as a selectable candidate, so
# "same allocation + gamma" can win), and gamma is folded into the selected checkpoint; no GPTQ reassignment.
# Output: checkpoints/ablation_gamma_only
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BANK="$CKPT/bank"; BASE="$CKPT/base_k5"; GAMMA="$LOGS/costs_gamma"; WORK="$LOGS/alloc_gamma_only"; OUT="$CKPT/ablation_gamma_only"
PIN="${PIN:-$LOGS/pins/pin_math500.json}"; PIN_LEVEL="${PIN_LEVEL:-8}"; TARGET_BITS="${TARGET_BITS:-2.0}"
MOMENTS="$(cat "$LOGS/fisher_moments.path")"
stage "$GAMMA/bank/.done" costs_gamma \
  "$PYTHON" -u fisher_anchor.py costs --model-dir "$MODEL_DIR" --moments "$MOMENTS" --bank "bank=$BANK" --levels "${LEVELS:-3,4,5,6,7,8,9}" \
    --out-dir "$GAMMA" --gamma --device cuda:0
stage "$WORK/.allocate_done" allocate_gamma_only \
  "$PYTHON" -u fisher_anchor.py allocate --costs "$GAMMA/bank" --target-bits "$TARGET_BITS" --initial-level 4 \
    --pin "$PIN" --pin-level "$PIN_LEVEL" --include "pinned_nogamma=$CKPT/stage12_pinned/optimization_summary.json" --out "$WORK/candidates.json"
stage "$OUT/.quant_done" select_gamma_only \
  "$PYTHON" -u fisher_anchor.py select --candidates "$WORK/candidates.json" --bank-dir "$BANK" --physical-base "$BASE" \
    --data-dir "$DATA/calibration" --work-dir "$WORK" --out-dir "$OUT" --gamma-dir "$GAMMA/bank" --device cuda:0
stage "$WORK/.gamma_verified" verify_gamma_only \
  "$PYTHON" -u verify_gamma_applied.py --checkpoint "$OUT" --bank-dir "$BANK" --gamma-dir "$GAMMA/bank"
say "gamma-only ablation checkpoint: $OUT"

#!/usr/bin/env bash
# Step 6 - Stage II with the termination experts pinned at K = 8 (3 bits) outside the bit budget: the other experts
# still average TARGET_BITS, so the whole model lands at 2.0 + |P|/N bits (2.0218 for 223 pins). PIN_IN_BUDGET=1
# instead keeps the whole-model average at TARGET_BITS. Output: checkpoints/stage12_pinned (the Stage 3 template and
# the "Stage I+II only" ablation)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BANK="$CKPT/bank"; BASE="$CKPT/base_k5"; COSTS="$LOGS/costs/bank"; WORK="$LOGS/alloc_pinned"; OUT="$CKPT/stage12_pinned"
PIN="${PIN:-$LOGS/pins/pin_math500.json}"; PIN_LEVEL="${PIN_LEVEL:-8}"; TARGET_BITS="${TARGET_BITS:-2.0}"
EXTRA=(); [ "${PIN_IN_BUDGET:-0}" = 1 ] && EXTRA=(--pin-in-budget)
stage "$WORK/.allocate_done" allocate_pinned \
  "$PYTHON" -u fisher_anchor.py allocate --costs "$COSTS" --target-bits "$TARGET_BITS" --initial-level 4 \
    --pin "$PIN" --pin-level "$PIN_LEVEL" "${EXTRA[@]}" --out "$WORK/candidates.json"
stage "$OUT/.quant_done" select_pinned \
  "$PYTHON" -u fisher_anchor.py select --candidates "$WORK/candidates.json" --bank-dir "$BANK" --physical-base "$BASE" \
    --data-dir "$DATA/calibration" --work-dir "$WORK" --out-dir "$OUT" --device cuda:0
stage "$WORK/.verified" verify_pinned \
  "$PYTHON" -u verify_pinned_level.py --checkpoint "$OUT" --bank-dir "$BANK" --pin "$PIN" --pin-level "$PIN_LEVEL"
say "pinned Stage I+II checkpoint: $OUT ($(grep -o '"final_average_logical_bits": [0-9.]*' "$OUT/optimization_summary.json"))"

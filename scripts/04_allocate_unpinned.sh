#!/usr/bin/env bash
# Step 4 - Stage II without pins: the multiple-choice knapsack over the Fisher costs (and the plain output cost) for a
# few trust regions around K0 = 4, each candidate installed from the bank and the one with the lowest held-out
# calibration NLL kept. This unpinned 2-bit model is one of the two routing-trace sources of step 5 and the
# "no pins" ablation. Output: checkpoints/stage12_unpinned
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BANK="$CKPT/bank"; BASE="$CKPT/base_k5"; COSTS="$LOGS/costs/bank"; WORK="$LOGS/alloc_unpinned"; OUT="$CKPT/stage12_unpinned"
TARGET_BITS="${TARGET_BITS:-2.0}"
stage "$WORK/.allocate_done" allocate_unpinned \
  "$PYTHON" -u fisher_anchor.py allocate --costs "$COSTS" --target-bits "$TARGET_BITS" --initial-level 4 --out "$WORK/candidates.json"
stage "$OUT/.quant_done" select_unpinned \
  "$PYTHON" -u fisher_anchor.py select --candidates "$WORK/candidates.json" --bank-dir "$BANK" --physical-base "$BASE" \
    --data-dir "$DATA/calibration" --work-dir "$WORK" --out-dir "$OUT" --device cuda:0
say "unpinned Stage I+II checkpoint: $OUT ($(grep -o '"selected_candidate": "[^"]*"' "$OUT/optimization_summary.json"))"

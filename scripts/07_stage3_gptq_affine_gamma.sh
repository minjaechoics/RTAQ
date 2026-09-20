#!/usr/bin/env bash
# Step 7 - Stage III on the pinned allocation. For every expert at its allocated level: routing-weighted curvature
# from the BF16 routed inputs (all calibration rows; the last 15 are held out to choose the damping), GPTQ code
# reassignment inside the expert's fixed codebook, H-weighted refit of the row offset/scale, then one output gain
# gamma folded into down_proj. Bit width, codebooks and storage layout are unchanged.
# Output: checkpoints/rtaq_final (the main checkpoint of the paper)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BANK="$CKPT/bank"; TEMPLATE="${TEMPLATE:-$CKPT/stage12_pinned}"; OUT="${OUT:-$CKPT/rtaq_final}"
INPUTS="$LOGS/expert_inputs"; AFFINE="$LOGS/affine_final"; PIN="${PIN:-$LOGS/pins/pin_math500.json}"
stage "$INPUTS/.done" collect_expert_inputs env CUDA_VISIBLE_DEVICES="$FIRST_GPU" \
  "$PYTHON" -u collect_moe_inputs.py --model-dir "$MODEL_DIR" --data "$DATA/calibration/calibration_packed2048.jsonl" \
    --out-dir "$INPUTS" --device cuda:0
stage "$OUT/.quant_done" stage3 \
  "$PYTHON" -u apply_gptq_gamma.py --levels "$TEMPLATE/optimization_summary.json" --bank-dir "$BANK" --model-dir "$MODEL_DIR" \
    --moments "$INPUTS" --template "$TEMPLATE" --work-dir "$OUT.work" --out-dir "$OUT" --damp auto --holdout-rows 15 \
    --refit-affine --affine-dir "$AFFINE" --min-tokens 64 --devices "$(cuda_list)" --delete-work-as-assembled
PINARG=(); [ -f "$PIN" ] && PINARG=(--pin "$PIN")
stage "$OUT/.grid_verified" verify_stage3 \
  "$PYTHON" -u verify_gptq_gamma.py --checkpoint "$OUT" --bank-dir "$BANK" --template "$TEMPLATE" --affine-dir "$AFFINE" "${PINARG[@]}"
rm -rf "$OUT.work"
say "final checkpoint: $OUT"

#!/usr/bin/env bash
# Step 3 - Stage I and the Stage II cost table.
#   bank           every routed expert at K = 3..9: activation-weighted histogram, exact 1-D k-means codebook (one DP,
#                  backtracked per K), 5 rounds of code assignment / affine fit, S-weighted affine relocation (K = 3 is
#                  the symmetric ternary path). Stored as packed codes + codebook + bf16 row affine parameters.
#   physical_base  a bf16 checkpoint with every expert at the uniform level K = 5, into which candidate allocations
#                  are installed for selection; non-expert tensors are kept once and hard-linked
#   fisher         one BF16 forward+backward per calibration document (16 x 2048 tokens; 32 x 1024 if the 2048-token
#                  rows do not fit), storing routed inputs, block-output gradients and routing weights
#   costs          C[l, e, K] of Eq. 8 for every expert and level, plus the plain output error
# Output: checkpoints/bank, checkpoints/base_k5, logs/costs/bank (the cost table)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BANK="$CKPT/bank"; BASE="$CKPT/base_k5"; NONEXP="$CKPT/nonexpert_base"; CAL="$DATA/calibration"; COSTS="$LOGS/costs"
LEVELS="${LEVELS:-3,4,5,6,7,8,9}"
NROWS=$(wc -l < "$CAL/calibration_packed2048.jsonl")
"$PYTHON" - "$LOGS/uniform_level5.json" <<'PY'
import json, sys
json.dump({f"{l}:{e}": 5 for l in range(40) for e in range(256)}, open(sys.argv[1], "w"))
PY
stage "$BANK/.bank_done" bank \
  "$PYTHON" -u build_qwen_glmstyle_proxy_v1_bank.py --scheme asymmetric_norot --bank-dir "$BANK" --model-dir "$MODEL_DIR" \
    --calibset "$CAL/calibration_packed2048.jsonl" --levels "$LEVELS" --nsamples "$NROWS" --seqlen 2048 --device cuda:0 --num-threads 20
stage "$BASE/.quant_done" physical_base \
  "$PYTHON" -u materialize_qwen_glmstyle_proxy.py --model-dir "$MODEL_DIR" --bank-dir "$BANK" \
    --levels-json "$LOGS/uniform_level5.json" --out-dir "$BASE" --nonexpert-base "$NONEXP" --case uniform_level5 --device cuda:0
MOMENTS=""
for setting in "16 2048 $LOGS/fisher_moments" "32 1024 $LOGS/fisher_moments_1024"; do
  set -- $setting; rows=$1 tokens=$2 dir=$3
  if [ -e "$dir/.done" ]; then MOMENTS="$dir"; say "SKIP fisher_$tokens (done)"; break; fi
  say "START fisher_$tokens rows=$rows (master sharded over $NGPU GPU(s))"
  "$PYTHON" -u fisher_anchor.py collect --model-dir "$MODEL_DIR" --data "$CAL/calibration.pt" --out-dir "$dir" \
    --rows "$rows" --max-tokens "$tokens" --grad-checkpointing --device-map balanced --device cuda:0 >>"$LOGS/fisher_$tokens.log" 2>&1
  rc=$?; say "EXIT fisher_$tokens rc=$rc"
  if [ $rc -eq 0 ] && [ -e "$dir/.done" ]; then MOMENTS="$dir"; break; fi
  say "fisher collection at $tokens tokens failed; trying the next setting"
done
[ -n "$MOMENTS" ] || { say "ABORT: Fisher moment collection failed at every setting"; exit 1; }
echo "$MOMENTS" > "$LOGS/fisher_moments.path"
stage "$COSTS/bank/.done" costs \
  "$PYTHON" -u fisher_anchor.py costs --model-dir "$MODEL_DIR" --moments "$MOMENTS" --bank "bank=$BANK" --levels "$LEVELS" \
    --out-dir "$COSTS" --device cuda:0
say "bank and cost table ready"

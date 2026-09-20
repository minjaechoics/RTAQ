#!/usr/bin/env bash
# Step 2 - the calibration set.
#   prepare       document-disjoint validation splits (probe / convergence / final) and a stratified candidate pool
#   on-policy     the target model re-answers every pool prompt, replacing the source datasets' assistant turns
#   route-scan    BF16 top-k routing counts for every candidate, one shard per GPU in $GPUS
#   route-select  128 documents chosen greedily for routed-expert coverage, packed into 2048-token rows
#
# The on-policy rewrite is what makes the routing statistics describe the model being quantized rather than whatever
# models wrote the source corpus. On Nemotron-3-Nano it moves the correlation between the calibration routing load
# and the model's load on its own reasoning traces from 0.51 to 0.71, and is worth about 8 AIME points on the final
# 2-bit checkpoint. Set ONPOLICY=0 to reproduce the off-policy ablation.
# Output: data/calibration/{calibration_packed2048.jsonl, calibration.pt, convergence_validation.pt, final_validation.pt, ...}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CAL="$DATA/calibration"
stage "$CAL/.prepared" cal_prepare \
  "$PYTHON" -u prepare_bipea_v3_data.py prepare --source "$DATA/corpus" --output "$CAL"

if [ "${ONPOLICY:-1}" = 1 ]; then
  stage "$CAL/.onpolicy" cal_onpolicy \
    "$PYTHON" -u make_calibration_onpolicy.py --output "$CAL" --model "$MODEL_DIR" \
      --tensor_parallel_size "$NGPU" --seq 2048
else
  say "ONPOLICY=0: keeping the source corpus' assistant turns (off-policy ablation)"
fi
if [ ! -e "$CAL/.scanned" ]; then
  say "route-scan on $NGPU GPU(s)"
  i=0; pids=()
  for gpu in $(echo "$GPUS" | tr ',' ' '); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u prepare_bipea_v3_data.py route-scan --output "$CAL" --model "$MODEL_DIR" \
      --device cuda:0 --shard-idx "$i" --num-shards "$NGPU" >"$LOGS/route_scan_$i.log" 2>&1 &
    pids+=("$!"); i=$((i + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || { say "route-scan failed (see logs/route_scan_*.log)"; exit 1; }; done
  touch "$CAL/.scanned"
fi
stage "$CAL/.selected" cal_select \
  "$PYTHON" -u prepare_bipea_v3_data.py route-select --output "$CAL" --num-shards "$NGPU"
say "calibration set ready: $(wc -l < "$CAL/calibration_packed2048.jsonl") packed rows in $CAL"

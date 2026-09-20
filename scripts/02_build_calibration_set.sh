#!/usr/bin/env bash
# Step 2 - the calibration set.
#   prepare       document-disjoint validation splits (probe / convergence / final) and a stratified candidate pool
#   route-scan    BF16 top-k routing counts for every candidate, one shard per GPU in $GPUS
#   route-select  128 documents chosen greedily for routed-expert coverage, packed into 2048-token rows
# Output: data/calibration/{calibration_packed2048.jsonl, calibration.pt, convergence_validation.pt, final_validation.pt, ...}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CAL="$DATA/calibration"
stage "$CAL/.prepared" cal_prepare \
  "$PYTHON" -u prepare_bipea_v3_data.py prepare --source "$DATA/corpus" --output "$CAL"
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

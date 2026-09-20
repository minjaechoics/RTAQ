#!/usr/bin/env bash
# Step 5 - the termination-expert set P (Eq. 9).
# MATH-500 is generated (40K-token cap) by the BF16 model and by the unpinned 2-bit model of step 4; each response is
# teacher-forced back through its own model's routers, and every layer/expert's selection rate is measured at the
# closing position (the one that predicts </think>) and on 200 positions sampled inside the reasoning trace.
# P = union over the two sources of {rho_close >= 0.8 and rho_reason < 0.3}. GPQA-Diamond is not used here, so it
# stays held out from the pin selection. Output: logs/pins/pin_math500.json (a list of [layer, expert])
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PINS="$LOGS/pins"; mkdir -p "$PINS"
for pair in "bf16:$MODEL_DIR" "unpinned:$CKPT/stage12_unpinned"; do
  name=${pair%%:*}; model=${pair#*:}
  stage "$RESULTS/math500_$name/.done" math500_$name \
    "$PYTHON" -u run_math_suite_vllm_tp.py --model_dir "$model" --output_dir "$RESULTS/math500_$name" --benchmarks math500 \
      --max_gen_toks 40000 --max_model_len 42048 --tensor_parallel_size "$TP" --max_num_seqs 64 --moe_backend triton \
      --gpu_memory_utilization 0.92
  stage "$PINS/.probe_$name" probe_$name env CUDA_VISIBLE_DEVICES="$FIRST_GPU" \
    "$PYTHON" -u probe_termination_routing.py --model_dir "$model" --source_dir "$RESULTS/math500_$name" --benchmark math500 \
      --source_format math_suite --out "$PINS/math500_$name.json" --device cuda:0
done
stage "$PINS/.pins_done" build_pins \
  "$PYTHON" -u build_pins.py --out "$PINS/pin_math500.json" math500_bf16="$PINS/math500_bf16.npz" math500_unpinned="$PINS/math500_unpinned.npz"
say "pin set: $("$PYTHON" -c "import json; print(len(json.load(open('$PINS/pin_math500.json'))))") experts -> $PINS/pin_math500.json (statistics in pin_math500.report.json)"

#!/usr/bin/env bash
# Step 8 - the sampled avg@3 protocol of the paper for any checkpoint.
#   usage: 08_evaluate.sh TAG MODEL_DIR [gsm8k aime25 aime26 gpqa198 lcb]      (default: all five)
# temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, 3 samples per problem, no loop abort; caps: GSM8K 40K,
# AIME 2025/2026 100K, GPQA-Diamond 70K, LiveCodeBench v6 (25.02-25.05) 100K. An answer counts only when the response
# stopped on its own inside the cap. Results: results/TAG_<bench>/{summary.json, scores.jsonl, generations/}
# GPQA-Diamond needs access to the gated dataset Idavidrein/gpqa (accept its terms, then `hf auth login`).
# LiveCodeBench needs the official grader: the LiveCodeBench repository is cloned into third_party/ on first use.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAG=${1:?usage: 08_evaluate.sh TAG MODEL_DIR [benchmarks...]}; MODEL=${2:?}; shift 2
BENCHES=("$@"); [ ${#BENCHES[@]} -gt 0 ] || BENCHES=(gsm8k aime25 aime26 gpqa198 lcb)
[ -d "$MODEL" ] || MODEL="$("$PYTHON" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")"
SAMPLING=(--num_samples 3 --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --seed 0 --no_loop_abort)
COMMON=(--tensor_parallel_size "$TP" --moe_backend triton --gpu_memory_utilization 0.92)
for bench in "${BENCHES[@]}"; do
  case "$bench" in
    gsm8k)  stage "$RESULTS/${TAG}_gsm8k_40k_s3/.done" "eval_${TAG}_gsm8k" \
              "$PYTHON" -u run_sampled_eval_vllm.py --benchmark gsm8k --model_dir "$MODEL" --output_dir "$RESULTS/${TAG}_gsm8k_40k_s3" \
                --max_gen_toks 40000 --max_model_len 42048 --max_num_seqs 128 "${SAMPLING[@]}" "${COMMON[@]}" ;;
    aime25|aime26)
            stage "$RESULTS/${TAG}_${bench}_100k_s3/.done" "eval_${TAG}_${bench}" \
              "$PYTHON" -u run_sampled_eval_vllm.py --benchmark "$bench" --model_dir "$MODEL" --output_dir "$RESULTS/${TAG}_${bench}_100k_s3" \
                --max_gen_toks 100000 --max_model_len 102048 --max_num_seqs 48 "${SAMPLING[@]}" "${COMMON[@]}" ;;
    gpqa198)
            stage "$DATA/gpqa198/.done" gpqa_docs \
              "$PYTHON" -u prepare_gpqa_docs.py --order "$RTAQ_ROOT/assets/gpqa198_choice_order.json" --out "$DATA/gpqa198" --model-dir "$MODEL_DIR"
            stage "$RESULTS/${TAG}_gpqa198_70k_s3/.done" "eval_${TAG}_gpqa198" \
              "$PYTHON" -u run_sampled_eval_vllm.py --benchmark gpqa198 --model_dir "$MODEL" --output_dir "$RESULTS/${TAG}_gpqa198_70k_s3" \
                --gpqa_doc_cache "$DATA/gpqa198" --gpqa_rerender --max_gen_toks 70000 --max_model_len 72048 --max_num_seqs 64 \
                "${SAMPLING[@]}" "${COMMON[@]}" ;;
    lcb)    LCB="$RTAQ_ROOT/third_party/LiveCodeBench"
            if [ ! -d "$LCB" ]; then
              say "cloning LiveCodeBench into third_party/"
              git clone -q https://github.com/LiveCodeBench/LiveCodeBench "$LCB" && git -C "$LCB" checkout -q 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 || exit 1
            fi
            stage "$DATA/livecodebench/lcb_codegen_v6_2502_2505.jsonl" lcb_problems \
              "$PYTHON" -u lcb_codegen.py build --start 2025-02-01 --end 2025-05-01 --out "$DATA/livecodebench/lcb_codegen_v6_2502_2505.jsonl"
            stage "$RESULTS/${TAG}_lcb_v6_100k_s3/.done" "eval_${TAG}_lcb" \
              "$PYTHON" -u run_sampled_eval_vllm.py --benchmark lcb_codegen --model_dir "$MODEL" --output_dir "$RESULTS/${TAG}_lcb_v6_100k_s3" \
                --lcb_problems "$DATA/livecodebench/lcb_codegen_v6_2502_2505.jsonl" --max_gen_toks 100000 --max_model_len 102048 \
                --max_num_seqs 48 "${SAMPLING[@]}" "${COMMON[@]}" ;;
    *) say "unknown benchmark $bench"; exit 2 ;;
  esac
done
for d in "$RESULTS"/${TAG}_*_s3; do [ -f "$d/summary.json" ] && "$PYTHON" -c "
import json, sys; s = json.load(open('$d/summary.json')); print('%-45s avg@3 %.1f  termination %.1f%%  mean tokens %.0f' % ('$(basename "$d")', 100*s['avg@3'], 100*s['termination_rate'], s['mean_generated_tokens']))"; done

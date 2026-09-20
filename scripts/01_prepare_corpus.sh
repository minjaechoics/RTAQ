#!/usr/bin/env bash
# Step 1 - the GPQA-free document pool that the calibration set is drawn from.
# Downloads the public source datasets at the revisions the paper used (about 250 GB of raw shards) and builds the
# corpus with prepare_nogpqa_training_corpus.py: source quotas, completeness and repetition filters, chat-template
# rendering with the model's tokenizer, one tokenised sequence per document, a source-stratified validation split.
# Output: data/corpus/{train.jsonl, validation.jsonl, metadata.json}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
HF="$DATA/hf_datasets"; mkdir -p "$HF"
dl() {  # dl REPO REVISION SUBDIR INCLUDE...
  local repo=$1 rev=$2 sub=$3; shift 3
  local inc=(); local p; for p in "$@"; do inc+=(--include "$p"); done
  [ -e "$HF/$sub/.done" ] && { say "SKIP download $sub"; return 0; }
  say "download $repo @ ${rev:0:8} -> data/hf_datasets/$sub"
  hf download "$repo" --repo-type dataset --revision "$rev" "${inc[@]}" --local-dir "$HF/$sub" >>"$LOGS/download_$sub.log" 2>&1 \
    && touch "$HF/$sub/.done" || { say "download of $repo failed (see logs/download_$sub.log)"; exit 1; }
}
dl open-r1/OpenR1-Math-220k                          e4e141ec9dea9f8326f4d347be56105859b2bd68 openr1_math_220k    'data/*.parquet'
dl nvidia/Nemotron-Post-Training-Dataset-v2          5c89e01dd720ae0f4058445ed49c5fb68a03c76e nemotron_ptd_v2     'data/math-*' 'data/code-*' 'data/stem-*' 'data/chat-*'
dl nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 main                                     nemotron_sft_chat_v2 'data/reasoning_off.jsonl'
dl nvidia/Nemotron-Science-v1                        82e1af468197076b4f0f392c239274eac032adc7 nemotron_science_v1 'data/MCQ.jsonl' 'data/RQA.jsonl'
dl abisee/cnn_dailymail                              96df5e686bee6baa90b8bee7c28b81fa3fa6223d cnn_dailymail       '3.0.0/train-*.parquet'
# nvidia/Nemotron-SFT-Math-v3 is streamed by the builder itself (its single file is 154 GB; only a filtered slice is kept).
stage "$DATA/corpus/.done" corpus \
  "$PYTHON" -u prepare_nogpqa_training_corpus.py --openr1_dir "$HF/openr1_math_220k" --output_dir "$DATA/corpus"
say "corpus ready: $DATA/corpus"

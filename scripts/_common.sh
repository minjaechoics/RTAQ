# Sourced by every driver. Three environment variables describe the machine; everything else derives from them.
#   RTAQ_ROOT  this repository (default: the parent of scripts/)
#   MODEL_DIR  a local Qwen3.6-35B-A3B snapshot, or a Hub id that is downloaded on first use
#   PYTHON     an interpreter with requirements.txt installed
#   GPUS       comma-separated GPU indices to use (default: every GPU nvidia-smi lists)
# Every driver is resumable: a stage that left its marker behind is skipped on the next run.
set -uo pipefail
RTAQ_ROOT="${RTAQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3.6-35B-A3B}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, )}"
GPUS="${GPUS:-0}"
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
FIRST_GPU="${GPUS%%,*}"
TP=1; while [ $((TP * 2)) -le "$NGPU" ]; do TP=$((TP * 2)); done   # vLLM tensor parallelism: largest power of two that fits
DATA="$RTAQ_ROOT/data"; CKPT="$RTAQ_ROOT/checkpoints"; LOGS="$RTAQ_ROOT/logs"; RESULTS="$RTAQ_ROOT/results"
mkdir -p "$DATA" "$CKPT" "$LOGS" "$RESULTS"
export RTAQ_ROOT PYTHON CUDA_VISIBLE_DEVICES="$GPUS" TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
if [ ! -d "$MODEL_DIR" ]; then   # a Hub id: fetch the snapshot once and use its local path from here on
  MODEL_DIR="$("$PYTHON" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_DIR'))")" || exit 1
fi
export MODEL_DIR
cd "$RTAQ_ROOT/src"
say() { echo "[$(date -Is)] $*" | tee -a "$LOGS/driver.log"; }
stage() {  # stage MARKER NAME COMMAND...   runs COMMAND with its output in logs/NAME.log unless MARKER already exists
  local marker=$1 name=$2; shift 2
  if [ -e "$marker" ]; then say "SKIP $name ($marker exists)"; return 0; fi
  mkdir -p "$(dirname "$marker")"
  say "START $name"
  "$@" >>"$LOGS/$name.log" 2>&1; local rc=$?
  say "EXIT $name rc=$rc"
  [ $rc -eq 0 ] || { say "ABORT at $name (see $LOGS/$name.log)"; exit $rc; }
  touch "$marker"
}
cuda_list() { local out=""; for i in $(seq 0 $((NGPU - 1))); do out="$out${out:+,}cuda:$i"; done; echo "$out"; }

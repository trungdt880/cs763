#!/usr/bin/env bash
# Phase 3 ablation runner. Launches a 6-run matrix (3 seeds x 2 conditions)
# of 0.6B Qwen3 fine-tunes on 2-GPU pools.
#
# Usage:
#   scripts/run_phase3.sh <matrix> [GPU_LIST]
# where <matrix> is one of: hard | pii | mask | dp_eps8
# and GPU_LIST defaults to "6,7" (waiv6 free GPUs).
#
# Each run trains, then evaluates with eval_fast.py on the same GPU it trained
# on. We run 2 in parallel (one per GPU) and serialize the remaining 4 in
# 2-run waves, so wall-clock is ~3x single-run training time per matrix.
set -euo pipefail

MATRIX=${1:?"usage: run_phase3.sh <hard|pii|mask|dp_eps8> [GPU_LIST]"}
GPUS=${2:-"6,7"}
IFS=',' read -ra GPU_ARR <<<"$GPUS"
N_GPUS=${#GPU_ARR[@]}

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"
# Cache-dir env vars: keep HF/torch caches off AFS. Defaults to a project-local
# .caches/ dir; override via $MYHOME (e.g. point at a fast scratch volume).
if [ -z "${MYHOME:-}" ]; then
  export MYHOME="$PROJECT_DIR/.caches"
fi
mkdir -p "$MYHOME"
export TMPDIR="$MYHOME/.tmp"
export HF_HOME="$MYHOME/.cache/huggingface"
export TORCH_HOME="$MYHOME/.cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
source .venv/bin/activate
mkdir -p logs

# Per-matrix flags applied to every run.
case "$MATRIX" in
hard)
  TRAIN=train.py
  FLAGS="--difficulty hard"
  ;;
pii)
  TRAIN=train.py
  FLAGS="--canary_format pii"
  ;;
mask)
  TRAIN=train.py
  FLAGS="--mask_prompt"
  ;;
dp_eps8)
  TRAIN=train_dp.py
  FLAGS="--target_epsilon 8 --max_grad_norm 1.0 --max_physical_batch 2 --num_epochs 5"
  # On 24GB 3090, full per-sample-grad fp32 0.6B fits at mpb=2 with
  # expandable_segments; 5 epochs (vs 10) keeps wall-clock manageable.
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ;;
*)
  echo "Unknown matrix: $MATRIX (expected hard|pii|mask|dp_eps8)"
  exit 2
  ;;
esac
SUFFIX="$MATRIX"

MODEL=qwen3_0p6b
PER_DEVICE_BATCH=16
GRAD_ACCUM=2 # eff batch = 32; matches Phase 2 0.6B config

# Build the run list: 3 seeds x 2 conditions = 6 runs (or fewer via SEED_LIST env).
# DP-SGD takes ~4h per wave on 24GB 3090s, so we typically reduce to 1 seed.
SEED_LIST="${SEED_LIST:-0 1 2}"
RUNS=()
for SEED in $SEED_LIST; do
  for COND in answer cot; do
    RUNS+=("$MODEL $COND $SEED")
  done
done

run_one() {
  local gpu=$1
  local model=$2
  local cond=$3
  local seed=$4
  local run="${model}__${cond}__seed${seed}__${SUFFIX}"
  echo "  [GPU $gpu] launching $run"
  CUDA_VISIBLE_DEVICES=$gpu python "$TRAIN" \
    --config configs.yaml --model "$model" --condition "$cond" --seed "$seed" \
    --per_device_batch $PER_DEVICE_BATCH --grad_accum $GRAD_ACCUM \
    --run_suffix "$SUFFIX" \
    $FLAGS \
    >"logs/${run}.train.log" 2>&1 &&
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_fast.py \
      --results_dir "results/$run" --gpus $gpu \
      >"logs/${run}.eval.log" 2>&1
  echo "  [GPU $gpu] DONE $run"
}

# Wave-launcher: process RUNS in chunks of N_GPUS, one per GPU per wave.
i=0
total=${#RUNS[@]}
while [ $i -lt $total ]; do
  for ((g = 0; g < N_GPUS && i < total; g++, i++)); do
    set -- ${RUNS[$i]}
    run_one ${GPU_ARR[$g]} $1 $2 $3 &
  done
  echo "Wave dispatched. Waiting..."
  wait
done

echo ""
echo "All ${total} $MATRIX runs complete."
echo "Result dirs: results/${MODEL}__*__${SUFFIX}/"
echo "Logs:        logs/*.${SUFFIX}.*.log"

#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
mkdir -p logs

# 1.7B matrix: 6 runs (3 seeds x 2 conditions), 1 GPU each.
# per_device_batch=8, grad_accum=4 → effective batch=32 (matches 0.6B runs).
# grad_checkpointing=true (set in configs.yaml for qwen3_1p7b).
# Expected: ~60-90 min training + ~30-60 min eval per run.
# All 6 run in parallel → wall-clock = time of 1 run.
#
# NOTE: if Qwen/Qwen3-1.7B-Base doesn't exist on HF, try changing
# hf_name in configs.yaml to Qwen/Qwen3-1.7B instead.

for entry in \
  "0 qwen3_1p7b answer 0" \
  "1 qwen3_1p7b cot    0" \
  "2 qwen3_1p7b answer 1" \
  "3 qwen3_1p7b cot    1" \
  "4 qwen3_1p7b answer 2" \
  "5 qwen3_1p7b cot    2"; do
  set -- $entry
  GPU=$1
  MODEL=$2
  COND=$3
  SEED=$4
  RUN="${MODEL}__${COND}__seed${SEED}"
  (
    echo "=== TRAIN $RUN on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --config configs.yaml --model $MODEL --condition $COND --seed $SEED \
      --per_device_batch 8 --grad_accum 4 &&
      echo "=== EVAL $RUN on GPU $GPU ===" &&
      CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_fast.py \
        --results_dir results/$RUN --gpus $GPU
  ) >logs/${RUN}.log 2>&1 &
  echo "  launched $RUN on GPU $GPU (pid $!)"
done

echo ""
echo "6 runs launched in parallel on GPUs 0-5. Monitor with:"
echo "  tail -f logs/qwen3_1p7b__cot__seed0.log"
echo "  ls results/qwen3_1p7b*/metrics.json"
echo ""
echo "Waiting for all to complete..."
wait
echo ""
echo "All done. Regenerate plots (includes both 0.6B and 1.7B):"
echo "  python plots.py"

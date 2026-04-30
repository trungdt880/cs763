#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
source .venv/bin/activate
mkdir -p logs

# 0.6B matrix: 5 remaining runs (cot seed0 already done).
# 1 GPU per run, per_device_batch=32, effective batch=32.
for entry in \
  "0 qwen3_0p6b answer 0" \
  "1 qwen3_0p6b cot    1" \
  "2 qwen3_0p6b answer 1" \
  "3 qwen3_0p6b cot    2" \
  "4 qwen3_0p6b answer 2"; do
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
      --per_device_batch 32 --grad_accum 1 &&
      echo "=== EVAL $RUN on GPU $GPU ===" &&
      CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_fast.py \
        --results_dir results/$RUN --gpus $GPU
  ) >logs/${RUN}.log 2>&1 &
  echo "  launched $RUN on GPU $GPU (pid $!)"
done

echo ""
echo "5 runs launched in parallel. Monitor with:"
echo "  tail -f logs/qwen3_0p6b__answer__seed0.log"
echo "  ls results/*/metrics.json    # appears as each run finishes eval"
echo ""
echo "Waiting for all to complete..."
wait
echo ""
echo "All done. Generate plots:"
echo "  python plots.py"
echo "  python scripts/inspect_metrics.py results/qwen3_0p6b__answer__seed0 --checkpoint final"

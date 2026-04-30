#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

echo "=== TRAIN qwen3_0p6b__cot__seed0 on GPU 5 ==="
CUDA_VISIBLE_DEVICES=5 python train.py \
    --config configs.yaml --model qwen3_0p6b --condition cot --seed 0 \
    --per_device_batch 32 --grad_accum 1

echo "=== EVAL qwen3_0p6b__cot__seed0 on GPU 5 ==="
CUDA_VISIBLE_DEVICES=5 python scripts/eval_fast.py \
    --results_dir results/qwen3_0p6b__cot__seed0 --gpus 5

echo "Done. Inspect with:"
echo "  python scripts/inspect_metrics.py results/qwen3_0p6b__cot__seed0 --checkpoint final"

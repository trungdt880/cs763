#!/bin/bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs.yaml \
    --model qwen3_0p6b --condition cot --seed 0 \
    --per_device_batch 32 --grad_accum 1

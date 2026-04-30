#!/usr/bin/env bash
# Wait for DP-eval on seed 0 to finish (GPUs 6,7 free), then launch seeds 1,2.
set -euo pipefail
cd "$(dirname "$0")/.."

while [ ! -f results/qwen3_0p6b__answer__seed0__dp_eps8/metrics.json ] ||
    [ ! -f results/qwen3_0p6b__cot__seed0__dp_eps8/metrics.json ]; do
    echo "[dp_more $(date '+%H:%M:%S')] waiting for seed-0 eval"
    sleep 30
done
echo "[dp_more] seed 0 eval done, launching seeds 1,2"
SEED_LIST="1 2" bash scripts/run_phase3.sh dp_eps8 6,7
echo "[dp_more] all DP seeds done."

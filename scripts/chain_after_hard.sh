#!/usr/bin/env bash
# Wait for hard matrix to finish, then run dp_eps8 matrix on the same GPUs.
set -euo pipefail
GPUS=${1:-"6,7"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

while true; do
    n=$(find results -maxdepth 2 -path "results/qwen3_0p6b__*__seed*__hard/metrics.json" 2>/dev/null | wc -l)
    echo "[chain_hard_dp $(date '+%H:%M:%S')] hard: $n/6"
    if [ "$n" -ge "6" ]; then break; fi
    sleep 60
done
echo "[chain_hard_dp] launching dp_eps8 matrix on $GPUS"
bash scripts/run_phase3.sh dp_eps8 "$GPUS"
echo "[chain_hard_dp] dp_eps8 matrix done."

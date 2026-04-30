#!/usr/bin/env bash
# Wait for pii matrix to finish, then run mask matrix on the same GPUs.
set -euo pipefail
GPUS=${1:-"4,5"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

while true; do
    n=$(find results -maxdepth 2 -path "results/qwen3_0p6b__*__seed*__pii/metrics.json" 2>/dev/null | wc -l)
    echo "[chain_pii_mask $(date '+%H:%M:%S')] pii: $n/6"
    if [ "$n" -ge "6" ]; then break; fi
    sleep 60
done
echo "[chain_pii_mask] launching mask matrix on $GPUS"
bash scripts/run_phase3.sh mask "$GPUS"
echo "[chain_pii_mask] mask matrix done."

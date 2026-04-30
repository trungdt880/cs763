#!/usr/bin/env bash
# Wait for the hard matrix to finish, then run pii, mask, and dp matrices in
# sequence on the same GPU pool. Designed to be launched as a background job
# alongside scripts/run_phase3.sh hard.
set -euo pipefail

GPUS=${1:-"6,7"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

wait_matrix() {
  local suffix=$1
  local need=6
  while true; do
    local n=$(ls -d results/qwen3_0p6b__*__seed*__${suffix} 2>/dev/null | xargs -I{} test -f {}/metrics.json && {} 2>/dev/null | wc -l || true)
    n=$(find results -maxdepth 2 -path "results/qwen3_0p6b__*__seed*__${suffix}/metrics.json" 2>/dev/null | wc -l)
    echo "[chain $(date '+%H:%M:%S')] $suffix matrix: $n/$need metrics.json files"
    if [ "$n" -ge "$need" ]; then break; fi
    sleep 60
  done
}

echo "[chain] waiting for hard matrix to finish..."
wait_matrix hard

echo "[chain] launching pii matrix"
bash scripts/run_phase3.sh pii "$GPUS"

echo "[chain] launching mask matrix"
bash scripts/run_phase3.sh mask "$GPUS"

echo "[chain] launching dp_eps8 matrix"
bash scripts/run_phase3.sh dp_eps8 "$GPUS"

echo "[chain] all phase 3 matrices complete."

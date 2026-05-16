#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-sparse-local-small}"
OUT_DIR="${2:-results/local_small}"

while tmux has-session -t "${SESSION}" 2>/dev/null; do
  date '+%F %T waiting for training session...'
  sleep 300
done

echo "training session ${SESSION} ended; starting evaluation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" CONFIG_PREFIX=local_small scripts/eval_small_suite.sh "${OUT_DIR}"

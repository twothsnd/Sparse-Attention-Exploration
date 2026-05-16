#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-sparse-local-nsa}"
OUT_DIR="${2:-results/local_small_nsa}"

while tmux has-session -t "${SESSION}" 2>/dev/null; do
  date '+%F %T waiting for local nsa training session...'
  sleep 300
done

echo "training session ${SESSION} ended; starting local nsa evaluation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" scripts/eval_local_nsa_suite.sh "${OUT_DIR}"

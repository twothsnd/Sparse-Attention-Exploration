#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-sparse-fair-nsa}"
OUT_DIR="${2:-results/local_small_fair_nsa}"

while tmux has-session -t "${SESSION}" 2>/dev/null; do
  date '+%F %T waiting for fair nsa training session...'
  sleep 300
done

echo "training session ${SESSION} ended; starting fair nsa evaluation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" scripts/eval_fair_nsa_suite.sh "${OUT_DIR}"

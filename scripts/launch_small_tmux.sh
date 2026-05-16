#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-sparse-small}"
OUT_DIR="${2:-results/small}"
CONFIG_PREFIX="${CONFIG_PREFIX:-small}"

tmux new-session -d -s "${SESSION}" \
  "cd $(pwd) && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} CONFIG_PREFIX=${CONFIG_PREFIX} scripts/run_small_suite.sh ${OUT_DIR}"

echo "launched tmux session: ${SESSION}"
echo "attach with: tmux attach -t ${SESSION}"
echo "logs: ${OUT_DIR}/logs"

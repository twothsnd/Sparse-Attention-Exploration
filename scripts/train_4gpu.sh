#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/small_nsa.yaml}"
OUT_DIR="${2:-results}"

torchrun --standalone --nproc_per_node=4 -m sparse_lm.train \
  --config "${CONFIG}" \
  --out_dir "${OUT_DIR}"


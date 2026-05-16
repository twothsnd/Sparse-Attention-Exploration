#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CONFIG_PREFIX="${CONFIG_PREFIX:-small}"

OUT_DIR="${1:-results/small}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "configs/${CONFIG_PREFIX}_full.yaml"
  "configs/${CONFIG_PREFIX}_sliding.yaml"
  "configs/${CONFIG_PREFIX}_bigbird.yaml"
  "configs/${CONFIG_PREFIX}_nsa.yaml"
  "configs/${CONFIG_PREFIX}_nsa_gated.yaml"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  echo "== train ${name} =="
  torchrun --standalone --nproc_per_node=4 -m sparse_lm.train \
    --config "${config}" \
    --out_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${name}.train.log"
done

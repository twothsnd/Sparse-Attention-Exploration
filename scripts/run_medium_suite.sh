#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#CUDA_DEVICE_LIST[@]}}"

OUT_DIR="${1:-results/medium}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  configs/medium_full.yaml
  configs/medium_bigbird.yaml
  configs/medium_nsa.yaml
  configs/medium_nsa_gated.yaml
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  echo "== train ${name} =="
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m sparse_lm.train \
    --config "${config}" \
    --out_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${name}.train.log"
done

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUT_DIR="${1:-results/local_small_nsa}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  configs/local_small_nsa.yaml
  configs/local_small_nsa_gated.yaml
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  echo "== train ${name} =="
  torchrun --standalone --nproc_per_node=8 -m sparse_lm.train \
    --config "${config}" \
    --out_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${name}.train.log"
done

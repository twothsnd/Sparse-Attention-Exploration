#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CONFIG_PREFIX="${CONFIG_PREFIX:-small}"

OUT_DIR="${1:-results/small}"
mkdir -p "${OUT_DIR}"
rm -f results/eval.csv results/retrieval.csv results/benchmark.csv results/budget.csv

CONFIGS=(
  "configs/${CONFIG_PREFIX}_full.yaml"
  "configs/${CONFIG_PREFIX}_sliding.yaml"
  "configs/${CONFIG_PREFIX}_bigbird.yaml"
  "configs/${CONFIG_PREFIX}_nsa.yaml"
  "configs/${CONFIG_PREFIX}_nsa_gated.yaml"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  ckpt="${OUT_DIR}/${name}.pt"
  echo "== eval ${name} =="
  python -m sparse_lm.evaluate \
    --config "${config}" \
    --ckpt "${ckpt}" \
    --steps 100 \
    --csv results/eval.csv

  echo "== retrieval ${name} =="
  python -m sparse_lm.retrieval \
    --config "${config}" \
    --ckpt "${ckpt}" \
    --seq_len 2048 \
    --num_pairs 64 \
    --eval_steps 100 \
    --csv results/retrieval.csv

  echo "== benchmark ${name} =="
  python -m sparse_lm.benchmark \
    --config "${config}" \
    --seq_lens 512 1024 2048 \
    --batch_size 1 \
    --steps 20 \
    --csv results/benchmark.csv
done

python -m sparse_lm.budget \
  --configs "${CONFIGS[@]}" \
  --seq_lens 512 1024 2048 \
  --csv results/budget.csv

python -m sparse_lm.summarize \
  --results_dir "${OUT_DIR}" \
  --benchmark_csv results/benchmark.csv \
  --eval_csv results/eval.csv \
  --retrieval_csv results/retrieval.csv \
  --budget_csv results/budget.csv \
  --out results/summary.md

echo "summary: results/summary.md"

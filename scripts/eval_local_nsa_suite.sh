#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUT_DIR="${1:-results/local_small_nsa}"
mkdir -p "${OUT_DIR}"

CONFIGS=(
  configs/local_small_nsa.yaml
  configs/local_small_nsa_gated.yaml
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  ckpt="${OUT_DIR}/${name}.pt"
  echo "== eval ${name} =="
  python -m sparse_lm.evaluate \
    --config "${config}" \
    --ckpt "${ckpt}" \
    --steps 100 \
    --csv results/local_nsa_eval.csv

  echo "== retrieval ${name} =="
  python -m sparse_lm.retrieval \
    --config "${config}" \
    --ckpt "${ckpt}" \
    --seq_len 2048 \
    --num_pairs 64 \
    --eval_steps 100 \
    --csv results/local_nsa_retrieval.csv

  echo "== benchmark ${name} =="
  python -m sparse_lm.benchmark \
    --config "${config}" \
    --seq_lens 512 1024 2048 \
    --batch_size 1 \
    --steps 20 \
    --csv results/local_nsa_benchmark.csv
done

python -m sparse_lm.budget \
  --configs "${CONFIGS[@]}" \
  --seq_lens 512 1024 2048 \
  --csv results/local_nsa_budget.csv

python -m sparse_lm.summarize \
  --results_dir "${OUT_DIR}" \
  --benchmark_csv results/local_nsa_benchmark.csv \
  --eval_csv results/local_nsa_eval.csv \
  --retrieval_csv results/local_nsa_retrieval.csv \
  --budget_csv results/local_nsa_budget.csv \
  --out results/local_nsa_summary.md

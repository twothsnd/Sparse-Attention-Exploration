#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/smoke
rm -f results/smoke/*.csv results/smoke/*.jsonl results/smoke/*.pt results/smoke/summary.md

python -m pytest tests
python -m sparse_lm.train --config configs/tiny_full.yaml --synthetic --max_steps 5 --out_dir results/smoke
python -m sparse_lm.train --config configs/tiny_nsa.yaml --synthetic --max_steps 5 --out_dir results/smoke
python -m sparse_lm.benchmark --config configs/tiny_nsa.yaml --seq_lens 64 128 --batch_size 1 --steps 2 --csv results/smoke/benchmark.csv
python -m sparse_lm.retrieval --config configs/tiny_sliding.yaml --seq_len 64 --batch_size 4 --num_pairs 4 --eval_steps 2 --csv results/smoke/retrieval.csv
python -m sparse_lm.budget --configs configs/tiny_full.yaml configs/tiny_sliding.yaml configs/tiny_nsa.yaml --seq_lens 128 --csv results/smoke/budget.csv
python -m sparse_lm.summarize --results_dir results/smoke --benchmark_csv results/smoke/benchmark.csv --retrieval_csv results/smoke/retrieval.csv --budget_csv results/smoke/budget.csv --out results/smoke/summary.md

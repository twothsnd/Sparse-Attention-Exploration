# Experiment Plan

## Main Comparison

Formal runs must use aligned token budgets. For each scale, keep `batch_size`,
`seq_len`, `max_steps`, GPU count, data source, and tokenizer aligned across
attention types unless an OOM forces a documented change.

Run the local small baseline rerun first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_local_small_baselines.sh results/local_small_rerun_baselines
```

If the small runs are stable, run the medium subset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_medium_suite.sh results/medium
```

## Benchmarks

Run forward-only scaling benchmarks:

```bash
python -m sparse_lm.benchmark --config configs/medium_full.yaml --seq_lens 1024 2048 4096 --batch_size 1 --steps 20
python -m sparse_lm.benchmark --config configs/medium_bigbird.yaml --seq_lens 1024 2048 4096 --batch_size 1 --steps 20
python -m sparse_lm.benchmark --config configs/medium_nsa.yaml --seq_lens 1024 2048 4096 --batch_size 1 --steps 20
python -m sparse_lm.benchmark --config configs/medium_nsa_gated.yaml --seq_lens 1024 2048 4096 --batch_size 1 --steps 20
```

Run theoretical attention-budget analysis:

```bash
python -m sparse_lm.budget \
  --configs configs/medium_full.yaml configs/medium_bigbird.yaml configs/medium_nsa.yaml configs/medium_nsa_gated.yaml \
  --seq_lens 1024 2048 4096 \
  --csv results/budget.csv
```

Run synthetic retrieval probes:

```bash
python -m sparse_lm.retrieval --config configs/small_full.yaml --ckpt results/small/small_full.pt --seq_len 2048 --num_pairs 64 --csv results/retrieval.csv
python -m sparse_lm.retrieval --config configs/small_bigbird.yaml --ckpt results/small/small_bigbird.pt --seq_len 2048 --num_pairs 64 --csv results/retrieval.csv
python -m sparse_lm.retrieval --config configs/small_nsa.yaml --ckpt results/small/small_nsa.pt --seq_len 2048 --num_pairs 64 --csv results/retrieval.csv
python -m sparse_lm.retrieval --config configs/small_nsa_gated.yaml --ckpt results/small/small_nsa_gated.pt --seq_len 2048 --num_pairs 64 --csv results/retrieval.csv
```

## What To Log

- config path
- GPU count and GPU type
- start/end time
- training token budget
- final train loss
- validation loss/perplexity
- benchmark latency, throughput, and peak memory
- any OOM or instability

Generate a Markdown summary after collecting CSV files:

```bash
python -m sparse_lm.summarize \
  --results_dir results/small \
  --benchmark_csv results/benchmark.csv \
  --eval_csv results/eval.csv \
  --retrieval_csv results/retrieval.csv \
  --budget_csv results/budget.csv \
  --out results/summary.md
```

## Report Claims To Keep Precise

- This repo implements NSA principles: local attention, compressed block attention, selected-block attention, and causal historical block selection.
- It does not claim NSA paper-level hardware speedups because it does not use custom Triton/cuDNN kernels.
- Efficiency comparisons should be framed as PyTorch prototype behavior plus theoretical attention budget analysis.

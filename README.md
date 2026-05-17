# Sparse Attention LM

This repository implements a small GPT-style causal language model for comparing sparse attention mechanisms under the same training and evaluation harness.

The attention code is written directly in PyTorch. External packages are used for infrastructure only: tokenization, dataset loading, YAML configs, progress bars, and tests.

## Implemented Attention Types

- `full`: standard causal full self-attention.
- `sliding`: causal sliding-window self-attention.
- `bigbird`: causal block-sparse BigBird-style mask with local, global, and deterministic random blocks.
- `nsa`: Native Sparse Attention-style module with local, compressed-block, and selected-block branches.
- `nsa_gated`: optional ablation that learns a gate over the three NSA branches. This is not a paper baseline; treat it as an exploratory variant.

Important limitation: this is an algorithmic PyTorch prototype. It validates causal masking, branch structure, training behavior, and theoretical attention budgets, but it does not include custom Triton/CUDA kernels and should not be used to claim kernel-level NSA speedups.

## Repository Layout

```text
sparse_lm/
  attention/          attention implementations
  model.py            GPT blocks and model wrapper
  train.py            single-GPU and torchrun/DDP training
  evaluate.py         validation loss and perplexity
  benchmark.py        forward latency, throughput, peak memory
  budget.py           theoretical visible-token budget
  retrieval.py        synthetic long-context retrieval probe
  summarize.py        Markdown summary generator
configs/              tiny/small/medium and local WikiText configs
scripts/              smoke and training-suite helpers
tests/                correctness smoke tests
reports/              experiment plan and technical notes
```

Large local artifacts are intentionally ignored by Git: `data/`, `results/`, checkpoints, logs, caches, and session metadata.

## Setup

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
python -m pytest tests
```

The code runs on CPU for tests and synthetic smoke runs. Real experiments should use CUDA.

## Data

Configs support two data paths:

1. HuggingFace datasets through `datasets.load_dataset`.
2. Local text files, JSONL files, or parquet files through `data.source: local_text`.

For the current local experiments, WikiText-103 parquet files are expected under:

```text
data/wikitext-103/wikitext-103-v1/
  train-00000-of-00002.parquet
  train-00001-of-00002.parquet
  validation-00000-of-00001.parquet
  test-00000-of-00001.parquet
```

The local configs use the GPT-2 tokenizer and read the `text` column.

## Quick Smoke Test

```bash
python -m sparse_lm.train --config configs/tiny_full.yaml --synthetic --max_steps 20 --out_dir results/smoke
python -m sparse_lm.benchmark --config configs/tiny_nsa.yaml --seq_lens 512 1024 --csv results/smoke/benchmark.csv
python -m sparse_lm.budget --configs configs/tiny_full.yaml configs/tiny_sliding.yaml configs/tiny_bigbird.yaml configs/tiny_nsa.yaml --seq_lens 512 1024 --csv results/smoke/budget.csv
```

## Training

Single config:

```bash
torchrun --standalone --nproc_per_node=4 -m sparse_lm.train \
  --config configs/local_small_nsa.yaml \
  --out_dir results/local_small_nsa
```

Run the local small baseline rerun:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_local_small_baselines.sh results/local_small_rerun_baselines
```

Run the medium comparison suite:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_medium_suite.sh results/medium
```

## Evaluation And Summaries

Generate a Markdown summary from collected logs and CSV files:

```bash
python -m sparse_lm.summarize \
  --results_dir results/local_small \
  --benchmark_csv results/benchmark.csv \
  --eval_csv results/eval.csv \
  --retrieval_csv results/retrieval.csv \
  --budget_csv results/budget.csv \
  --out results/summary.md
```

## Core Experimental Claims

This project should report three separate quantities:

- Language-model quality: validation loss and perplexity at a fixed data/token budget.
- Prototype runtime: PyTorch forward/training throughput and memory.
- Attention budget: theoretical average visible tokens per query and ratio against full causal attention.

Keep these separate in the report. A sparse pattern can have a lower theoretical budget while still running slower in this prototype because the implementation uses regular PyTorch tensor operations rather than fused sparse kernels.

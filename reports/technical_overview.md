# Technical Overview

## Goal

The project compares causal sparse attention mechanisms in a controlled GPT-style language-modeling setup. The goal is not to build a production kernel, but to implement the mechanisms correctly enough to study their modeling behavior, theoretical attention budget, and PyTorch prototype performance.

## Model

All runs use the same decoder-only Transformer architecture:

- token embedding plus learned absolute position embedding
- repeated pre-norm Transformer blocks
- one attention module per block
- GELU MLP with 4x hidden expansion
- tied token embedding and language-model head
- cross-entropy next-token prediction objective

The local small configs use:

- vocabulary: GPT-2 tokenizer, `50257`
- sequence length: `2048`
- layers: `6`
- heads: `8`
- embedding width: `512`
- parameters: about `48.8M`
- training data: local WikiText-103 parquet

## Attention Mechanisms

### Full Causal Attention

`full` is the reference baseline. Each query at position `i` attends to all keys `j <= i`. This has the strongest information access and the highest theoretical budget: average visible tokens are roughly `(seq_len + 1) / 2`.

### Sliding Window Attention

`sliding` restricts each query to a causal recent window. For a window size `w`, position `i` attends to keys:

```text
max(0, i - w + 1) <= j <= i
```

This is a strong local baseline for language modeling because most short-range syntax and phrase-level dependencies are local.

### BigBird-Style Causal Attention

`bigbird` uses a block-sparse causal mask with:

- local neighboring blocks
- global prefix blocks
- deterministic pseudo-random historical blocks

The implementation is causal: no query sees future blocks or future tokens. The random pattern is deterministic so repeated runs are comparable.

### NSA-Style Attention

`nsa` implements three causal branches:

1. Local branch: sliding-window attention over recent original tokens.
2. Compressed branch: attention over summaries of completed historical blocks.
3. Selected branch: top-k completed blocks are selected using query-to-block-summary scores, then attention runs over original tokens from those selected blocks.

For query position `i`, a block is eligible only when the block end is `<= i`. This is the main causal correctness condition for the compressed and selected branches. Selected-token attention also applies the token-level causal mask.

The branch outputs are concatenated and projected back to the model width.

### NSA Gated Ablation

`nsa_gated` replaces concatenation with a learned softmax gate over the local, compressed, and selected branches. This is an exploratory ablation implemented for comparison, not a baseline from the NSA paper.

## Metrics

Training logs contain:

- train loss
- tokens processed
- tokens per second
- step and elapsed time

Evaluation reports:

- validation loss
- perplexity

Benchmark reports:

- forward latency
- forward tokens per second
- peak allocated GPU memory

Budget analysis reports:

- average visible tokens per query
- maximum visible tokens
- ratio to full causal attention

## Interpretation Rules

Do not merge theoretical budget and measured latency into one claim. This repository does not implement fused sparse kernels, so dense PyTorch overhead can dominate runtime. The correct framing is:

- budget numbers estimate the mathematical sparsity of the attention pattern
- benchmark numbers measure this prototype
- validation loss measures modeling quality at a specific token budget

## Current Experiment Direction

The small local suite first compares `full`, `sliding`, `bigbird`, `nsa`, and `nsa_gated`. Because the first NSA run used a smaller per-rank batch and saw fewer tokens, the fair NSA rerun uses `local_small_nsa_b4.yaml` so the NSA token budget is comparable to the other local-small runs.

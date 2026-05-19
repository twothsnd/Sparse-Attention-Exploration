# Medium WikiText-103 Results

These medium results use the local WikiText-103 parquet dataset and the same token budget for all runs. The original FineWeb streaming plan was replaced by local WikiText-103 because the remote streaming path repeatedly hit network/download instability during multi-GPU training.

All four models were trained with the medium GPT-style configuration on 4 A100 GPUs. Each run reached step 29,980 and consumed 982,417,408 training tokens. Checkpoints are intentionally not tracked in Git because each `.pt` file is hundreds of megabytes.

## Setup

| item | value |
| --- | --- |
| dataset | WikiText-103 local parquet |
| train files | `data/wikitext-103/wikitext-103-v1/train-*.parquet` |
| validation files | `data/wikitext-103/wikitext-103-v1/validation-*.parquet` |
| tokenizer | GPT-2 |
| model family | GPT-style causal LM |
| train GPUs | 4 A100 GPUs |
| train sequence length | 4096 |
| evaluation sequence length | 4096 |
| token budget per run | 982,417,408 tokens |

## Training

| run | attention | params | step | train loss | train tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| medium_full | full | 126,762,240 | 29,980 | 3.0619 | 68,066 | 982,417,408 |
| medium_bigbird | bigbird | 126,762,240 | 29,980 | 3.0259 | 52,541 | 982,417,408 |
| medium_nsa | nsa | 140,918,016 | 29,980 | 3.1353 | 33,685 | 982,417,408 |
| medium_nsa_gated | nsa_gated | 126,789,924 | 29,980 | 3.0727 | 33,452 | 982,417,408 |

## Validation

Validation uses WikiText-103 validation parquet, sequence length 4096, batch size 2, and 100 evaluation steps.

| run | seq_len | validation loss | perplexity |
| --- | --- | --- | --- |
| medium_full | 4096 | 3.0137 | 20.36 |
| medium_bigbird | 4096 | 3.0611 | 21.35 |
| medium_nsa | 4096 | 3.1010 | 22.22 |
| medium_nsa_gated | 4096 | 3.0633 | 21.40 |

## Retrieval

Synthetic key-value retrieval uses sequence length 4096, 128 key-value pairs, batch size 8, and 100 evaluation steps.

| run | seq_len | pairs | loss | accuracy |
| --- | --- | --- | --- | --- |
| medium_full | 4096 | 128 | 14.9187 | 0.0000 |
| medium_bigbird | 4096 | 128 | 13.4316 | 0.0000 |
| medium_nsa | 4096 | 128 | 16.1220 | 0.0000 |
| medium_nsa_gated | 4096 | 128 | 16.7415 | 0.0000 |

## Forward Benchmark

Batch size is 1. These are PyTorch prototype numbers, not fused sparse-kernel numbers.

| run | seq_len | latency ms | tok/s | peak GB |
| --- | --- | --- | --- | --- |
| medium_full | 1024 | 26.18 | 39,116 | 0.68 |
| medium_full | 2048 | 64.20 | 31,901 | 1.26 |
| medium_full | 4096 | 190.75 | 21,473 | 3.56 |
| medium_bigbird | 1024 | 103.43 | 9,900 | 0.68 |
| medium_bigbird | 2048 | 111.22 | 18,415 | 1.26 |
| medium_bigbird | 4096 | 471.69 | 8,684 | 3.56 |
| medium_nsa | 1024 | 69.92 | 14,644 | 0.75 |
| medium_nsa | 2048 | 115.37 | 17,752 | 1.37 |
| medium_nsa | 4096 | 379.89 | 10,782 | 3.80 |
| medium_nsa_gated | 1024 | 61.16 | 16,742 | 0.70 |
| medium_nsa_gated | 2048 | 112.82 | 18,153 | 1.32 |
| medium_nsa_gated | 4096 | 376.47 | 10,880 | 3.75 |

## Theoretical Attention Budget

Budget ratio is total visible query-key pairs divided by full causal attention at the same sequence length.

| run | attention | seq_len | avg visible | max visible | ratio |
| --- | --- | --- | --- | --- | --- |
| medium_full | full | 4096 | 2048.5 | 4096.0 | 1.0000 |
| medium_bigbird | bigbird | 4096 | 275.5 | 320.0 | 0.1345 |
| medium_nsa | nsa | 4096 | 874.7 | 960.0 | 0.4270 |
| medium_nsa_gated | nsa_gated | 4096 | 874.7 | 960.0 | 0.4270 |

## Notes

- Full attention has the best validation perplexity in this medium WikiText-103 run: 20.36 at sequence length 4096.
- BigBird has the lowest theoretical 4096-token attention budget, 13.45% of full causal attention, but the current dense-mask PyTorch prototype is slower than full attention in the forward benchmark.
- NSA uses 42.70% of full causal attention at 4096 tokens in this configuration. It has more parameters than the other medium runs because the compressed and selected branches add projection parameters.
- `nsa_gated` is an exploratory ablation with learned branch mixing. It is not a baseline from the NSA paper.
- The retrieval probe produced zero accuracy for all four medium runs under this setup, so it should be reported as a stress-test result rather than evidence that one attention pattern solved retrieval.

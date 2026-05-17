# Local Small Results

These WikiText-103 local-small results use the aligned 8-GPU local configurations. All five runs use the same 20,000-step schedule and the same observed token budget, about 1.309B tokens. `nsa_gated` is an ablation and should not be treated as a baseline from the NSA paper.

## Training

| run | attention | params | step | train loss | train tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| local_small_full | full | 45,683,200 | 19,980 | 4.1634 | 224,396 | 1,309,474,816 |
| local_small_sliding | sliding | 45,683,200 | 19,980 | 4.1303 | 355,563 | 1,309,474,816 |
| local_small_bigbird | bigbird | 45,683,200 | 19,980 | 4.2100 | 323,587 | 1,309,474,816 |
| local_small_nsa | nsa | 48,828,928 | 19,980 | 4.2763 | 337,170 | 1,309,474,816 |
| local_small_nsa_gated | nsa_gated | 45,692,434 | 19,980 | 4.1475 | 332,435 | 1,309,474,816 |

## Validation

| run | seq_len | validation loss | perplexity |
| --- | --- | --- | --- |
| local_small_full | 2048 | 3.9211 | 50.46 |
| local_small_sliding | 2048 | 3.9047 | 49.63 |
| local_small_bigbird | 2048 | 3.9605 | 52.48 |
| local_small_nsa | 2048 | 4.0493 | 57.36 |
| local_small_nsa_gated | 2048 | 3.8995 | 49.38 |

## Retrieval

Synthetic key-value retrieval uses sequence length 2048, 64 key-value pairs, batch size 32, and 100 evaluation steps.

| run | seq_len | pairs | loss | accuracy |
| --- | --- | --- | --- | --- |
| local_small_full | 2048 | 64 | 12.0278 | 0.0000 |
| local_small_sliding | 2048 | 64 | 11.2726 | 0.0000 |
| local_small_bigbird | 2048 | 64 | 11.0591 | 0.0025 |
| local_small_nsa | 2048 | 64 | 10.9022 | 0.0094 |
| local_small_nsa_gated | 2048 | 64 | 12.5688 | 0.0000 |

## Forward Benchmark

Batch size is 1. Numbers are from the PyTorch prototype, not custom sparse kernels.

| run | seq_len | latency ms | tok/s | peak GB |
| --- | --- | --- | --- | --- |
| local_small_full | 512 | 5.81 | 88,065 | 0.28 |
| local_small_full | 1024 | 10.34 | 99,060 | 0.37 |
| local_small_full | 2048 | 23.93 | 85,569 | 0.70 |
| local_small_sliding | 512 | 5.99 | 85,474 | 0.28 |
| local_small_sliding | 1024 | 9.94 | 102,988 | 0.37 |
| local_small_sliding | 2048 | 24.25 | 84,460 | 0.70 |
| local_small_bigbird | 512 | 14.10 | 36,323 | 0.28 |
| local_small_bigbird | 1024 | 24.52 | 41,756 | 0.37 |
| local_small_bigbird | 2048 | 45.70 | 44,809 | 0.70 |
| local_small_nsa | 512 | 11.60 | 44,152 | 0.29 |
| local_small_nsa | 1024 | 15.76 | 64,961 | 0.39 |
| local_small_nsa | 2048 | 42.94 | 47,696 | 0.75 |
| local_small_nsa_gated | 512 | 17.16 | 29,843 | 0.28 |
| local_small_nsa_gated | 1024 | 17.36 | 58,980 | 0.37 |
| local_small_nsa_gated | 2048 | 42.55 | 48,128 | 0.74 |

## Theoretical Attention Budget

Budget ratio is total visible query-key pairs divided by full causal attention at the same sequence length.

| run | attention | seq_len | avg visible | max visible | ratio |
| --- | --- | --- | --- | --- | --- |
| local_small_full | full | 2048 | 1024.5 | 2048.0 | 1.0000 |
| local_small_sliding | sliding | 2048 | 448.1 | 512.0 | 0.4374 |
| local_small_bigbird | bigbird | 2048 | 266.5 | 320.0 | 0.2601 |
| local_small_nsa | nsa | 2048 | 699.8 | 800.0 | 0.6830 |
| local_small_nsa_gated | nsa_gated | 2048 | 699.8 | 800.0 | 0.6830 |

## Notes

- Sliding window is the best non-ablation model on validation loss in this local-small setting.
- BigBird has the lowest 2048-token theoretical attention budget, but the dense-mask PyTorch prototype is slower than full and sliding attention.
- NSA has a lower 2048-token theoretical attention budget than full attention and the best synthetic retrieval score among the non-ablation sparse variants, but this implementation does not claim kernel-level speedup.

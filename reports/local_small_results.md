# Local Small Results

These WikiText-103 local-small results use the current aligned local configurations. `nsa_gated` is an ablation and should not be treated as a baseline from the NSA paper.

## Training

| run | attention | params | step | train loss | train tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| local_small_full | full | 45,683,200 | 19,980 | 4.1487 | 153,646 | 654,737,408 |
| local_small_sliding | sliding | 45,683,200 | 19,980 | 4.1540 | 153,847 | 654,737,408 |
| local_small_bigbird | bigbird | 45,683,200 | 19,980 | 4.2220 | 133,556 | 654,737,408 |
| local_small_nsa | nsa | 48,828,928 | 19,980 | 4.2763 | 337,170 | 1,309,474,816 |
| local_small_nsa_gated | nsa_gated | 45,692,434 | 19,980 | 4.1475 | 332,435 | 1,309,474,816 |

## Validation

| run | seq_len | validation loss | perplexity |
| --- | --- | --- | --- |
| local_small_full | 2048 | 3.8960 | 49.20 |
| local_small_sliding | 2048 | 3.9104 | 49.92 |
| local_small_bigbird | 2048 | 3.9659 | 52.77 |
| local_small_nsa | 2048 | 4.0493 | 57.36 |
| local_small_nsa_gated | 2048 | 3.8995 | 49.38 |

## Retrieval

| run | seq_len | pairs | loss | accuracy |
| --- | --- | --- | --- | --- |
| local_small_nsa | 2048 | 64 | 10.9803 | 0.0088 |
| local_small_nsa_gated | 2048 | 64 | 12.5058 | 0.0000 |

## Forward Benchmark

Batch size is 1. Numbers are from the PyTorch prototype, not custom sparse kernels.

| run | seq_len | latency ms | tok/s | peak GB |
| --- | --- | --- | --- | --- |
| local_small_full | 512 | 6.70 | 76,411 | 0.28 |
| local_small_full | 1024 | 20.17 | 50,761 | 0.37 |
| local_small_full | 2048 | 40.97 | 49,990 | 0.70 |
| local_small_sliding | 512 | 6.24 | 82,000 | 0.28 |
| local_small_sliding | 1024 | 18.61 | 55,030 | 0.37 |
| local_small_sliding | 2048 | 40.04 | 51,151 | 0.70 |
| local_small_bigbird | 512 | 16.85 | 30,379 | 0.28 |
| local_small_bigbird | 1024 | 34.18 | 29,957 | 0.37 |
| local_small_bigbird | 2048 | 72.90 | 28,092 | 0.70 |
| local_small_nsa | 512 | 9.12 | 56,159 | 0.29 |
| local_small_nsa | 1024 | 15.68 | 65,287 | 0.39 |
| local_small_nsa | 2048 | 42.89 | 47,748 | 0.75 |
| local_small_nsa_gated | 512 | 9.14 | 56,019 | 0.28 |
| local_small_nsa_gated | 1024 | 15.58 | 65,713 | 0.37 |
| local_small_nsa_gated | 2048 | 42.52 | 48,167 | 0.74 |

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

- Sliding window is very competitive in this small WikiText setting.
- BigBird has the lowest 2048-token theoretical budget, but its dense-mask PyTorch prototype is slower than full attention.
- NSA has a lower theoretical attention budget than full attention at sequence length 2048, but this PyTorch implementation does not claim kernel-level speedup.

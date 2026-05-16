# Local Small Preliminary Results

These are preliminary WikiText-103 local-small results collected before the fair NSA rerun completed. They are useful for tracking progress, but the first NSA and NSA-gated runs used a smaller per-rank batch size and therefore saw roughly half as many tokens as the full, sliding, and BigBird runs.

## Training

| run | attention | params | step | train loss | train tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| local_small_full | full | 45,683,200 | 19,980 | 4.1487 | 153,646 | 654,737,408 |
| local_small_sliding | sliding | 45,683,200 | 19,980 | 4.1540 | 153,847 | 654,737,408 |
| local_small_bigbird | bigbird | 45,683,200 | 19,980 | 4.2220 | 133,556 | 654,737,408 |
| local_small_nsa | nsa | 48,828,928 | 19,980 | 4.3151 | 76,786 | 327,368,704 |
| local_small_nsa_gated | nsa_gated | 45,692,434 | 19,980 | 4.3514 | 78,911 | 327,368,704 |

## Validation

| run | seq_len | validation loss | perplexity |
| --- | --- | --- | --- |
| local_small_full | 2048 | 3.8960 | 49.20 |
| local_small_sliding | 2048 | 3.9104 | 49.92 |
| local_small_bigbird | 2048 | 3.9659 | 52.77 |
| local_small_nsa | 2048 | 4.6309 | 102.61 |
| local_small_nsa_gated | 2048 | 4.5660 | 96.16 |

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
| local_small_nsa | 512 | 13.08 | 39,136 | 0.29 |
| local_small_nsa | 1024 | 25.01 | 40,946 | 0.39 |
| local_small_nsa | 2048 | 71.07 | 28,818 | 0.75 |
| local_small_nsa_gated | 512 | 17.84 | 28,693 | 0.28 |
| local_small_nsa_gated | 1024 | 24.68 | 41,493 | 0.37 |
| local_small_nsa_gated | 2048 | 74.28 | 27,572 | 0.74 |

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
- NSA needs the fair `local_small_nsa_b4` rerun before final quality claims, because the preliminary NSA run saw half the token budget.

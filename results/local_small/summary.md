# Experiment Summary

## Training

| run | attn | params | step | loss | tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| local_small_bigbird | bigbird | 45683200 | 19980 | 4.2100 | 323587 | 1309474816 |
| local_small_full | full | 45683200 | 19980 | 4.1634 | 224396 | 1309474816 |
| local_small_nsa | nsa | 48828928 | 19980 | 4.2763 | 337170 | 1309474816 |
| local_small_nsa_gated | nsa_gated | 45692434 | 19980 | 4.1475 | 332435 | 1309474816 |
| local_small_sliding | sliding | 45683200 | 19980 | 4.1303 | 355563 | 1309474816 |


## Benchmark

| run | attn | seq_len | latency_ms | tok/s | peak_gb |
| --- | --- | --- | --- | --- | --- |
| local_small_full | full | 512 | 5.81 | 88065 | 0.28 |
| local_small_full | full | 1024 | 10.34 | 99060 | 0.37 |
| local_small_full | full | 2048 | 23.93 | 85569 | 0.70 |
| local_small_sliding | sliding | 512 | 5.99 | 85474 | 0.28 |
| local_small_sliding | sliding | 1024 | 9.94 | 102988 | 0.37 |
| local_small_sliding | sliding | 2048 | 24.25 | 84460 | 0.70 |
| local_small_bigbird | bigbird | 512 | 14.10 | 36323 | 0.28 |
| local_small_bigbird | bigbird | 1024 | 24.52 | 41756 | 0.37 |
| local_small_bigbird | bigbird | 2048 | 45.70 | 44809 | 0.70 |
| local_small_nsa | nsa | 512 | 11.60 | 44152 | 0.29 |
| local_small_nsa | nsa | 1024 | 15.76 | 64961 | 0.39 |
| local_small_nsa | nsa | 2048 | 42.94 | 47696 | 0.75 |
| local_small_nsa_gated | nsa_gated | 512 | 17.16 | 29843 | 0.28 |
| local_small_nsa_gated | nsa_gated | 1024 | 17.36 | 58980 | 0.37 |
| local_small_nsa_gated | nsa_gated | 2048 | 42.55 | 48128 | 0.74 |


## Validation

| run | seq_len | loss | ppl |
| --- | --- | --- | --- |
| local_small_full | 2048 | 3.9211 | 50.46 |
| local_small_sliding | 2048 | 3.9047 | 49.63 |
| local_small_bigbird | 2048 | 3.9605 | 52.48 |
| local_small_nsa | 2048 | 4.0493 | 57.36 |
| local_small_nsa_gated | 2048 | 3.8995 | 49.38 |


## Retrieval

| run | seq_len | pairs | loss | accuracy |
| --- | --- | --- | --- | --- |
| local_small_full | 2048 | 64 | 12.0278 | 0.0000 |
| local_small_sliding | 2048 | 64 | 11.2726 | 0.0000 |
| local_small_bigbird | 2048 | 64 | 11.0591 | 0.0025 |
| local_small_nsa | 2048 | 64 | 10.9022 | 0.0094 |
| local_small_nsa_gated | 2048 | 64 | 12.5688 | 0.0000 |


## Attention Budget

| run | attn | seq_len | avg_visible | max_visible | ratio |
| --- | --- | --- | --- | --- | --- |
| local_small_full | full | 512 | 256.5 | 512.0 | 1.0000 |
| local_small_full | full | 1024 | 512.5 | 1024.0 | 1.0000 |
| local_small_full | full | 2048 | 1024.5 | 2048.0 | 1.0000 |
| local_small_sliding | sliding | 512 | 256.5 | 512.0 | 1.0000 |
| local_small_sliding | sliding | 1024 | 384.2 | 512.0 | 0.7498 |
| local_small_sliding | sliding | 2048 | 448.1 | 512.0 | 0.4374 |
| local_small_bigbird | bigbird | 512 | 200.5 | 320.0 | 0.7817 |
| local_small_bigbird | bigbird | 1024 | 244.5 | 320.0 | 0.4771 |
| local_small_bigbird | bigbird | 2048 | 266.5 | 320.0 | 0.2601 |
| local_small_nsa | nsa | 512 | 436.5 | 776.0 | 1.7018 |
| local_small_nsa | nsa | 1024 | 608.0 | 784.0 | 1.1864 |
| local_small_nsa | nsa | 2048 | 699.8 | 800.0 | 0.6830 |
| local_small_nsa_gated | nsa_gated | 512 | 436.5 | 776.0 | 1.7018 |
| local_small_nsa_gated | nsa_gated | 1024 | 608.0 | 784.0 | 1.1864 |
| local_small_nsa_gated | nsa_gated | 2048 | 699.8 | 800.0 | 0.6830 |


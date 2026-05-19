# Experiment Summary

## Training

| run | attn | params | step | loss | tok/s | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| medium_bigbird | bigbird | 126762240 | 29980 | 3.0259 | 52541 | 982417408 |
| medium_full | full | 126762240 | 29980 | 3.0619 | 68066 | 982417408 |
| medium_nsa | nsa | 140918016 | 29980 | 3.1353 | 33685 | 982417408 |
| medium_nsa_gated | nsa_gated | 126789924 | 29980 | 3.0727 | 33452 | 982417408 |


## Benchmark

| run | attn | seq_len | latency_ms | tok/s | peak_gb |
| --- | --- | --- | --- | --- | --- |
| medium_full | full | 1024 | 26.18 | 39116 | 0.68 |
| medium_full | full | 2048 | 64.20 | 31901 | 1.26 |
| medium_full | full | 4096 | 190.75 | 21473 | 3.56 |
| medium_bigbird | bigbird | 1024 | 103.43 | 9900 | 0.68 |
| medium_bigbird | bigbird | 2048 | 111.22 | 18415 | 1.26 |
| medium_bigbird | bigbird | 4096 | 471.69 | 8684 | 3.56 |
| medium_nsa | nsa | 1024 | 69.92 | 14644 | 0.75 |
| medium_nsa | nsa | 2048 | 115.37 | 17752 | 1.37 |
| medium_nsa | nsa | 4096 | 379.89 | 10782 | 3.80 |
| medium_nsa_gated | nsa_gated | 1024 | 61.16 | 16742 | 0.70 |
| medium_nsa_gated | nsa_gated | 2048 | 112.82 | 18153 | 1.32 |
| medium_nsa_gated | nsa_gated | 4096 | 376.47 | 10880 | 3.75 |


## Validation

| run | seq_len | loss | ppl |
| --- | --- | --- | --- |
| medium_full | 4096 | 3.0137 | 20.36 |
| medium_bigbird | 4096 | 3.0611 | 21.35 |
| medium_nsa | 4096 | 3.1010 | 22.22 |
| medium_nsa_gated | 4096 | 3.0633 | 21.40 |


## Retrieval

| run | seq_len | pairs | loss | accuracy |
| --- | --- | --- | --- | --- |
| medium_full | 4096 | 128 | 14.9187 | 0.0000 |
| medium_bigbird | 4096 | 128 | 13.4316 | 0.0000 |
| medium_nsa | 4096 | 128 | 16.1220 | 0.0000 |
| medium_nsa_gated | 4096 | 128 | 16.7415 | 0.0000 |


## Attention Budget

| run | attn | seq_len | avg_visible | max_visible | ratio |
| --- | --- | --- | --- | --- | --- |
| medium_full | full | 1024 | 512.5 | 1024.0 | 1.0000 |
| medium_full | full | 2048 | 1024.5 | 2048.0 | 1.0000 |
| medium_full | full | 4096 | 2048.5 | 4096.0 | 1.0000 |
| medium_bigbird | bigbird | 1024 | 244.5 | 320.0 | 0.4771 |
| medium_bigbird | bigbird | 2048 | 266.5 | 320.0 | 0.2601 |
| medium_bigbird | bigbird | 4096 | 275.5 | 320.0 | 0.1345 |
| medium_nsa | nsa | 1024 | 692.1 | 912.0 | 1.3505 |
| medium_nsa | nsa | 2048 | 805.8 | 928.0 | 0.7866 |
| medium_nsa | nsa | 4096 | 874.7 | 960.0 | 0.4270 |
| medium_nsa_gated | nsa_gated | 1024 | 692.1 | 912.0 | 1.3505 |
| medium_nsa_gated | nsa_gated | 2048 | 805.8 | 928.0 | 0.7866 |
| medium_nsa_gated | nsa_gated | 4096 | 874.7 | 960.0 | 0.4270 |


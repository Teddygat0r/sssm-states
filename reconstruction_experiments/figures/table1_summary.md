| Model | State (D_k x D_v) | Eff.rank@99% (med) | Eff.rank@99% (p99) | Control eff.rank (med) | cos@k16 (mean) | relFro@k16 (med) | relFro@k16 (p99) | Mem.reduction@k16 |
|---|---|---|---|---|---|---|---|---|
| Mamba2-1.3B | 64x128 | 5 | 33 | 60 | 0.9982 | 1.40e-04 | 5.84e-02 | 62.5% |
| Nemotron-3-Nano-4B | 80x128 | 2 | 13 | 72 | 0.9997 | 5.83e-05 | 6.57e-03 | 67.5% |
| Qwen3.5-4B | 128x128 | 9 | 48 | 99 | 0.9945 | 2.21e-03 | 1.13e-01 | 75.0% |
| DeltaNet-1.3B | 128x128 | 24 | 77 | 99 | 0.9811 | 1.87e-02 | 2.28e-01 | 75.0% |
| GatedDeltaNet-1.3B | 256x256 | 10 | 146 | 198 | 0.9619 | 6.47e-04 | 4.57e-01 | 87.5% |

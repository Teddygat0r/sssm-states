# Single-step compression results

Same 100 chat and 100 code prompts, prefixes 256/full, 128 greedy tokens, probes every four steps. Probes are discarded; attention and convolution states are unchanged. Per-head INT8 uses one affine scale and zero-point per matrix, per head, per layer. No SVD is run in per-head-only mode. Batch size is two in that mode; numerical differences from the previous eight-row probe batch may affect reference trajectories.

| Model | Prefix | Method | N | KL mean | Median | p90 | p99 | Max | Top-1 agreement | Mean squared relative Frobenius error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeltaNet-1.3B | -1 | int8_head_affine | 6400 | 0.026073838 | 0.0037197014 | 0.056526642 | 0.40185022 | 2.814496 | 0.96515625 | 0.0063523019 |
| DeltaNet-1.3B | 256 | int8_head_affine | 6400 | 0.10413489 | 0.01534244 | 0.2636022 | 1.3130625 | 4.2492623 | 0.94 | 0.0070374135 |
| GatedDeltaNet-1.3B | -1 | int8_head_affine | 6400 | 0.0010235897 | 0.00021545377 | 0.0023757108 | 0.0097778849 | 0.085948765 | 0.99453125 | 0.0064451453 |
| GatedDeltaNet-1.3B | 256 | int8_head_affine | 6400 | 0.00051077955 | 0.00014625618 | 0.0014101584 | 0.0041809119 | 0.036639616 | 0.99453125 | 0.006754254 |
| Mamba2-1.3B | -1 | int8_head_affine | 6400 | 0.15777495 | 0.037719768 | 0.4000147 | 1.6982783 | 14.395677 | 0.91140625 | 0.0015280013 |
| Mamba2-1.3B | 256 | int8_head_affine | 6400 | 0.0022704455 | 0.00040909811 | 0.0054388493 | 0.022994798 | 0.69064516 | 0.98953125 | 0.001401357 |
| Nemotron-3-Nano-4B | -1 | int8_head_affine | 6400 | 0.43271258 | 0.029216083 | 1.0585525 | 6.5927882 | 15.629841 | 0.8684375 | 0.0016405262 |
| Nemotron-3-Nano-4B | 256 | int8_head_affine | 6400 | 0.41915584 | 0.021602821 | 0.96714562 | 7.3239026 | 17.908401 | 0.855 | 0.0017032631 |
| Qwen3.5-4B | -1 | int8_head_affine | 6400 | 0.015842894 | 0.00039788245 | 0.014477276 | 0.12892258 | 13.13943 | 0.97796875 | 0.0049686975 |
| Qwen3.5-4B | 256 | int8_head_affine | 6400 | 0.017224816 | 0.0010140556 | 0.016649315 | 0.17662324 | 7.9314976 | 0.9765625 | 0.0049920498 |

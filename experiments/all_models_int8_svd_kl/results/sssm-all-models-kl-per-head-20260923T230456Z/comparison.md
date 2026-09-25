# Single-step compression KL

| Model | Prefix | Method | N | Mean KL | p99 KL | Top-1 agreement |
|---|---|---|---:|---:|---:|---:|
| DeltaNet-1.3B | full | int8_head_affine | 6400 | 0.0260738 | 0.40185 | 96.516% |
| DeltaNet-1.3B | 256 | int8_head_affine | 6400 | 0.104135 | 1.31306 | 94.000% |
| GatedDeltaNet-1.3B | full | int8_head_affine | 6400 | 0.00102359 | 0.00977788 | 99.453% |
| GatedDeltaNet-1.3B | 256 | int8_head_affine | 6400 | 0.00051078 | 0.00418091 | 99.453% |
| Mamba2-1.3B | full | int8_head_affine | 6400 | 0.157775 | 1.69828 | 91.141% |
| Mamba2-1.3B | 256 | int8_head_affine | 6400 | 0.00227045 | 0.0229948 | 98.953% |
| Nemotron-3-Nano-4B | full | int8_head_affine | 6400 | 0.432713 | 6.59279 | 86.844% |
| Nemotron-3-Nano-4B | 256 | int8_head_affine | 6400 | 0.419156 | 7.3239 | 85.500% |
| Qwen3.5-4B | full | int8_head_affine | 6400 | 0.0158429 | 0.128923 | 97.797% |
| Qwen3.5-4B | 256 | int8_head_affine | 6400 | 0.0172248 | 0.176623 | 97.656% |

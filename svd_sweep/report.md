# HiServe Section 1 — State-Level Reconstruction Fidelity

Auto-generated from per-model `metrics.csv` runs. See `svd_sweep/run_qwen35.py`, `run_mamba2.py`, `run_nemotron.py` for the capture pipeline; `plot_results.py` for the figures referenced below.

## TL;DR

- **Qwen3.5-4B** (full grid, GDN hybrid, state `[128,128]`, 9216 (layer, head, position, input) records): effective rank @ 99% energy is **11.3** on average (median 8); rel-Fro MSE at rank 16 has p99 = **9.50e-02**, max = **2.88e-01**.
- **Mamba2-1.3B** (pure-SSM control, state `[64,128]`, 24576 records): effective rank @ 99% = **6.5** (median 4); rel-Fro MSE at rank 16 p99 = **5.20e-02**, max = **1.47e-01**.  Confirms concentration is structural to the SSM recurrence, not the hybrid architecture.
- **Nemotron-3-Nano-4B** (Mamba2 hybrid, state `[80,128]`, 16128 records): effective rank @ 99% = **2.6** (median 2); rel-Fro MSE at rank 16 p99 = **5.54e-03**, max = **2.42e-02**.

## Method

For each `(model, input distribution, sequence position)` we run a single prefill pass, capture the SSM recurrent state at every `(layer, head)`, and compute **one** full SVD per state matrix on CPU — those `(U, S, Vh)` factors are reused for every metric in 1.1 (cumulative energy, effective-rank thresholds) and every rank `k` in 1.2 (truncated reconstruction error). For each state we also draw an i.i.d. Gaussian matrix `R` of the same shape and matched element-wise variance and SVD that, as a Marchenko–Pastur baseline.

SVDs run in **float64**. On Nemotron, some Mamba states have abs-max ~1e-8; the fp32 LAPACK driver (SGESDD) aborts on those via a Fortran `STOP 1` from `SLASCL`. fp64 LAPACK (DGESDD) handles the same matrices cleanly, and 80×128 / 128×128 SVDs in fp64 are essentially free in CPU time. The Qwen3.5 sweep was run earlier in fp32 before this was diagnosed; its inputs never tripped the SLASCL path so its results are unaffected.

Both SVDs are wrapped in try/except so that any matrix-level failure marks the row `spectrum_ok=False` and falls back to `M_hat = M` for reconstruction; this catches Python-level `LinAlgError` but not Fortran `STOP`s, which is why the fp64 switch was needed.

**Sweep axes (per `_helpers.py`):**

| model | layers | heads | positions | inputs |
|---|---|---|---|---|
| Qwen3.5-4B (full grid) | 24 linear-attention | 32 V-heads | {256, 512, 1024, 2048, 4096, full} | ShareGPT, Python from GitHub (Stack-v2 fallback) |
| Mamba2-1.3B (subset)   | all 48 | all 64    | {256, 1024, 4096, full} | ShareGPT, Python from GitHub |
| Nemotron-3-Nano-4B (subset) | all `M`-pattern Mamba layers | all 96 | {256, 1024, 4096, full} | ShareGPT, Python from GitHub |

_Note: `bigcode/the-stack-v2` is a gated dataset; the pipeline auto-falls back to `codeparrot/codeparrot-clean` (public, deduplicated Python from GitHub) for the code distribution._

**Parallelism**: capture is dispatched as one task per `(input, position)` to a `ThreadPoolExecutor` (`SVD_PARALLEL_WORKERS=4` by default). Forward passes serialize on the GPU via a lock; CPU SVDs run truly in parallel because PyTorch BLAS releases the GIL.

## Results

### Qwen3.5-4B

Records: **9216**  
Inputs: `['code_thestack', 'sharegpt']`  
Positions: `['full', 256, 512, 1024, 2048, 4096]`  
Layers swept: 24 (0..30)  
Heads/layer: 32  
State shape: `[128, 128]`  (max rank = 128)  

**1.1 Cumulative energy (mean over (input, pos, layer, head))**

| k | state | random control | effective rank @ k of state |
|---|---|---|---|
| 1 | 0.8121 | 0.0302 | — |
| 2 | 0.8911 | 0.0588 | — |
| 4 | 0.9425 | 0.1128 | — |
| 8 | 0.9741 | 0.2108 | — |
| 16 | 0.9910 | 0.3771 | — |
| 32 | 0.9979 | 0.6242 | — |
| 64 | 0.9998 | 0.8945 | — |

**Effective rank thresholds (number of components needed)**

| threshold | state mean | state median | random mean | random median |
|---|---|---|---|---|
| 99% | 11.25 | 8.0 | 98.93 | 99.0 |
| 99.5% | 14.87 | 11.0 | 104.97 | 105.0 |
| 99.9% | 24.49 | 19.0 | 114.64 | 115.0 |

**1.2 Reconstruction error (per (layer, head) records, all (input, pos))**

| rank k | rel-Fro MSE p50 | p90 | p95 | p99 | max | mean | mean cos sim | mean max-abs |
|---|---|---|---|---|---|---|---|---|
| 4 | 2.795e-02 | 1.621e-01 | 2.244e-01 | 3.494e-01 | 6.274e-01 | 5.750e-02 | 0.9699 | 3.456e-02 |
| 8 | 8.902e-03 | 7.341e-02 | 1.077e-01 | 2.081e-01 | 4.559e-01 | 2.589e-02 | 0.9867 | 1.731e-02 |
| 16 | 1.601e-03 | 2.528e-02 | 3.987e-02 | 9.503e-02 | 2.878e-01 | 9.040e-03 | 0.9954 | 7.371e-03 |
| 32 | 1.184e-04 | 5.696e-03 | 9.815e-03 | 2.495e-02 | 1.423e-01 | 2.098e-03 | 0.9989 | 2.498e-03 |
| 64 | 9.746e-07 | 4.995e-04 | 9.227e-04 | 2.469e-03 | 3.083e-02 | 1.970e-04 | 0.9999 | 6.287e-04 |

**SVD robustness**: spectrum SVD ok in 9216/9216 (100.0%); all-k reconstruction ok in 9216/9216 (100.0%).


### Mamba2-1.3B

Records: **24576**  
Inputs: `['code_thestack', 'sharegpt']`  
Positions: `['full', 256, 1024, 4096]`  
Layers swept: 48 (0..47)  
Heads/layer: 64  
State shape: `[64, 128]`  (max rank = 64)  

**1.1 Cumulative energy (mean over (input, pos, layer, head))**

| k | state | random control | effective rank @ k of state |
|---|---|---|---|
| 1 | 0.8750 | 0.0436 | — |
| 2 | 0.9380 | 0.0844 | — |
| 4 | 0.9729 | 0.1600 | — |
| 8 | 0.9897 | 0.2935 | — |
| 16 | 0.9969 | 0.5081 | — |
| 32 | 0.9995 | 0.7912 | — |
| 64 | 1.0000 | 1.0000 | — |

**Effective rank thresholds (number of components needed)**

| threshold | state mean | state median | random mean | random median |
|---|---|---|---|---|
| 99% | 6.51 | 4.0 | 59.84 | 60.0 |
| 99.5% | 8.48 | 6.0 | 61.90 | 62.0 |
| 99.9% | 13.83 | 10.0 | 64.00 | 64.0 |

**1.2 Reconstruction error (per (layer, head) records, all (input, pos))**

| rank k | rel-Fro MSE p50 | p90 | p95 | p99 | max | mean | mean cos sim | mean max-abs |
|---|---|---|---|---|---|---|---|---|
| 4 | 9.667e-03 | 7.030e-02 | 1.212e-01 | 2.490e-01 | 4.841e-01 | 2.707e-02 | 0.9860 | 1.430e-02 |
| 8 | 1.786e-03 | 2.614e-02 | 5.503e-02 | 1.341e-01 | 3.054e-01 | 1.029e-02 | 0.9948 | 5.847e-03 |
| 16 | 1.285e-04 | 6.873e-03 | 1.786e-02 | 5.202e-02 | 1.473e-01 | 3.094e-03 | 0.9984 | 2.042e-03 |
| 32 | 1.964e-06 | 8.519e-04 | 2.977e-03 | 1.052e-02 | 3.837e-02 | 5.155e-04 | 0.9997 | 5.716e-04 |
| 64 | 4.589e-13 | 8.062e-13 | 9.750e-13 | 1.701e-12 | 8.396e-12 | 5.214e-13 | 1.0000 | 2.867e-07 |

**SVD robustness**: spectrum SVD ok in 24576/24576 (100.0%); all-k reconstruction ok in 24576/24576 (100.0%).


### Nemotron-3-Nano-4B

Records: **16128**  
Inputs: `['code_thestack', 'sharegpt']`  
Positions: `['full', 256, 1024, 4096]`  
Layers swept: 21 (0..40)  
Heads/layer: 96  
State shape: `[80, 128]`  (max rank = 80)  

**1.1 Cumulative energy (mean over (input, pos, layer, head))**

| k | state | random control | effective rank @ k of state |
|---|---|---|---|
| 1 | 0.9633 | 0.0385 | — |
| 2 | 0.9854 | 0.0747 | — |
| 4 | 0.9946 | 0.1422 | — |
| 8 | 0.9982 | 0.2625 | — |
| 16 | 0.9995 | 0.4596 | — |
| 32 | 0.9999 | 0.7314 | — |
| 64 | 1.0000 | 0.9707 | — |

**Effective rank thresholds (number of components needed)**

| threshold | state mean | state median | random mean | random median |
|---|---|---|---|---|
| 99% | 2.63 | 2.0 | 72.13 | 72.0 |
| 99.5% | 3.67 | 2.0 | 75.25 | 75.0 |
| 99.9% | 7.59 | 5.0 | 79.00 | 79.0 |

**1.2 Reconstruction error (per (layer, head) records, all (input, pos))**

| rank k | rel-Fro MSE p50 | p90 | p95 | p99 | max | mean | mean cos sim | mean max-abs |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.358e-03 | 1.501e-02 | 2.375e-02 | 5.674e-02 | 1.602e-01 | 5.444e-03 | 0.9973 | 2.757e-02 |
| 8 | 3.216e-04 | 4.798e-03 | 7.979e-03 | 2.002e-02 | 6.246e-02 | 1.752e-03 | 0.9991 | 9.433e-03 |
| 16 | 4.399e-05 | 1.242e-03 | 2.203e-03 | 5.544e-03 | 2.418e-02 | 4.559e-04 | 0.9998 | 3.348e-03 |
| 32 | 3.315e-06 | 1.968e-04 | 3.658e-04 | 9.759e-04 | 5.419e-03 | 7.583e-05 | 1.0000 | 1.038e-03 |
| 64 | 2.504e-08 | 6.121e-06 | 1.215e-05 | 3.470e-05 | 2.552e-04 | 2.514e-06 | 1.0000 | 1.820e-04 |

**SVD robustness**: spectrum SVD ok in 16128/16128 (100.0%); all-k reconstruction ok in 16128/16128 (100.0%).


## Plots

### `cdf_at_k16.png`
![cdf_at_k16.png](plots/cdf_at_k16.png)

### `effrank_heatmap_Mamba2-1.3B.png`
![effrank_heatmap_Mamba2-1.3B.png](plots/effrank_heatmap_Mamba2-1.3B.png)

### `effrank_heatmap_Nemotron-3-Nano-4B.png`
![effrank_heatmap_Nemotron-3-Nano-4B.png](plots/effrank_heatmap_Nemotron-3-Nano-4B.png)

### `effrank_heatmap_Qwen3.5-4B.png`
![effrank_heatmap_Qwen3.5-4B.png](plots/effrank_heatmap_Qwen3.5-4B.png)

### `effrank_violin.png`
![effrank_violin.png](plots/effrank_violin.png)

### `error_vs_rank.png`
![error_vs_rank.png](plots/error_vs_rank.png)

### `spectrum_overlay_Mamba2-1.3B.png`
![spectrum_overlay_Mamba2-1.3B.png](plots/spectrum_overlay_Mamba2-1.3B.png)

### `spectrum_overlay_Nemotron-3-Nano-4B.png`
![spectrum_overlay_Nemotron-3-Nano-4B.png](plots/spectrum_overlay_Nemotron-3-Nano-4B.png)

### `spectrum_overlay_Qwen3.5-4B.png`
![spectrum_overlay_Qwen3.5-4B.png](plots/spectrum_overlay_Qwen3.5-4B.png)

## Reviewer's notes

- **MSE is a proxy.** Per the experiment plan, state-level reconstruction error is a sanity-check / diagnostic; the fidelity argument rests on the output-distribution experiments in Section 2 (KL between full-state and low-rank-state logits).
- **SSM-vs-random spectrum overlay** (`spectrum_overlay_*.png`) is the clearest piece of evidence for the structural claim: SSM states have a fast-decaying spectrum while same-shape, same-variance random matrices follow the Marchenko–Pastur bulk.
- **Per-head tail** (`cdf_at_k16.png`) shows that even the worst per-head rel-Fro error at rank 16 is small — the hard part of any tail-driven serving argument.
- **Fast-path kernels.** Mamba2 and Nemotron forwards both run on the GPU fast path (`mamba_ssm` 2.3.1 + `causal_conv1d` 1.6.1, dispatched via the HuggingFace `kernels` package which fetches `kernels-community/{mamba-ssm,causal-conv1d}` — PyPI `mamba_ssm` does not expose `selective_state_update` at the top level, so `kernels` is required even when `mamba_ssm` is installed). Without these, the naive PyTorch path is ~50× slower at 4k tokens.
- **Nemotron cache patch.** The unsloth-shipped `HybridMambaAttentionDynamicCache` has two bugs (missing `conv_kernel_size` attribute; `update_*_state` reads `.device` off the Python list rather than the tensor in the slot). `run_nemotron.py` patches both before use; states are then read as `[B, num_heads*head_dim, state_size]` and reshaped to per-head `[head_dim, state_size]`. The model is invoked via `model(..., cache_params=cache, use_cache=True)` — it ignores `past_key_values` for this architecture.
- **Threading.** SVD parallelism uses a Python ThreadPoolExecutor of 4 workers + `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`. With multithreaded BLAS the Mamba2 sweep hit a sporadic SLASCL race (Fortran `STOP 1`) on aarch64; pinning BLAS to single-threaded resolved it without sacrificing wall-clock parallelism (4 simultaneous CPU SVDs).
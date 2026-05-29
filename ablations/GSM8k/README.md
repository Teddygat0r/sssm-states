# Qwen3.5-4B recurrent-state compression on GSM8K

How far can Qwen3.5-4B's linear-attention **recurrent state** be compressed
*after prefill* before its math reasoning degrades? We prefill each GSM8K
prompt, replace every linear-attention layer's recurrent state with a
compressed-then-reconstructed copy, then let the model generate the answer, and
score it with lm-eval's built-in `gsm8k` grader (5-shot, greedy).

The model is a hybrid: 24 linear-attention layers (`Qwen3_5GatedDeltaNet`) whose
post-prefill recurrent state has shape `[B, H=32, D_k=128, D_v=128]` per layer,
plus 8 full-attention layers (left untouched).

> **Note on state location.** Qwen3.5 stores recurrent state *per layer* at
> `cache.layers[li].recurrent_states`, **not** at the top-level
> `cache.recurrent_states` (which is `None`). Compressing the top-level
> attribute is a silent no-op. `HFLM_compress._maybe_compress` handles this
> correctly by iterating `cache.layers`.

## Compression families

| family | description | effective bits/elem |
| --- | --- | --- |
| baseline | uncompressed fp32 state | 32 |
| `q4` / `q8` per-head | uniform affine fake-quant, per-`(layer, head)` scale/zero-point | 4 / 8 |
| SVD rank-r | GPU randomized SVD (`rsvd_eigh.py`), r ∈ {1,2,4,8,12,16,24} | 32·r·(D_k+D_v+1)/(D_k·D_v) |
| Hadamard + q4 | rotate state by 128×128 Walsh-Hadamard on both inner dims, q4, rotate back (QuaRot/SpinQuant-style) | 4 |

## Files

| file | role |
| --- | --- |
| `compression_ops.py` | the four compressors as pure tensor ops; `python compression_ops.py` runs a self-test |
| `hflm_compress.py` | `HFLM_compress` — lm-eval `HFLM` subclass with a pluggable compressor applied between prefill and generation |
| `run_gsm8k_compression.py` | sweep driver (12 configs); streams `gsm8k_results.csv` + per-config dumps to `raw/` |
| `plot_results.py` | builds `gsm8k_summary.csv` and `gsm8k_ablation.{png,pdf}` (accuracy vs effective bits) |
| `analyze_hadamard_failure.py` | explains the Hadamard collapse on a real captured state (see below) |
| `bench_throughput.py` | prefill/decode throughput + compression-overhead diagnostic |

Results that are committed: `gsm8k_results.csv`, `gsm8k_summary.csv`,
`gsm8k_ablation.{png,pdf}`. The per-config sample dumps (`raw/`, ~150 MB) and run
logs are git-ignored; regenerate them by rerunning the sweep.

## Reproduce

```bash
source .venv/bin/activate
python ablations/GSM8k/run_gsm8k_compression.py            # full sweep (1319 prompts × 12 configs)
python ablations/GSM8k/run_gsm8k_compression.py --limit 8  # smoke test
python ablations/GSM8k/plot_results.py                     # figures + summary table
```

## Results (full GSM8K test set, 5-shot, greedy, strict exact-match)

| config | bits/elem | strict | flex |
| --- | --- | --- | --- |
| baseline (fp32) | 32 | 0.850 | 0.848 |
| q8 per-head | 8 | 0.854 | 0.851 |
| q4 per-head | 4 | 0.807 | 0.801 |
| **Hadamard + q4** | 4 | **0.363** | 0.390 |
| SVD r1 | 0.50 | 0.732 | 0.711 |
| SVD r2 | 1.00 | 0.828 | 0.822 |
| SVD r4 | 2.01 | 0.850 | 0.847 |
| SVD r8 | 4.02 | 0.863 | 0.861 |
| SVD r12 | 6.02 | 0.848 | 0.848 |
| SVD r16 | 8.03 | 0.852 | 0.850 |
| SVD r24 | 12.05 | 0.862 | 0.861 |

### Findings

- **8-bit per-head quant is lossless** (0.854 ≈ baseline 0.850).
- **SVD r4 already matches baseline** at ~16× compression, and r8 slightly
  exceeds it — the recurrent state is effectively very low rank (r=1 alone keeps
  0.732). This is the headline positive result.
- **Hadamard + q4 collapses to 0.363** — far below naive q4 (0.807) at the same
  4-bit budget. This is *not* a bug; see below.

### Why Hadamard catastrophically fails (and why it is not a bug)

`analyze_hadamard_failure.py` reproduces the result on a real captured state.
The rotation math is correct (`H@H − I` ≈ 6e-8, exact round-trip) and Hadamard
actually reconstructs the state **better** than naive q4 by every static metric:

| metric (4-bit, real state) | naive q4 | Hadamard q4 |
| --- | --- | --- |
| Frobenius rel-error | 0.355 | **0.257** |
| rel-error on top-0.1% largest entries | 0.150 | **0.093** |
| rel-error in rank-4 signal subspace | 0.240 | **0.046** |

The collapse is a **decode-dynamics** failure, not a reconstruction failure.
The recurrent state is wildly outlier-dominated (absmax/std ≈ 440, excess
kurtosis ≈ 17000) and those few huge entries *are* the low-rank signal the model
reads. Hadamard spreads each outlier's energy across all coordinates; quantizing
in the rotated domain and rotating back injects low-magnitude but **dense,
broadband** noise into directions that are normally clean. Qwen3.5 reacts to
that broadband perturbation by falling into a `<think>`-token repetition loop —
in the full sweep, `hadamard_q4` emitted a mean of **31** (max **384**)
`<think>` blocks per generation and **47.5%** of rollouts hit the 768-token cap
without ever producing `#### N`, so they score 0 on strict-match
(`0.85 × (1 − 0.475) ≈ 0.36`). Naive per-head quant avoids this entirely: it
leaves the outlier in the top quant bin and dumps its error onto the
unimportant near-null entries, keeping decode dynamics stable.

**Takeaway:** for an outlier-dominated, low-rank state, outlier-spreading
rotations (QuaRot/SpinQuant) are counterproductive — the outliers are the
signal. Frobenius/subspace reconstruction error does not predict generative
behavior here. To isolate quantization quality from decode stability, evaluate
teacher-forced perplexity/KL or add a repetition penalty rather than scoring
free generation.

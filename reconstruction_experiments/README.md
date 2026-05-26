# Reconstruction experiments

State-level reconstruction-fidelity sweep across five recurrent / SSM models,
reusing the exact metric + SVD pipeline from `../svd_sweep/_helpers.py`.

## Models

| key | label | HF id | architecture | per-head state |
|---|---|---|---|---|
| `qwen35` | Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Gated-DeltaNet hybrid | `[128, 128]` |
| `mamba2` | Mamba2-1.3B | `AntonV/mamba2-1.3b-hf` | pure Mamba2 | `[64, 128]` |
| `nemotron` | Nemotron-3-Nano-4B | `unsloth/NVIDIA-Nemotron-3-Nano-4B` | Mamba2 hybrid (built-in `nemotron_h`, sdpa) | `[80, 128]` |
| `deltanet` | DeltaNet-1.3B | `fla-hub/delta_net-1.3B-100B` | pure DeltaNet (FLA) | `[128, 128]` |
| `gated_deltanet` | GatedDeltaNet-1.3B | `m-a-p/1.3B-100B-GatedDeltaNet-pure` | pure Gated-DeltaNet (FLA) | `[256, 256]` |

## Metrics (per layer, head, position, prompt)

- `cos_flat_k{1..64}` — flat cosine similarity of the rank-k reconstruction
- `rel_fro_mse_k{1..64}` — relative Frobenius (squared) error
- `max_abs_k*`, `mean_abs_k*` — per-element abs error
- `energy_at_k*` (+ `_random`) — cumulative SVD energy, state vs. matched random control
- `num_rank990 / 995 / 999` (+ `_random`) — effective rank at 99 / 99.5 / 99.9 % energy

Outputs land in `results/recon_<model>_<timestamp>/` as `metrics.csv` (one row
per (layer, head, position, prompt)) and `spectra.pt` (full singular spectra).

## Position sweep

All five models share the same prefill-length grid so the comparison is
apples-to-apples: `{256, 512, 1024, 2048, 4096}` plus a `full` task for every
prompt longer than 4096 tokens — 6 tasks per (sufficiently long) prompt. Each
position is an independent forward pass that prefills the first `T` tokens and
snapshots the recurrent state there. Override the fixed positions with
`SWEEP_POSITIONS="256,1024,4096"` (the `full` task is always added on top).

## Prompts

Two distributions, fetched and chunked by `load_prompts()`:
- **ShareGPT** chat — `Aeala/ShareGPT_Vicuna_unfiltered`
- **code** — `bigcode/the-stack-v2`, auto-falling-back to `codeparrot/codeparrot-clean`

Each chunk concatenates consecutive source docs to ≥ `SWEEP_MIN_CHARS_PER_PROMPT`
chars. `SWEEP_N_PROMPTS` chunks per distribution.

## Run

```bash
# all five models, 200 prompts/distribution
./run_all.sh

# a subset
./run_all.sh mamba2 deltanet

# one model directly
SWEEP_N_PROMPTS=200 ../.venv/bin/python run_reconstruction.py --model nemotron
```

### Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `SWEEP_N_PROMPTS` | 200 (in `run_all.sh`) | prompts per distribution |
| `SWEEP_MIN_CHARS_PER_PROMPT` | 24000 | min chars per chunk |
| `SWEEP_MAX_FULL_TOKENS` | 65536 | cap on the "full" position |
| `SVD_PARALLEL_WORKERS` | 4 | task threads (forward serializes, SVD parallel) |
| `OMP/OPENBLAS/MKL_NUM_THREADS` | 1 | pin BLAS (avoids aarch64 LAPACK SVD race) |

## SVD algorithm (hybrid)

Reconstruction error is measured with the **randomized** `torch.svd_lowrank`,
matching the deployed §2 KL / downstream-eval path:

    q = min(k + RECON_OVERSAMPLE, min(D_k, D_v)),   niter = RECON_NITER
    (defaults: oversample=4, niter=1)

so `cos_flat_k*`, `rel_fro_mse_k*`, `max_abs_k*`, `mean_abs_k*` reflect the
actual algorithm rather than an idealized exact truncation. See
`_lowrank_metrics.py`.

The spectrum-side columns — `energy_at_k*`, `num_rank990/995/999`, and the
**random control** — are still computed with the **exact full SVD**
(`torch.linalg.svdvals`, float64). A low-rank sketch fundamentally can't produce
them: the energy denominator sums over *all* singular values, and the
Marchenko–Pastur control is near-full-rank by design.

| knob | env var | default |
|---|---|---|
| oversample | `RECON_OVERSAMPLE` | 4 |
| power iterations | `RECON_NITER` | 1 |
| reconstruction dtype | `RECON_DTYPE` | `float64` (set `float32` to mirror the deployed path exactly) |

## Notes

- Qwen3.5's fast causal-conv is disabled so captured states are exact.
- The m-a-p GatedDeltaNet checkpoint has an `attn.D` skip param FLA ignores;
  it doesn't enter the recurrent-state computation, so it's fine here.

# KL-under-state-compression generation experiments

How much does compressing a recurrent/SSM model's **entire recurrent state** to
a low rank distort what it generates? For five architectures we prefill a
prompt, greedily generate tokens, and at sampled steps measure the
next-token-distribution divergence

    KL( P[full state]  ||  P[rank-k state] )

against the uncompressed model. Compression is applied to every recurrent layer
at once (the deployed scenario); attention KV and short conv states are left
exact. This complements `../reconstruction_experiments/` (which measures
*state* reconstruction error) by measuring the *downstream generative* cost.

## Models (shared with `reconstruction_experiments`)

| key | label | arch | per-head state |
|---|---|---|---|
| `mamba2` | Mamba2-1.3B | pure Mamba2 | [64,128] |
| `qwen35` | Qwen3.5-4B | Gated-DeltaNet hybrid | [128,128] |
| `nemotron` | Nemotron-3-Nano-4B | Mamba2 hybrid | [80,128] |
| `deltanet` | DeltaNet-1.3B (FLA) | pure DeltaNet | [128,128] |
| `gated_deltanet` | GatedDeltaNet-1.3B (FLA) | pure Gated-DeltaNet | [256,256] |

Loaders + per-architecture state access are imported from
`reconstruction_experiments/run_reconstruction.py`.

## Design

- **Distributions:** ShareGPT chat + The-Stack code, via
  `svd_sweep/_helpers.load_prompts()` (same chunks as the reconstruction sweep).
- **Prefill lengths (`SWEEP_POSITIONS`, default `256`):** plus a `full` task per
  prompt — i.e. a **short (256-token)** and a **full-length (~6–8k-token)**
  context, so we see KL as a function of how much the fixed-size state has
  absorbed.
- **Generation:** `KL_GEN_TOKENS` (128) greedy tokens on the true rollout.
- **Sampling:** KL measured every `KL_EVERY` (4) steps for ranks `KL_RANKS`
  (`4,8,16`).

### How the batched KL readout works (and why it's correct)

At a measured step the true recurrent state `S` (per layer, per head) is taken
from the live cache. All recurrent layers are stacked and compressed in **one**
batched `randomized_svd_eigh` call (the repo-root `rsvd_eigh`: eigh-on-Gram +
mixed-precision Cholesky-QR), sketched at the largest rank and truncated to each
requested rank. This is the decisive optimization: ~25 ms vs ~2 s/step for
per-layer `torch.svd_lowrank`, on-GPU. The cache is then repeated to batch
`N=len(ranks)+1`; row 0 keeps the true state, rows `1..R` get the rank-k
reconstructions (KV/conv left identical). One forward yields next-token logits
for all rows, and we read `KL(row0 || row_r)`. The rollout continues from row 0.

`batch_repeat_interleave` from transformers cannot be used here: the recurrent
`LinearAttentionLayer` doesn't implement it (only the attention `DynamicLayer`
does), so it crashes on the first recurrent layer. `kl_core` instead repeats /
selects / injects by hand over the per-layer tensor fields, and threads the
cache via the model-correct kwarg (`cache_params` for mamba2, `past_key_values`
otherwise — using the wrong one silently drops the cache).

**Validated invariants** (`_selftest.py`, passes on mamba2 + qwen35):
- all-true batch → rows bit-identical (no per-row nondeterminism);
- row 0 is invariant to what compressed rows 1.. contain (no cross-row
  contamination), so the KL reference is exact and the in-batch noise floor is 0;
- reconstruction error is monotone decreasing in rank (per-token KL need not be);
- the rollout continues after selecting row 0.

## Output

`results/kl_<model>_<ts>/`
- `kl_metrics.csv` — one row per `(prompt, position, gen_step, rank)`:
  `model, family, input_id, input_kind, prompt_id, position, prefill_tokens,
  gen_step, context_len, rank, kl, state_rel_fro`.
- `run_config.json` — all knobs + per-task timing.

## Run

```bash
# one model
SWEEP_N_PROMPTS=100 ../.venv/bin/python run_kl_generation.py --model mamba2

# calibration (few prompts, real per-step config)
../.venv/bin/python run_kl_generation.py --model qwen35 --limit-prompts 3

# all five
./run_all.sh
# cheaper
KL_EVERY=8 SWEEP_N_PROMPTS=50 ./run_all.sh
```

### Knobs (env)

| var | default | meaning |
|---|---|---|
| `KL_GEN_TOKENS` | 128 | tokens generated per prefill |
| `KL_EVERY` | 4 | measure KL every N generated tokens |
| `KL_RANKS` | 4,8,16 | ranks compared each measured step |
| `SWEEP_POSITIONS` | 256 | prefill length(s); a `full` task is always added |
| `SWEEP_N_PROMPTS` | 100 | prompts per distribution |
| `KL_STOP_ON_EOS` | 0 | stop a rollout at EOS (default: fixed length) |
| `KL_OVERSAMPLE` / `KL_NITER` | 8 / 2 | `rsvd_eigh` sketch oversample / power iters |

# `randomized_svd_eigh` robustness on Qwen3.5-4B recurrent states

**Date:** 2026-05-11
**Code:** `benchmarks/rsvd_eigh_robustness.py`
**Run dir:** `eval_results/rsvd_robustness_20260511_093633/`

## What was tested

The randomized SVD in `rsvd_eigh.py` exposes three orthonormalization paths
(`chol_min`, `cholqr2`, `house`) and an optional low-precision power-iteration
dtype. The docstring claims `chol_min` is fast but fragile, `cholqr2` is the
robust default, and `house` is the most robust. This experiment stress-tests
all three on real recurrent-state matrices produced by `Qwen/Qwen3.5-4B`
prefill on MMLU prompts.

## Setup

- **Model:** `Qwen/Qwen3.5-4B`, loaded in bfloat16 on CUDA.
- **Architecture:** 32 hidden layers, of which 24 are linear-attention
  (Qwen3.5 GatedDeltaNet) and produce a recurrent state. Each such state
  has shape `[B=1, num_v_heads=32, head_k_dim=128, head_v_dim=128]`.
- **Prompts:** 30 prompts sampled uniformly at random (seed=42) from
  `cais/mmlu` (split `test`), formatted as standard multiple-choice prompts
  and truncated to ≤ 512 tokens. Token counts ranged ~50–500.
- **Batch:** for each forward pass, all 24 linear layers' recurrent states
  were concatenated along the head axis to give a single
  `[24 × 32, 128, 128] = [768, 128, 128]` tensor and passed in **one** SVD
  call.
- **SVD parameters:** `rank=16`, `oversample=8` (sketch dim `q=24`),
  `n_iter=2`. These are the defaults from `rsvd_eigh.py`.
- **Sweep:** every combination of
  `orth ∈ {chol_min, cholqr2, house}` × `power_dtype ∈ {float32, bfloat16}`.
- **Ground truth:** dense per-state truncated SVD via `torch.linalg.svd`
  in fp32, truncated to the top 16 singular values/vectors.

## How "failure" is defined

Three categories, each recorded per-prompt:

1. **Exception raised** by `randomized_svd_eigh`.
2. **NaN / Inf** in any of `U`, `S`, `Vh`.
3. **Excess reconstruction error**: for at least one of the 768 states in
   the batch, the relative Frobenius reconstruction error
   `‖A − U·diag(S)·Vh‖_F / ‖A‖_F`
   is materially worse than the true rank-16 truncated SVD. Two thresholds
   are tracked, the second is a strict-but-noisy informational flag:
   - **absolute-gap >5pp**: `err − err_optimal > 0.05`. This is the
     primary failure threshold — a real loss of reconstruction quality.
   - **relative-gap >50%**: `(err − err_optimal) / max(err_optimal, 1e-3) > 0.5`.
     This fires when the optimum itself is already tiny (a state that is
     effectively ≤ rank-16); any small absolute slack then balloons in
     relative terms. Reported but not treated as a hard failure.

## Headline numbers

Per `(method, power_dtype)`, across **30 prompts × 768 states = 23,040**
state-level SVDs (per row):

| method     | power_dtype | prompts raised | NaN/Inf | states with abs-gap >5pp | states with rel-gap >50% | mean recon err | mean runtime (ms) |
|------------|-------------|---------------:|--------:|-------------------------:|-------------------------:|---------------:|------------------:|
| `chol_min` | float32     |        **30/30** |       0 |                       n/a |                       n/a |              — |       ~1 ms (fail-fast) |
| `chol_min` | bfloat16    |        **30/30** |       0 |                       n/a |                       n/a |              — |       ~1 ms (fail-fast) |
| `cholqr2`  | float32     |         0/30 |       0 |                **0 / 23,040** |                  5 / 23,040 |         0.0487 |             127.0 |
| `cholqr2`  | bfloat16    |         0/30 |       0 |                **0 / 23,040** |                625 / 23,040 |         0.0488 |             114.2 |
| `house`    | float32     |         0/30 |       0 |                **0 / 23,040** |                  4 / 23,040 |         0.0487 |             110.5 |
| `house`    | bfloat16    |         0/30 |       0 |                **0 / 23,040** |                648 / 23,040 |         0.0488 |             109.7 |

The mean recon error (~0.049) is essentially the rank-16 truncation
optimum on these states — see the gap row below.

| method     | power_dtype | mean(gap vs optimum) | worst-case gap (abs) | mean top-1 σ error |
|------------|-------------|---------------------:|---------------------:|-------------------:|
| `cholqr2`  | float32     |             4.18e-05 |              0.0021 |          4.33e-04 |
| `cholqr2`  | bfloat16   |             1.38e-04 |              0.0030 |          4.29e-04 |
| `house`    | float32     |             4.23e-05 |              0.0020 |          4.33e-04 |
| `house`    | bfloat16   |             1.38e-04 |              0.0026 |          4.29e-04 |

## Findings

### 1. `chol_min` is unusable on Qwen3.5-4B states

`chol_min` raised `torch.linalg.LinAlgError` on **every single prompt** in
**both** dtypes (60/60 calls). The exception comes from `torch.linalg.cholesky`
on the sketch's Gram matrix `Y^T Y`, with the leading-minor failure index
distributed across `5..24` (i.e. somewhere in the 24×24 Gram is always
non-PD-by-numerical-floor).

This matches the algorithm's documented failure mode, but the threshold for
triggering it is much looser than the docstring implies. The docstring says
this happens when "rank(A) < q (≈ 23)". In practice, **every** batch we
tested contains at least one state whose Gram is too ill-conditioned for
plain Cholesky, even though **no** state's full SVD shows formal rank
deficiency at `1e-6 · σ_max`. The driver is the singular-value tail: across
prompts, on average ~110 of 768 states (≈14%) have `σ_{q-1}/σ_0 < 1e-4`
and ~240 states (≈31%) have `σ_{q-1}/σ_0 < 1e-3`. That's enough soft
ill-conditioning to break unjittered Cholesky on the sketch.

**Practical implication:** never use `orth="chol_min"` on Qwen3.5-4B
recurrent states — not because of catastrophic states, but because the
*typical* state is conditioned just badly enough for plain Cholesky.

### 2. `cholqr2` and `house` are robust and numerically equivalent here

Neither method ever raised; neither ever produced NaN/Inf. The recon error
matches the rank-16 truncation optimum to within `2×10⁻³` in absolute terms
across **all 23,040 states** in either dtype. The mean across-state gap to
the optimum is `~4×10⁻⁵` in fp32 and `~1.4×10⁻⁴` in bf16. Top-1 singular
value reconstruction error is `~4×10⁻⁴` (independent of dtype).

The relative-gap >50% flag fires for ~625 states in bf16 (~2.7%) but only
~5 in fp32 (~0.02%). All of these have abs-gap < 0.003 (i.e., the optimum
itself was already < 0.006, so the state is effectively rank-≤16 and a tiny
absolute slack inflates the relative gap). These are not real failures and
do not indicate algorithm trouble.

### 3. bf16 power-iteration is fine in practice

Switching the power matmuls to bf16 buys ~10–15% wall-clock and costs
about a 3× larger absolute gap (`~4×10⁻⁵ → ~1.4×10⁻⁴`) and a 3× larger
worst-case absolute gap (`~0.002 → ~0.003`). Neither matters for the
intended downstream use (rank-16 compression of states whose tail spectrum
is in the range we measured), but if you need fp32-level accuracy on the
near-zero residual states, keep `power_dtype=torch.float32`.

### 4. State conditioning (context for interpretability)

Across the 30 prompts (23,040 states), measured from the dense ground-truth SVD:

- `min(σ_{q-1}/σ_0)` per prompt ranged `4.4×10⁻⁶ … 7.2×10⁻⁶` — i.e. at
  least one state in each batch has a sketch-dimension singular value six
  orders of magnitude smaller than its top singular value.
- `median(σ_{q-1}/σ_0)` per prompt ranged `7.9×10⁻³ … 1.4×10⁻²` — most
  states are well-conditioned at the sketch dimension.
- **No state** at the `1e-6` threshold was formally rank-deficient at q,
  but the tail conditioning is what kills `chol_min`.

## Recommendation

- **Default**: use `orth="cholqr2"` with `power_dtype=torch.float32` for
  Qwen3.5-4B states. This is the cheapest setting we tested that has zero
  observed failures.
- **bf16 power**: acceptable for downstream rank-16 compression, slightly
  faster, slightly noisier in the near-zero recon regime.
- **Never `chol_min`**: in this workload it raises 100% of the time, and
  with the standard rank/oversample settings it does not appear
  recoverable by simply increasing oversample (the failure happens at
  arbitrary minor orders across the sketch dim).
- Consider tightening the `chol_min` docstring's failure-mode description:
  in practice the trigger is "any state in the batch with
  `σ_{q-1}/σ_0` below the float floor for Cholesky," which fires roughly
  100% of the time on Qwen3.5-4B even at oversample=8.

## Addendum — `cholqr2` is always taking its fallback (follow-up run)

A follow-up run (`benchmarks/rsvd_eigh_cholqr2_retry.py`) instrumented
`rsvd_eigh._chol_min` to count successful and raising calls, and measured
both random-Gaussian and real-state batches with 3 warmup + 10 timing
iterations per setting.

**Random Gaussian `[768, 128, 128]`** (no rank deficiency, fast path expected):

| method     | dtype | mean (ms) | chol_min ok / raised |
|------------|-------|----------:|---------------------:|
| `chol_min` | fp32  |       9.7 |           39 / 0     |
| `chol_min` | bf16  |       7.7 |           39 / 0     |
| `cholqr2`  | fp32  |      12.1 |           78 / 0     |
| `cholqr2`  | bf16  |      10.0 |           78 / 0     |
| `house`    | fp32  |     108.9 |             —        |
| `house`    | bf16  |     108.4 |             —        |

These match the docstring's `chol_min ~9.4 ms / cholqr2 ~13 ms` claims.

**Real Qwen3.5-4B states**, 5 prompts × `_chol_min` counters:

| prompt              | method    | dtype | mean (ms) | chol_min ok / raised |
|---------------------|-----------|-------|----------:|---------------------:|
| mmlu_64             | `cholqr2` | fp32  |     115.7 |          **0 / 39 (100%)** |
| mmlu_64             | `cholqr2` | bf16  |     113.4 |          **0 / 39 (100%)** |
| (4 other prompts)   | `cholqr2` | fp32  |   114–116 |          **0 / 39 (100%)** |
| all 5 prompts       | `house`   | fp32  |   110–111 |             —        |

So `cholqr2 ≈ house` on real states because **every inner Cholesky raises**,
forcing the Householder-QR fallback. The reported `cholqr2` time in the
main robustness sweep above (~114 ms) is the fallback path's wall-clock,
not the documented fast path's. On random full-rank-at-q matrices the fast
path is reached and runs at the expected ~12 ms.

**Additional finding — fallback can also fail in bf16.** On
`mmlu_9740 (philosophy)`, `cholqr2 bf16` propagated `LinAlgError` from
the fallback path itself (i.e. `torch.linalg.qr` on the bf16 sketch raised
on this batch). `house bf16` succeeded on the same prompt in the same
run. The takeaway: under bf16 power, neither `cholqr2` nor `house` is
100% safe on these states; if you must use bf16, prefer `house` and be
prepared to retry in fp32 on the rare raise. Under fp32 power, `cholqr2`
and `house` are both 100% safe in this sample, with identical recon
error and identical runtime.

Updated recommendation: on Qwen3.5-4B states there is **no practical
benefit** to choosing `cholqr2` over `house`; the inner-Cholesky fast
path never fires, so they execute the same code at the same cost. Use
`house` directly and skip the try/except overhead.

## Artifacts in this directory

- `REPORT.md` — this file.
- `aggregate.csv` / `aggregate.json` — per-`(method, dtype)` summary.
- `per_prompt_results.csv` — full per-prompt per-method metrics (180 rows).
- `failures.csv` — every recorded failure with its concrete cause (129 rows
  — 60 chol_min exceptions + 69 informational relative-gap excesses).

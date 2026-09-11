# `chol_v6` test on Qwen3.5-4B recurrent states

**Date:** 2026-05-11
**Code:** `benchmarks/rsvd_eigh_chol_v6_test.py`
**Run dir:** `eval_results/rsvd_chol_v6_20260511_185431/`

## What was tested

A focused comparison of the new `chol_v6` orthonormalization (three-tier
ladder: scaled Cholesky jitter → eigh SPD repair → Householder fallback)
against `cholqr2` and `house` on the same Qwen3.5-4B recurrent-state
workload as the main robustness report. 10 prompts × 6 method/dtype
combinations × (3 warmup + 10 timed iters) per (prompt, method, dtype).

For `chol_v6` we additionally measure the **orthonormality defect** of the
returned `U`: `‖UᵀU − I‖_F / √k` per state. The docstring warns this can
be ≈ jitter size; we wanted to know how large it actually gets in practice.

## Aggregate results (10 prompts × 768 states/prompt = 7,680 state-level SVDs per row)

| method     | dtype | runs | raised | mean runtime (ms) | mean recon err | mean p99 err | worst gap vs optimum | mean orth defect of U | max orth defect of U | states with abs-gap >5pp |
|------------|-------|-----:|-------:|------------------:|---------------:|-------------:|---------------------:|----------------------:|---------------------:|-------------------------:|
| **`chol_v6`** | fp32 | 10 | **5 / 10** |    **7.81** |        0.0607 |       0.2066 |              +0.0371 |              **0.97** |             **0.97** |                        0 |
| **`chol_v6`** | bf16 | 10 | **2 / 10** |    **6.02** |        0.0632 |       0.2156 |              +0.0371 |              **0.97** |             **0.97** |                        0 |
| `cholqr2`  | fp32 | 10 |   0 / 10 |           113.97 |        0.0507 |       0.2133 |              +0.0014 |              8.13e-07 |             8.48e-07 |                        0 |
| `cholqr2`  | bf16 | 10 |   0 / 10 |           113.41 |        0.0507 |       0.2133 |              +0.0017 |              8.26e-07 |             9.61e-07 |                        0 |
| `house`    | fp32 | 10 |   0 / 10 |           109.72 |        0.0507 |       0.2133 |              +0.0017 |              8.45e-07 |             9.25e-07 |                        0 |
| `house`    | bf16 | 10 |   0 / 10 |           108.81 |        0.0507 |       0.2133 |              +0.0017 |              8.20e-07 |             9.40e-07 |                        0 |

## Findings

### 1. `chol_v6` is **14–19× faster** than `cholqr2`/`house` when it doesn't raise

Steady-state runtime per `[768, 128, 128]` call:

- `chol_v6` fp32: **7.8 ms**
- `chol_v6` bf16: **6.0 ms**
- `cholqr2` fp32/bf16: **113–114 ms**  (forced fallback path on these states)
- `house` fp32/bf16:   **109–110 ms**

This matches the docstring claim that `chol_v6` stays on the fast CholQR
path even on rank-deficient sketches. It's the first orth method we've
seen that actually hits the documented sub-15 ms regime on real Qwen3.5-4B
states.

### 2. But it raises 20–50% of the time — and the raise is from **`eigh`**, not from `_chol_v6`

`chol_v6` fp32 raised on 5 of 10 prompts; `chol_v6` bf16 raised on 2 of 10.

The traceback every time looks like:

```
_LinAlgError: linalg.eigh: (Batch element NNN): The algorithm failed to converge
```

This is **not** the chol_v6 jitter ladder failing — the ladder always
succeeds at returning *some* `Q`. The raise comes from the **next step**
inside `randomized_svd_eigh`:

```python
Bproj = Q.T @ A.float()
C = Bproj @ Bproj.T
evals, evecs = torch.linalg.eigh(C)   # ← this is what raises
```

When `Q` is nearly singular (because the jitter ladder over-jittered to
get a Cholesky), `C` is so ill-conditioned that cuSOLVER's batched `eigh`
doesn't converge. So the user-visible behavior is "chol_v6 raised", but
the actual mechanism is "jitter ladder produced a degenerate Q that broke
the downstream eigh".

**bf16 is more robust than fp32 here** (2 raises vs 5). This is the
opposite of the usual relationship — probably because bf16's larger
quantization floor accidentally re-conditions the projected Gram. Don't
read this as a recommendation, just a curiosity.

### 3. `U` is essentially **non-orthonormal** when `chol_v6` succeeds

Every successful `chol_v6` call returned a `U` with
`‖UᵀU − I‖_F / √k ≈ 0.97` (`max == mean == p99` in our 7,680-state sample).
For reference, `cholqr2` and `house` return `U` with this same defect
≈ `8×10⁻⁷` — i.e. **6 orders of magnitude tighter**.

A defect of 0.97 means `UᵀU` is roughly the zero matrix, i.e. `U`'s columns
have shrunk to near-zero norm. This is because the jitter ladder typically
ends with `jitter ≫ ‖G‖`, so `R = chol(G + jitter·I) ≈ √jitter · I` and
`Q = Y · R⁻¹ ≈ Y / √jitter` is tiny. `U = Q · Ub` inherits the same tiny
scale.

The downstream `S` and `Vh` rescale to compensate, so the **reconstruction
product `U·diag(S)·Vh`** still approximates `A` reasonably (see (4)). But
the individual factors are not the SVD: `S` is not the singular values,
`U` is not orthonormal, `Vh` rows are not orthonormal.

**If downstream code relies on `UᵀU = I`** (e.g. for energy-preserving
rotations, basis alignment, sign-flip resolution, or projecting onto a
true orthonormal subspace), `chol_v6` is **not safe**. If you only need
`A ≈ X·Y` as a low-rank factorization where the factor structure doesn't
matter, it's usable subject to (4).

### 4. Reconstruction error is meaningfully worse than `cholqr2`/`house`

Mean recon error across states:
- `chol_v6`:  0.061 (fp32) / 0.063 (bf16)
- `cholqr2`/`house`: 0.051 (both dtypes)

So the average state is reconstructed **~1.0–1.2 percentage points worse**
than the rank-16 truncation optimum. The worst single state is **3.7
percentage points worse** (`max_gap_abs = +0.0371`, vs `cholqr2`/`house` at
≤ +0.002).

Zero states crossed our 5pp-absolute-gap "hard failure" threshold, so
the approximation is still useful — but it is clearly not the same
accuracy point as `cholqr2`/`house`. If you're using `chol_v6` you're
trading ~3pp of worst-case rank-16 reconstruction quality for the 14×
speedup.

### 5. The p99 recon error is unchanged

`p99 recon err` is the same (~0.213) across all three methods. The
chol_v6 hit is concentrated in the **worst** few states — the ones with
the smallest spectral tail. The bulk of the batch reconstructs as well as
with `cholqr2`/`house`.

## Putting it next to the main report's headline table

| method     | dtype | raises (prompts) | mean recon err | worst gap | orth(U) defect | runtime (ms) |
|------------|-------|------------------|----------------|----------:|---------------:|-------------:|
| `chol_min` | fp32 | **30/30**        | —              |        — |              — |          ~1 (raises fast) |
| `chol_min` | bf16 | **30/30**        | —              |        — |              — |          ~1 (raises fast) |
| `chol_v6`  | fp32 | **5/10**         | 0.061          | +0.0371  |       **0.97** |      **7.8** |
| `chol_v6`  | bf16 | **2/10**         | 0.063          | +0.0371  |       **0.97** |      **6.0** |
| `cholqr2`  | fp32 | 0/30             | 0.051          | +0.0021  |       8.13e-07 |        127 |
| `cholqr2`  | bf16 | 0/30             | 0.051          | +0.0030  |       8.26e-07 |        114 |
| `house`    | fp32 | 0/30             | 0.051          | +0.0020  |       8.45e-07 |        110 |
| `house`    | bf16 | 0/30             | 0.051          | +0.0026  |       8.20e-07 |        110 |

## Recommendation

`chol_v6` is **not a drop-in replacement** for `cholqr2`/`house` on
Qwen3.5-4B states. Use it only if **all** of these are true:

1. You can absorb a 20–50% per-call raise rate (or wrap the call in a
   try/except that retries with `house`).
2. You only care about `A ≈ X·Y` as a low-rank factor, not about the
   SVD-shape properties (orthonormality, true singular values).
3. The ~3pp worst-case extra reconstruction error is acceptable.
4. The 14× speedup matters.

If any of those isn't true, stay with `house` (or `cholqr2`, which on
these states is identical to `house` because it always takes the
Householder fallback).

The docstring's claim that `chol_v6` "stays on the fast CholQR path even
when the sketch's Gram is rank-deficient" is technically true — the
ladder always returns a `Q`. But the silent accuracy degradation is much
larger than the docstring's "≈ jitter size" suggests, and the downstream
`eigh` step is not robust to the degenerate `Q` that the ladder produces.

A possible fix worth considering: in `_chol_v6`, after the jitter ladder
succeeds, check whether `‖QᵀQ − I‖` exceeds a threshold (say 1e-3); if
so, run one CholQR pass on `Q` to clean it up, or fall through to
Householder. That would preserve the 14× speedup for the well-conditioned
sketches in the batch and only pay the Householder cost when the ladder
actually compromised orthonormality. (Since the current implementation
applies the ladder to the entire batch, you'd need a partial-batch
re-orth path to avoid pulling the well-conditioned sketches down with
the rank-deficient ones.)

## Artifacts in this directory

- `REPORT.md` — this file.
- `results.csv` — per-(prompt, method, dtype) metrics including
  orth-defect mean/max/p99.

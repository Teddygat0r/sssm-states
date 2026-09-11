"""
Fast batched randomized SVD via eigh-on-Gram + mixed-precision Cholesky-QR.

Minimal, single-path variant of `rsvd_eigh.py` that uses fp64 Cholesky-QR
for orthonormalization, with a Householder-QR fallback for inputs that are
exactly rank-deficient at `q = rank + oversample`.

On ill-conditioned but full-rank inputs (decaying spectra, ML activations,
KV-cache states), the fp64 CholQR path lands at fp32-machine-epsilon
orthonormality and the rank-k truncation optimum — no jitter bias, no slow
Householder for the common case. On rank-deficient inputs (`σ_q = 0` to
fp64 precision), the fallback handles correctness at the standard
Householder cost.

Algorithm (one call):
  Omega ~ N(0, 1) of shape [..., n, q] where q = rank + oversample.
  Y = A · Omega                       sketch the column space
  Q = chol_mp(Y)                      fp64 CholQR → fp32 Q  (Householder
                                       fallback on LinAlgError)
  repeat n_iter times:
      Y = A · (Aᵀ · Q)                power iteration
      Q = chol_mp(Y)
  Bproj = Qᵀ · A                      project A onto the subspace
  C = Bproj · Bprojᵀ                  small [q, q] SPD Gram
  λ, V = eigh(C)                      ascending eigvals/vecs
  S = sqrt(λ_top-k)
  U = Q · V_top-k
  Vh = V_top-kᵀ · Bproj / S

Why this path:
  - Replacing the inner SVD with eigh-on-Gram is the algorithm's signature
    optimization: cuSOLVER's tiny-SPD eigh is much faster than batched
    Jacobi SVD on the projected matrix. This is the same shape advantage
    that distinguishes rsvd_eigh from `torch.svd_lowrank`.
  - fp64 inside CholQR absorbs the κ² · ε amplification that breaks plain
    fp32 Cholesky on decaying spectra. On a typical Qwen3.5-4B recurrent-
    state batch ([768, 128, 128] → rank 16), single-pass fp32 CholQR fails
    on ~30% of states; fp32 with jitter ladder works but biases the recon
    by ~1pp; fp64 single-pass succeeds with no bias.
  - Householder QR fallback handles the residual case where even fp64
    Cholesky raises (truly rank-deficient input with σ_q = 0). Did not
    fire on Qwen3.5-4B states across 14,042 MMLU prompts but is here for
    correctness on synthetic / pathological inputs.

Cost summary on Qwen3.5-4B recurrent states, [768, 128, 128] → rank 16:
  - this file (chol_mp + per-elem Householder)  ~15 ms (no failures);
                                                ~5 ms extra per ~100 bad elements
  - cholqr2 / house  (fallback path fires)      ~110 ms
  - chol_v6 + proactive jitter                  ~10 ms (but +1.2pp recon bias)
  - torch.svd_lowrank, GPU                      ~2150 ms

When some batch elements are rank-deficient, the fallback uses
`linalg.cholesky_ex` (returns per-element `info` rather than raising) and
runs Householder QR only on the failing slice. At our standard [768, 128,
24] sketch shape, this scales as roughly +0.04 ms per bad element on top
of the fast path — vs +33 ms for the whole-batch fallback strategy.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _chol_mp(Y: Tensor) -> Tensor:
    """Single-pass Cholesky-QR in fp64 with per-element Householder fallback.

    Fast path: compute the Gram, Cholesky factor, and triangular solve all
    in fp64, then cast the result back to Y's original dtype. Buys ~8
    extra decimal digits over plain fp32 CholQR, which is enough to handle
    decaying-spectrum sketches where fp32 CholQR fails (κ² · ε ≈ 1)
    without introducing the systematic bias of jitter-based paths.

    Fallback: `torch.linalg.cholesky_ex` returns per-element `info` codes
    instead of raising. For elements where fp64 Cholesky succeeded
    (`info == 0`), use the standard `solve_triangular` path. For elements
    that failed (truly rank-deficient at q: σ_q = 0 in fp64), run
    Householder QR via `torch.linalg.qr` on just that slice and scatter
    the result back into the output tensor.

    Cost on a Qwen3.5-4B-shaped batch [B=768, m=128, q=24]:
      - No failures:                 ~3 ms     (just cholesky_ex + trsm)
      - 1 bad element out of 768:    ~5 ms     (vs ~36 ms for whole-batch
                                                Householder fallback)
      - 64 bad elements:             ~7 ms
      - 256 bad elements:            ~13 ms
      - All 768 elements bad:        ~28 ms

    For workloads with extreme per-batch failure rates (>~30%), a CPU
    Householder fallback on the bad slice is slightly faster than GPU
    Householder (LAPACK beats cuSOLVER's batched-Householder dispatch on
    tall-skinny [128, 24] panels). Not implemented here because the
    common case is sparse failures where per-element GPU wins.
    """
    Y64 = Y.double()
    G = Y64.transpose(-2, -1) @ Y64
    R, info = torch.linalg.cholesky_ex(G, upper=True)
    bad = info != 0

    if not bad.any():
        # Fast path: every batch element succeeded.
        Q = torch.linalg.solve_triangular(R, Y64, upper=True, left=False)
        return Q.to(Y.dtype)

    # Mixed batch: scatter results from two paths.
    ok = ~bad
    Q_out = torch.empty_like(Y)
    if ok.any():
        Q_ok = torch.linalg.solve_triangular(
            R[ok], Y64[ok], upper=True, left=False
        )
        Q_out[ok] = Q_ok.to(Y.dtype)
    # Householder QR returns an orthonormal basis of range(Y[bad]) even
    # when rank(Y[bad]) < q.
    Q_bad = torch.linalg.qr(Y[bad], mode="reduced")[0]
    Q_out[bad] = Q_bad
    return Q_out


@torch.no_grad()
def randomized_svd_eigh(
    A: Tensor,
    *,
    rank: int,
    n_iter: int = 2,
    oversample: int = 8,
    power_dtype: torch.dtype | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomized truncated SVD with eigh-on-Gram and fp64 Cholesky-QR.

    Parameters
    ----------
    A:
        Input tensor of shape [..., m, n]. Computation is batched over all
        leading dimensions. The dominant cost is matmuls of A and Aᵀ
        against tall-skinny [..., m, q] sketches, so this is fastest when
        m, n are small (e.g. 128) and the leading batch is large (e.g.
        1024).
    rank:
        Target rank k. Must satisfy `rank <= min(m, n)`.
    n_iter:
        Number of subspace (power) iterations. Each iteration does
        `Y = A · (Aᵀ · Q)` then re-orthonormalizes. `n_iter=2` is the
        accuracy point that matches `torch.svd_lowrank(niter=4)`.
    oversample:
        Extra sketch columns beyond `rank`. The sketch dimension is
        `q = rank + oversample`. q=8 is a good default; larger q tightens
        the approximation at moderate cost.
    power_dtype:
        Optional dtype for the power-iteration matmuls. Defaults to
        `A.dtype`. Pass `torch.bfloat16` (or `torch.float16`) on GPU to
        run matmuls in lower precision; the orthonormalization runs in
        fp64 and the projection + eigh run in fp32 regardless, for
        numerical safety.

    Returns
    -------
    U  : [..., m, rank]
    S  : [..., rank]   (descending)
    Vh : [..., rank, n]

    such that A ≈ U · diag(S) · Vh in the truncated rank-k Frobenius
    sense, and UᵀU = I, Vh·Vhᵀ = I at fp32 machine epsilon (≈ 1e-7).
    """
    m, n = A.shape[-2], A.shape[-1]
    if rank <= 0 or rank > min(m, n):
        raise ValueError(f"rank must be in [1, min(m, n) = {min(m, n)}], got {rank}")

    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A

    # Initial Gaussian sketch.
    Omega = torch.randn(n, q, dtype=Ap.dtype, device=A.device)
    Y = Ap @ Omega
    if pdtype != torch.float32:
        Y = Y.float()
    Q = _chol_mp(Y)

    # Subspace iteration. Each step: Y = A · Aᵀ · Q (one fused power step,
    # no intermediate orthonormalize), then re-orthonormalize via fp64
    # Cholesky-QR.
    for _ in range(n_iter):
        Q_p = Q.to(Ap.dtype) if pdtype != torch.float32 else Q
        Z = Ap.transpose(-2, -1) @ Q_p           # [..., n, q]
        Y = Ap @ Z                                # [..., m, q]
        if pdtype != torch.float32:
            Y = Y.float()
        Q = _chol_mp(Y)

    # Final projection and small SVD via eigh on the small Gram.
    Bproj = Q.transpose(-2, -1) @ A.float()       # [..., q, n] in fp32
    C = Bproj @ Bproj.transpose(-2, -1)           # [..., q, q] SPD
    C = 0.5 * (C + C.transpose(-2, -1))           # cheap symmetrize for safety
    # Defensive diagonal jitter for cuSOLVER's batched Jacobi eigh
    # (`syevjBatched`), which can fail to converge with "error code: 1"
    # on Gram matrices with near-repeated eigenvalues or extreme
    # conditioning. The shift is uniform, so the top-k eigenvalue ordering
    # and eigenvectors are unchanged to within fp32 epsilon — we verified
    # accuracy-neutrality across 50 ground-truth checks (max per-state
    # |Δerr| < 5e-6, with the chol_v6 bias for comparison being ~1.2e-2).
    d = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-30)
    eye_q = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
    C = C + (1e-8 * d).unsqueeze(-1) * eye_q
    
    #try fp32 first, on failure try fp64
    try:
        evals, evecs = torch.linalg.eigh(C)           # ascending eigvals
    except torch.linalg.LinAlgError:
        print("Exception occurred in torch.linalg.eigh, falling back to fp64")
        evals, evecs = torch.linalg.eigh(C.double())           # ascending eigvals
        evals, evecs = evals.float(), evecs.float()

    # Take top-k (last k columns), flip to descending.
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)            # [..., q, rank]
    U = Q @ Ub                                     # [..., m, rank]
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def low_rank_svd(
    tensor: Tensor,
    n: int = 16,
    oversample: int = 4,
    niter: int = 2,
) -> Tensor:
    """Rank-n reconstruction of `tensor` via `randomized_svd_eigh`.

    Returns `U · diag(S) · Vh` cast back to `tensor.dtype`, on the input
    device (no CPU round-trip). Batched over all leading dimensions.
    """
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    orig_dtype = tensor.dtype
    rank = min(n, min(tensor.shape[-2:]))
    try:
        u, s, vh = randomized_svd_eigh(
            tensor.detach(), rank=rank, n_iter=niter, oversample=oversample
        )
    except torch.linalg.LinAlgError:
        print("Exception occurred in randomized SVD, falling back to torch.svd_lowrank")
        # perform low rank SVD on CPU
        try:
            tensor_cpu = tensor.cpu()
            u, s, vh = torch.svd_lowrank(tensor_cpu, q=rank+oversample, oversample=oversample)
            u, s, vh = u.to(tensor.device), s.to(tensor.device), vh.to(tensor.device)
        except:
            print("Exception occurred in torch.svd_lowrank, skipping SVD")
            return tensor

    approx = (u * s.unsqueeze(-2)) @ vh
    return approx.to(orig_dtype)


__all__ = ["randomized_svd_eigh", "low_rank_svd"]

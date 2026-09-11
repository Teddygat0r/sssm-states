"""
Benchmark per-element Householder fallback vs whole-batch fallback for
the rank-deficient case in chol_mp.

Hypothesis: cholesky_ex returns per-element `info` codes. If only k out of
B elements are rank-deficient, we only need Householder on those k. Cost
should scale linearly with k, not B.

Variants to compare:
  A. baseline (try/except + whole-batch Householder)
  B. cholesky_ex + per-element Householder fallback
  C. CPU Householder for the failing slice (transferred back)

Test grid: B=768 with k in {0, 1, 8, 64, 256, 768} rank-deficient elements.
"""

from __future__ import annotations

import argparse
from time import perf_counter
import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _chol_mp_baseline(Y):
    """try/except + whole-batch Householder fallback."""
    try:
        Y64 = Y.double()
        G = Y64.transpose(-2, -1) @ Y64
        R = torch.linalg.cholesky(G, upper=True)
        Q = torch.linalg.solve_triangular(R, Y64, upper=True, left=False)
        return Q.to(Y.dtype)
    except torch.linalg.LinAlgError:
        return torch.linalg.qr(Y, mode="reduced")[0]


@torch.no_grad()
def _chol_mp_per_elem(Y):
    """cholesky_ex + per-element Householder fallback."""
    Y64 = Y.double()
    G = Y64.transpose(-2, -1) @ Y64
    R, info = torch.linalg.cholesky_ex(G, upper=True)
    bad = info != 0
    if not bad.any():
        Q = torch.linalg.solve_triangular(R, Y64, upper=True, left=False)
        return Q.to(Y.dtype)
    ok = ~bad
    Q_out = torch.empty_like(Y)
    if ok.any():
        Q_ok = torch.linalg.solve_triangular(R[ok], Y64[ok], upper=True, left=False)
        Q_out[ok] = Q_ok.to(Y.dtype)
    if bad.any():
        Q_bad = torch.linalg.qr(Y[bad], mode="reduced")[0]
        Q_out[bad] = Q_bad
    return Q_out


@torch.no_grad()
def _chol_mp_per_elem_cpu(Y):
    """cholesky_ex + per-element Householder fallback on CPU for the bad slice."""
    Y64 = Y.double()
    G = Y64.transpose(-2, -1) @ Y64
    R, info = torch.linalg.cholesky_ex(G, upper=True)
    bad = info != 0
    if not bad.any():
        Q = torch.linalg.solve_triangular(R, Y64, upper=True, left=False)
        return Q.to(Y.dtype)
    ok = ~bad
    Q_out = torch.empty_like(Y)
    if ok.any():
        Q_ok = torch.linalg.solve_triangular(R[ok], Y64[ok], upper=True, left=False)
        Q_out[ok] = Q_ok.to(Y.dtype)
    if bad.any():
        Y_bad_cpu = Y[bad].cpu()
        Q_bad_cpu = torch.linalg.qr(Y_bad_cpu, mode="reduced")[0]
        Q_out[bad] = Q_bad_cpu.to(Y.device)
    return Q_out


def make_test(B, m, q, n_bad, device="cuda", dtype=torch.float32, seed=0):
    """Make a [B, m, q] sketch where n_bad elements are rank-deficient.

    Y for the good elements is generated as a Gaussian. Y for the bad
    elements is generated as L @ R with L: [m, r], R: [r, q] for r < q,
    making rank(Y[i]) = r exactly.
    """
    torch.manual_seed(seed)
    Y = torch.randn(B, m, q, device=device, dtype=dtype)
    if n_bad > 0:
        r = max(1, q // 4)  # rank-6 when q=24
        L = torch.randn(n_bad, m, r, device=device, dtype=dtype)
        R = torch.randn(n_bad, r, q, device=device, dtype=dtype)
        Y[:n_bad] = L @ R
    return Y


def time_fn(fn, Y, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        fn(Y)
    _sync()
    t0 = perf_counter()
    for _ in range(n_iter):
        fn(Y)
    _sync()
    return (perf_counter() - t0) / n_iter * 1000


def main():
    B, m, q = 768, 128, 24
    ks = [0, 1, 8, 64, 256, 768]
    variants = [
        ("baseline (whole-batch fallback)", _chol_mp_baseline),
        ("per-elem GPU fallback         ", _chol_mp_per_elem),
        ("per-elem CPU fallback         ", _chol_mp_per_elem_cpu),
    ]

    print(f"Y shape [{B}, {m}, {q}], q={q}")
    print(f"{'n_bad':>8}", end="")
    for name, _ in variants:
        print(f"{name:>40}", end="")
    print()
    for k in ks:
        Y = make_test(B, m, q, k)
        print(f"{k:>8}", end="")
        for name, fn in variants:
            ms = time_fn(fn, Y)
            print(f"{ms:>40.2f} ms", end="")
        print()


if __name__ == "__main__":
    main()

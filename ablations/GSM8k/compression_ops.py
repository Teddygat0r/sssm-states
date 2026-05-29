"""State compression operators for Qwen3.5-4B recurrent state.

All operators take a tensor of shape `[..., D_k, D_v]` (typically
`[L_lin * B, H, D_k, D_v]` for Qwen3.5) and return a same-shape tensor
that is a compressed-then-reconstructed copy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsvd_eigh import randomized_svd_eigh  # noqa: E402


# --- Low-rank SVD ------------------------------------------------------------

@torch.no_grad()
def svd_compress(state: Tensor, rank: int) -> Tensor:
    """Rank-`rank` reconstruction via GPU randomized SVD."""
    orig_dtype = state.dtype
    max_rank = min(state.shape[-2], state.shape[-1])
    k = max(1, min(rank, max_rank))
    A = state.detach()
    if A.dtype != torch.float32:
        A = A.float()
    U, S, Vh = randomized_svd_eigh(
        A, rank=k, n_iter=1, oversample=4, power_dtype=torch.float32
    )
    approx = (U * S.unsqueeze(-2)) @ Vh
    return approx.to(orig_dtype)


def make_svd_compressor(rank: int) -> Callable[[Tensor], Tensor]:
    def _f(state: Tensor) -> Tensor:
        return svd_compress(state, rank)
    _f.__name__ = f"svd_r{rank}"
    return _f


# --- Per-head uniform affine fake-quant --------------------------------------

@torch.no_grad()
def fake_quant_per_head(state: Tensor, n_bits: int) -> Tensor:
    """Per-`(..., H)` affine fake-quant over the last two dims.

    Matches state_spectrum_sweep/run_q4_vs_svd_sweep.py:154-167.
    """
    qmin = 0
    qmax = (1 << n_bits) - 1
    orig_dtype = state.dtype
    f = state.detach()
    if f.dtype != torch.float32:
        f = f.float()
    flat = f.flatten(start_dim=-2)                              # [..., D_k*D_v]
    t_min = flat.amin(dim=-1, keepdim=True)
    t_max = flat.amax(dim=-1, keepdim=True)
    scale = ((t_max - t_min) / qmax).clamp_min(1e-12)
    zero_point = torch.round(-t_min / scale).clamp(qmin, qmax)
    q = torch.round(flat / scale + zero_point).clamp(qmin, qmax)
    deq = (q - zero_point) * scale
    return deq.reshape_as(f).to(orig_dtype)


def make_quant_compressor(n_bits: int) -> Callable[[Tensor], Tensor]:
    def _f(state: Tensor) -> Tensor:
        return fake_quant_per_head(state, n_bits)
    _f.__name__ = f"q{n_bits}_per_head"
    return _f


# --- Hadamard rotation + quantization ----------------------------------------

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Normalized symmetric Walsh-Hadamard matrix of size `n` (power of two).

    The returned `H` satisfies `H = H.T` and `H @ H = I` to fp32 epsilon.
    Cached per `(n, device, dtype)`.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard_matrix requires n to be a power of two; got {n}")
    key = (n, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    H = H / (n ** 0.5)
    _HADAMARD_CACHE[key] = H
    return H


@torch.no_grad()
def hadamard_quant(state: Tensor, n_bits: int) -> Tensor:
    """Rotate state along both inner dims with normalized Hadamard, per-head
    quantize, rotate back. Both rotations are orthogonal (and symmetric), so
    the round-trip is exact when n_bits is large enough.
    """
    if state.dim() < 2:
        raise ValueError(f"expected rank>=2, got shape {tuple(state.shape)}")
    orig_dtype = state.dtype
    D_k, D_v = state.shape[-2], state.shape[-1]
    f = state.detach()
    if f.dtype != torch.float32:
        f = f.float()
    Hk = hadamard_matrix(D_k, f.device, f.dtype)
    Hv = hadamard_matrix(D_v, f.device, f.dtype)
    rotated = Hk @ f @ Hv
    quantized = fake_quant_per_head(rotated, n_bits)
    restored = Hk @ quantized @ Hv
    return restored.to(orig_dtype)


def make_hadamard_quant_compressor(n_bits: int) -> Callable[[Tensor], Tensor]:
    def _f(state: Tensor) -> Tensor:
        return hadamard_quant(state, n_bits)
    _f.__name__ = f"hadamard_q{n_bits}"
    return _f


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Self-test on {device}")

    H = hadamard_matrix(128, device)
    err = (H @ H - torch.eye(128, device=device)).abs().max().item()
    print(f"  H@H - I max abs error (128): {err:.2e}")
    assert err < 1e-5, "Hadamard not orthonormal"

    state = torch.randn(24, 32, 128, 128, device=device, dtype=torch.bfloat16)

    # SVD round-trip at full rank should be near-exact.
    out = svd_compress(state, rank=128)
    err = (out.float() - state.float()).norm() / state.float().norm()
    print(f"  svd_r128 round-trip rel-Frob: {err.item():.2e}")

    # Per-head quant degrades as bits drop.
    for bits in (8, 4):
        out = fake_quant_per_head(state, bits)
        err = (out.float() - state.float()).norm() / state.float().norm()
        print(f"  q{bits}_per_head rel-Frob:   {err.item():.4f}")

    # Hadamard + 4-bit should produce a different (typically smaller) error
    # than naive 4-bit. The round-trip without quant noise is exact:
    out_no_quant = hadamard_matrix(128, device) @ (hadamard_matrix(128, device) @ state.float() @ hadamard_matrix(128, device)) @ hadamard_matrix(128, device)
    err = (out_no_quant - state.float()).norm() / state.float().norm()
    print(f"  hadamard round-trip (no quant) rel-Frob: {err.item():.2e}")
    assert err.item() < 1e-3, "Hadamard round-trip not identity"

    out = hadamard_quant(state, 4)
    err = (out.float() - state.float()).norm() / state.float().norm()
    print(f"  hadamard_q4 rel-Frob:      {err.item():.4f}")

    # Sanity: SVD at intermediate ranks
    for r in (1, 4, 16, 24):
        out = svd_compress(state, r)
        err = (out.float() - state.float()).norm() / state.float().norm()
        print(f"  svd_r{r:<3} rel-Frob:        {err.item():.4f}")

    print("OK")

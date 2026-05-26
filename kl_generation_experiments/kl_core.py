"""
Core machinery for the KL-under-state-compression generation experiment.

The instantaneous-KL measurement: at a generation step the model has a true
recurrent state S (per recurrent layer, per head). We build a batched decode
where batch row 0 carries the true state and rows 1..R carry rank-k_r
reconstructions of S (attention KV / conv states are left exact). One forward
gives logits for all rows; KL(row0 || row_r) is the distributional cost of
compressing the whole recurrent state to rank k_r at this step.

Cache handling note: these models (mamba2 / qwen3.5-gdn / nemotron via
transformers, deltanet / gated_deltanet via FLA) store state in the unified
`DynamicCache.layers[*]`. The built-in `batch_repeat_interleave` cannot be used:
the recurrent `LinearAttentionLayer` does not implement it (only the attention
`DynamicLayer` does), so calling it crashes on the first recurrent layer. We
therefore repeat / select / inject by hand over the known tensor fields.
"""
from __future__ import annotations

import inspect
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# The repo-root rsvd_eigh (eigh-on-Gram + mixed-precision Cholesky-QR) is the
# fast batched randomized SVD: ~15ms vs ~2150ms for torch.svd_lowrank on a
# [768,128,128]->16 batch, and it runs on-GPU (no CPU round-trip). It's the
# decisive optimization for the per-step compression here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from rsvd_eigh import randomized_svd_eigh  # noqa: E402


# --------------------------------------------------------------------------- #
# cache I/O kwarg (mamba2 uses `cache_params=`, the rest use `past_key_values=`)
# --------------------------------------------------------------------------- #

def detect_cache_kwarg(model) -> str:
    """The forward() keyword used to thread the cache for this model. mamba2's
    forward takes `cache_params`; hybrids / FLA models take `past_key_values`.
    Using the wrong one silently drops the cache (verified: garbage logits)."""
    params = inspect.signature(model.forward).parameters
    if "past_key_values" in params:
        return "past_key_values"
    if "cache_params" in params:
        return "cache_params"
    raise ValueError("model.forward exposes neither past_key_values nor cache_params")


def forward_step(model, input_ids, cache, pos: int, cache_kwarg: str):
    """One decode step threading `cache` via the model-correct kwarg. Returns the
    output; read the advanced cache with `getattr(out, cache_kwarg)`."""
    cp = torch.tensor([pos], device=input_ids.device)
    kwargs = {cache_kwarg: cache, "cache_position": cp, "use_cache": True}
    with torch.inference_mode():
        return model(input_ids=input_ids, **kwargs)

# Per-layer tensor fields that carry a batch dimension, across all five models.
#   attention layer (DynamicLayer):       keys, values
#   linear/ssm layer (LinearAttention*):  conv_states, recurrent_states
_BATCH_TENSOR_FIELDS = ("keys", "values", "conv_states", "recurrent_states")


# --------------------------------------------------------------------------- #
# recurrent-state access (handles both the transformers and FLA layer layouts)
# --------------------------------------------------------------------------- #

def _layer_recurrent_tensor(layer) -> torch.Tensor | None:
    """Return the layer's recurrent-state tensor [B, H, D_k, D_v], or None.

    transformers builtin layers expose `.recurrent_states`; FLA layers keep it
    in a `.state` dict under 'recurrent_state'."""
    t = getattr(layer, "recurrent_states", None)
    if torch.is_tensor(t):
        return t
    st = getattr(layer, "state", None)
    if isinstance(st, dict):
        t = st.get("recurrent_state")
        if torch.is_tensor(t):
            return t
    return None


def recurrent_layer_indices(cache) -> list[int]:
    """Indices of layers that carry a (>=4D) recurrent state tensor."""
    out = []
    for li, layer in enumerate(cache.layers):
        t = _layer_recurrent_tensor(layer)
        if t is not None and t.dim() >= 4:
            out.append(li)
    return out


def read_recurrent_states(cache, layer_indices) -> dict[int, torch.Tensor]:
    """{layer_idx: state[H, D_k, D_v]} taken from batch row 0, as float32."""
    states: dict[int, torch.Tensor] = {}
    for li in layer_indices:
        t = _layer_recurrent_tensor(cache.layers[li])
        states[li] = t[0].detach().to(dtype=torch.float32)
    return states


def write_recurrent_row(cache, li: int, row: int, value: torch.Tensor) -> None:
    """Overwrite batch `row` of layer `li`'s recurrent state in place."""
    t = _layer_recurrent_tensor(cache.layers[li])
    t[row] = value.to(dtype=t.dtype, device=t.device)


# --------------------------------------------------------------------------- #
# manual batch repeat / select over the unified DynamicCache
# --------------------------------------------------------------------------- #

def _apply_to_batch_tensors(layer, fn) -> None:
    """Apply `fn` (a batch-dim transform) in place to every batched tensor a
    layer carries: attribute-stored (keys/values/conv_states/recurrent_states),
    FLA dict-stored (state['recurrent_state']), AND tensors nested inside
    tuples/lists in the state dict (FLA's state['conv_state'] is a 3-tuple of
    q/k/v conv states — missing these is what NaN'd the FLA batched forward)."""
    for f in _BATCH_TENSOR_FIELDS:
        t = getattr(layer, f, None)
        if torch.is_tensor(t):
            setattr(layer, f, fn(t))
    st = getattr(layer, "state", None)
    if isinstance(st, dict):
        for key, v in st.items():
            if torch.is_tensor(v):
                st[key] = fn(v)
            elif isinstance(v, (tuple, list)) and any(torch.is_tensor(x) for x in v):
                st[key] = type(v)(fn(x) if torch.is_tensor(x) else x for x in v)


def cache_repeat_(cache, n: int) -> None:
    """Repeat every batched tensor in the cache n times along dim 0 (in place)."""
    for layer in cache.layers:
        _apply_to_batch_tensors(layer, lambda t: t.repeat_interleave(n, dim=0))
        if hasattr(layer, "max_batch_size") and isinstance(layer.max_batch_size, int):
            layer.max_batch_size = layer.max_batch_size * n


def cache_select_row_(cache, row: int) -> None:
    """Keep only batch index `row` (in place), restoring batch size 1."""
    for layer in cache.layers:
        _apply_to_batch_tensors(layer, lambda t: t[row : row + 1, ...].contiguous())
        if hasattr(layer, "max_batch_size") and isinstance(layer.max_batch_size, int):
            layer.max_batch_size = 1


# --------------------------------------------------------------------------- #
# low-rank reconstruction (matches the deployed randomized svd_lowrank path)
# --------------------------------------------------------------------------- #

def lowrank_recon_state(state: torch.Tensor, k: int,
                        *, oversample: int = 4, niter: int = 1) -> torch.Tensor:
    """Rank-k randomized-SVD reconstruction of a [H, D_k, D_v] state (batched
    over heads). Returns the same shape/dtype as `state`. (Single-layer helper,
    used by the self-test; the runner uses lowrank_recon_all.)"""
    H, D_k, D_v = state.shape
    max_rank = min(D_k, D_v)
    kk = max(1, min(k, max_rank))
    q = min(kk + oversample, max_rank)
    A = state.to(dtype=torch.float32)
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)       # U[H,Dk,q] S[H,q] V[H,Dv,q]
    U, S, V = U[..., :kk], S[..., :kk], V[..., :kk]
    M_hat = (U * S[..., None, :]) @ V.transpose(-2, -1)    # [H, Dk, Dv]
    return M_hat.to(dtype=state.dtype)


def lowrank_recon_all(states: dict[int, torch.Tensor], ranks,
                      *, oversample: int = 8, niter: int = 2
                      ) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, float]]:
    """Batched rank reconstructions for *all* recurrent layers at once.

    Calling svd_lowrank per (layer, rank) is hundreds of launch-bound tiny SVDs
    (~19s/step on mamba2). Instead we stack every layer's heads (grouped by
    [D_k, D_v]) into one tensor and run a *single* `randomized_svd_eigh` at the
    largest requested rank on-GPU, then truncate that one sketch to each rank
    (descending S). On the mamba2 stack [3072,64,128]->16 this is ~25ms.

    Returns (recons, rel_fro) where
      recons[rank][layer_idx] = recon[H, D_k, D_v] (inputs' device/dtype), and
      rel_fro[rank]           = mean relative Frobenius^2 error over all heads.
    The rel_fro is computed batched here (one sync per rank) rather than looping
    per layer in the caller. Truncating a top-kmax sketch to a smaller k is >=
    as accurate as a dedicated rank-k sketch.
    """
    ranks = tuple(sorted(ranks))
    kmax = max(ranks)
    out: dict[int, dict[int, torch.Tensor]] = {k: {} for k in ranks}
    relfro_sum: dict[int, float] = {k: 0.0 for k in ranks}
    total_n = 0
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for li, s in states.items():
        groups[(int(s.shape[1]), int(s.shape[2]))].append(li)

    for (D_k, D_v), lis in groups.items():
        max_rank = min(D_k, D_v)
        sizes = [int(states[li].shape[0]) for li in lis]
        dt = states[lis[0]].dtype
        A = torch.cat([states[li] for li in lis], dim=0).to(dtype=torch.float32)  # [Ntot,D_k,D_v]
        total_n += A.shape[0]
        den = A.flatten(1).pow(2).sum(1).clamp_min(1e-30)
        r_eff = min(kmax, max_rank)
        try:
            U, S, Vh = randomized_svd_eigh(A, rank=r_eff, n_iter=niter, oversample=oversample)
        except torch.linalg.LinAlgError:
            # extremely rare; fall back to CPU svd_lowrank for this group
            q = min(r_eff + oversample, max_rank)
            U, S, Vt = torch.svd_lowrank(A.cpu(), q=q, niter=max(1, niter))
            U, S, Vh = U.to(A.device), S.to(A.device), Vt.transpose(-2, -1).to(A.device)
        for k in ranks:
            kk = max(1, min(k, r_eff))
            Uk, Sk, Vhk = U[..., :kk], S[..., :kk], Vh[..., :kk, :]
            M_hat = (Uk * Sk.unsqueeze(-2)) @ Vhk                  # [Ntot,D_k,D_v] fp32
            relfro_sum[k] += float(((A - M_hat).flatten(1).pow(2).sum(1) / den).sum().item())
            M_hat = M_hat.to(dtype=dt)
            off = 0
            for li, h in zip(lis, sizes):
                out[k][li] = M_hat[off:off + h]
                off += h
    rel_fro = {k: relfro_sum[k] / max(1, total_n) for k in ranks}
    return out, rel_fro


def rel_fro_mse(state: torch.Tensor, recon: torch.Tensor) -> float:
    """Mean (over heads) relative Frobenius squared error of a reconstruction."""
    a = state.to(torch.float32).flatten(1)
    b = recon.to(torch.float32).flatten(1)
    num = (a - b).pow(2).sum(dim=1)
    den = a.pow(2).sum(dim=1).clamp_min(1e-30)
    return float((num / den).mean().item())


# --------------------------------------------------------------------------- #
# KL
# --------------------------------------------------------------------------- #

def kl_full_vs_approx(logits_full: torch.Tensor, logits_approx: torch.Tensor) -> float:
    """KL(P_full || P_approx) for one next-token step. Each logits arg is [V].
    Matches the convention in state_spectrum_sweep/run_experiment_kl_parallel.py
    (kl_div(input=approx_logprobs, target=full_logprobs, log_target=True))."""
    lf = torch.log_softmax(logits_full.float().unsqueeze(0), dim=-1)
    la = torch.log_softmax(logits_approx.float().unsqueeze(0), dim=-1)
    return float(F.kl_div(la, lf, reduction="batchmean", log_target=True).item())

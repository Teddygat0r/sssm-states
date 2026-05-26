"""
Randomized-SVD variant of `_helpers.compute_task_metrics_batched`.

The *reconstruction* columns (rel_fro_mse_k*, max_abs_k*, mean_abs_k*,
cos_flat_k*) are produced with `torch.svd_lowrank` using the SAME parameters as
the deployed §2 KL / downstream-eval path:

    q = min(k + OVERSAMPLE, min(D_k, D_v)),  niter = NITER       (oversample=4, niter=1)

so the reported reconstruction error reflects the actual randomized algorithm,
not an idealized exact truncated SVD.

Everything else is computed with the EXACT full SVD, because a low-rank sketch
cannot produce it:
  * cumulative energy (energy_at_k*) — denominator is the sum over ALL singular
    values; we still get this exactly from the full spectrum.
  * effective-rank thresholds (num_rank990/995/999) — can exceed any small q.
  * the matched random (Marchenko–Pastur) control — near-full-rank by design.

The output row schema and spectra are identical to the exact kernel, so the
CSV / spectra.pt are drop-in compatible with the existing analysis/plots.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from _helpers import (  # shared exact-SVD infrastructure
    K_GRID,
    ENERGY_THRESHOLDS,
    SVD_DEVICE,
    SVD_DTYPE,
    _batched_cumulative_energy,
    _batched_num_rank_at_threshold,
)

# Randomized-SVD parameters — match the deployed KL/eval path.
OVERSAMPLE = int(os.getenv("RECON_OVERSAMPLE", "4"))
NITER = int(os.getenv("RECON_NITER", "1"))
# dtype for the svd_lowrank reconstruction. fp64 by default for stability
# (the rest of the pipeline is fp64); set RECON_DTYPE=float32 to mirror the
# deployed path exactly.
RECON_DTYPE = torch.float32 if os.getenv("RECON_DTYPE", "float64") == "float32" else torch.float64


def compute_task_metrics_batched_lowrank(
    states_by_layer: dict[int, torch.Tensor],
    *,
    model_label: str, family: str, prompt: dict,
    position: int, actual_tokens: int,
    seed: int,
    k_grid: tuple[int, ...] = K_GRID,
    energy_thresholds: tuple[float, ...] = ENERGY_THRESHOLDS,
    device: str = SVD_DEVICE,
    dtype: torch.dtype = SVD_DTYPE,
) -> tuple[list[dict], list, list, list[dict]]:
    """Drop-in replacement for compute_task_metrics_batched with randomized
    `svd_lowrank` reconstruction. Same args / return schema."""
    layer_indices = sorted(states_by_layer.keys())
    chunks: list[torch.Tensor] = []
    layer_head_meta: list[tuple[int, int]] = []
    for li in layer_indices:
        s = states_by_layer[li]                 # [H, D_k, D_v]
        chunks.append(s)
        for h in range(int(s.shape[0])):
            layer_head_meta.append((li, h))
    if not chunks:
        return [], [], [], []
    M_batch = torch.cat(chunks, dim=0).to(device=device, dtype=dtype)  # [N, D_k, D_v]
    N, D_k, D_v = M_batch.shape
    max_rank = min(D_k, D_v)

    # Matched random control: iid Gaussian, per-matrix element std.
    elem_std = M_batch.flatten(-2).std(dim=-1, unbiased=False)  # [N]
    g = torch.Generator(device=device).manual_seed(int(seed))
    R_batch = torch.randn(M_batch.shape, generator=g, dtype=dtype, device=device) \
              * elem_std.view(N, 1, 1)

    # ---- 1.1 spectrum: EXACT full SVD (state) + svdvals (random control) ----
    spectrum_ok = True
    try:
        S = torch.linalg.svdvals(M_batch)                          # [N, k]
    except Exception as e:
        print(f"    WARN: batched svdvals failed on state: {type(e).__name__}: {e}")
        S = None
        spectrum_ok = False
    try:
        S_R = torch.linalg.svdvals(R_batch)                        # [N, k]
    except Exception as e:
        print(f"    WARN: batched svdvals failed on random: {type(e).__name__}: {e}")
        S_R = None
        spectrum_ok = False

    M_flat = M_batch.flatten(-2)
    fro_norm = M_flat.to(dtype=torch.float64).norm(dim=-1)         # [N]
    fro_norm2 = M_flat.to(dtype=torch.float64).pow(2).sum(dim=-1).clamp_min(1e-30)

    if S is not None:
        cum_S = _batched_cumulative_energy(S)
        num_rank_S = {thr: _batched_num_rank_at_threshold(cum_S, thr).cpu()
                      for thr in energy_thresholds}
        energy_S_at_k = {}
        for k in k_grid:
            kk = max(1, min(k, S.shape[-1]))
            energy_S_at_k[k] = cum_S[..., kk - 1].cpu()
        S_cpu = S.cpu()
    else:
        num_rank_S = {thr: None for thr in energy_thresholds}
        energy_S_at_k = {k: None for k in k_grid}
        S_cpu = None

    if S_R is not None:
        cum_R = _batched_cumulative_energy(S_R)
        num_rank_R = {thr: _batched_num_rank_at_threshold(cum_R, thr).cpu()
                      for thr in energy_thresholds}
        energy_R_at_k = {}
        for k in k_grid:
            kk = max(1, min(k, S_R.shape[-1]))
            energy_R_at_k[k] = cum_R[..., kk - 1].cpu()
        S_R_cpu = S_R.cpu()
    else:
        num_rank_R = {thr: None for thr in energy_thresholds}
        energy_R_at_k = {k: None for k in k_grid}
        S_R_cpu = None

    # ---- 1.2 reconstruction: RANDOMIZED svd_lowrank (oversample, niter) ----
    recon = {k: None for k in k_grid}
    recon_all_ok = True
    M_lr = M_batch.to(dtype=RECON_DTYPE)
    for k in k_grid:
        kk = max(1, min(k, max_rank))
        q = min(kk + OVERSAMPLE, max_rank)
        try:
            U_k, S_k, V_k = torch.svd_lowrank(M_lr, q=q, niter=NITER)
        except Exception as e:
            print(f"    WARN: svd_lowrank failed (k={k}, q={q}): {type(e).__name__}: {e}")
            recon_all_ok = False
            continue
        U_k, S_k, V_k = U_k[..., :kk], S_k[..., :kk], V_k[..., :kk]
        # M ~= U diag(S) V^T   (svd_lowrank returns V, not Vh)
        M_hat = ((U_k * S_k[..., None, :]) @ V_k.transpose(-2, -1)).to(dtype=dtype)
        diff = M_batch - M_hat
        diff_flat = diff.flatten(-2).to(dtype=torch.float64)
        rel_fro = diff_flat.pow(2).sum(dim=-1) / fro_norm2
        max_abs = diff.flatten(-2).abs().amax(dim=-1).to(dtype=torch.float64)
        mean_abs = diff.flatten(-2).abs().mean(dim=-1).to(dtype=torch.float64)
        cos_flat = F.cosine_similarity(M_flat.to(dtype=torch.float64),
                                       M_hat.flatten(-2).to(dtype=torch.float64), dim=-1)
        recon[k] = {
            "rel_fro_mse": rel_fro.cpu(),
            "max_abs": max_abs.cpu(),
            "mean_abs": mean_abs.cpu(),
            "cos_flat": cos_flat.cpu(),
        }

    del M_batch, R_batch, M_lr
    if device == "cuda":
        torch.cuda.empty_cache()

    fro_norm_cpu = fro_norm.cpu()
    elem_std_cpu = elem_std.to(dtype=torch.float64).cpu()

    input_id = prompt.get("input_id", prompt["id"])
    input_kind = prompt["kind"]
    prompt_id = prompt["id"]
    rows: list[dict] = []
    sv_state_list: list = []
    sv_random_list: list = []
    sv_index: list[dict] = []
    for idx, (li, h) in enumerate(layer_head_meta):
        row: dict = {
            "model": model_label, "family": family,
            "input_id": input_id, "input_kind": input_kind, "prompt_id": prompt_id,
            "position": position, "actual_tokens": actual_tokens,
            "layer": li, "head": h,
            "D_k": D_k, "D_v": D_v, "max_rank": max_rank,
            "fro_norm_state": float(fro_norm_cpu[idx].item()),
            "elem_std_state": float(elem_std_cpu[idx].item()),
            "spectrum_ok": spectrum_ok,
        }
        for k in k_grid:
            row[f"energy_at_k{k}"] = (
                float(energy_S_at_k[k][idx].item()) if energy_S_at_k[k] is not None else float("nan")
            )
            row[f"energy_at_k{k}_random"] = (
                float(energy_R_at_k[k][idx].item()) if energy_R_at_k[k] is not None else float("nan")
            )
        for thr in energy_thresholds:
            tag = str(int(round(thr * 1000)))
            row[f"num_rank{tag}"] = (
                int(num_rank_S[thr][idx].item()) if num_rank_S[thr] is not None else -1
            )
            row[f"num_rank{tag}_random"] = (
                int(num_rank_R[thr][idx].item()) if num_rank_R[thr] is not None else -1
            )
        for k in k_grid:
            if recon[k] is not None:
                row[f"rel_fro_mse_k{k}"] = float(recon[k]["rel_fro_mse"][idx].item())
                row[f"max_abs_k{k}"] = float(recon[k]["max_abs"][idx].item())
                row[f"mean_abs_k{k}"] = float(recon[k]["mean_abs"][idx].item())
                row[f"cos_flat_k{k}"] = float(recon[k]["cos_flat"][idx].item())
            else:
                row[f"rel_fro_mse_k{k}"] = float("nan")
                row[f"max_abs_k{k}"] = float("nan")
                row[f"mean_abs_k{k}"] = float("nan")
                row[f"cos_flat_k{k}"] = float("nan")
        row["recon_all_ok"] = recon_all_ok
        rows.append(row)
        sv_state_list.append(S_cpu[idx] if S_cpu is not None else None)
        sv_random_list.append(S_R_cpu[idx] if S_R_cpu is not None else None)
        sv_index.append({
            "input_id": input_id, "input_kind": input_kind, "prompt_id": prompt_id,
            "position": position, "actual_tokens": actual_tokens,
            "layer": li, "head": h,
        })
    return rows, sv_state_list, sv_random_list, sv_index

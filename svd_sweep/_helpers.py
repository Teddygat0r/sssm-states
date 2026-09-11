"""
Shared helpers for the state-level SVD sweep (Sections 1.1 + 1.2).

We compute, per (model, input, position, layer, head), one SVD of the SSM
state tensor S_t of shape [D_k, D_v] and one SVD of an iid-Gaussian random
matrix R of the same shape and matched element-wise variance.

From those spectra we derive:
  - cumulative energy fraction at rank k for k in K_GRID
  - effective rank thresholds (99 / 99.5 / 99.9 percent of energy)
  - relative Frobenius MSE, per-element max-abs and mean-abs error,
    flat cosine similarity for truncated-SVD reconstruction at each rank
  - a random-matrix control spectrum (Marchenko-Pastur baseline)

Per-record outputs are written to a CSV and the full spectra are written to
a torch `.pt` file so that downstream plotting can do CDFs / overlays.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


# Number of worker threads for the (input, position) task pool.  Threads share
# the loaded model; CPU batched SVDs run truly in parallel because PyTorch BLAS
# releases the GIL.  Forward-pass GPU work serializes naturally on the device.
NUM_WORKERS = max(1, int(os.getenv("SVD_PARALLEL_WORKERS", "4")))


SVD_DEVICE = os.getenv("SVD_DEVICE", "cpu")
# float64 instead of float32: some Nemotron Mamba states have abs_max ~ 1e-8,
# which trips SGESDD's SLASCL scaling and aborts the process with Fortran STOP 1.
# fp64 LAPACK handles the same matrices cleanly.
#
# Note on device: on this aarch64/Blackwell box CPU fp64 batched SVD is
# ~13x faster than GPU fp64 (~4s vs ~53s for [3072, 64, 128]). cuSOLVER
# fp64 falls back to per-matrix gesvd which is launch-bound at this size.
# Set SVD_DEVICE=cuda only if you are running with much larger matrices
# (or fp32) where the GPU path actually wins.
SVD_DTYPE = torch.float64

# Ranks at which to report cumulative energy / reconstruction error.
K_GRID = (1, 2, 4, 8, 16, 32, 64)

# Energy thresholds for the "effective rank" metric (smallest k whose
# cumulative squared-singular-value energy exceeds the threshold).
ENERGY_THRESHOLDS = (0.99, 0.995, 0.999)

# Positions (in tokens) at which we extract the recurrent state.  "full"
# means use the entire available prompt and is added as -1 elsewhere.
POSITIONS_FULL = (256, 512, 1024, 2048, 4096)
POSITIONS_SUBSET = (256, 1024, 4096)


# -----------------------------------------------------------------------------
# spectrum metrics
# -----------------------------------------------------------------------------

def singular_values(matrix: torch.Tensor) -> torch.Tensor | None:
    """Full singular-value spectrum, returned on CPU/float32 sorted descending.

    Returns None if the SVD raises (LinAlgError, NaN/Inf input, etc.). Callers
    must treat None as "skip this matrix's spectrum metrics".
    """
    M = matrix.detach().to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    try:
        return torch.linalg.svdvals(M)
    except Exception as e:
        print(f"    WARN: svdvals failed on shape {tuple(M.shape)}: {type(e).__name__}: {e}")
        return None


def full_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """One full SVD of `matrix`. Returns (U, S, Vh) on CPU/float32, or None on
    failure. Cached and reused for every k in K_GRID -- no per-k re-decomposition.
    """
    M = matrix.detach().to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U, S, Vh
    except Exception as e:
        print(f"    WARN: svd failed on shape {tuple(M.shape)}: {type(e).__name__}: {e}")
        return None


def cumulative_energy(sv: torch.Tensor) -> torch.Tensor:
    """Cumulative fraction of squared singular-value energy."""
    s2 = sv.to(dtype=torch.float64).pow(2)
    total = s2.sum().clamp_min(1e-30)
    return torch.cumsum(s2, dim=0) / total


def energy_at_rank(sv: torch.Tensor, k: int) -> float:
    if k <= 0:
        return 0.0
    cum = cumulative_energy(sv)
    if k >= cum.numel():
        return float(cum[-1].item())
    return float(cum[k - 1].item())


def num_rank_at_threshold(sv: torch.Tensor, threshold: float) -> int:
    cum = cumulative_energy(sv)
    idx = (cum >= threshold).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return int(sv.numel())
    return int(idx[0].item()) + 1


def truncated_reconstruct_from_svd(
    U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, k: int
) -> torch.Tensor:
    """Slice a cached full SVD to a rank-k reconstruction. No new SVD."""
    kk = max(1, min(k, S.numel()))
    return (U[:, :kk] * S[:kk]) @ Vh[:kk]


def reconstruction_metrics_from_svd(
    M: torch.Tensor,
    svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    k: int,
) -> dict:
    """Rank-k reconstruction error using a *cached* SVD of M.

    On SVD failure (svd is None) we "return the original state":
        M_hat = M  -> rel_fro=0, max_abs=0, mean_abs=0, cos=1.0
    and the row carries svd_ok=False so failures are traceable downstream.
    """
    M = M.to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    if svd is None:
        M_hat = M.clone()
        ok = False
    else:
        U, S, Vh = svd
        M_hat = truncated_reconstruct_from_svd(U, S, Vh, k)
        ok = True
    diff = M - M_hat
    M_norm2 = M.to(dtype=torch.float64).pow(2).sum().clamp_min(1e-30)
    rel_fro = float((diff.to(dtype=torch.float64).pow(2).sum() / M_norm2).item())
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    cos = float(
        F.cosine_similarity(M.flatten().unsqueeze(0), M_hat.flatten().unsqueeze(0)).item()
    )
    return {
        "rel_fro_mse": rel_fro,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cos_flat": cos,
        "svd_ok": ok,
    }


def random_matched_matrix(matrix: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """iid Gaussian matrix with element-wise std matched to `matrix`."""
    M = matrix.detach().to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    sigma = float(M.std(unbiased=False).item())
    return torch.randn(M.shape, generator=generator, dtype=SVD_DTYPE) * sigma


# -----------------------------------------------------------------------------
# record schema
# -----------------------------------------------------------------------------

@dataclass
class StateRecord:
    """One state matrix S_t of shape [D_k, D_v] tagged with metadata."""

    model: str
    family: str
    input_id: str          # bucket: "sharegpt" / "code_thestack"
    input_kind: str        # "chat" / "code"
    prompt_id: str         # unique per source document (e.g., "sharegpt_0042")
    position: int          # -1 for "full"
    actual_tokens: int     # number of tokens prefilled to produce this state
    layer: int
    head: int
    matrix: torch.Tensor   # [D_k, D_v]


# -----------------------------------------------------------------------------
# Batched per-task compute (GPU)
# -----------------------------------------------------------------------------

def _batched_cumulative_energy(sv: torch.Tensor) -> torch.Tensor:
    """sv: [..., k] singular values sorted desc. Returns cumulative squared
    energy fraction along the last dim."""
    s2 = sv.to(dtype=torch.float64).pow(2)
    total = s2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.cumsum(s2, dim=-1) / total


def _batched_num_rank_at_threshold(cum_energy: torch.Tensor, threshold: float) -> torch.Tensor:
    """cum_energy: [N, k]. Returns the smallest k_i for which cum >= threshold,
    or k if no rank reaches the threshold."""
    K = cum_energy.shape[-1]
    mask = cum_energy >= threshold
    any_hit = mask.any(dim=-1)
    first = mask.int().argmax(dim=-1)
    return torch.where(any_hit, first + 1, torch.full_like(first, K))


def compute_task_metrics_batched(
    states_by_layer: dict[int, torch.Tensor],
    *,
    model_label: str, family: str, prompt: dict,
    position: int, actual_tokens: int,
    seed: int,
    k_grid: tuple[int, ...] = K_GRID,
    energy_thresholds: tuple[float, ...] = ENERGY_THRESHOLDS,
    device: str = SVD_DEVICE,
    dtype: torch.dtype = SVD_DTYPE,
) -> tuple[list[dict], list[torch.Tensor | None], list[torch.Tensor | None], list[dict]]:
    """ONE batched SVD per task instead of `num_layers * num_heads` per-matrix calls.

    Stacks every per-(layer, head) state matrix into a `[N, D_k, D_v]` tensor,
    moves it to `device`, runs a single `torch.linalg.svd` (state) and a single
    `torch.linalg.svdvals` (matched random control), then computes all
    cumulative-energy / effective-rank / reconstruction-error metrics in
    vectorised batched form. Per-(layer, head) rows are reconstructed at the
    end so the CSV / spectra schema is unchanged.
    """
    layer_indices = sorted(states_by_layer.keys())
    chunks: list[torch.Tensor] = []
    layer_head_meta: list[tuple[int, int]] = []
    for li in layer_indices:
        s = states_by_layer[li]                 # [H, D_k, D_v] on CPU/float32
        chunks.append(s)
        for h in range(int(s.shape[0])):
            layer_head_meta.append((li, h))
    if not chunks:
        return [], [], [], []
    M_batch = torch.cat(chunks, dim=0).to(device=device, dtype=dtype)  # [N, D_k, D_v]
    N, D_k, D_v = M_batch.shape
    max_rank = min(D_k, D_v)

    # Matched random control: iid Gaussian with element-wise std per matrix.
    elem_std = M_batch.flatten(-2).std(dim=-1, unbiased=False)  # [N]
    g = torch.Generator(device=device).manual_seed(int(seed))
    R_batch = torch.randn(M_batch.shape, generator=g, dtype=dtype, device=device) \
              * elem_std.view(N, 1, 1)

    # Two GPU calls per task: batched SVD on state, batched svdvals on random.
    spectrum_ok = True
    try:
        U, S, Vh = torch.linalg.svd(M_batch, full_matrices=False)   # S: [N, k]
    except Exception as e:
        print(f"    WARN: batched SVD failed on shape {tuple(M_batch.shape)}: {type(e).__name__}: {e}")
        U = S = Vh = None
        spectrum_ok = False
    try:
        S_R = torch.linalg.svdvals(R_batch)                          # [N, k]
    except Exception as e:
        print(f"    WARN: batched svdvals failed on random: {type(e).__name__}: {e}")
        S_R = None
        spectrum_ok = False

    # Per-record state norms (used for relative Frobenius MSE).
    M_flat = M_batch.flatten(-2)
    fro_norm = M_flat.to(dtype=torch.float64).norm(dim=-1)           # [N]
    fro_norm2 = M_flat.to(dtype=torch.float64).pow(2).sum(dim=-1).clamp_min(1e-30)

    # 1.1 -- cumulative energies + thresholds (batched).
    if S is not None:
        cum_S = _batched_cumulative_energy(S)
        num_rank_S = {thr: _batched_num_rank_at_threshold(cum_S, thr).cpu()
                      for thr in energy_thresholds}
        energy_S_at_k: dict[int, torch.Tensor] = {}
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
        energy_R_at_k: dict[int, torch.Tensor] = {}
        for k in k_grid:
            kk = max(1, min(k, S_R.shape[-1]))
            energy_R_at_k[k] = cum_R[..., kk - 1].cpu()
        S_R_cpu = S_R.cpu()
    else:
        num_rank_R = {thr: None for thr in energy_thresholds}
        energy_R_at_k = {k: None for k in k_grid}
        S_R_cpu = None

    # 1.2 -- truncated reconstruction error per k, all batched on GPU.
    recon = {k: None for k in k_grid}
    recon_all_ok = U is not None and S is not None and Vh is not None
    if recon_all_ok:
        for k in k_grid:
            kk = max(1, min(k, S.shape[-1]))
            M_hat = (U[:, :, :kk] * S[:, None, :kk]) @ Vh[:, :kk, :]
            diff = M_batch - M_hat
            diff_flat = diff.flatten(-2).to(dtype=torch.float64)
            rel_fro = (diff_flat.pow(2).sum(dim=-1) / fro_norm2)
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

    # Free GPU intermediates.
    del U, Vh, M_batch, R_batch
    if device == "cuda":
        torch.cuda.empty_cache()

    fro_norm_cpu = fro_norm.cpu()
    elem_std_cpu = elem_std.to(dtype=torch.float64).cpu()

    # Reconstruct per-record rows.
    input_id = prompt.get("input_id", prompt["id"])
    input_kind = prompt["kind"]
    prompt_id = prompt["id"]
    rows: list[dict] = []
    sv_state_list: list[torch.Tensor | None] = []
    sv_random_list: list[torch.Tensor | None] = []
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


def compute_record_metrics(
    rec: StateRecord, generator: torch.Generator, k_grid: Iterable[int] = K_GRID
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Run SVD + random control + reconstruction at every k in k_grid.

    Returns (csv_row, sv_state[max_rank], sv_random[max_rank]).
    """
    M = rec.matrix.to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    R = random_matched_matrix(M, generator)

    # ONE full SVD per state matrix.  S is reused for every Section-1.1 metric
    # (cumulative energy, effective-rank thresholds) and U,S,Vh are sliced for
    # every k in K_GRID for Section-1.2 reconstruction.
    svd_M = full_svd(M)
    sv_S = svd_M[1] if svd_M is not None else None
    # The random control only needs singular values (no reconstruction on it).
    sv_R = singular_values(R)
    max_rank = int(min(M.shape))
    spectrum_ok = sv_S is not None and sv_R is not None

    row: dict = {
        "model": rec.model,
        "family": rec.family,
        "input_id": rec.input_id,
        "input_kind": rec.input_kind,
        "prompt_id": rec.prompt_id,
        "position": rec.position,
        "actual_tokens": rec.actual_tokens,
        "layer": rec.layer,
        "head": rec.head,
        "D_k": int(M.shape[0]),
        "D_v": int(M.shape[1]),
        "max_rank": max_rank,
        "fro_norm_state": float(M.to(dtype=torch.float64).norm().item()),
        "elem_std_state": float(M.std(unbiased=False).item()),
        "spectrum_ok": spectrum_ok,
    }
    for k in k_grid:
        row[f"energy_at_k{k}"] = energy_at_rank(sv_S, k) if sv_S is not None else float("nan")
        row[f"energy_at_k{k}_random"] = (
            energy_at_rank(sv_R, k) if sv_R is not None else float("nan")
        )
    for thr in ENERGY_THRESHOLDS:
        tag = str(int(round(thr * 1000)))   # 990 / 995 / 999
        row[f"num_rank{tag}"] = (
            num_rank_at_threshold(sv_S, thr) if sv_S is not None else -1
        )
        row[f"num_rank{tag}_random"] = (
            num_rank_at_threshold(sv_R, thr) if sv_R is not None else -1
        )
    all_recon_ok = svd_M is not None
    for k in k_grid:
        m = reconstruction_metrics_from_svd(M, svd_M, k)
        all_recon_ok = all_recon_ok and bool(m["svd_ok"])
        row[f"rel_fro_mse_k{k}"] = m["rel_fro_mse"]
        row[f"max_abs_k{k}"] = m["max_abs"]
        row[f"mean_abs_k{k}"] = m["mean_abs"]
        row[f"cos_flat_k{k}"] = m["cos_flat"]
    row["recon_all_ok"] = all_recon_ok
    return row, sv_S, sv_R


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_spectra(path: Path, sv_state: list[torch.Tensor | None],
                 sv_random: list[torch.Tensor | None],
                 index: list[dict]) -> None:
    """Save the full spectra + their (model, input, pos, layer, head) index.

    Entries where the SVD failed are stored as a NaN row (so downstream code
    can mask them out via torch.isnan)."""
    if not sv_state:
        return
    max_rank = max(int(s.numel()) for s in sv_state if s is not None)
    n = len(sv_state)
    state_mat = torch.full((n, max_rank), float("nan"), dtype=torch.float32)
    rand_mat = torch.full((n, max_rank), float("nan"), dtype=torch.float32)
    for i, (s, r) in enumerate(zip(sv_state, sv_random)):
        if s is not None:
            state_mat[i, : s.numel()] = s.to(dtype=torch.float32)
        if r is not None:
            rand_mat[i, : r.numel()] = r.to(dtype=torch.float32)
    torch.save({"state": state_mat, "random": rand_mat, "index": index}, path)


# -----------------------------------------------------------------------------
# prompt loaders -- two distributions per Section 1.1
# -----------------------------------------------------------------------------

def _try_load(loader: Callable[[], list[dict]], label: str) -> list[dict]:
    try:
        out = loader()
        print(f"  loaded {label}: {len(out)} prompt(s)")
        return out
    except Exception as e:
        print(f"  WARN: failed to load {label}: {type(e).__name__}: {e}")
        return []


N_PROMPTS = max(1, int(os.getenv("SWEEP_N_PROMPTS", "1")))
# Each built prompt accumulates source documents until it crosses
# MIN_CHARS_PER_PROMPT; ~24000 chars is enough that most prompts will exceed
# 4096 tokens once tokenized (English / Python both run ~3-4 chars/tok).
MIN_CHARS_PER_PROMPT = int(os.getenv("SWEEP_MIN_CHARS_PER_PROMPT", "24000"))


def _accumulate_prompts(text_iter: Iterable[str], n_prompts: int,
                         min_chars_per_prompt: int, joiner: str) -> list[str]:
    """Greedily group an iterable of source documents into `n_prompts` strings,
    each at least `min_chars_per_prompt` characters. Returns fewer strings if
    the iterator is exhausted early."""
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for t in text_iter:
        if not t:
            continue
        cur.append(t)
        cur_len += len(t)
        if cur_len >= min_chars_per_prompt:
            out.append(joiner.join(cur))
            if len(out) >= n_prompts:
                return out
            cur = []
            cur_len = 0
    if cur and len(out) < n_prompts and cur_len >= min_chars_per_prompt // 2:
        out.append(joiner.join(cur))
    return out


def _build_sharegpt(n_prompts: int = N_PROMPTS,
                    min_chars: int = MIN_CHARS_PER_PROMPT) -> list[dict]:
    """Build `n_prompts` ShareGPT prompts, each a concatenation of consecutive
    conversations summing to >= min_chars characters. Falls back to OpenAssistant
    if the ShareGPT mirror is unreachable."""
    from datasets import load_dataset

    def _iter_sharegpt():
        ds = load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V4.3_unfiltered_cleaned_split.json",
            split="train",
        )
        for ex in ds:
            convo = ex.get("conversations") or ex.get("conversation") or []
            text = "\n\n".join(c.get("value", "") for c in convo if isinstance(c, dict))
            if text:
                yield text

    def _iter_oasst():
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        for ex in ds:
            t = ex.get("text") or ""
            if t:
                yield t

    try:
        texts = _accumulate_prompts(_iter_sharegpt(), n_prompts, min_chars, "\n\n=====\n\n")
    except Exception as e:
        print(f"    ShareGPT load failed ({type(e).__name__}: {e}); falling back to oasst1")
        texts = _accumulate_prompts(_iter_oasst(), n_prompts, min_chars, "\n\n=====\n\n")
    if not texts:
        raise RuntimeError("no ShareGPT prompts built")
    return [{"id": f"sharegpt_{i:04d}", "input_id": "sharegpt",
             "kind": "chat", "title": f"ShareGPT chunk {i}", "text": t}
            for i, t in enumerate(texts)]


def _build_code(n_prompts: int = N_PROMPTS,
                min_chars: int = MIN_CHARS_PER_PROMPT) -> list[dict]:
    """Build `n_prompts` code prompts. Prefer The Stack v2; fall back to
    codeparrot/codeparrot-clean (public Python from GitHub) on failure."""
    from datasets import load_dataset

    def _iter_v2():
        ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
        for ex in ds:
            t = ex.get("content") or ex.get("text") or ""
            if t:
                yield t

    def _iter_codeparrot():
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        for ex in ds:
            t = ex.get("content") or ex.get("text") or ""
            if t:
                yield t

    try:
        texts = _accumulate_prompts(_iter_v2(), n_prompts, min_chars, "\n\n# ----- # \n\n")
    except Exception as e:
        print(f"    The Stack v2 load failed ({type(e).__name__}: {e}); "
              f"falling back to codeparrot/codeparrot-clean (public Python from GitHub)")
        texts = _accumulate_prompts(_iter_codeparrot(), n_prompts, min_chars, "\n\n# ----- # \n\n")
    if not texts:
        raise RuntimeError("no code prompts built")
    return [{"id": f"code_thestack_{i:04d}", "input_id": "code_thestack",
             "kind": "code", "title": f"code chunk {i}", "text": t}
            for i, t in enumerate(texts)]


def load_prompts() -> list[dict]:
    """Two input distributions: ShareGPT-style chat, and code from The Stack.
    Returns N_PROMPTS prompts per distribution (env var SWEEP_N_PROMPTS, default 1)."""
    out: list[dict] = []
    out.extend(_try_load(_build_sharegpt, "ShareGPT"))
    out.extend(_try_load(_build_code, "code"))
    if not out:
        raise RuntimeError("no prompt sources succeeded")
    return out


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

@dataclass
class Task:
    """One independent unit of work: do a prefill at position T for one prompt."""

    prompt: dict
    position: int            # -1 == full
    actual_tokens: int       # T
    ids: torch.Tensor        # shape [T]


def _task_seed(prompt_id: str, position: int) -> int:
    """Deterministic per-task seed so random-control matrices are reproducible
    regardless of thread-completion order."""
    h = abs(hash((prompt_id, int(position)))) % (2 ** 31)
    return int(h)


def _process_task(
    task: Task,
    *,
    model_label: str,
    family: str,
    state_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
    forward_lock,
    progress_prefix: str = "",
    metrics_fn: Callable = compute_task_metrics_batched,
) -> tuple[list[dict], list, list, list[dict]]:
    """One thread-pool worker: forward-pass + per-(layer, head) SVD metrics.

    state_fn(ids) is the per-model adapter that runs the prefill and returns
    {layer_idx: state_tensor [H, D_k, D_v]} on CPU/float32.

    metrics_fn is the per-task metric kernel; it defaults to the exact batched
    SVD `compute_task_metrics_batched` but can be swapped (e.g. for a randomized
    `svd_lowrank` reconstruction) as long as it keeps the same signature/outputs.

    Returns (rows, sv_state_list, sv_random_list, sv_index_list).
    """
    seed = _task_seed(task.prompt["id"], task.position)

    t0 = perf_counter()
    with forward_lock:
        states = state_fn(task.ids)
    fwd_s = perf_counter() - t0

    t1 = perf_counter()
    rows, sv_state_list, sv_random_list, sv_index = metrics_fn(
        states_by_layer=states,
        model_label=model_label, family=family,
        prompt=task.prompt, position=task.position, actual_tokens=task.actual_tokens,
        seed=seed,
    )
    svd_s = perf_counter() - t1
    print(
        f"{progress_prefix}[{task.prompt['id']} pos={task.position} T={task.actual_tokens}]"
        f"  fwd={fwd_s:.2f}s  svd={svd_s:.2f}s  records={len(rows)}"
    )
    return rows, sv_state_list, sv_random_list, sv_index


def run_tasks_parallel(
    tasks: list[Task],
    *,
    model_label: str,
    family: str,
    state_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
    num_workers: int = NUM_WORKERS,
    metrics_fn: Callable = compute_task_metrics_batched,
) -> tuple[list[dict], list, list, list[dict]]:
    """Submit all tasks to a thread pool.  Returns merged (rows, sv_state, sv_random, sv_index).

    Pattern matches state_spectrum_sweep/run_experiment_kl_parallel.py:
    model is loaded once and shared across workers; futures are gathered with
    as_completed; per-task results are concatenated in the main thread.

    metrics_fn lets a caller substitute a different per-task metric kernel
    (default: exact batched SVD). It must match compute_task_metrics_batched's
    signature and return schema.
    """
    import threading
    forward_lock = threading.Lock()
    workers = max(1, min(num_workers, len(tasks)))
    print(f"\nRunning {len(tasks)} tasks across {workers} worker threads "
          f"(SVD_PARALLEL_WORKERS={NUM_WORKERS})")

    rows: list[dict] = []
    sv_state_all: list = []
    sv_random_all: list = []
    sv_index_all: list[dict] = []

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, task in enumerate(tasks, start=1):
            fut = ex.submit(
                _process_task, task,
                model_label=model_label, family=family,
                state_fn=state_fn, forward_lock=forward_lock,
                progress_prefix=f"  [{i:>3}/{len(tasks)}] ",
                metrics_fn=metrics_fn,
            )
            futures[fut] = i
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                r, sS, sR, idx = fut.result()
            except Exception as exc:
                i = futures[fut]
                print(f"  task {i} FAILED: {type(exc).__name__}: {exc}")
                continue
            rows.extend(r)
            sv_state_all.extend(sS)
            sv_random_all.extend(sR)
            sv_index_all.extend(idx)

    return rows, sv_state_all, sv_random_all, sv_index_all


def make_run_dir(root: Path, prefix: str) -> Path:
    from datetime import datetime

    run_dir = root / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_token_count(prompt: dict, tokenizer, min_tokens: int) -> torch.Tensor:
    """Tokenize `prompt['text']` and return the full id tensor. The `min_tokens`
    arg is informational: short prompts are returned as-is and the caller is
    responsible for filtering positions that don't fit. (We used to raise; with
    SWEEP_N_PROMPTS > 1 not every chunk is guaranteed to reach the longest
    target, and we'd rather drop those positions than kill the run.)"""
    ids = tokenizer(prompt["text"], return_tensors="pt").input_ids[0]
    return ids


MAX_FULL_TOKENS = int(os.getenv("SWEEP_MAX_FULL_TOKENS", "65536"))


def build_tasks(prompts: list[dict], tokenizer, positions_target: tuple[int, ...]) -> list["Task"]:
    """Build (prompt × position) tasks. Each prompt produces one task per
    position in `positions_target` that fits, plus a `-1` ("full") task if the
    prompt is strictly longer than the largest target. The full position is
    capped at MAX_FULL_TOKENS (default 65536) so very long prompts (>~100k
    tokens) don't trip Triton SSD-scan illegal-memory-access bugs in mamba_ssm
    on Nemotron's hybrid path. Prompts shorter than the smallest target are
    skipped with a warning."""
    longest = max(positions_target)
    tasks: list[Task] = []
    skipped = 0
    for prompt in prompts:
        ids_full = ensure_token_count(prompt, tokenizer, min_tokens=longest)
        full_len = int(ids_full.shape[0])
        full_capped = min(full_len, MAX_FULL_TOKENS)
        positions = [p for p in positions_target if p <= full_capped]
        if not positions:
            skipped += 1
            continue
        if full_capped > positions[-1]:
            positions.append(-1)
        for pos in positions:
            T = full_capped if pos == -1 else pos
            tasks.append(Task(prompt=prompt, position=pos, actual_tokens=T, ids=ids_full[:T]))
    if skipped:
        print(f"  WARN: skipped {skipped} prompt(s) shorter than the smallest target position {positions_target[0]}")
    return tasks


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

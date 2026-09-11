"""
Decision-rule data: chol_mp vs chol_mp2 on (a) Qwen3.5-4B states and (b)
a synthetic rank-deficient input designed to force the fp64 Cholesky to
raise.

Measured per variant:
  1. raise rate
  2. per-call wall-clock
  3. reconstruction error (mean, p99, worst gap vs dense ground truth)
  4. orthonormality defect of U: ||UᵀU - I||_F / sqrt(rank), per state.
     This is the key differentiator — chol_mp2 does two CholQR passes, so
     it should be at fp32 machine epsilon. chol_mp's single pass might
     leave a larger residual, especially on ill-conditioned input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

NEW_RSVD = Path("/home/joshuaz/kv-svd/svd_methods")
sys.path.insert(0, str(NEW_RSVD))
from rsvd_eigh import _ORTH_FNS  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16
RANK = 16
OVERSAMPLE = 8


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _rsvd(A, *, orth, rank=RANK, n_iter=2, oversample=OVERSAMPLE, seed=0):
    orth_fn = _ORTH_FNS[orth]
    m, n = A.shape[-2], A.shape[-1]
    q = min(rank + oversample, m, n)
    torch.manual_seed(seed)
    Omega = torch.randn(n, q, dtype=torch.float32, device=A.device)
    Af = A.float()
    Y = Af @ Omega
    Q = orth_fn(Y)
    for _ in range(n_iter):
        Z = Af.transpose(-2, -1) @ Q
        Y = Af @ Z
        Q = orth_fn(Y)
    Bproj = Q.transpose(-2, -1) @ Af
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(C)
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def _per_state_recon_err(A, U, S, Vh):
    recon = (U * S.unsqueeze(-2)) @ Vh
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return num / den


@torch.no_grad()
def _orth_defect(U):
    k = U.shape[-1]
    G = U.transpose(-2, -1) @ U
    eye = torch.eye(k, device=U.device, dtype=U.dtype)
    diff = (G - eye).reshape(U.shape[0], -1).norm(dim=-1)
    return (diff / (k ** 0.5)).cpu()


def _find_linear_layers(model):
    out = []
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            out.append(li)
    return out


def _format_mmlu(ex):
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    return (f"The following is a multiple choice question about "
            f"{ex['subject'].replace('_', ' ')}.\n\n"
            f"Question: {ex['question']}\n\n{body}\n\nAnswer:")


def time_fn(fn, A, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        try:
            fn(A)
        except Exception:
            return float("nan"), True
    _sync()
    t0 = perf_counter()
    raised = False
    for _ in range(n_iter):
        try:
            fn(A)
        except Exception:
            raised = True
            break
    _sync()
    return (perf_counter() - t0) / n_iter * 1000, raised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    linear_layers = _find_linear_layers(model)

    ds = load_dataset("cais/mmlu", "all", split="test")
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(ds), generator=g)[: args.n_prompts].tolist()

    methods = ["chol_mp", "chol_mp2"]
    err_means = {m: [] for m in methods}
    gap_maxes = {m: [] for m in methods}
    defect_meds = {m: [] for m in methods}
    defect_maxes = {m: [] for m in methods}
    raises = {m: 0 for m in methods}
    times = {m: [] for m in methods}

    # ---- Part 1: real Qwen3.5-4B states ---------------------------------
    print("\n" + "=" * 100)
    print(f"Part 1 — real Qwen3.5-4B recurrent states, {args.n_prompts} MMLU prompts")
    print("=" * 100)
    for pi, i in enumerate(idxs):
        ex = ds[i]
        prompt = _format_mmlu(ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.max_seq_len].to(DEVICE)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        _sync()
        cache = out.past_key_values
        states = torch.cat([cache.layers[li].recurrent_states.squeeze(0).detach()
                             for li in linear_layers], dim=0).float()
        del out, cache

        gtU, gtS, gtVh = torch.linalg.svd(states, full_matrices=False)
        gt_err = _per_state_recon_err(states, gtU[..., :RANK], gtS[..., :RANK], gtVh[..., :RANK, :])
        del gtU, gtS, gtVh

        row_info = f"[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})"

        for m in methods:
            try:
                ms, _ = time_fn(lambda A, m=m: _rsvd(A, orth=m), states)
                times[m].append(ms)
                # Accuracy + orth defect (single trial).
                U, S, Vh = _rsvd(states, orth=m)
                err = _per_state_recon_err(states, U, S, Vh).cpu()
                gap = err - gt_err.cpu()
                defect = _orth_defect(U)
                err_means[m].append(float(err.mean()))
                gap_maxes[m].append(float(gap.max()))
                defect_meds[m].append(float(defect.median()))
                defect_maxes[m].append(float(defect.max()))
            except Exception as e:
                raises[m] += 1
                print(f"  {row_info}  {m}: RAISED {type(e).__name__}")

        print(f"  {row_info}")
        for m in methods:
            print(f"    {m:<10}  t={times[m][-1]:6.2f} ms  "
                  f"err_mean={err_means[m][-1]:.4f}  gap_max={gap_maxes[m][-1]:+.4f}  "
                  f"orth_def(median)={defect_meds[m][-1]:.2e}  "
                  f"orth_def(max)={defect_maxes[m][-1]:.2e}")
        del states, gt_err

    # ---- Part 2: synthetic rank-deficient input -------------------------
    print()
    print("=" * 100)
    print("Part 2 — synthetic truly rank-deficient input (σ_q = 0 exactly)")
    print("=" * 100)
    # Build a [768, 128, 128] batch where each state is a random rank-r matrix.
    # For r < q=24, the Gram of the sketch is exactly rank-r, and fp64
    # Cholesky should still raise on it.
    rng = torch.Generator(device=DEVICE).manual_seed(0)
    for r in [4, 8, 16, 20, 24]:
        # Generate random rank-r matrices: A = L @ R where L is [128, r], R is [r, 128].
        L = torch.randn(768, 128, r, generator=rng, device=DEVICE, dtype=torch.float32)
        R = torch.randn(768, r, 128, generator=rng, device=DEVICE, dtype=torch.float32)
        A = L @ R
        # Ground truth.
        gtU, gtS, gtVh = torch.linalg.svd(A, full_matrices=False)
        # Truncate to min(r, RANK) — what we should recover.
        kk = min(r, RANK)
        gt_err = _per_state_recon_err(A, gtU[..., :kk], gtS[..., :kk], gtVh[..., :kk, :])
        for m in methods:
            try:
                U, S, Vh = _rsvd(A, orth=m, rank=kk, oversample=OVERSAMPLE)
                err = _per_state_recon_err(A, U, S, Vh).cpu()
                gap = err - gt_err.cpu()
                defect = _orth_defect(U)
                tag = (f"ok   err_mean={float(err.mean()):.4f}  "
                       f"gap_max={float(gap.max()):+.4f}  "
                       f"orth_def(max)={float(defect.max()):.2e}")
            except Exception as e:
                tag = f"RAISED {type(e).__name__}: {str(e)[:80]}"
            print(f"  true_rank={r:>2d}  {m:<10}  {tag}")

    # ---- Aggregate Part 1 ----------------------------------------------
    def avg(xs): return sum(xs)/len(xs) if xs else float("nan")
    print()
    print("=" * 100)
    print(f"Part 1 aggregate over {args.n_prompts} prompts")
    print("=" * 100)
    print(f"{'variant':<10}{'raise':>8}{'time_ms':>10}{'err_mean':>12}"
          f"{'gap_max':>12}{'orth_def_med':>14}{'orth_def_max':>14}")
    for m in methods:
        print(f"{m:<10}{raises[m]:>6}/{args.n_prompts:<2}{avg(times[m]):>10.2f}"
              f"{avg(err_means[m]):>12.4f}{max(gap_maxes[m]) if gap_maxes[m] else float('nan'):>+12.4f}"
              f"{avg(defect_meds[m]):>14.2e}{max(defect_maxes[m]) if defect_maxes[m] else float('nan'):>14.2e}")


if __name__ == "__main__":
    main()

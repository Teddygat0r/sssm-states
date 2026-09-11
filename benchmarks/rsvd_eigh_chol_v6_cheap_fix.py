"""
Test the cheap fix for chol_v6 fp32 raises: add a tiny diagonal jitter to
`C = Bproj @ Bproj.T` before `eigh(C)`. Since we only keep the top-k
eigenvalues anyway, a uniform shift doesn't change the rank-k truncation.

Three variants timed against the baseline:
  (A) baseline                : current code, eigh(C)
  (B) try/except retry        : eigh(C); on raise, retry with jitter
  (C) proactive jitter        : always C += 1e-8 * mean_diag * I, then eigh
  (D) proactive jitter fp64   : promote C to fp64 + jitter then back

We also re-test svd-on-Bproj as the upper-correctness reference. For each
variant: raise count over 10 prompts, mean recon error vs baseline, and
mean wall-clock per call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsvd_eigh import _ORTH_FNS  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _build_Q_and_Bproj(A, *, rank=16, n_iter=2, oversample=8, orth="chol_v6", power_dtype=None, seed=0):
    """Build Q and Bproj — shared prologue for all final-step variants."""
    orth_fn = _ORTH_FNS[orth]
    m, n = A.shape[-2], A.shape[-1]
    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A
    torch.manual_seed(seed)
    Omega = torch.randn(n, q, dtype=Ap.dtype, device=A.device)
    Y = Ap @ Omega
    if pdtype != torch.float32:
        Y = Y.float()
    Q = orth_fn(Y)
    for _ in range(n_iter):
        Q_p = Q.to(Ap.dtype) if pdtype != torch.float32 else Q
        Z = Ap.transpose(-2, -1) @ Q_p
        Y = Ap @ Z
        if pdtype != torch.float32:
            Y = Y.float()
        Q = orth_fn(Y)
    Bproj = Q.transpose(-2, -1) @ A.float()
    return Q, Bproj


@torch.no_grad()
def _finalize_eigh(Q, Bproj, rank, jitter_factor=0.0):
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    if jitter_factor > 0:
        d = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-30)
        eye = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
        C = C + (jitter_factor * d).unsqueeze(-1) * eye
    evals, evecs = torch.linalg.eigh(C)
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def _finalize_eigh_retry(Q, Bproj, rank):
    """Try eigh(C); on raise, retry with proactive jitter."""
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    try:
        evals, evecs = torch.linalg.eigh(C)
    except torch.linalg.LinAlgError:
        d = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-30)
        eye = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
        C = C + (1e-6 * d).unsqueeze(-1) * eye
        evals, evecs = torch.linalg.eigh(C)
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def _finalize_eigh_fp64(Q, Bproj, rank, jitter_factor=1e-8):
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    Cd = C.double()
    if jitter_factor > 0:
        d = torch.diagonal(Cd, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-30)
        eye = torch.eye(Cd.shape[-1], device=Cd.device, dtype=Cd.dtype)
        Cd = Cd + (jitter_factor * d).unsqueeze(-1) * eye
    evals, evecs = torch.linalg.eigh(Cd)
    evals = evals.float(); evecs = evecs.float()
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def _finalize_svd(Q, Bproj, rank):
    Ub_small, Sk_full, Vh_full = torch.linalg.svd(Bproj, full_matrices=False)
    U = Q @ Ub_small[..., :, :rank]
    Sk = Sk_full[..., :rank]
    Vh = Vh_full[..., :rank, :]
    return U, Sk, Vh


@torch.no_grad()
def recon_err(A, U, S, Vh):
    recon = (U * S.unsqueeze(-2)) @ Vh
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return num / den


VARIANTS = [
    ("A_eigh_baseline",       lambda Q, B: _finalize_eigh(Q, B, 16, 0.0)),
    ("B_eigh_retry",          lambda Q, B: _finalize_eigh_retry(Q, B, 16)),
    ("C_eigh_jitter_1e-8",    lambda Q, B: _finalize_eigh(Q, B, 16, 1e-8)),
    ("C_eigh_jitter_1e-6",    lambda Q, B: _finalize_eigh(Q, B, 16, 1e-6)),
    ("D_eigh_fp64_jit_1e-8",  lambda Q, B: _finalize_eigh_fp64(Q, B, 16, 1e-8)),
    ("E_svd_on_Bproj",        lambda Q, B: _finalize_svd(Q, B, 16)),
]


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


def time_finalize(fn, Q, Bproj, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        try:
            fn(Q, Bproj)
        except Exception:
            return float("nan"), True
    _sync()
    t0 = perf_counter()
    raised = False
    for _ in range(n_iter):
        try:
            fn(Q, Bproj)
        except Exception:
            raised = True
            break
    _sync()
    return (perf_counter() - t0) / n_iter * 1000, raised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=10)
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

    print(f"Sampling {args.n_prompts} MMLU prompts ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(ds), generator=g)[:args.n_prompts].tolist()

    timings = {name: [] for name, _ in VARIANTS}
    raises = {name: 0 for name, _ in VARIANTS}
    rec_means = {name: [] for name, _ in VARIANTS}
    rec_max_gaps = {name: [] for name, _ in VARIANTS}

    for pi, i in enumerate(idxs):
        ex = ds[i]
        prompt = _format_mmlu(ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.max_seq_len].to(DEVICE)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        cache = out.past_key_values
        states = torch.cat([cache.layers[li].recurrent_states.squeeze(0).detach()
                             for li in linear_layers], dim=0).float()
        del out, cache

        # Ground truth.
        gtU, gtS, gtVh = torch.linalg.svd(states, full_matrices=False)
        gt_err = recon_err(states, gtU[..., :16], gtS[..., :16], gtVh[..., :16, :])

        # Build Q and Bproj once per prompt, time only the final step.
        Q, Bproj = _build_Q_and_Bproj(states, orth="chol_v6", power_dtype=torch.float32)

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  gt_err_mean={float(gt_err.mean()):.4f}")
        for name, fn in VARIANTS:
            ms, raised = time_finalize(fn, Q, Bproj)
            timings[name].append(ms)
            # Accuracy: run once (not in timing loop) to get recon if it doesn't raise.
            try:
                U, S, Vh = fn(Q, Bproj)
                err = recon_err(states, U, S, Vh)
                rec_means[name].append(float(err.mean()))
                rec_max_gaps[name].append(float((err - gt_err).max()))
                tag = f"ok   t={ms:7.2f} ms  err_mean={rec_means[name][-1]:.4f}  max_gap={rec_max_gaps[name][-1]:+.4f}"
            except Exception as e:  # noqa: BLE001
                raises[name] += 1
                tag = f"RAISED  t={ms:7.2f} ms  ({str(e)[:60]})"
            print(f"    {name:<22}  {tag}")

        del states, Q, Bproj, gtU, gtS, gtVh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def avg(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    print()
    print("=" * 110)
    print(f"Final-step variants for chol_v6 fp32, averaged over {args.n_prompts} prompts (B=768)")
    print("=" * 110)
    print(f"{'variant':<24}{'time (ms)':>13}{'raises':>10}{'mean_recon_err':>18}{'max_gap_vs_optimum':>22}")
    for name, _ in VARIANTS:
        t = avg(timings[name])
        re = avg(rec_means[name])
        gp = max(rec_max_gaps[name]) if rec_max_gaps[name] else float("nan")
        print(f"{name:<24}{t:>13.2f}{raises[name]:>10}{re:>18.4f}{gp:>+22.4f}")


if __name__ == "__main__":
    main()

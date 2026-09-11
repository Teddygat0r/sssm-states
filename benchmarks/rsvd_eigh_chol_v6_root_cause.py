"""
Pin down the actual root cause of the chol_v6 raise:

  The chol_v6 ladder produces Q with non-orthonormal columns for ~5% of
  states even at the minimum jitter (base_eps=1e-5). The downstream
    Bproj = Q.T @ A
    C = Bproj @ Bproj.T
    eigh(C)
  step then operates on a mixed-magnitude C whose batched Jacobi eigh
  sometimes fails to converge.

Test two candidate fixes — replacing the final small eigh-on-Gram with a
direct SVD on Bproj, which is more robust to scale anisotropy:
  1. `svd_on_Bproj`: replace eigh(B B^T) with svd(B). Robust to mixed
     column norms; ~2× the work but still tiny since B is [q, n].
  2. `colnorm_then_eigh`: column-normalize Q to unit columns before
     building Bproj. This forces C to have ~unit-magnitude entries
     and should fix the eigh convergence issue.

For 8 MMLU prompts at B=768 fp32 chol_v6, we count raise rates under:
  - the current algorithm (baseline)
  - replace eigh(B B^T) with svd(B)
  - column-normalize Q before Bproj, then eigh(B B^T) as before
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rsvd_eigh  # noqa: E402
from rsvd_eigh import _ORTH_FNS  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


@torch.no_grad()
def rsvd_baseline(A, *, rank=16, n_iter=2, oversample=8, orth="chol_v6", power_dtype=None):
    """Same as randomized_svd_eigh — repeat here so we can mutate the last step."""
    orth_fn = _ORTH_FNS[orth]
    m, n = A.shape[-2], A.shape[-1]
    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A
    torch.manual_seed(0)
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
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(C)
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def rsvd_svd_on_Bproj(A, *, rank=16, n_iter=2, oversample=8, orth="chol_v6", power_dtype=None):
    """Replace the final eigh(B B^T) with svd(B)."""
    orth_fn = _ORTH_FNS[orth]
    m, n = A.shape[-2], A.shape[-1]
    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A
    torch.manual_seed(0)
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
    Bproj = Q.transpose(-2, -1) @ A.float()    # [..., q, n]
    # Direct SVD on Bproj (rank-q is small).
    Ub_small, Sk_full, Vh_full = torch.linalg.svd(Bproj, full_matrices=False)
    # Ub_small: [..., q, q], Sk_full: [..., q], Vh_full: [..., q, n]
    U = Q @ Ub_small[..., :, :rank]            # [..., m, rank]
    Sk = Sk_full[..., :rank]
    Vh = Vh_full[..., :rank, :]
    return U, Sk, Vh


@torch.no_grad()
def rsvd_colnorm_eigh(A, *, rank=16, n_iter=2, oversample=8, orth="chol_v6", power_dtype=None):
    """Column-normalize Q after orth, then eigh(B B^T) as before."""
    orth_fn = _ORTH_FNS[orth]
    m, n = A.shape[-2], A.shape[-1]
    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A
    torch.manual_seed(0)
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
    # Column-normalize Q (forces ||q_i||_2 = 1 per column).
    Q_norm = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    Q = Q / Q_norm
    Bproj = Q.transpose(-2, -1) @ A.float()
    C = Bproj @ Bproj.transpose(-2, -1)
    C = 0.5 * (C + C.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(C)
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)
    U = Q @ Ub
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def recon_err(A, U, S, Vh):
    recon = (U * S.unsqueeze(-2)) @ Vh
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return num / den


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

    n_base_raise = n_svd_raise = n_cn_raise = 0
    rows = []

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

        # Dense ground truth for recon comparison.
        gtU, gtS, gtVh = torch.linalg.svd(states, full_matrices=False)
        gt_err = recon_err(states, gtU[..., :16], gtS[..., :16], gtVh[..., :16, :])
        gt_mean = float(gt_err.mean())

        results = {}
        for label, fn in [("baseline", rsvd_baseline),
                          ("svd_on_Bproj", rsvd_svd_on_Bproj),
                          ("colnorm_eigh", rsvd_colnorm_eigh)]:
            try:
                U, S, Vh = fn(states, rank=16, n_iter=2, oversample=8,
                              orth="chol_v6", power_dtype=torch.float32)
                err = recon_err(states, U, S, Vh)
                results[label] = ("ok", float(err.mean()), float((err - gt_err).max()))
            except Exception as e:  # noqa: BLE001
                results[label] = ("RAISED", float("nan"), float("nan"))
                results[label + "_msg"] = str(e)[:140]

        # Tally.
        if results["baseline"][0] == "RAISED": n_base_raise += 1
        if results["svd_on_Bproj"][0] == "RAISED": n_svd_raise += 1
        if results["colnorm_eigh"][0] == "RAISED": n_cn_raise += 1

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  gt_err_mean={gt_mean:.4f}")
        for label in ["baseline", "svd_on_Bproj", "colnorm_eigh"]:
            status, m, mx = results[label]
            if status == "RAISED":
                print(f"    {label:<14}  RAISED ({results.get(label+'_msg','')[:80]})")
            else:
                print(f"    {label:<14}  ok   err_mean={m:.4f}   max_gap_abs={mx:+.4f}")

        rows.append({"prompt_id": f"mmlu_{i}", "subject": ex["subject"],
                     **{l: results[l][0] for l in ["baseline", "svd_on_Bproj", "colnorm_eigh"]}})

        del states, gtU, gtS, gtVh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print(f"Raise rates over {args.n_prompts} prompts (chol_v6 fp32 at B=768):")
    print(f"  baseline (current code)      : {n_base_raise}/{args.n_prompts} raised")
    print(f"  fix A: svd(B) instead of eigh: {n_svd_raise}/{args.n_prompts} raised")
    print(f"  fix B: column-normalize Q    : {n_cn_raise}/{args.n_prompts} raised")
    print("=" * 80)


if __name__ == "__main__":
    main()

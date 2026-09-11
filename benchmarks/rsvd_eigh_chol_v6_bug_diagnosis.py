"""
Diagnose whether `_chol_v6`'s scalar `eps` is the source of batched silent
non-orthonormality and the downstream eigh raise.

Hypothesis: when one rotten state in a batch of 768 needs eps=10 to pass
Cholesky, the scalar-eps loop boosts eps for the entire batch. All 768
states end up over-jittered, producing Q with tiny columns.

Test:
  1. For each prompt, compute Q from `_chol_v6` (a) per-state at B=1 and
     (b) on the full B=768 batch. Track the orth defect ||QᵀQ - I||/√q
     of every state under both regimes.
  2. Track the final `eps` reached in the scalar-eps ladder for the batched
     run.
  3. Implement a fixed version with per-element eps, re-run the batched
     case, and compare orth defects + eigh raise rate.
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
from rsvd_eigh import _house_qr, randomized_svd_eigh  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


# ---- instrumented & fixed chol_v6 ----------------------------------------

@torch.no_grad()
def chol_v6_instrumented(
    Y: torch.Tensor,
    base_eps=1e-5,
    max_eps=10.0,
    max_tries=6,
    use_eigh_repair=True,
):
    """Same as `_chol_v6` but returns the final eps reached and the iteration count."""
    Yf = Y.float()
    G = Yf.transpose(-2, -1) @ Yf
    G = 0.5 * (G + G.transpose(-2, -1))
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    eps = base_eps
    for it in range(max_tries):
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye, upper=True)
        if (info == 0).all():
            Q = torch.linalg.solve_triangular(R, Yf, upper=True, left=False)
            return Q.to(Y.dtype), eps, it, "tier1"
        eps = min(eps * 10.0, max_eps)
    if use_eigh_repair:
        try:
            L, V = torch.linalg.eigh(G)
            L = torch.clamp(L, min=max(1e-4, eps))
            G_spd = V @ torch.diag_embed(L) @ V.transpose(-2, -1)
            R = torch.linalg.cholesky(G_spd, upper=True)
            Q = torch.linalg.solve_triangular(R, Yf, upper=True, left=False)
            return Q.to(Y.dtype), eps, max_tries, "tier2_eigh"
        except Exception:
            pass
    return _house_qr(Yf).to(Y.dtype), eps, max_tries, "tier3_house"


@torch.no_grad()
def chol_v6_fixed(
    Y: torch.Tensor,
    base_eps=1e-5,
    max_eps=10.0,
    max_tries=6,
):
    """`_chol_v6` with per-element eps. Only grows eps for elements that
    failed Cholesky at the current eps; keeps successful elements at their
    minimum-needed jitter.
    """
    Yf = Y.float()
    G = Yf.transpose(-2, -1) @ Yf
    G = 0.5 * (G + G.transpose(-2, -1))
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    # Per-element eps.
    eps_e = torch.full(G.shape[:-2] + (1, 1), base_eps, device=G.device, dtype=G.dtype)

    R = None
    for _ in range(max_tries):
        R, info = torch.linalg.cholesky_ex(G + eps_e * scale * eye, upper=True)
        if (info == 0).all():
            Q = torch.linalg.solve_triangular(R, Yf, upper=True, left=False)
            return Q.to(Y.dtype), eps_e.squeeze().detach().cpu()
        # Grow eps only for elements that failed.
        failed = (info != 0)
        # info shape: G.shape[:-2]. Broadcast to eps_e shape by unsqueezing trailing dims.
        while failed.dim() < eps_e.dim():
            failed = failed.unsqueeze(-1)
        eps_e = torch.where(failed, eps_e * 10.0, eps_e).clamp_max(max_eps)
    # Final attempt with the final eps_e.
    R, info = torch.linalg.cholesky_ex(G + eps_e * scale * eye, upper=True)
    if (info == 0).all():
        Q = torch.linalg.solve_triangular(R, Yf, upper=True, left=False)
        return Q.to(Y.dtype), eps_e.squeeze().detach().cpu()
    # Fall back to Householder for remaining bad elements.
    return _house_qr(Yf).to(Y.dtype), eps_e.squeeze().detach().cpu()


@torch.no_grad()
def orth_defect(Q):
    # ||Q.T Q - I||_F / sqrt(q)
    q = Q.shape[-1]
    G = Q.transpose(-2, -1) @ Q
    eye = torch.eye(q, device=Q.device, dtype=Q.dtype)
    diff = (G - eye).reshape(Q.shape[0], -1).norm(dim=-1)
    return (diff / (q ** 0.5)).cpu()


# ---- helpers ---------------------------------------------------------------

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


def _make_sketch(A: torch.Tensor, rank=16, oversample=8, n_iter=2):
    """Reproduce `randomized_svd_eigh`'s sketch Y up to the last orth step."""
    q = rank + oversample
    n = A.shape[-1]
    torch.manual_seed(0)
    Omega = torch.randn(n, q, dtype=torch.float32, device=A.device)
    Y = A.float() @ Omega
    # We need one more orthonormalization to reach the same Y the algorithm
    # would orth at the final step before eigh. For diagnostic purposes, we
    # measure orth defect of Q from `_chol_v6(Y)` here directly.
    return Y


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=4)
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
        N = states.shape[0]

        # Build the same initial sketch Y the algorithm uses.
        Y_batch = _make_sketch(states)
        # Batched call: scalar-eps ladder.
        Qb, eps_reached, n_iter_reached, tier = chol_v6_instrumented(Y_batch)
        defect_batched = orth_defect(Qb)

        # Per-state calls: scalar-eps ladder (but with only one element, so
        # effectively per-element).
        defect_per = torch.zeros(N, dtype=torch.float32)
        eps_per = torch.zeros(N, dtype=torch.float32)
        for n in range(N):
            Yn = Y_batch[n].unsqueeze(0)
            Qn, eps_n, _, _ = chol_v6_instrumented(Yn)
            defect_per[n] = orth_defect(Qn).item()
            eps_per[n] = eps_n

        # Fixed (per-element eps) batched call.
        Qf, eps_e = chol_v6_fixed(Y_batch)
        defect_fixed = orth_defect(Qf)

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  N={N}")
        print(f"  Scalar-eps batched: final eps={eps_reached:.0e} after {n_iter_reached} iters, tier={tier}")
        print(f"    orth defect of Q: min={float(defect_batched.min()):.2e}  "
              f"median={float(defect_batched.median()):.2e}  "
              f"p99={float(defect_batched.quantile(0.99)):.2e}  "
              f"max={float(defect_batched.max()):.2e}")
        # Histogram of per-state eps from the per-state runs.
        eps_unique = sorted(set(float(x) for x in eps_per.tolist()))
        eps_hist = [(e, int((eps_per == e).sum().item())) for e in eps_unique]
        print(f"  Per-state ladder eps histogram: {eps_hist}")
        print(f"    orth defect of Q (per-state): min={float(defect_per.min()):.2e}  "
              f"median={float(defect_per.median()):.2e}  "
              f"p99={float(defect_per.quantile(0.99)):.2e}  "
              f"max={float(defect_per.max()):.2e}")
        if isinstance(eps_e, torch.Tensor) and eps_e.numel() > 0:
            eps_e_flat = eps_e.flatten()
            eu = sorted(set(float(x) for x in eps_e_flat.tolist()))
            eh = [(e, int((eps_e_flat == e).sum().item())) for e in eu]
            print(f"  Fixed (per-elem) eps histogram: {eh}")
        print(f"    orth defect of Q (fixed): min={float(defect_fixed.min()):.2e}  "
              f"median={float(defect_fixed.median()):.2e}  "
              f"p99={float(defect_fixed.quantile(0.99)):.2e}  "
              f"max={float(defect_fixed.max()):.2e}")

        # Also test the downstream raise: run full randomized_svd_eigh with
        # the fixed orth function, count raises.
        # We can monkey-patch _ORTH_FNS["chol_v6"] for this prompt.
        orig = rsvd_eigh._ORTH_FNS["chol_v6"]
        try:
            # current chol_v6 (scalar eps)
            scalar_raised = False
            try:
                _ = randomized_svd_eigh(states, rank=16, n_iter=2, oversample=8,
                                        orth="chol_v6", power_dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                scalar_raised = True
                scalar_err = str(e)[:120]

            # Fixed version
            rsvd_eigh._ORTH_FNS["chol_v6"] = lambda Y: chol_v6_fixed(Y)[0]
            fixed_raised = False
            try:
                _ = randomized_svd_eigh(states, rank=16, n_iter=2, oversample=8,
                                        orth="chol_v6", power_dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                fixed_raised = True
                fixed_err = str(e)[:120]
        finally:
            rsvd_eigh._ORTH_FNS["chol_v6"] = orig

        scalar_tag = f"RAISED ({scalar_err})" if scalar_raised else "ok"
        fixed_tag = f"RAISED ({fixed_err})" if fixed_raised else "ok"
        print(f"  Downstream randomized_svd_eigh:")
        print(f"    scalar-eps (current code): {scalar_tag}")
        print(f"    per-elem-eps (fixed)     : {fixed_tag}")

        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

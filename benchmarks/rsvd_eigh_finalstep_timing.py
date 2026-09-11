"""
Time the final small decomp in `randomized_svd_eigh`: `eigh(B B^T)` vs.
`svd(B)`, for `chol_v6`, `cholqr2`, and `house` at B=768 on Qwen3.5-4B
recurrent states (10 prompts × 3 warmup + 10 timed iters each).

What we are measuring: only the wall-clock cost of swapping the final
decomp. Everything before it (sketch, power iter, orth) is unchanged.
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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def rsvd_eigh_on_Gram(A, *, rank=16, n_iter=2, oversample=8, orth="chol_v6", power_dtype=None):
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
    Ub_small, Sk_full, Vh_full = torch.linalg.svd(Bproj, full_matrices=False)
    U = Q @ Ub_small[..., :, :rank]
    Sk = Sk_full[..., :rank]
    Vh = Vh_full[..., :rank, :]
    return U, Sk, Vh


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


def time_call(fn, states, orth, pdt, n_warmup=3, n_iter=10):
    # Warm up
    for _ in range(n_warmup):
        try:
            fn(states, rank=16, n_iter=2, oversample=8, orth=orth, power_dtype=pdt)
        except Exception:
            return float("nan"), True
    _sync()
    raised = False
    t0 = perf_counter()
    for _ in range(n_iter):
        try:
            fn(states, rank=16, n_iter=2, oversample=8, orth=orth, power_dtype=pdt)
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

    method_specs = [
        ("chol_v6", torch.float32),
        ("chol_v6", torch.bfloat16),
        ("cholqr2", torch.float32),
        ("cholqr2", torch.bfloat16),
        ("house",   torch.float32),
        ("house",   torch.bfloat16),
    ]

    timings = {}  # (orth, pdt, variant) -> [ms per prompt]
    raises = {}   # (orth, pdt, variant) -> count

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
        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  N={states.shape[0]}")
        for orth, pdt in method_specs:
            for label, fn in [("eigh_on_Gram", rsvd_eigh_on_Gram),
                              ("svd_on_Bproj", rsvd_svd_on_Bproj)]:
                ms, raised = time_call(fn, states, orth, pdt)
                key = (orth, str(pdt).replace("torch.", ""), label)
                timings.setdefault(key, []).append(ms)
                raises.setdefault(key, 0)
                if raised:
                    raises[key] += 1
            # Print pair side-by-side.
            e_key = (orth, str(pdt).replace("torch.", ""), "eigh_on_Gram")
            s_key = (orth, str(pdt).replace("torch.", ""), "svd_on_Bproj")
            em = timings[e_key][-1]
            sm = timings[s_key][-1]
            er = "(R)" if raises[e_key] > 0 and pi == 0 else ""  # never mind
            tag = ""
            if em == em and sm == sm:
                tag = f"  Δ = {sm-em:+.2f} ms  ({(sm-em)/em*100:+.1f}%)"
            print(f"    {orth:<9} pdt={str(pdt).replace('torch.',''):<9}  "
                  f"eigh_on_Gram={em:7.2f} ms  svd_on_Bproj={sm:7.2f} ms{tag}")

        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate.
    def avg(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    print()
    print("=" * 100)
    print(f"Average per-call timing across {args.n_prompts} prompts (B=768)")
    print("=" * 100)
    print(f"{'method':<10}{'pdt':<10}{'eigh_on_Gram (ms)':>20}{'svd_on_Bproj (ms)':>20}{'delta ms':>12}{'delta %':>10}{'raises (e/s)':>14}")
    for orth, pdt in method_specs:
        pdt_name = str(pdt).replace("torch.", "")
        em = avg(timings[(orth, pdt_name, "eigh_on_Gram")])
        sm = avg(timings[(orth, pdt_name, "svd_on_Bproj")])
        er = raises[(orth, pdt_name, "eigh_on_Gram")]
        sr = raises[(orth, pdt_name, "svd_on_Bproj")]
        d = sm - em if em == em and sm == sm else float("nan")
        dp = d / em * 100 if em == em and em > 0 else float("nan")
        print(f"{orth:<10}{pdt_name:<10}{em:>20.2f}{sm:>20.2f}{d:>+12.2f}{dp:>+9.1f}%   {er:>2d} / {sr:<2d}")


if __name__ == "__main__":
    main()

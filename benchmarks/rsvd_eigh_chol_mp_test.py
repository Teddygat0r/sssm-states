"""
Test the new fp64 orth methods (`chol_mp`, `chol_mp2`) on Qwen3.5-4B
recurrent states, alongside `chol_v6` (baseline + my 1e-8 jitter fix).

Imports the latest rsvd_eigh from /home/joshuaz/kv-svd/svd_methods/.

Variants (all fp32 power_dtype, GPU):
  - chol_v6_baseline   : current chol_v6 path, no jitter applied to C
  - chol_v6_jitter1e-8 : chol_v6 + my proactive jitter fix on C
  - chol_mp            : new fp64 single CholQR
  - chol_mp2           : new fp64 CholQR twice + Householder fallback

For each MMLU prompt:
  1. Prefill on GPU. Stack states → [768, 128, 128] fp32.
  2. Run all four variants. Catch raises, time each call.
  3. Every `accuracy_every` prompts, also compute dense rank-16 truncated
     SVD on GPU as ground truth and per-state Frobenius recon error for
     each variant. Track gap-vs-optimum mean and worst.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

# Import from the NEW location with the chol_mp/chol_mp2 additions.
NEW_RSVD = Path("/home/joshuaz/kv-svd/svd_methods")
sys.path.insert(0, str(NEW_RSVD))
import rsvd_eigh as new_rsvd  # noqa: E402
from rsvd_eigh import _ORTH_FNS  # noqa: E402

assert "chol_mp" in _ORTH_FNS, f"new rsvd_eigh not loaded — orth fns are {list(_ORTH_FNS.keys())}"
assert "chol_mp2" in _ORTH_FNS

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


@torch.no_grad()
def _rsvd(A, *, orth, rank=RANK, n_iter=2, oversample=OVERSAMPLE,
          jitter_factor=0.0, seed=0):
    """Inlined version of randomized_svd_eigh so we can optionally inject
    the jitter into C. Uses the orth functions from new_rsvd."""
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
def _per_state_recon_err(A, U, S, Vh):
    recon = (U * S.unsqueeze(-2)) @ Vh
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jitter_factor", type=float, default=1e-8)
    ap.add_argument("--accuracy_every", type=int, default=10,
                    help="Compute ground-truth SVD comparison every N prompts.")
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "eval_results" / datetime.now().strftime(
        "rsvd_chol_mp_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Loaded rsvd_eigh from: {new_rsvd.__file__}")
    print(f"Available orth methods: {list(_ORTH_FNS.keys())}")
    print(f"n_prompts={args.n_prompts}  jitter_factor={args.jitter_factor:.0e}")
    print(f"rsvd_eigh n_iter=2, oversample=8  →  sketch dim q=24\n")

    print(f"Loading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    linear_layers = _find_linear_layers(model)

    print(f"\nLoading MMLU ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g)
    idxs = perm[: args.n_prompts].tolist()
    print(f"  {len(ds)} examples in test split, sampling first {args.n_prompts} from a seed={args.seed} permutation")

    variants = [
        ("chol_v6_baseline",      lambda A: _rsvd(A, orth="chol_v6", jitter_factor=0.0)),
        ("chol_v6_jitter_1e-8",   lambda A: _rsvd(A, orth="chol_v6", jitter_factor=args.jitter_factor)),
        ("chol_mp",               lambda A: _rsvd(A, orth="chol_mp",  jitter_factor=0.0)),
        ("chol_mp2",              lambda A: _rsvd(A, orth="chol_mp2", jitter_factor=0.0)),
    ]

    per_prompt = []
    failures = []
    n_raised = {n: 0 for n, _ in variants}
    n_acc = {n: 0 for n, _ in variants}
    err_means = {n: [] for n, _ in variants}
    err_p99s = {n: [] for n, _ in variants}
    gap_means = {n: [] for n, _ in variants}
    gap_maxes = {n: [] for n, _ in variants}
    gap_p99s = {n: [] for n, _ in variants}
    times = {n: [] for n, _ in variants}
    diff_baseline_vs_jit = []  # max |err_baseline - err_jitter| per accuracy check

    overall_start = perf_counter()

    for pi, i in enumerate(idxs):
        ex = ds[i]
        prompt = _format_mmlu(ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.max_seq_len].to(DEVICE)
        T = int(ids.shape[1])
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        _sync()
        cache = out.past_key_values
        states = torch.cat([cache.layers[li].recurrent_states.squeeze(0).detach()
                             for li in linear_layers], dim=0).float()
        del out, cache
        N = states.shape[0]

        do_acc = (pi % args.accuracy_every == 0)
        gt_err = None
        if do_acc:
            gtU, gtS, gtVh = torch.linalg.svd(states, full_matrices=False)
            gt_err = _per_state_recon_err(states, gtU[..., :RANK], gtS[..., :RANK], gtVh[..., :RANK, :])
            del gtU, gtS, gtVh

        row = {
            "prompt_idx_in_perm": pi,
            "mmlu_index": int(i),
            "subject": ex["subject"],
            "seq_len": T,
            "accuracy_checked": do_acc,
        }

        # Cache per-state recon err for cross-variant comparison.
        recon_errs = {}

        for name, fn in variants:
            _sync()
            t0 = perf_counter()
            try:
                U, S, Vh = fn(states)
                _sync()
                t = perf_counter() - t0
                row[f"{name}_raised"] = False
                row[f"{name}_t_ms"] = t * 1000
                times[name].append(t * 1000)

                if do_acc:
                    err = _per_state_recon_err(states, U, S, Vh).cpu()
                    gap = err - gt_err.cpu()
                    err_means[name].append(float(err.mean()))
                    err_p99s[name].append(float(err.quantile(0.99)))
                    gap_means[name].append(float(gap.mean()))
                    gap_maxes[name].append(float(gap.max()))
                    gap_p99s[name].append(float(gap.quantile(0.99)))
                    row[f"{name}_err_mean"] = float(err.mean())
                    row[f"{name}_err_p99"] = float(err.quantile(0.99))
                    row[f"{name}_gap_mean"] = float(gap.mean())
                    row[f"{name}_gap_max"] = float(gap.max())
                    n_acc[name] += 1
                    recon_errs[name] = err
                del U, S, Vh
            except Exception as e:  # noqa: BLE001
                _sync()
                t = perf_counter() - t0
                n_raised[name] += 1
                row[f"{name}_raised"] = True
                row[f"{name}_t_ms"] = t * 1000
                row[f"{name}_err"] = type(e).__name__ + ": " + str(e)[:160]
                failures.append({
                    "prompt_idx_in_perm": pi, "mmlu_index": int(i),
                    "subject": ex["subject"], "variant": name,
                    "err": row[f"{name}_err"], "seq_len": T,
                })

        # If we did accuracy check and both chol_v6 variants succeeded, record
        # the max per-state |Δerr| as a check for jitter-vs-baseline drift.
        if (do_acc
            and "chol_v6_baseline" in recon_errs
            and "chol_v6_jitter_1e-8" in recon_errs):
            d = float((recon_errs["chol_v6_baseline"] - recon_errs["chol_v6_jitter_1e-8"]).abs().max())
            diff_baseline_vs_jit.append(d)
            row["jitter_vs_baseline_max_abs"] = d

        per_prompt.append(row)

        if (pi + 1) % args.print_every == 0 or (pi + 1) == len(idxs):
            elapsed = perf_counter() - overall_start
            eta = elapsed / (pi + 1) * (len(idxs) - pi - 1)
            r_counts = "  ".join([f"{n}={n_raised[n]}" for n, _ in variants])
            print(f"  [{pi+1:>4d}/{len(idxs)}]  raises: {r_counts}   "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        del states
        if gt_err is not None:
            del gt_err
        gc.collect()
        if torch.cuda.is_available() and pi % 50 == 49:
            torch.cuda.empty_cache()

    # Write CSVs.
    pp_path = out_dir / "per_prompt.csv"
    with open(pp_path, "w", newline="") as f:
        keys: list[str] = []
        for r in per_prompt:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_prompt:
            w.writerow(r)
    f_path = out_dir / "failures.csv"
    with open(f_path, "w", newline="") as f:
        keys = ["prompt_idx_in_perm", "mmlu_index", "subject", "variant", "err", "seq_len"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in failures:
            w.writerow(r)

    elapsed = perf_counter() - overall_start

    def avg(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    def safemax(xs):
        xs = [x for x in xs if x == x]
        return max(xs) if xs else float("nan")

    summary = []
    summary.append("=" * 120)
    summary.append(f"chol_v6 vs chol_mp/chol_mp2 on {args.n_prompts} MMLU prompts × Qwen3.5-4B states")
    summary.append("=" * 120)
    summary.append(f"  jitter_factor    : {args.jitter_factor:.0e}")
    summary.append(f"  rsvd_eigh.n_iter : 2 (default)")
    summary.append(f"  wall-clock       : {elapsed:.0f}s ({elapsed/len(idxs)*1000:.0f} ms/prompt)")
    summary.append(f"  accuracy checks  : {n_acc[variants[0][0]]} per variant")
    summary.append("")
    summary.append(f"{'variant':<24}{'raise_rate':>14}{'time_avg_ms':>14}"
                   f"{'err_mean_avg':>16}{'gap_mean_avg':>16}{'gap_p99_avg':>16}{'gap_max_worst':>16}")
    for name, _ in variants:
        rr = f"{n_raised[name]}/{args.n_prompts}"
        t = avg(times[name])
        em = avg(err_means[name])
        gm = avg(gap_means[name])
        gp9 = avg(gap_p99s[name])
        gx = safemax(gap_maxes[name])
        summary.append(f"{name:<24}{rr:>14}{t:>14.2f}"
                       f"{em:>16.5f}{gm:>+16.6f}{gp9:>+16.5f}{gx:>+16.4f}")

    if diff_baseline_vs_jit:
        d_avg = sum(diff_baseline_vs_jit) / len(diff_baseline_vs_jit)
        d_max = max(diff_baseline_vs_jit)
        summary.append("")
        summary.append("Jitter vs no-jitter consistency check (chol_v6 path, accuracy-checked prompts only):")
        summary.append(f"  n checks                : {len(diff_baseline_vs_jit)}")
        summary.append(f"  mean of max per-state |Δerr| : {d_avg:.3e}")
        summary.append(f"  worst of max per-state |Δerr|: {d_max:.3e}")

    s = "\n".join(summary)
    print()
    print(s)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(s + "\n")
    print(f"\nWrote {pp_path}")
    print(f"Wrote {f_path}")
    print(f"Wrote {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()

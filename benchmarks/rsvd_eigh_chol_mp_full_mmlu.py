"""
Run prefill on EVERY prompt in MMLU test (14,042 examples) and test both
`chol_mp` and `chol_mp2` on the resulting Qwen3.5-4B states. Record any
raises with the failing prompt + spectral signature of the state batch.

No accuracy comparison at this scale — we just want to know if the fp64
path fails anywhere in MMLU. We already verified accuracy ≈ optimal at
smaller scale.

For each MMLU prompt (in canonical dataset order — not a random sample):
  1. Prefill on GPU.
  2. Stack 24 linear-attn layers × 32 v-heads = 768 recurrent states
     into a [768, 128, 128] fp32 batch.
  3. Try chol_mp; record raise.
  4. Try chol_mp2; record raise.
  5. On any raise, dump a brief spectral summary of the batch so we can
     characterize what triggered it.

Outputs:
  - failures.csv        every raised call with prompt id, subject, error
  - summary.txt         aggregate raise counts + (if any) failure analysis
  - progress.csv        sparse per-100-prompt progress checkpoints
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
def _spectrum_summary(states):
    """Cheap-ish summary of the state batch's spectral health.
    Returns a dict of scalars for the worst-conditioned state."""
    sv = torch.linalg.svdvals(states)            # [N, 128]
    sv_max = sv[:, 0]
    sv_q = sv[:, RANK + OVERSAMPLE - 1]
    sv_min = sv[:, -1]
    ratio_q = (sv_q / sv_max.clamp_min(1e-30))
    ratio_min = (sv_min / sv_max.clamp_min(1e-30))
    fro = states.reshape(states.shape[0], -1).norm(dim=-1)
    absmax = states.reshape(states.shape[0], -1).abs().amax(dim=-1)
    worst_q = int(ratio_q.argmin().item())
    return {
        "n_states": int(states.shape[0]),
        "fro_min": float(fro.min().item()),
        "fro_median": float(fro.median().item()),
        "fro_max": float(fro.max().item()),
        "absmax_min": float(absmax.min().item()),
        "absmax_max": float(absmax.max().item()),
        "sv_max_min": float(sv_max.min().item()),
        "sv_max_max": float(sv_max.max().item()),
        "ratio_q_min": float(ratio_q.min().item()),
        "ratio_q_median": float(ratio_q.median().item()),
        "ratio_min_min": float(ratio_min.min().item()),
        "worst_state_idx": worst_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--start", type=int, default=0, help="Start index in MMLU test")
    ap.add_argument("--stop", type=int, default=None,
                    help="Stop index in MMLU test (exclusive). Defaults to all.")
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--checkpoint_every", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "eval_results" / datetime.now().strftime(
        "rsvd_chol_mp_full_mmlu_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    linear_layers = _find_linear_layers(model)

    print(f"\nLoading MMLU test split ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    total = len(ds)
    stop = args.stop if args.stop is not None else total
    n = stop - args.start
    print(f"  {total} examples; running indices [{args.start}, {stop}) → {n} prompts")

    failures = []
    n_raised = {"chol_mp": 0, "chol_mp2": 0}
    progress = []
    overall_start = perf_counter()

    for offset in range(n):
        idx = args.start + offset
        ex = ds[idx]
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

        spec = None  # only computed if a raise occurs (it's ~50 ms)

        for m in ("chol_mp", "chol_mp2"):
            try:
                _ = _rsvd(states, orth=m)
            except Exception as e:  # noqa: BLE001
                n_raised[m] += 1
                if spec is None:
                    spec = _spectrum_summary(states)
                fr = {
                    "mmlu_index": idx,
                    "subject": ex["subject"],
                    "variant": m,
                    "err": type(e).__name__ + ": " + str(e)[:200],
                    "seq_len": T,
                    **spec,
                }
                failures.append(fr)
                print(f"  RAISE  idx={idx}  {ex['subject']:<30}  {m}: {type(e).__name__}: {str(e)[:80]}")

        if (offset + 1) % args.print_every == 0 or (offset + 1) == n:
            elapsed = perf_counter() - overall_start
            rate = (offset + 1) / elapsed
            eta = (n - offset - 1) / rate
            print(f"  [{offset+1:>5d}/{n}]  raises: chol_mp={n_raised['chol_mp']}  "
                  f"chol_mp2={n_raised['chol_mp2']}   "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  ETA={eta:.0f}s")

        if (offset + 1) % args.checkpoint_every == 0:
            elapsed = perf_counter() - overall_start
            progress.append({
                "offset": offset + 1, "mmlu_index": idx,
                "raised_chol_mp": n_raised["chol_mp"],
                "raised_chol_mp2": n_raised["chol_mp2"],
                "elapsed_s": elapsed,
            })
            # Write current state of failures and progress.
            with open(out_dir / "failures.csv", "w", newline="") as f:
                if failures:
                    keys = list(failures[0].keys())
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in failures:
                        w.writerow(r)
            with open(out_dir / "progress.csv", "w", newline="") as f:
                keys = ["offset", "mmlu_index", "raised_chol_mp", "raised_chol_mp2", "elapsed_s"]
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in progress:
                    w.writerow(r)

        del states
        gc.collect()
        if torch.cuda.is_available() and offset % 100 == 99:
            torch.cuda.empty_cache()

    elapsed = perf_counter() - overall_start

    # Write final outputs.
    if failures:
        with open(out_dir / "failures.csv", "w", newline="") as f:
            keys = list(failures[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in failures:
                w.writerow(r)
    else:
        # Empty failures CSV for clarity.
        with open(out_dir / "failures.csv", "w") as f:
            f.write("# No failures across all prompts.\n")

    summary = []
    summary.append("=" * 96)
    summary.append(f"Full-MMLU raise-rate test: chol_mp vs chol_mp2 on Qwen3.5-4B")
    summary.append("=" * 96)
    summary.append(f"  n_prompts        : {n}  (MMLU test indices {args.start}..{stop-1})")
    summary.append(f"  wall-clock       : {elapsed:.0f}s ({elapsed/n*1000:.0f} ms/prompt avg)")
    summary.append(f"  chol_mp raises   : {n_raised['chol_mp']}/{n}  "
                   f"({100*n_raised['chol_mp']/n:.3f}%)")
    summary.append(f"  chol_mp2 raises  : {n_raised['chol_mp2']}/{n}  "
                   f"({100*n_raised['chol_mp2']/n:.3f}%)")
    if failures:
        summary.append("")
        summary.append("Failure breakdown:")
        from collections import Counter
        by_var = Counter([f["variant"] for f in failures])
        for v, c in by_var.most_common():
            summary.append(f"  {v}: {c}")
        summary.append("")
        summary.append("Failing subjects (any variant):")
        by_sub = Counter([f["subject"] for f in failures])
        for s, c in by_sub.most_common(20):
            summary.append(f"  {s:<40s}  {c}")
    s = "\n".join(summary)
    print()
    print(s)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(s + "\n")
    print(f"\nWrote {out_dir/'summary.txt'}")
    print(f"Wrote {out_dir/'failures.csv'}")


if __name__ == "__main__":
    main()

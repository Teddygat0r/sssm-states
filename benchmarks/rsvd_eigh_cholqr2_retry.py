"""
Investigate why `cholqr2` takes ~114-127 ms on Qwen3.5-4B states when the
docstring claims ~13 ms.

Hypothesis: on these states the inner `_chol_min` always raises, so
`_chol_qr2` always falls back to Householder QR — i.e. we are measuring
the fallback path, not the fast path.

We:
  1. Monkey-patch `_chol_min` to count successes vs. raises.
  2. Re-run cholqr2 on a few MMLU prompts and report the fallback rate.
  3. As a sanity check, time the fast path on a same-shape random Gaussian
     tensor to confirm we can hit ~13 ms when no fallback is needed.
  4. Also time `chol_min`-only on the random tensor to recover the
     ~9 ms claim from the docstring.
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
from rsvd_eigh import randomized_svd_eigh  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


# ---- instrumentation ------------------------------------------------------

_counters = {"chol_min_ok": 0, "chol_min_raised": 0}
_orig_chol_min = rsvd_eigh._chol_min


def _counted_chol_min(Y):
    try:
        out = _orig_chol_min(Y)
        _counters["chol_min_ok"] += 1
        return out
    except torch.linalg.LinAlgError:
        _counters["chol_min_raised"] += 1
        raise


def _reset_counts():
    _counters["chol_min_ok"] = 0
    _counters["chol_min_raised"] = 0


def install():
    rsvd_eigh._chol_min = _counted_chol_min
    rsvd_eigh._ORTH_FNS["chol_min"] = _counted_chol_min


def uninstall():
    rsvd_eigh._chol_min = _orig_chol_min
    rsvd_eigh._ORTH_FNS["chol_min"] = _orig_chol_min


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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_n(fn, n_warmup=2, n_iter=10):
    for _ in range(n_warmup):
        fn()
    _sync()
    t0 = perf_counter()
    for _ in range(n_iter):
        fn()
    _sync()
    return (perf_counter() - t0) / n_iter


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=5)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    install()

    print("=" * 78)
    print("Part 1 — Time random-Gaussian batch (no fallback expected)")
    print("=" * 78)
    torch.manual_seed(0)
    rand = torch.randn(768, 128, 128, device=DEVICE, dtype=torch.float32)
    # Warm-up + measurement, fp32 power
    for label, orth, pdt in [
        ("chol_min  fp32 ", "chol_min", torch.float32),
        ("chol_min  bf16 ", "chol_min", torch.bfloat16),
        ("cholqr2   fp32 ", "cholqr2",  torch.float32),
        ("cholqr2   bf16 ", "cholqr2",  torch.bfloat16),
        ("house     fp32 ", "house",    torch.float32),
        ("house     bf16 ", "house",    torch.bfloat16),
    ]:
        _reset_counts()
        try:
            def go():
                return randomized_svd_eigh(rand, rank=16, n_iter=2, oversample=8,
                                           orth=orth, power_dtype=pdt)
            t = _time_n(go, n_warmup=3, n_iter=10) * 1000
            print(f"  {label}  mean={t:6.2f} ms over 10 iters   "
                  f"chol_min ok={_counters['chol_min_ok']:4d}  raised={_counters['chol_min_raised']:4d}")
        except Exception as e:
            print(f"  {label}  RAISED: {type(e).__name__}: {str(e)[:80]}")

    # ----------------------- Part 2: real Qwen3.5-4B states --------------
    print()
    print("=" * 78)
    print("Part 2 — Time on real Qwen3.5-4B recurrent states (with fallback tracking)")
    print("=" * 78)
    print(f"Loading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    linear_layers = _find_linear_layers(model)

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

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  batch={tuple(states.shape)}")
        for label, orth, pdt in [
            ("chol_min  fp32 ", "chol_min", torch.float32),
            ("chol_min  bf16 ", "chol_min", torch.bfloat16),
            ("cholqr2   fp32 ", "cholqr2",  torch.float32),
            ("cholqr2   bf16 ", "cholqr2",  torch.bfloat16),
            ("house     fp32 ", "house",    torch.float32),
            ("house     bf16 ", "house",    torch.bfloat16),
        ]:
            _reset_counts()
            try:
                def go():
                    return randomized_svd_eigh(states, rank=16, n_iter=2, oversample=8,
                                               orth=orth, power_dtype=pdt)
                t = _time_n(go, n_warmup=3, n_iter=10) * 1000
                # n_warmup=3 + n_iter=10 = 13 invocations.
                # For cholqr2, each call attempts chol_min twice (once for sketch,
                # once after first power iter, etc — actually n_iter=2 means 3 Q
                # constructions per call), but only the FIRST chol_min attempt
                # raises; on raise, fallback runs and no further chol_min runs.
                # So counters reflect ok/raised per top-level call.
                tot = _counters['chol_min_ok'] + _counters['chol_min_raised']
                raise_pct = (100.0 * _counters['chol_min_raised'] / tot) if tot else 0.0
                print(f"  {label}  mean={t:7.2f} ms   "
                      f"chol_min calls: ok={_counters['chol_min_ok']:4d}  "
                      f"raised={_counters['chol_min_raised']:4d}  "
                      f"({raise_pct:.0f}% raised)")
            except Exception as e:
                tot = _counters['chol_min_ok'] + _counters['chol_min_raised']
                print(f"  {label}  RAISED: {type(e).__name__}  "
                      f"chol_min: ok={_counters['chol_min_ok']}  raised={_counters['chol_min_raised']}")


if __name__ == "__main__":
    main()

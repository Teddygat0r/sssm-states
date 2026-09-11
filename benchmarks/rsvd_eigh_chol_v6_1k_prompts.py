"""
Large-scale raise-rate test: chol_v6 fp32 with and without proactive jitter
on 1000 MMLU prompts × Qwen3.5-4B prefill.

Per prompt:
  1. Prefill (cache only — we discard logits).
  2. Stack the 24 linear-attn layers × 32 v-heads = 768 recurrent states
     into a [768, 128, 128] fp32 batch.
  3. Run randomized_svd_eigh with orth=chol_v6, power_dtype=fp32, twice:
       a. Baseline    : current rsvd_eigh.py code, eigh(C) no jitter.
       b. Jitter      : eigh(C + 1e-8 * mean_diag(C) * I)
  4. Record raise / OK for each variant.
  5. On every 50th prompt, also compute the dense rank-16 truncated SVD and
     check that the jitter variant's reconstruction error matches baseline
     within a small tolerance — sanity check that the jitter isn't silently
     changing the answer.

Output:
  - per_prompt.csv : 1000 rows
  - failures.csv   : every raised call with the eigh error message
  - summary.txt    : aggregate raise rate, top failing-prompt characteristics.
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
def _rsvd_chol_v6(A, *, rank=16, n_iter=2, oversample=8, jitter_factor=0.0, seed=0):
    """Inlined randomized_svd_eigh for chol_v6 fp32 with an optional proactive jitter."""
    orth_fn = _ORTH_FNS["chol_v6"]
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
    ap.add_argument("--accuracy_every", type=int, default=50,
                    help="Compute ground-truth SVD comparison every N prompts.")
    ap.add_argument("--print_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "eval_results" / datetime.now().strftime(
        "rsvd_chol_v6_1k_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"n_prompts={args.n_prompts}  jitter_factor={args.jitter_factor:.0e}")

    print(f"\nLoading {MODEL_ID} ...")
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

    per_prompt = []
    failures = []

    n_base_raised = 0
    n_jit_raised = 0
    n_jit_recoveries = 0      # baseline raised, jitter ok
    n_jit_only_raised = 0     # jitter raised but baseline didn't (shouldn't happen)
    n_acc_checked = 0
    n_acc_mismatch = 0

    overall_start = perf_counter()

    for pi, i in enumerate(idxs):
        ex = ds[i]
        prompt = _format_mmlu(ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.max_seq_len].to(DEVICE)
        T = int(ids.shape[1])
        t_pre = perf_counter()
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        _sync()
        prefill_s = perf_counter() - t_pre
        cache = out.past_key_values
        states = torch.cat([cache.layers[li].recurrent_states.squeeze(0).detach()
                             for li in linear_layers], dim=0).float()
        del out, cache

        N = states.shape[0]
        # Cheap per-state stats for failure-case analysis.
        abs_max = float(states.reshape(N, -1).abs().amax(dim=-1).max().item())
        fro_sum = float(states.reshape(N, -1).norm(dim=-1).sum().item())
        # Per-state Frobenius statistics — useful for correlating with failures
        fro = states.reshape(N, -1).norm(dim=-1)
        fro_min = float(fro.min().item())
        fro_median = float(fro.median().item())
        fro_max = float(fro.max().item())

        row = {
            "prompt_idx_in_perm": pi,
            "mmlu_index": int(i),
            "subject": ex["subject"],
            "seq_len": T,
            "n_states": N,
            "prefill_s": prefill_s,
            "abs_max": abs_max,
            "fro_min": fro_min,
            "fro_median": fro_median,
            "fro_max": fro_max,
            "baseline_raised": False,
            "baseline_err": "",
            "jitter_raised": False,
            "jitter_err": "",
            "accuracy_checked": False,
            "max_recon_err_diff": 0.0,
        }

        # Variant A: baseline (no jitter).
        try:
            U_b, S_b, V_b = _rsvd_chol_v6(states, jitter_factor=0.0)
        except Exception as e:  # noqa: BLE001
            row["baseline_raised"] = True
            row["baseline_err"] = type(e).__name__ + ": " + str(e)[:160]
            n_base_raised += 1
            U_b = S_b = V_b = None
            failures.append({
                "prompt_idx_in_perm": pi, "mmlu_index": int(i),
                "subject": ex["subject"], "variant": "baseline",
                "err": row["baseline_err"], "seq_len": T,
                "abs_max": abs_max, "fro_min": fro_min, "fro_median": fro_median,
            })

        # Variant B: proactive jitter.
        try:
            U_j, S_j, V_j = _rsvd_chol_v6(states, jitter_factor=args.jitter_factor)
        except Exception as e:  # noqa: BLE001
            row["jitter_raised"] = True
            row["jitter_err"] = type(e).__name__ + ": " + str(e)[:160]
            n_jit_raised += 1
            U_j = S_j = V_j = None
            failures.append({
                "prompt_idx_in_perm": pi, "mmlu_index": int(i),
                "subject": ex["subject"], "variant": "jitter",
                "err": row["jitter_err"], "seq_len": T,
                "abs_max": abs_max, "fro_min": fro_min, "fro_median": fro_median,
            })

        # Recovery / regression accounting.
        if row["baseline_raised"] and not row["jitter_raised"]:
            n_jit_recoveries += 1
        if not row["baseline_raised"] and row["jitter_raised"]:
            n_jit_only_raised += 1

        # Periodic accuracy check.
        if (pi % args.accuracy_every == 0
            and not row["baseline_raised"]
            and not row["jitter_raised"]):
            e_b = _per_state_recon_err(states, U_b, S_b, V_b)
            e_j = _per_state_recon_err(states, U_j, S_j, V_j)
            max_diff = float((e_b - e_j).abs().max().item())
            row["accuracy_checked"] = True
            row["max_recon_err_diff"] = max_diff
            n_acc_checked += 1
            if max_diff > 1e-4:
                n_acc_mismatch += 1

        per_prompt.append(row)

        if (pi + 1) % args.print_every == 0 or (pi + 1) == len(idxs):
            elapsed = perf_counter() - overall_start
            eta = elapsed / (pi + 1) * (len(idxs) - pi - 1)
            print(f"  [{pi+1:>4d}/{len(idxs)}]  "
                  f"baseline_raised={n_base_raised:>3d}  jitter_raised={n_jit_raised:>3d}  "
                  f"recoveries={n_jit_recoveries:>3d}  "
                  f"acc_checked={n_acc_checked}  acc_mismatch={n_acc_mismatch}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        del states
        if U_b is not None: del U_b, S_b, V_b
        if U_j is not None: del U_j, S_j, V_j
        gc.collect()
        if torch.cuda.is_available() and pi % 50 == 49:
            torch.cuda.empty_cache()

    # Write CSVs.
    pp_path = out_dir / "per_prompt.csv"
    with open(pp_path, "w", newline="") as f:
        keys = list(per_prompt[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_prompt:
            w.writerow(r)
    f_path = out_dir / "failures.csv"
    with open(f_path, "w", newline="") as f:
        keys = ["prompt_idx_in_perm", "mmlu_index", "subject", "variant", "err",
                "seq_len", "abs_max", "fro_min", "fro_median"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in failures:
            w.writerow(r)

    elapsed = perf_counter() - overall_start
    summary = []
    summary.append("=" * 80)
    summary.append(f"chol_v6 fp32 — 1000-prompt raise-rate experiment")
    summary.append("=" * 80)
    summary.append(f"  n_prompts          : {len(idxs)}")
    summary.append(f"  jitter_factor      : {args.jitter_factor:.0e}")
    summary.append(f"  wall-clock         : {elapsed:.0f}s  ({elapsed/len(idxs)*1000:.0f} ms/prompt)")
    summary.append("")
    summary.append(f"  baseline raised    : {n_base_raised}/{len(idxs)} ({100*n_base_raised/len(idxs):.2f}%)")
    summary.append(f"  jitter raised      : {n_jit_raised}/{len(idxs)} ({100*n_jit_raised/len(idxs):.2f}%)")
    summary.append(f"  recoveries by jit  : {n_jit_recoveries}  (baseline raised, jitter ok)")
    summary.append(f"  jitter regressions : {n_jit_only_raised}  (jitter raised, baseline ok)")
    summary.append("")
    summary.append(f"  accuracy checks    : {n_acc_checked}")
    summary.append(f"  accuracy mismatches: {n_acc_mismatch}  (recon err diff > 1e-4)")
    if failures:
        from collections import Counter
        subj_counts = Counter([r["subject"] for r in failures if r["variant"] == "baseline"])
        summary.append("")
        summary.append("Top MMLU subjects in baseline failures:")
        for s, c in subj_counts.most_common(10):
            summary.append(f"  {s:<40s}  {c}")
    s_text = "\n".join(summary)
    print()
    print(s_text)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(s_text + "\n")
    print(f"\nWrote {pp_path}")
    print(f"Wrote {f_path}")
    print(f"Wrote {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()

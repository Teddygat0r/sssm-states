"""
Per-state trigger analysis for `chol_v6` fp32 on Qwen3.5-4B recurrent states.

For each of N prompts:
  1. Run forward pass, get the [768, 128, 128] state batch.
  2. Compute the dense SVD of every state to record its spectrum.
  3. Call `randomized_svd_eigh(state[None], orth='chol_v6', power_dtype=fp32)`
     on **each state individually** — so a raise is attributable to *that*
     state, not "somewhere in the batch".
  4. Record which states raised, with the exact exception text.
  5. Compare the spectrum of raising vs non-raising states.

The goal: identify the spectral signature that predicts a raise so the
user can decide when chol_v6 fp32 is safe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsvd_eigh import randomized_svd_eigh  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


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
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--oversample", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "eval_results" / datetime.now().strftime(
        "rsvd_chol_v6_trigger_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    q = args.rank + args.oversample
    print(f"rank={args.rank}, oversample={args.oversample}, sketch dim q={q}")

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

    all_rows = []  # one per (prompt, state) — 768 per prompt

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

        # Dense SVD spectrum per state.
        sv = torch.linalg.svdvals(states)        # [N, 128]
        # Per-state diagnostics.
        sv_max = sv[:, 0]                         # σ_0
        sv_q = sv[:, q - 1]                       # σ_{q-1}
        sv_rank = sv[:, args.rank - 1]            # σ_{rank-1}
        sv_min = sv[:, -1]                        # σ_min
        sv_tail_after_rank_l2 = (sv[:, args.rank:] ** 2).sum(dim=-1).sqrt()  # ‖tail‖₂ past rank
        sv_at_q_ratio = sv_q / sv_max.clamp_min(1e-30)
        sv_min_ratio = sv_min / sv_max.clamp_min(1e-30)
        sv_rank_ratio = sv_rank / sv_max.clamp_min(1e-30)

        sv_max = sv_max.cpu()
        sv_q = sv_q.cpu()
        sv_rank = sv_rank.cpu()
        sv_min = sv_min.cpu()
        sv_at_q_ratio = sv_at_q_ratio.cpu()
        sv_min_ratio = sv_min_ratio.cpu()
        sv_rank_ratio = sv_rank_ratio.cpu()
        sv_tail_after_rank_l2 = sv_tail_after_rank_l2.cpu()

        # Frobenius norm and absmax of state itself.
        abs_max = states.reshape(N, -1).abs().amax(dim=-1).cpu()
        fro = states.reshape(N, -1).norm(dim=-1).cpu()

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  N={N}  T={ids.shape[1]}")
        n_raised = 0
        n_ok = 0
        # Run chol_v6 fp32 per-state. Reuse the same RNG seed for fairness.
        for n in range(N):
            A_n = states[n].unsqueeze(0).contiguous()   # [1, 128, 128]
            torch.manual_seed(0)
            try:
                U, S, Vh = randomized_svd_eigh(
                    A_n, rank=args.rank, n_iter=2, oversample=args.oversample,
                    orth="chol_v6", power_dtype=torch.float32,
                )
                has_nan = bool(torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(Vh).any())
                raised = False
                err_msg = ""
            except Exception as e:  # noqa: BLE001
                raised = True
                has_nan = False
                err_msg = type(e).__name__ + ": " + str(e)[:200]
            if raised:
                n_raised += 1
            else:
                n_ok += 1
            all_rows.append({
                "prompt_id": f"mmlu_{i}",
                "subject": ex["subject"],
                "state_idx": n,
                "raised": raised,
                "has_nan": has_nan,
                "err_msg": err_msg,
                "sigma_max": float(sv_max[n]),
                "sigma_rank_minus1": float(sv_rank[n]),
                "sigma_q_minus1": float(sv_q[n]),
                "sigma_min": float(sv_min[n]),
                "sigma_rank_ratio": float(sv_rank_ratio[n]),
                "sigma_q_ratio": float(sv_at_q_ratio[n]),
                "sigma_min_ratio": float(sv_min_ratio[n]),
                "tail_l2_after_rank": float(sv_tail_after_rank_l2[n]),
                "state_absmax": float(abs_max[n]),
                "state_fro": float(fro[n]),
            })
        print(f"  per-state chol_v6 fp32: raised={n_raised}/{N}  ok={n_ok}/{N}")

        del states, sv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV.
    csv_path = out_dir / "per_state.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}  ({len(all_rows)} rows)")

    # Quantile analysis.
    def q(xs, qq):
        xs = sorted(xs); n = len(xs)
        if n == 0: return float("nan")
        idx = max(0, min(n - 1, int(round(qq * (n - 1)))))
        return xs[idx]

    raised = [r for r in all_rows if r["raised"]]
    ok = [r for r in all_rows if not r["raised"]]
    print()
    print("=" * 110)
    print(f"Per-state chol_v6 fp32 summary across {args.n_prompts} prompts × 768 states")
    print(f"  total raised: {len(raised)}/{len(all_rows)} ({100*len(raised)/len(all_rows):.1f}%)")
    print(f"  total OK    : {len(ok)}/{len(all_rows)} ({100*len(ok)/len(all_rows):.1f}%)")
    print("=" * 110)

    print()
    print(f"{'metric':<28}{'group':<10}{'min':>14}{'p1':>14}{'p10':>14}{'p50':>14}{'p90':>14}{'p99':>14}{'max':>14}")
    for name in ["sigma_max", "sigma_rank_ratio", "sigma_q_ratio", "sigma_min_ratio",
                 "tail_l2_after_rank", "state_absmax", "state_fro"]:
        for label, group in [("raised", raised), ("ok", ok)]:
            xs = [r[name] for r in group]
            if not xs:
                continue
            print(f"{name:<28}{label:<10}"
                  f"{min(xs):>14.4g}{q(xs,0.01):>14.4g}{q(xs,0.10):>14.4g}{q(xs,0.50):>14.4g}"
                  f"{q(xs,0.90):>14.4g}{q(xs,0.99):>14.4g}{max(xs):>14.4g}")

    # Best-single-feature classifier: threshold on sigma_q_ratio.
    if raised and ok:
        # For different thresholds on sigma_q_ratio, compute precision/recall.
        # Predict "will raise" iff sigma_q_ratio < threshold.
        thresholds = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        print()
        print("Single-feature trigger: state raises iff sigma_q_ratio = σ[q-1]/σ_0 < T")
        print(f"{'threshold T':<14}{'precision':>12}{'recall':>10}{'predicted_raise':>20}")
        for t in thresholds:
            pred_raise_pos = sum(1 for r in raised if r["sigma_q_ratio"] < t)
            pred_raise_neg = sum(1 for r in ok     if r["sigma_q_ratio"] < t)
            total_pred = pred_raise_pos + pred_raise_neg
            prec = pred_raise_pos / total_pred if total_pred else 0.0
            rec  = pred_raise_pos / len(raised)
            print(f"{t:<14.0e}{prec:>12.3f}{rec:>10.3f}{total_pred:>20d}")


if __name__ == "__main__":
    main()

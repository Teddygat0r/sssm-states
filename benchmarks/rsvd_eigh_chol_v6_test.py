"""
Robustness + accuracy + speed test for `orth="chol_v6"` (the new ladder)
vs. `cholqr2` and `house` on Qwen3.5-4B recurrent states.

The chol_v6 docstring warns that the jitter ladder can return a non-
orthonormal Q on rank-deficient inputs (orth defect ∝ jitter size). This
script measures that defect directly via `||UᵀU - I||_F / sqrt(k)` along
with the usual reconstruction error vs the dense truncated SVD optimum.

Per prompt we run a per-method timing loop (3 warmup + 10 iters) so the
runtime numbers are steady-state.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict, field
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


@torch.no_grad()
def _per_state_recon_errors(A, U, S, Vh):
    recon = U * S.unsqueeze(-2)
    recon = recon @ Vh
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return (num / den).cpu()


@torch.no_grad()
def _orth_defect(U):
    # ||UᵀU - I||_F / sqrt(k), per state.
    k = U.shape[-1]
    G = U.transpose(-2, -1) @ U
    eye = torch.eye(k, device=U.device, dtype=U.dtype)
    diff = (G - eye).reshape(U.shape[0], -1).norm(dim=-1)
    return (diff / (k ** 0.5)).cpu()


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_iter_time", type=int, default=10)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / datetime.now().strftime("rsvd_chol_v6_%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

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
        ("chol_v6",  torch.float32),
        ("chol_v6",  torch.bfloat16),
        ("cholqr2",  torch.float32),
        ("cholqr2",  torch.bfloat16),
        ("house",    torch.float32),
        ("house",    torch.bfloat16),
    ]

    rows: list[dict] = []

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

        # Dense truncated SVD ground truth.
        gtU, gtS, gtVh = torch.linalg.svd(states, full_matrices=False)
        gtU = gtU[..., :16]; gtS = gtS[..., :16]; gtVh = gtVh[..., :16, :]
        gt_opt_err = _per_state_recon_errors(states, gtU, gtS, gtVh)

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  batch={tuple(states.shape)}  T={ids.shape[1]}")

        for orth, pdt in method_specs:
            pdt_name = str(pdt).replace("torch.", "")
            raised = False
            err_type = ""
            err_msg = ""
            try:
                # Warmup
                for _ in range(args.n_warmup):
                    U, S, Vh = randomized_svd_eigh(
                        states, rank=16, n_iter=2, oversample=8,
                        orth=orth, power_dtype=pdt,
                    )
                _sync()
                t0 = perf_counter()
                for _ in range(args.n_iter_time):
                    U, S, Vh = randomized_svd_eigh(
                        states, rank=16, n_iter=2, oversample=8,
                        orth=orth, power_dtype=pdt,
                    )
                _sync()
                rt = (perf_counter() - t0) / args.n_iter_time
            except Exception as e:
                _sync()
                raised = True
                err_type = type(e).__name__
                err_msg = str(e)[:200]
                print(f"  {orth:<9} pdt={pdt_name:<9}  RAISED {err_type}: {err_msg[:80]}")
                rows.append({
                    "prompt_id": f"mmlu_{i}", "subject": ex["subject"],
                    "method": orth, "power_dtype": pdt_name,
                    "raised": True, "error_type": err_type, "error_msg": err_msg,
                    "runtime_ms": float("nan"),
                    "recon_err_mean": float("nan"),
                    "recon_err_p99": float("nan"),
                    "max_gap_abs": float("nan"),
                    "max_gap_rel": float("nan"),
                    "orth_defect_mean": float("nan"),
                    "orth_defect_max": float("nan"),
                    "orth_defect_p99": float("nan"),
                    "has_nan": False, "has_inf": False,
                })
                continue

            # Use the final (post-loop) U,S,Vh for accuracy stats.
            Uf, Sf, Vhf = U.float(), S.float(), Vh.float()
            has_nan = bool(torch.isnan(Uf).any() or torch.isnan(Sf).any() or torch.isnan(Vhf).any())
            has_inf = bool(torch.isinf(Uf).any() or torch.isinf(Sf).any() or torch.isinf(Vhf).any())

            if has_nan or has_inf:
                print(f"  {orth:<9} pdt={pdt_name:<9}  NaN/Inf  rt={rt*1000:.1f}ms")
                rows.append({
                    "prompt_id": f"mmlu_{i}", "subject": ex["subject"],
                    "method": orth, "power_dtype": pdt_name,
                    "raised": False, "error_type": "", "error_msg": "",
                    "runtime_ms": rt * 1000,
                    "recon_err_mean": float("nan"),
                    "recon_err_p99": float("nan"),
                    "max_gap_abs": float("nan"),
                    "max_gap_rel": float("nan"),
                    "orth_defect_mean": float("nan"),
                    "orth_defect_max": float("nan"),
                    "orth_defect_p99": float("nan"),
                    "has_nan": has_nan, "has_inf": has_inf,
                })
                continue

            err = _per_state_recon_errors(states, Uf, Sf, Vhf)
            gap = err - gt_opt_err
            rel = gap / gt_opt_err.clamp_min(1e-3)
            defect = _orth_defect(Uf)

            n_over_5pct = int((gap > 0.05).sum().item())
            print(f"  {orth:<9} pdt={pdt_name:<9}  rt={rt*1000:>6.1f}ms  "
                  f"err_mean={float(err.mean()):.4f}  err_p99={float(err.quantile(0.99)):.4f}  "
                  f"max_gap_abs={float(gap.max()):+.4f}  "
                  f"orth_defect mean={float(defect.mean()):.2e} p99={float(defect.quantile(0.99)):.2e} "
                  f"max={float(defect.max()):.2e}  "
                  f"states>5pct_abs={n_over_5pct}")
            rows.append({
                "prompt_id": f"mmlu_{i}", "subject": ex["subject"],
                "method": orth, "power_dtype": pdt_name,
                "raised": False, "error_type": "", "error_msg": "",
                "runtime_ms": rt * 1000,
                "recon_err_mean": float(err.mean()),
                "recon_err_p99": float(err.quantile(0.99)),
                "max_gap_abs": float(gap.max()),
                "max_gap_rel": float(rel.max()),
                "orth_defect_mean": float(defect.mean()),
                "orth_defect_max": float(defect.max()),
                "orth_defect_p99": float(defect.quantile(0.99)),
                "has_nan": False, "has_inf": False,
                "n_states_gap_over_5pct": n_over_5pct,
            })

        del states, gtU, gtS, gtVh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV.
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(rows[0].keys())
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Final aggregate.
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n_runs=0, n_raised=0, runtimes=[], errs=[],
                                    p99s=[], gaps=[], defects=[], over_5pct=0))
    for r in rows:
        k = (r["method"], r["power_dtype"])
        a = agg[k]
        a["n_runs"] += 1
        if r["raised"]:
            a["n_raised"] += 1
            continue
        if r.get("recon_err_mean") and r["recon_err_mean"] == r["recon_err_mean"]:
            a["runtimes"].append(r["runtime_ms"])
            a["errs"].append(r["recon_err_mean"])
            a["p99s"].append(r["recon_err_p99"])
            a["gaps"].append(r["max_gap_abs"])
            a["defects"].append(r["orth_defect_max"])
            a["over_5pct"] += r.get("n_states_gap_over_5pct", 0)

    print()
    print("=" * 120)
    print(f"Aggregate over {args.n_prompts} prompts")
    print("=" * 120)
    print(f"{'method':<10}{'pdt':<10}{'runs':>5}{'raised':>8}{'rt_avg(ms)':>13}"
          f"{'err_avg':>10}{'p99_avg':>10}{'gap_max':>10}"
          f"{'orth_def_avg':>14}{'orth_def_max':>14}{'over_5pct':>12}")
    for (m, pdt), a in sorted(agg.items()):
        if a["runtimes"]:
            rt = sum(a["runtimes"]) / len(a["runtimes"])
            er = sum(a["errs"]) / len(a["errs"])
            p9 = sum(a["p99s"]) / len(a["p99s"])
            gm = max(a["gaps"])
            dm = sum(a["defects"]) / len(a["defects"])
            dM = max(a["defects"])
        else:
            rt = er = p9 = gm = dm = dM = float("nan")
        print(f"{m:<10}{pdt:<10}{a['n_runs']:>5}{a['n_raised']:>8}{rt:>13.2f}"
              f"{er:>10.4f}{p9:>10.4f}{gm:>+10.4f}{dm:>14.3e}{dM:>14.3e}{a['over_5pct']:>12}")


if __name__ == "__main__":
    main()

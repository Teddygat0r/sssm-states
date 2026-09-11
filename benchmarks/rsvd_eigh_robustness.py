"""
Robustness test for `randomized_svd_eigh` (rsvd_eigh.py) on Qwen3.5-4B recurrent
states.

For each prompt sampled from MMLU (cais/mmlu/all):
  1. Prefill through Qwen/Qwen3.5-4B with use_cache=True.
  2. Stack all per-(linear-layer, v-head) recurrent states into a single
     [N, head_k_dim, head_v_dim] = [768, 128, 128] tensor.
  3. Call `randomized_svd_eigh` on the batch with each combination of
     orth in {chol_min, cholqr2, house} and power_dtype in {fp32, bf16}.
  4. Compare against the ground-truth rank-k truncated SVD computed via
     `torch.linalg.svd` on the same batch (in fp32).

A "failure" is any of:
  - the call raises an exception (e.g. LinAlgError from Cholesky)
  - any U, S, or Vh entry is NaN/Inf
  - per-state relative Frobenius reconstruction error exceeds the truncated
    rank-k optimum by more than 5% in absolute or 50% relative terms

We log per-state metrics and per-prompt aggregates, then dump them all to CSV
and a markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch

# Make the repo root importable so we can `import rsvd_eigh`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsvd_eigh import randomized_svd_eigh  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ------------------------------------------------------------------ helpers --

def _find_linear_layers(model) -> list[int]:
    out = []
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            out.append(li)
    return out


def _format_mmlu_prompt(ex: dict) -> str:
    """Format an MMLU example as a single prompt string."""
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    return (
        f"The following is a multiple choice question about "
        f"{ex['subject'].replace('_', ' ')}.\n\n"
        f"Question: {ex['question']}\n\n"
        f"{body}\n\n"
        f"Answer:"
    )


def _sample_mmlu_prompts(n: int, seed: int = 0) -> list[dict]:
    """Sample n random MMLU prompts from cais/mmlu split='test'."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
    out = []
    for i in idxs:
        ex = ds[i]
        out.append({
            "id": f"mmlu_{i}",
            "subject": ex["subject"],
            "text": _format_mmlu_prompt(ex),
        })
    return out


# ----------------------------------------------------------------- metrics --

@torch.no_grad()
def _state_stats(states_fp32: torch.Tensor) -> dict:
    """Cheap per-batch summary stats on the raw state batch."""
    # states_fp32: [N, m, n] on GPU
    flat = states_fp32.reshape(states_fp32.shape[0], -1)
    abs_max = flat.abs().amax(dim=-1)             # [N]
    fro = flat.norm(dim=-1)                       # [N]
    return {
        "abs_max_min": float(abs_max.min().item()),
        "abs_max_max": float(abs_max.max().item()),
        "abs_max_median": float(abs_max.median().item()),
        "fro_min": float(fro.min().item()),
        "fro_max": float(fro.max().item()),
        "fro_median": float(fro.median().item()),
    }


@torch.no_grad()
def _truncated_svd_groundtruth(A: torch.Tensor, rank: int):
    """Reference truncated SVD. A is [N, m, n] fp32. Returns U,S,Vh truncated to rank."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[..., :rank], S[..., :rank], Vh[..., :rank, :]


@torch.no_grad()
def _per_state_recon_errors(A: torch.Tensor, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor):
    """Per-state relative Frobenius reconstruction error: ||A - U diag(S) Vh||_F / ||A||_F."""
    recon = U * S.unsqueeze(-2)            # [..., m, k] scale columns of U by S
    recon = recon @ Vh                     # [..., m, n]
    diff = (A - recon).reshape(A.shape[0], -1)
    num = diff.norm(dim=-1)
    den = A.reshape(A.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
    return (num / den).cpu()


@torch.no_grad()
def _true_singular_values(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(A)


# ------------------------------------------------------------------ method --

@dataclass
class MethodResult:
    method: str
    power_dtype: str
    raised: bool
    error_type: str
    error_msg: str
    has_nan: bool
    has_inf: bool
    runtime_s: float
    # per-prompt error stats (only if not raised + no NaN)
    recon_err_min: float = float("nan")
    recon_err_max: float = float("nan")
    recon_err_mean: float = float("nan")
    recon_err_p99: float = float("nan")
    gap_vs_optimum_max: float = float("nan")
    gap_vs_optimum_mean: float = float("nan")
    rel_gap_max: float = float("nan")
    rel_gap_mean: float = float("nan")
    num_states_over_5pct_abs: int = 0
    num_states_over_50pct_rel: int = 0
    sv_top1_max_abs_err: float = float("nan")
    sv_topk_max_abs_err: float = float("nan")


@torch.no_grad()
def _run_one_method(
    states_fp32: torch.Tensor,
    rank: int,
    orth: str,
    power_dtype: torch.dtype,
    A_input_dtype: torch.dtype,
    gt_U, gt_S, gt_Vh, gt_optimum_err,
) -> MethodResult:
    """Run one method on the state batch and compute metrics."""
    A_in = states_fp32.to(A_input_dtype) if A_input_dtype != torch.float32 else states_fp32
    torch.cuda.synchronize() if states_fp32.is_cuda else None
    t0 = perf_counter()
    try:
        U, S, Vh = randomized_svd_eigh(
            A_in, rank=rank, n_iter=2, oversample=8,
            orth=orth, power_dtype=power_dtype,
        )
        if states_fp32.is_cuda:
            torch.cuda.synchronize()
        rt = perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        if states_fp32.is_cuda:
            torch.cuda.synchronize()
        rt = perf_counter() - t0
        return MethodResult(
            method=orth,
            power_dtype=str(power_dtype).replace("torch.", ""),
            raised=True,
            error_type=type(e).__name__,
            error_msg=str(e)[:240],
            has_nan=False,
            has_inf=False,
            runtime_s=rt,
        )

    has_nan = bool(torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(Vh).any())
    has_inf = bool(torch.isinf(U).any() or torch.isinf(S).any() or torch.isinf(Vh).any())

    if has_nan or has_inf:
        return MethodResult(
            method=orth,
            power_dtype=str(power_dtype).replace("torch.", ""),
            raised=False,
            error_type="",
            error_msg="",
            has_nan=has_nan,
            has_inf=has_inf,
            runtime_s=rt,
        )

    # Cast back to fp32 for fair comparison.
    U_f, S_f, Vh_f = U.float(), S.float(), Vh.float()
    err = _per_state_recon_errors(states_fp32, U_f, S_f, Vh_f)
    gap = (err - gt_optimum_err)                 # absolute gap
    # Relative gap: only meaningful when the optimum itself is non-trivial.
    # When the optimum is near zero (state is ~rank-16), a tiny absolute
    # excess becomes a huge relative number and is not actually a failure.
    safe_den = gt_optimum_err.clamp_min(1e-3)
    rel = (gap / safe_den)

    # singular-value comparison: top-1 and top-k max abs error.
    # S can be unordered? It's descending by construction. compare to gt_S.
    S_f_sorted, _ = torch.sort(S_f, dim=-1, descending=True)
    gt_S_sorted, _ = torch.sort(gt_S, dim=-1, descending=True)
    sv_diff = (S_f_sorted - gt_S_sorted).abs()
    sv_top1 = float(sv_diff[..., 0].max().item())
    sv_topk = float(sv_diff.max().item())

    return MethodResult(
        method=orth,
        power_dtype=str(power_dtype).replace("torch.", ""),
        raised=False,
        error_type="",
        error_msg="",
        has_nan=False,
        has_inf=False,
        runtime_s=rt,
        recon_err_min=float(err.min().item()),
        recon_err_max=float(err.max().item()),
        recon_err_mean=float(err.mean().item()),
        recon_err_p99=float(err.quantile(0.99).item()),
        gap_vs_optimum_max=float(gap.max().item()),
        gap_vs_optimum_mean=float(gap.mean().item()),
        rel_gap_max=float(rel.max().item()),
        rel_gap_mean=float(rel.mean().item()),
        num_states_over_5pct_abs=int((gap > 0.05).sum().item()),
        num_states_over_50pct_rel=int((rel > 0.50).sum().item()),
        sv_top1_max_abs_err=sv_top1,
        sv_topk_max_abs_err=sv_topk,
    )


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_seq_len", type=int, default=512,
                    help="Truncate (or no-op pad) prompts to this many tokens.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / datetime.now().strftime("rsvd_robustness_%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ---- prompts ---------------------------------------------------------
    print(f"Sampling {args.n_prompts} MMLU prompts ...")
    prompts = _sample_mmlu_prompts(args.n_prompts, seed=args.seed)

    # ---- model -----------------------------------------------------------
    print(f"Loading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    linear_layers = _find_linear_layers(model)
    cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
    num_v_heads = cfg.linear_num_value_heads
    print(f"  linear-attn layers: {linear_layers}  ({len(linear_layers)} of {cfg.num_hidden_layers})")
    print(f"  num_v_heads={num_v_heads}, head dims=({cfg.linear_key_head_dim},{cfg.linear_value_head_dim})")

    # ---- sweep -----------------------------------------------------------
    method_specs = [
        ("chol_min", torch.float32),
        ("chol_min", torch.bfloat16),
        ("cholqr2", torch.float32),
        ("cholqr2", torch.bfloat16),
        ("house",   torch.float32),
        ("house",   torch.bfloat16),
    ]

    per_prompt_rows: list[dict] = []
    failure_rows: list[dict] = []

    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi+1}/{len(prompts)}] {prompt['id']} ({prompt['subject']})")
        ids = tokenizer(prompt["text"], return_tensors="pt").input_ids
        if ids.shape[1] > args.max_seq_len:
            ids = ids[:, : args.max_seq_len]
        ids = ids.to(DEVICE)
        seq_len = int(ids.shape[1])

        t0 = perf_counter()
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_s = perf_counter() - t0

        cache = out.past_key_values
        # Stack recurrent states for each linear layer: each [1, V, Dk, Dv].
        per_layer_states = []
        for li in linear_layers:
            s = cache.layers[li].recurrent_states  # [1, V, Dk, Dv]
            per_layer_states.append(s.squeeze(0).detach())  # [V, Dk, Dv]
        batch = torch.cat(per_layer_states, dim=0)          # [L*V, Dk, Dv]
        batch_fp32 = batch.to(dtype=torch.float32)
        N = batch_fp32.shape[0]
        print(f"  prefill {prefill_s:.2f}s; T={seq_len}; batch shape={tuple(batch_fp32.shape)}")

        # Free model output now that we have the states.
        del out, cache, per_layer_states, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Per-state ground truth.
        t0 = perf_counter()
        gt_U, gt_S, gt_Vh = _truncated_svd_groundtruth(batch_fp32, args.rank)
        gt_optimum_err = _per_state_recon_errors(batch_fp32, gt_U, gt_S, gt_Vh)
        gt_full_sv = _true_singular_values(batch_fp32)
        gt_min_sv = gt_full_sv.min(dim=-1).values.cpu()
        gt_max_sv = gt_full_sv.max(dim=-1).values.cpu()
        gt_sv_at_q = gt_full_sv[..., args.rank + 8 - 1].cpu()   # sketch dim q-th singular value
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gt_s = perf_counter() - t0
        # Detect "rank-deficient at q" states (the chol_min trap).
        # Use multiple thresholds — the algorithm is sensitive to tiny tail SVs
        # because the Cholesky on the Gram fails when the tail is too small.
        eff_full_rank_1em6 = (gt_full_sv > gt_full_sv.max(dim=-1, keepdim=True).values * 1e-6).sum(dim=-1).cpu()
        eff_full_rank_1em4 = (gt_full_sv > gt_full_sv.max(dim=-1, keepdim=True).values * 1e-4).sum(dim=-1).cpu()
        eff_full_rank_1em3 = (gt_full_sv > gt_full_sv.max(dim=-1, keepdim=True).values * 1e-3).sum(dim=-1).cpu()
        rank_deficient_q_1em6 = int((eff_full_rank_1em6 < args.rank + 8).sum().item())
        rank_deficient_q_1em4 = int((eff_full_rank_1em4 < args.rank + 8).sum().item())
        rank_deficient_q_1em3 = int((eff_full_rank_1em3 < args.rank + 8).sum().item())
        # Smallest condition-number-style ratio sv[q-1]/sv[0] over states.
        cond_q = (gt_full_sv[..., args.rank + 8 - 1] /
                  gt_full_sv[..., 0].clamp_min(1e-30)).cpu()
        cond_q_min = float(cond_q.min().item())
        cond_q_median = float(cond_q.median().item())
        rank_deficient_q = rank_deficient_q_1em6   # legacy field
        print(f"  groundtruth SVD: {gt_s:.2f}s; "
              f"states with sv[q-1]/sv[0]<1e-6: {rank_deficient_q_1em6}/{N}, "
              f"<1e-4: {rank_deficient_q_1em4}/{N}, "
              f"<1e-3: {rank_deficient_q_1em3}/{N}; "
              f"min(sv[q-1]/sv[0])={cond_q_min:.2e}, median={cond_q_median:.2e}")

        stats = _state_stats(batch_fp32)
        print(f"  state abs_max: median={stats['abs_max_median']:.3g}, "
              f"min={stats['abs_max_min']:.3g}, max={stats['abs_max_max']:.3g}")

        for orth, pdt in method_specs:
            res = _run_one_method(
                batch_fp32, args.rank, orth, pdt,
                A_input_dtype=torch.float32,  # we always feed fp32 input; power_dtype controls internals
                gt_U=gt_U, gt_S=gt_S, gt_Vh=gt_Vh,
                gt_optimum_err=gt_optimum_err,
            )
            tag = f"{orth:<9} pdt={str(pdt).replace('torch.',''):<9}"
            if res.raised:
                print(f"    {tag}  RAISED {res.error_type}: {res.error_msg[:80]} (t={res.runtime_s*1000:.1f}ms)")
                failure_rows.append({
                    "prompt_id": prompt["id"],
                    "subject": prompt["subject"],
                    "method": res.method,
                    "power_dtype": res.power_dtype,
                    "failure_type": "exception",
                    "detail": f"{res.error_type}: {res.error_msg}",
                    "rank_deficient_q_states": rank_deficient_q,
                    "n_states": N,
                })
            elif res.has_nan or res.has_inf:
                print(f"    {tag}  NaN/Inf in output (NaN={res.has_nan}, Inf={res.has_inf})")
                failure_rows.append({
                    "prompt_id": prompt["id"],
                    "subject": prompt["subject"],
                    "method": res.method,
                    "power_dtype": res.power_dtype,
                    "failure_type": "nan_or_inf",
                    "detail": f"nan={res.has_nan}, inf={res.has_inf}",
                    "rank_deficient_q_states": rank_deficient_q,
                    "n_states": N,
                })
            else:
                print(f"    {tag}  err mean={res.recon_err_mean:.4f} "
                      f"p99={res.recon_err_p99:.4f} max_gap_abs={res.gap_vs_optimum_max:+.4f} "
                      f"max_gap_rel={res.rel_gap_max:+.3f} "
                      f"sv_top1_err={res.sv_top1_max_abs_err:.3e} "
                      f"t={res.runtime_s*1000:.1f}ms")
                if res.num_states_over_5pct_abs or res.num_states_over_50pct_rel:
                    failure_rows.append({
                        "prompt_id": prompt["id"],
                        "subject": prompt["subject"],
                        "method": res.method,
                        "power_dtype": res.power_dtype,
                        "failure_type": "excess_recon_error",
                        "detail": (
                            f"over_5pct_abs={res.num_states_over_5pct_abs}, "
                            f"over_50pct_rel={res.num_states_over_50pct_rel}, "
                            f"max_gap_abs={res.gap_vs_optimum_max:.4f}, "
                            f"max_gap_rel={res.rel_gap_max:.3f}"
                        ),
                        "rank_deficient_q_states": rank_deficient_q,
                        "n_states": N,
                    })

            per_prompt_rows.append({
                "prompt_id": prompt["id"],
                "subject": prompt["subject"],
                "seq_len": seq_len,
                "n_states": N,
                "rank_deficient_q_states": rank_deficient_q,
                "abs_max_min": stats["abs_max_min"],
                "abs_max_max": stats["abs_max_max"],
                "abs_max_median": stats["abs_max_median"],
                **asdict(res),
            })

        # cleanup per-prompt
        del batch_fp32, gt_U, gt_S, gt_Vh, gt_optimum_err, gt_full_sv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- write outputs ---------------------------------------------------
    csv_path = out_dir / "per_prompt_results.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(per_prompt_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_prompt_rows:
            w.writerow(r)

    fail_path = out_dir / "failures.csv"
    with open(fail_path, "w", newline="") as f:
        keys = ["prompt_id", "subject", "method", "power_dtype", "failure_type",
                "detail", "rank_deficient_q_states", "n_states"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in failure_rows:
            w.writerow(r)

    # Aggregate per (method, power_dtype).
    methods = sorted({(r["method"], r["power_dtype"]) for r in per_prompt_rows})
    agg_rows = []
    for m, pdt in methods:
        rs = [r for r in per_prompt_rows if r["method"] == m and r["power_dtype"] == pdt]
        n_runs = len(rs)
        n_raised = sum(1 for r in rs if r["raised"])
        n_naninf = sum(1 for r in rs if (not r["raised"]) and (r["has_nan"] or r["has_inf"]))
        n_excess = sum(
            1 for r in rs
            if (not r["raised"]) and not (r["has_nan"] or r["has_inf"])
            and (r["num_states_over_5pct_abs"] or r["num_states_over_50pct_rel"])
        )
        n_ok = n_runs - n_raised - n_naninf - n_excess
        ok_rs = [r for r in rs if not r["raised"] and not r["has_nan"] and not r["has_inf"]]
        def mean(name):
            xs = [r[name] for r in ok_rs if r[name] == r[name]]
            return sum(xs) / len(xs) if xs else float("nan")
        agg_rows.append({
            "method": m,
            "power_dtype": pdt,
            "n_runs": n_runs,
            "n_raised": n_raised,
            "n_nan_or_inf": n_naninf,
            "n_excess_recon_err": n_excess,
            "n_ok": n_ok,
            "mean_recon_err": mean("recon_err_mean"),
            "mean_recon_err_p99": mean("recon_err_p99"),
            "mean_gap_vs_optimum": mean("gap_vs_optimum_mean"),
            "mean_runtime_s": mean("runtime_s"),
            "mean_sv_top1_err": mean("sv_top1_max_abs_err"),
        })

    agg_path = out_dir / "aggregate.csv"
    with open(agg_path, "w", newline="") as f:
        keys = list(agg_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    # Print final summary.
    print("\n" + "=" * 96)
    print(
        f"Robustness summary  rank={args.rank}  n_prompts={args.n_prompts}  "
        f"states/prompt={per_prompt_rows[0]['n_states']}"
    )
    print("=" * 96)
    h = (
        f"{'method':<10}{'pdt':<10}{'n_ok':>6}{'n_raised':>10}"
        f"{'n_naninf':>10}{'n_excess':>10}{'mean_err':>12}{'mean_p99':>12}{'mean_t (ms)':>14}"
    )
    print(h)
    for r in agg_rows:
        print(
            f"{r['method']:<10}{r['power_dtype']:<10}{r['n_ok']:>6}{r['n_raised']:>10}"
            f"{r['n_nan_or_inf']:>10}{r['n_excess_recon_err']:>10}"
            f"{r['mean_recon_err']:>12.4f}{r['mean_recon_err_p99']:>12.4f}"
            f"{r['mean_runtime_s']*1000:>14.1f}"
        )

    # Save aggregate as JSON for the report writer.
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump({
            "config": {
                "model_id": MODEL_ID,
                "n_prompts": args.n_prompts,
                "max_seq_len": args.max_seq_len,
                "rank": args.rank,
                "oversample": 8,
                "n_iter": 2,
                "seed": args.seed,
                "states_per_prompt": per_prompt_rows[0]["n_states"],
            },
            "aggregate": agg_rows,
            "n_failures": len(failure_rows),
        }, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {fail_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {out_dir/'aggregate.json'}")


if __name__ == "__main__":
    main()

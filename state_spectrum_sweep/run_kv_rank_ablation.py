"""
K/V projection rank ablation.

Premise: in linear-attention models the recurrent state is

    S_h = sum_t  d_{t,h} * k_{t,h} v_{t,h}^T

where d_{t,h} is a (decay/gate) scalar and k_{t,h}, v_{t,h} are per-head
projections of the hidden state x_t. Since k_{t,h} in image(W_k_h) and
v_{t,h} in image(W_v_h), the row/column space of S_h is bounded:

    rank(S_h)  <=  min(rank(W_k_h), rank(W_v_h))

If the per-head W_k_h and W_v_h are themselves low-rank, the state inherits
low-rankness mechanically, with no spectral theorem about decay required.
This script measures:

  1. The per-head effective rank of W_k_h and W_v_h (the "data-driven baseline").
  2. The per-head effective rank of the recurrent state after a long prefill.
  3. For GatedDeltaNet: the per-head decay parameters (A_log, dt_bias) and
     the correlation between decay strength and the rank gap
        baseline_cap - state_eff_rank
     i.e. how much extra compression decay buys on top of the KV baseline.

Models:
  - DeltaNet:        fla-hub/delta_net-1.3B-100B   (16 heads x 128 head_dim)
  - GatedDeltaNet:   m-a-p/1.3B-100B-GatedDeltaNet-pure (8 heads x 256 head_dim)

Note: the conv1d on k/v is per-channel (depthwise, kernel=4) and so preserves
the column space of the linear projection. The KV-rank bound on S_h is unchanged
by the conv.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

import fla  # noqa: F401  (registers FLA classes)
from fla.models import DeltaNetForCausalLM, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


SEQ_LEN_FOR_STATE = 2048
PROMPT_TITLE = "Bob Dylan"  # one of the wikitext-2 articles used in earlier sweeps
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

MODELS = [
    {
        "label": "DeltaNet-1.3B",
        "model_id": "fla-hub/delta_net-1.3B-100B",
        "cls": DeltaNetForCausalLM,
        "family": "delta_net",
        "has_decay": False,
    },
    {
        "label": "GatedDeltaNet-1.3B",
        "model_id": "m-a-p/1.3B-100B-GatedDeltaNet-pure",
        "cls": GatedDeltaNetForCausalLM,
        "family": "gated_deltanet",
        "has_decay": True,
    },
]


# -------------------------------------------------------------- rank helpers --

def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    """Singular values on CPU/fp32."""
    return torch.linalg.svdvals(matrix.detach().to(device="cpu", dtype=torch.float32))


def _effective_rank_from_sv(sv: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank: exp(H(p)) with p_i = sigma_i^2 / sum sigma_j^2."""
    s2 = sv.pow(2)
    total = s2.sum().clamp_min(eps)
    p = s2 / total
    safe = p.clamp_min(eps)
    h = -(p * safe.log()).sum()
    return float(h.exp().item())


def _numerical_rank_at_energy(sv: torch.Tensor, frac: float) -> int:
    """Smallest k such that the top-k singular values capture >= frac of total energy."""
    s2 = sv.pow(2)
    total = s2.sum().clamp_min(1e-12)
    cum = torch.cumsum(s2, dim=0) / total
    k = int((cum >= frac).nonzero(as_tuple=True)[0][0].item()) + 1
    return k


# -------------------------------------------------------------- weight extract -

def _per_head_kv_weights(model, family: str) -> dict[str, torch.Tensor]:
    """Return per-head W_k and W_v with shapes [num_layers, num_heads, head_dim, hidden_size].

    The conv1d in the forward path is per-channel and preserves column space, so
    we measure W_k / W_v directly.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    if family == "delta_net":
        num_heads = cfg.num_heads
        head_dim_k = hidden // num_heads
        head_dim_v = hidden // num_heads
    elif family == "gated_deltanet":
        num_heads = cfg.num_heads
        head_dim_k = cfg.head_dim
        head_dim_v = cfg.head_dim
    else:
        raise ValueError(f"unknown family {family}")

    Wk_all = torch.empty(num_layers, num_heads, head_dim_k, hidden)
    Wv_all = torch.empty(num_layers, num_heads, head_dim_v, hidden)
    for li in range(num_layers):
        attn = model.model.layers[li].attn
        Wk = attn.k_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        Wv = attn.v_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        # Output of k_proj is [hidden] -> [num_heads * head_dim_k]; rows split per-head.
        Wk_per_head = Wk.view(num_heads, head_dim_k, hidden)
        Wv_per_head = Wv.view(num_heads, head_dim_v, hidden)
        Wk_all[li] = Wk_per_head
        Wv_all[li] = Wv_per_head
    return {"W_k": Wk_all, "W_v": Wv_all}


def _per_head_decay_params(model, family: str) -> dict[str, torch.Tensor] | None:
    if family != "gated_deltanet":
        return None
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_heads
    A_log = torch.empty(num_layers, num_heads)
    dt_bias = torch.empty(num_layers, num_heads)
    for li in range(num_layers):
        attn = model.model.layers[li].attn
        A_log[li] = attn.A_log.detach().to(device="cpu", dtype=torch.float32)
        dt_bias[li] = attn.dt_bias.detach().to(device="cpu", dtype=torch.float32)
    return {"A_log": A_log, "dt_bias": dt_bias}


# -------------------------------------------------------------- prompt loader -

def _load_wiki_article(title: str) -> str:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full = "".join(x["text"] for x in ds)
    parts = re.split(r"\n( = [^=].*? = )\n", full)
    by_title: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        t = parts[i].strip().strip("=").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        by_title[t] = body
    if title not in by_title:
        raise KeyError(f"article {title!r} not found in wikitext-2 train")
    return by_title[title]


# -------------------------------------------------------------- main per-model -

def _run_model(info: dict) -> dict:
    label = info["label"]
    print(f"\n=== {label} ({info['model_id']}) ===")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    model = info["cls"].from_pretrained(info["model_id"], torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    weights = _per_head_kv_weights(model, info["family"])
    Wk = weights["W_k"]  # [L, H, head_dim, hidden]
    Wv = weights["W_v"]
    L, H, head_dim_k, _ = Wk.shape
    head_dim_v = Wv.shape[2]
    print(f"  layers={L} heads={H} head_dim_k={head_dim_k} head_dim_v={head_dim_v}")

    decay = _per_head_decay_params(model, info["family"])

    # ---- Per-head W_k / W_v rank metrics ------------------------------------
    print("  computing per-head SVDs of W_k, W_v ...")
    weight_rows = []
    for li in range(L):
        for h in range(H):
            sv_k = _singular_values(Wk[li, h])
            sv_v = _singular_values(Wv[li, h])
            er_k = _effective_rank_from_sv(sv_k)
            er_v = _effective_rank_from_sv(sv_v)
            nr99_k = _numerical_rank_at_energy(sv_k, 0.99)
            nr99_v = _numerical_rank_at_energy(sv_v, 0.99)
            nr95_k = _numerical_rank_at_energy(sv_k, 0.95)
            nr95_v = _numerical_rank_at_energy(sv_v, 0.95)
            row = {
                "layer": li,
                "head": h,
                "head_dim_k": head_dim_k,
                "head_dim_v": head_dim_v,
                "max_rank_k": min(Wk.shape[2], Wk.shape[3]),
                "max_rank_v": min(Wv.shape[2], Wv.shape[3]),
                "eff_rank_Wk": er_k,
                "eff_rank_Wv": er_v,
                "num_rank99_Wk": nr99_k,
                "num_rank99_Wv": nr99_v,
                "num_rank95_Wk": nr95_k,
                "num_rank95_Wv": nr95_v,
                "baseline_cap_eff": min(er_k, er_v),
                "baseline_cap_num99": min(nr99_k, nr99_v),
            }
            if decay is not None:
                a_log = float(decay["A_log"][li, h].item())
                dtb = float(decay["dt_bias"][li, h].item())
                row.update({
                    "A_log": a_log,
                    "exp_A_log": math.exp(a_log),
                    "dt_bias": dtb,
                    # Effective static decay rate per token at baseline x=0:
                    #   g = -exp(A_log) * softplus(dt_bias)   (per FLA forward)
                    #   per-token retention factor exp(g)
                    "static_g": -math.exp(a_log) * math.log1p(math.exp(dtb)),
                })
            weight_rows.append(row)
    print(f"  done weight SVDs ({len(weight_rows)} rows)")

    # ---- Recurrent state per-head effective rank at long prefill ------------
    print(f"  prefilling {SEQ_LEN_FOR_STATE} tokens on '{PROMPT_TITLE}' ...")
    article_text = _load_wiki_article(PROMPT_TITLE)
    ids_full = tokenizer(article_text, return_tensors="pt").input_ids[0]
    if ids_full.shape[0] < SEQ_LEN_FOR_STATE:
        raise RuntimeError(
            f"prompt has only {ids_full.shape[0]} tokens, need >= {SEQ_LEN_FOR_STATE}"
        )
    ids = ids_full[:SEQ_LEN_FOR_STATE].unsqueeze(0).to(DEVICE)
    t0 = perf_counter()
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    print(f"  prefill in {perf_counter()-t0:.2f}s; computing per-head state SVDs ...")
    pkv = out.past_key_values

    state_rows = []
    for li, layer in enumerate(pkv.layers):
        state = layer.state.get("recurrent_state")
        if state is None:
            continue
        # state: [B=1, H, D_k, D_v]
        s_cpu = state.detach().to(device="cpu", dtype=torch.float32)
        for h in range(s_cpu.shape[1]):
            mat = s_cpu[0, h]  # [D_k, D_v]
            sv = torch.linalg.svdvals(mat)
            state_rows.append({
                "layer": li,
                "head": h,
                "eff_rank_state": _effective_rank_from_sv(sv),
                "num_rank99_state": _numerical_rank_at_energy(sv, 0.99),
                "num_rank95_state": _numerical_rank_at_energy(sv, 0.95),
                "state_frob_norm": float(sv.pow(2).sum().sqrt().item()),
            })

    del out, pkv, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Join weight + state per (layer, head) ------------------------------
    weight_index = {(r["layer"], r["head"]): r for r in weight_rows}
    joined = []
    for sr in state_rows:
        wr = weight_index.get((sr["layer"], sr["head"]))
        if wr is None:
            continue
        merged = dict(wr)
        merged.update(sr)
        # Slack relative to data-driven cap:
        merged["state_eff_under_cap"] = merged["baseline_cap_eff"] - merged["eff_rank_state"]
        merged["state_num99_under_cap"] = merged["baseline_cap_num99"] - merged["num_rank99_state"]
        joined.append(merged)

    return {
        "label": label,
        "family": info["family"],
        "model_id": info["model_id"],
        "joined_rows": joined,
        "n_layers": L,
        "n_heads": H,
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
    }


# -------------------------------------------------------------- summary stats -

def _correlate(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return float((cov / (x.std(unbiased=False) * y.std(unbiased=False))).item())


def _summarize_model(rec: dict) -> dict:
    rows = rec["joined_rows"]
    eff_k = [r["eff_rank_Wk"] for r in rows]
    eff_v = [r["eff_rank_Wv"] for r in rows]
    cap = [r["baseline_cap_eff"] for r in rows]
    state = [r["eff_rank_state"] for r in rows]
    nr99_k = [r["num_rank99_Wk"] for r in rows]
    nr99_v = [r["num_rank99_Wv"] for r in rows]
    nr99_cap = [r["baseline_cap_num99"] for r in rows]
    nr99_state = [r["num_rank99_state"] for r in rows]
    head_dim_k = rec["head_dim_k"]
    head_dim_v = rec["head_dim_v"]

    def _stats(name, xs):
        t = torch.tensor(xs, dtype=torch.float64)
        return {
            "name": name,
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "std": float(t.std(unbiased=False).item()),
        }

    summary = {
        "label": rec["label"],
        "family": rec["family"],
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
        "n_heads": rec["n_heads"],
        "n_layers": rec["n_layers"],
        "n_rows_layer_x_head": len(rows),
        "stats": {
            "eff_rank_Wk": _stats("eff_rank_Wk", eff_k),
            "eff_rank_Wv": _stats("eff_rank_Wv", eff_v),
            "baseline_cap_eff": _stats("baseline_cap_eff", cap),
            "eff_rank_state": _stats("eff_rank_state", state),
            "num_rank99_Wk": _stats("num_rank99_Wk", nr99_k),
            "num_rank99_Wv": _stats("num_rank99_Wv", nr99_v),
            "baseline_cap_num99": _stats("baseline_cap_num99", nr99_cap),
            "num_rank99_state": _stats("num_rank99_state", nr99_state),
        },
        "fraction_eff_rank_under_full": {
            "eff_Wk_over_max": float(torch.tensor(eff_k).mean().item()) / head_dim_k,
            "eff_Wv_over_max": float(torch.tensor(eff_v).mean().item()) / head_dim_v,
            "eff_state_over_max": float(torch.tensor(state).mean().item()) / min(head_dim_k, head_dim_v),
        },
        "correlations": {
            "eff_rank_state_vs_baseline_cap_eff": _correlate(state, cap),
            "num_rank99_state_vs_baseline_cap_num99": _correlate(nr99_state, nr99_cap),
        },
    }

    # Decay correlations only meaningful for GDN
    if "A_log" in rows[0]:
        a_log = [r["A_log"] for r in rows]
        exp_a = [r["exp_A_log"] for r in rows]
        slack = [r["state_eff_under_cap"] for r in rows]
        slack_num = [r["state_num99_under_cap"] for r in rows]
        summary["stats"]["A_log"] = _stats("A_log", a_log)
        summary["stats"]["exp_A_log"] = _stats("exp_A_log", exp_a)
        summary["correlations_decay"] = {
            "exp_A_log_vs_state_eff_under_cap": _correlate(exp_a, slack),
            "A_log_vs_state_eff_under_cap": _correlate(a_log, slack),
            "exp_A_log_vs_eff_rank_state": _correlate(exp_a, state),
            "exp_A_log_vs_num_rank99_state": _correlate(exp_a, nr99_state),
            "exp_A_log_vs_state_num99_under_cap": _correlate(exp_a, slack_num),
        }
    return summary


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("kv_rank_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")

    summaries = []
    all_rows: list[dict] = []
    for info in MODELS:
        rec = _run_model(info)
        summary = _summarize_model(rec)
        summaries.append(summary)
        # tag rows with model label and write per-model csv
        per_model_rows = []
        for r in rec["joined_rows"]:
            r2 = {"model": rec["label"], "family": rec["family"], **r}
            per_model_rows.append(r2)
            all_rows.append(r2)
        _write_csv(per_model_rows, run_dir / f"{rec['family']}_kv_rank.csv")

    # Combined CSV across both models (include all keys, fill missing with empty)
    if all_rows:
        keys = []
        seen = set()
        for r in all_rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        with open(run_dir / "kv_rank_combined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"  wrote {run_dir / 'kv_rank_combined.csv'} ({len(all_rows)} rows)")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "seq_len_for_state": SEQ_LEN_FOR_STATE,
        "prompt_title": PROMPT_TITLE,
        "summaries": summaries,
    }, indent=2))
    print(f"  wrote {summary_path}")

    # ---- Pretty print ------------------------------------------------------
    print()
    print("=" * 72)
    print("Per-model summary (means over layers x heads)")
    print("=" * 72)
    for s in summaries:
        print()
        print(f"{s['label']}  -- head_dim_k={s['head_dim_k']}, head_dim_v={s['head_dim_v']}, "
              f"n_heads={s['n_heads']}, n_layers={s['n_layers']}")
        st = s["stats"]
        print(f"  eff_rank_Wk          mean={st['eff_rank_Wk']['mean']:.2f}  "
              f"median={st['eff_rank_Wk']['median']:.2f}  ({st['eff_rank_Wk']['min']:.1f}–{st['eff_rank_Wk']['max']:.1f})")
        print(f"  eff_rank_Wv          mean={st['eff_rank_Wv']['mean']:.2f}  "
              f"median={st['eff_rank_Wv']['median']:.2f}  ({st['eff_rank_Wv']['min']:.1f}–{st['eff_rank_Wv']['max']:.1f})")
        print(f"  baseline_cap_eff     mean={st['baseline_cap_eff']['mean']:.2f}  "
              f"median={st['baseline_cap_eff']['median']:.2f}  ({st['baseline_cap_eff']['min']:.1f}–{st['baseline_cap_eff']['max']:.1f})")
        print(f"  eff_rank_state       mean={st['eff_rank_state']['mean']:.2f}  "
              f"median={st['eff_rank_state']['median']:.2f}  ({st['eff_rank_state']['min']:.1f}–{st['eff_rank_state']['max']:.1f})")
        print(f"  num_rank@99 Wk       mean={st['num_rank99_Wk']['mean']:.2f}")
        print(f"  num_rank@99 Wv       mean={st['num_rank99_Wv']['mean']:.2f}")
        print(f"  num_rank@99 cap      mean={st['baseline_cap_num99']['mean']:.2f}")
        print(f"  num_rank@99 state    mean={st['num_rank99_state']['mean']:.2f}")
        print(f"  fractions of full rank: Wk={s['fraction_eff_rank_under_full']['eff_Wk_over_max']:.3f}  "
              f"Wv={s['fraction_eff_rank_under_full']['eff_Wv_over_max']:.3f}  "
              f"state={s['fraction_eff_rank_under_full']['eff_state_over_max']:.3f}")
        print(f"  corr(state_eff_rank, baseline_cap_eff)        = {s['correlations']['eff_rank_state_vs_baseline_cap_eff']:.3f}")
        print(f"  corr(state_num99,    baseline_cap_num99)      = {s['correlations']['num_rank99_state_vs_baseline_cap_num99']:.3f}")
        if "correlations_decay" in s:
            cd = s["correlations_decay"]
            print(f"  decay analysis (per-head):")
            print(f"    A_log mean={st['A_log']['mean']:.3f} (range {st['A_log']['min']:.2f}..{st['A_log']['max']:.2f})  "
                  f"exp(A_log) mean={st['exp_A_log']['mean']:.3f}")
            print(f"    corr(exp(A_log), state_eff_rank)          = {cd['exp_A_log_vs_eff_rank_state']:.3f}")
            print(f"    corr(exp(A_log), state_eff_under_cap)     = {cd['exp_A_log_vs_state_eff_under_cap']:.3f}")
            print(f"    corr(exp(A_log), state_num99_under_cap)   = {cd['exp_A_log_vs_state_num99_under_cap']:.3f}")
    print()
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main()

"""
Activation-rank ablation: which mechanism makes the recurrent state low-rank?

Hypotheses to discriminate, given that the per-head W_k and W_v are nearly
full-rank:
  H1. The residual stream / activations themselves live on a low-dim
      manifold; even though W_k is nearly full-rank, K_t = W_k x_t spans a
      small subspace because x_t does.
  H2. The recurrence cancels redundant directions (delta rule + decay):
      activations span a large space, but few directions actually accumulate
      into the state.

For each layer, we measure the effective rank of:
  - X_stack:        residual-stream input to the attention block, [T, hidden]
  - K_stack[h]:     post-conv1d keys, [T, head_dim_k]
  - K_stack_l2[h]:  L2-normalized per-token (the form the kernel uses)
  - V_stack[h]:     post-conv1d values, [T, head_dim_v]
  - S[h]:           recurrent state, [head_dim_k, head_dim_v]

Comparison to the parameter cap (eff_rank_Wk / eff_rank_Wv) and to the state
rank tells us where the rank is actually lost.
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
import torch.nn.functional as F

import fla  # noqa: F401
from fla.models import DeltaNetForCausalLM, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


SEQ_LEN = 2048
PROMPT_TITLE = "Bob Dylan"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

MODELS = [
    {
        "label": "DeltaNet-1.3B",
        "model_id": "fla-hub/delta_net-1.3B-100B",
        "cls": DeltaNetForCausalLM,
        "family": "delta_net",
    },
    {
        "label": "GatedDeltaNet-1.3B",
        "model_id": "m-a-p/1.3B-100B-GatedDeltaNet-pure",
        "cls": GatedDeltaNetForCausalLM,
        "family": "gated_deltanet",
    },
]


# --- rank helpers ------------------------------------------------------------

def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.detach().to(device="cpu", dtype=torch.float32))


def _eff_rank(sv: torch.Tensor, eps: float = 1e-12) -> float:
    s2 = sv.pow(2)
    total = s2.sum().clamp_min(eps)
    p = s2 / total
    safe = p.clamp_min(eps)
    h = -(p * safe.log()).sum()
    return float(h.exp().item())


def _num_rank_at(sv: torch.Tensor, frac: float) -> int:
    s2 = sv.pow(2)
    total = s2.sum().clamp_min(1e-12)
    cum = torch.cumsum(s2, dim=0) / total
    idx = (cum >= frac).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return int(sv.numel())
    return int(idx[0].item()) + 1


# --- prompt loader -----------------------------------------------------------

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
        raise KeyError(f"article {title!r} not found")
    return by_title[title]


# --- per-head W rank (parameter cap) -----------------------------------------

def _per_head_weight_eff_ranks(model, family: str) -> dict[tuple[int, int], dict]:
    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_heads
    if family == "delta_net":
        D_k = cfg.hidden_size // H
        D_v = cfg.hidden_size // H
    elif family == "gated_deltanet":
        D_k = cfg.head_dim
        D_v = cfg.head_dim
    else:
        raise ValueError(family)
    out = {}
    for li in range(L):
        attn = model.model.layers[li].attn
        Wk = attn.k_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        Wv = attn.v_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        Wk = Wk.view(H, D_k, cfg.hidden_size)
        Wv = Wv.view(H, D_v, cfg.hidden_size)
        for h in range(H):
            sv_k = _singular_values(Wk[h])
            sv_v = _singular_values(Wv[h])
            out[(li, h)] = {
                "eff_rank_Wk": _eff_rank(sv_k),
                "eff_rank_Wv": _eff_rank(sv_v),
                "num_rank99_Wk": _num_rank_at(sv_k, 0.99),
                "num_rank99_Wv": _num_rank_at(sv_v, 0.99),
            }
    return out


# --- per-model run with hooks ------------------------------------------------

def _run_model(info: dict) -> dict:
    label = info["label"]
    print(f"\n=== {label} ({info['model_id']}) ===")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    model = info["cls"].from_pretrained(info["model_id"], torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_heads
    if info["family"] == "delta_net":
        D_k = cfg.hidden_size // H
        D_v = cfg.hidden_size // H
    else:
        D_k = cfg.head_dim
        D_v = cfg.head_dim
    print(f"  L={L} H={H} D_k={D_k} D_v={D_v}")

    print("  computing per-head parameter SVDs ...")
    weight_ranks = _per_head_weight_eff_ranks(model, info["family"])

    # Hooks to capture: residual stream into each attn block (per layer),
    # and post-conv1d keys / values (per layer).
    captures_x: dict[int, torch.Tensor] = {}
    captures_k: dict[int, torch.Tensor] = {}
    captures_v: dict[int, torch.Tensor] = {}
    handles = []

    def _attn_pre_hook(li):
        def hook(module, inputs, kwargs):
            # The attn forward expects hidden_states as first positional arg or kwarg.
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            captures_x[li] = x.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    def _kconv_hook(li):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captures_k[li] = x.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    def _vconv_hook(li):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captures_v[li] = x.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    for li in range(L):
        attn = model.model.layers[li].attn
        handles.append(attn.register_forward_pre_hook(_attn_pre_hook(li), with_kwargs=True))
        handles.append(attn.k_conv1d.register_forward_hook(_kconv_hook(li)))
        handles.append(attn.v_conv1d.register_forward_hook(_vconv_hook(li)))

    article = _load_wiki_article(PROMPT_TITLE)
    ids_full = tokenizer(article, return_tensors="pt").input_ids[0]
    if ids_full.shape[0] < SEQ_LEN:
        raise RuntimeError(f"need >= {SEQ_LEN} tokens, got {ids_full.shape[0]}")
    ids = ids_full[:SEQ_LEN].unsqueeze(0).to(DEVICE)
    print(f"  prefilling {SEQ_LEN} tokens ...")
    t0 = perf_counter()
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    print(f"  prefill in {perf_counter()-t0:.2f}s")
    pkv = out.past_key_values

    for h in handles:
        h.remove()

    # Per-layer residual-stream rank (head-agnostic).
    layer_x_metrics = {}
    for li in range(L):
        x = captures_x[li]                                  # [T, hidden]
        if x.dim() == 3:                                    # safety
            x = x.squeeze(0)
        sv_x = _singular_values(x)
        layer_x_metrics[li] = {
            "eff_rank_x": _eff_rank(sv_x),
            "num_rank99_x": _num_rank_at(sv_x, 0.99),
            "num_rank95_x": _num_rank_at(sv_x, 0.95),
            "x_frob": float(sv_x.pow(2).sum().sqrt().item()),
        }

    # Per-(layer, head) activation rank metrics.
    rows = []
    for li in range(L):
        k_full = captures_k[li]                             # [T, hidden]
        v_full = captures_v[li]
        # reshape to [T, H, D_k]
        K = k_full.view(k_full.shape[0], H, D_k)
        V = v_full.view(v_full.shape[0], H, D_v)
        # state per head
        state = pkv.layers[li].state.get("recurrent_state")
        s_cpu = state.detach().to(device="cpu", dtype=torch.float32).squeeze(0)  # [H, D_k, D_v]
        for h in range(H):
            K_h = K[:, h, :]                                # [T, D_k]
            V_h = V[:, h, :]
            # L2-normalized per-token key: matches what the kernel uses
            K_h_l2 = K_h / K_h.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            sv_k = _singular_values(K_h)
            sv_v = _singular_values(V_h)
            sv_kl2 = _singular_values(K_h_l2)
            sv_S = _singular_values(s_cpu[h])

            wr = weight_ranks[(li, h)]
            x_metric = layer_x_metrics[li]
            row = {
                "model": label,
                "family": info["family"],
                "layer": li,
                "head": h,
                "head_dim_k": D_k,
                "head_dim_v": D_v,
                "T": int(K_h.shape[0]),
                "max_rank_k": min(K_h.shape),
                "max_rank_v": min(V_h.shape),
                "max_rank_state": min(D_k, D_v),
                # Parameter cap (per-head)
                "eff_rank_Wk": wr["eff_rank_Wk"],
                "eff_rank_Wv": wr["eff_rank_Wv"],
                "num_rank99_Wk": wr["num_rank99_Wk"],
                "num_rank99_Wv": wr["num_rank99_Wv"],
                # Residual stream (shared across heads at this layer)
                "eff_rank_x": x_metric["eff_rank_x"],
                "num_rank99_x": x_metric["num_rank99_x"],
                "num_rank95_x": x_metric["num_rank95_x"],
                # Empirical activation span per head
                "eff_rank_K": _eff_rank(sv_k),
                "num_rank99_K": _num_rank_at(sv_k, 0.99),
                "num_rank95_K": _num_rank_at(sv_k, 0.95),
                "eff_rank_K_l2": _eff_rank(sv_kl2),
                "num_rank99_K_l2": _num_rank_at(sv_kl2, 0.99),
                "eff_rank_V": _eff_rank(sv_v),
                "num_rank99_V": _num_rank_at(sv_v, 0.99),
                # Recurrent state per head
                "eff_rank_state": _eff_rank(sv_S),
                "num_rank99_state": _num_rank_at(sv_S, 0.99),
                "num_rank95_state": _num_rank_at(sv_S, 0.95),
            }
            rows.append(row)

    del out, pkv, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "family": info["family"],
        "model_id": info["model_id"],
        "rows": rows,
        "n_layers": L,
        "n_heads": H,
        "head_dim_k": D_k,
        "head_dim_v": D_v,
    }


# --- summary stats -----------------------------------------------------------

def _stats(name, vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return {
        "name": name,
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def _correlate(xs, ys):
    if len(xs) < 2:
        return float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return float((cov / (x.std(unbiased=False) * y.std(unbiased=False))).item())


def _summarize(rec: dict) -> dict:
    rows = rec["rows"]
    keys_for_stats = [
        "eff_rank_x", "num_rank99_x",
        "eff_rank_Wk", "eff_rank_Wv",
        "eff_rank_K", "num_rank99_K",
        "eff_rank_K_l2", "num_rank99_K_l2",
        "eff_rank_V", "num_rank99_V",
        "eff_rank_state", "num_rank99_state",
    ]
    return {
        "label": rec["label"],
        "family": rec["family"],
        "head_dim_k": rec["head_dim_k"],
        "head_dim_v": rec["head_dim_v"],
        "n_heads": rec["n_heads"],
        "n_layers": rec["n_layers"],
        "stats": {k: _stats(k, [r[k] for r in rows]) for k in keys_for_stats},
        "fraction_of_full_rank": {
            # x is hidden_size = 2048; report fraction over min(T, hidden_size)
            "x_eff_over_minTH": _stats(
                "x_eff_over_minTH",
                [r["eff_rank_x"] / min(r["T"], r["head_dim_k"] * rec["n_heads"]) for r in rows],
            )["mean"],
            "K_eff_over_min": _stats(
                "K_eff_over_min", [r["eff_rank_K"] / r["max_rank_k"] for r in rows],
            )["mean"],
            "K_l2_eff_over_min": _stats(
                "K_l2_eff_over_min", [r["eff_rank_K_l2"] / r["max_rank_k"] for r in rows],
            )["mean"],
            "V_eff_over_min": _stats(
                "V_eff_over_min", [r["eff_rank_V"] / r["max_rank_v"] for r in rows],
            )["mean"],
            "Wk_eff_over_min": _stats(
                "Wk_eff_over_min", [r["eff_rank_Wk"] / r["max_rank_k"] for r in rows],
            )["mean"],
            "state_eff_over_min": _stats(
                "state_eff_over_min", [r["eff_rank_state"] / r["max_rank_state"] for r in rows],
            )["mean"],
        },
        "correlations_per_head": {
            "state_eff_vs_K_eff": _correlate(
                [r["eff_rank_state"] for r in rows],
                [r["eff_rank_K"] for r in rows],
            ),
            "state_eff_vs_K_l2_eff": _correlate(
                [r["eff_rank_state"] for r in rows],
                [r["eff_rank_K_l2"] for r in rows],
            ),
            "state_eff_vs_V_eff": _correlate(
                [r["eff_rank_state"] for r in rows],
                [r["eff_rank_V"] for r in rows],
            ),
            "K_eff_vs_Wk_eff": _correlate(
                [r["eff_rank_K"] for r in rows],
                [r["eff_rank_Wk"] for r in rows],
            ),
        },
    }


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
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("kv_activation_rank_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")

    summaries = []
    all_rows = []
    for info in MODELS:
        rec = _run_model(info)
        summary = _summarize(rec)
        summaries.append(summary)
        _write_csv(rec["rows"], run_dir / f"{rec['family']}_kv_activation.csv")
        all_rows.extend(rec["rows"])

    if all_rows:
        keys = list(all_rows[0].keys())
        with open(run_dir / "kv_activation_combined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"  wrote {run_dir / 'kv_activation_combined.csv'} ({len(all_rows)} rows)")

    (run_dir / "summary.json").write_text(json.dumps({
        "seq_len": SEQ_LEN,
        "prompt_title": PROMPT_TITLE,
        "summaries": summaries,
    }, indent=2))

    print()
    print("=" * 78)
    print("Per-model summary -- means over layers x heads")
    print("=" * 78)
    for s in summaries:
        st = s["stats"]
        print(f"\n{s['label']}  head_dim_k={s['head_dim_k']}, head_dim_v={s['head_dim_v']}, "
              f"H={s['n_heads']}, L={s['n_layers']}")
        print(f"  eff_rank residual x       mean={st['eff_rank_x']['mean']:7.2f}  "
              f"({st['eff_rank_x']['min']:.0f}–{st['eff_rank_x']['max']:.0f})")
        print(f"  eff_rank Wk (param cap)   mean={st['eff_rank_Wk']['mean']:7.2f}  "
              f"of max {s['head_dim_k']}")
        print(f"  eff_rank Wv (param cap)   mean={st['eff_rank_Wv']['mean']:7.2f}  "
              f"of max {s['head_dim_v']}")
        print(f"  eff_rank K_t  (post-conv) mean={st['eff_rank_K']['mean']:7.2f}  "
              f"({st['eff_rank_K']['min']:.1f}–{st['eff_rank_K']['max']:.1f})")
        print(f"  eff_rank K_t  (L2-normed) mean={st['eff_rank_K_l2']['mean']:7.2f}")
        print(f"  eff_rank V_t  (post-conv) mean={st['eff_rank_V']['mean']:7.2f}  "
              f"({st['eff_rank_V']['min']:.1f}–{st['eff_rank_V']['max']:.1f})")
        print(f"  eff_rank state            mean={st['eff_rank_state']['mean']:7.2f}  "
              f"({st['eff_rank_state']['min']:.1f}–{st['eff_rank_state']['max']:.1f})")
        print(f"  num_rank99 K_t            mean={st['num_rank99_K']['mean']:7.2f}")
        print(f"  num_rank99 V_t            mean={st['num_rank99_V']['mean']:7.2f}")
        print(f"  num_rank99 state          mean={st['num_rank99_state']['mean']:7.2f}")
        print(f"  num_rank99 residual x     mean={st['num_rank99_x']['mean']:7.2f}")
        f = s["fraction_of_full_rank"]
        print(f"  fractions: Wk={f['Wk_eff_over_min']:.3f}  K={f['K_eff_over_min']:.3f}  "
              f"K_l2={f['K_l2_eff_over_min']:.3f}  V={f['V_eff_over_min']:.3f}  "
              f"state={f['state_eff_over_min']:.3f}")
        c = s["correlations_per_head"]
        print(f"  corr(state_eff, K_eff)    = {c['state_eff_vs_K_eff']:+.3f}")
        print(f"  corr(state_eff, K_l2_eff) = {c['state_eff_vs_K_l2_eff']:+.3f}")
        print(f"  corr(state_eff, V_eff)    = {c['state_eff_vs_V_eff']:+.3f}")
        print(f"  corr(K_eff,  Wk_eff)      = {c['K_eff_vs_Wk_eff']:+.3f}")
    print()
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main()

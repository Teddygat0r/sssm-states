"""
Multi-prompt version of run_kv_activation_rank.py.

Runs the same activation-rank measurements (residual stream x, post-conv K_t,
post-conv V_t, L2-normed K_t, recurrent state) across the 6 diverse prompts
used in earlier sweeps:
  - 4 Wikipedia articles (Bob Dylan, Battle of Romani, Missouri River, Roger Federer)
  - 1 HumanEval-concatenated code prompt
  - 1 SQuAD-v2 concat Q&A prompt

For each (model, prompt) we record per-(layer, head) effective ranks, plus
the two-stage decomposition factor (param_cap / K_eff) and (K_eff / state_eff).

Goal: confirm that the H1/H2 split observed on the Bob Dylan article holds
across prompt domains.
"""

from __future__ import annotations

import csv
import gc
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

import fla  # noqa: F401
from fla.models import DeltaNetForCausalLM, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


SEQ_LEN = 2048
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

WIKI_TARGETS = [
    "Bob Dylan",
    "Battle of Romani",
    "Missouri River",
    "Roger Federer",
]


# ---- rank helpers -----------------------------------------------------------

def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.detach().to(device="cpu", dtype=torch.float32))


def _eff_rank(sv: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank, with fp64 + max-normalization to avoid overflow."""
    sv = sv.to(dtype=torch.float64)
    m = sv.max()
    if not torch.isfinite(m) or m <= 0:
        return float("nan")
    sv_n = sv / m
    s2 = sv_n.pow(2)
    total = s2.sum()
    if total <= 0 or not torch.isfinite(total):
        return float("nan")
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


# ---- prompts ---------------------------------------------------------------

def _load_wiki_articles(targets):
    print(f"Loading wikitext-2-raw-v1 train; matching {len(targets)} target articles ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full = "".join(x["text"] for x in ds)
    parts = re.split(r"\n( = [^=].*? = )\n", full)
    by_title: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        t = parts[i].strip().strip("=").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        by_title[t] = body
    out = []
    for t in targets:
        if t not in by_title:
            raise KeyError(t)
        out.append({
            "id": f"wiki_{t.lower().replace(' ', '_')}",
            "kind": "wiki", "title": t, "text": by_title[t],
        })
        print(f"  - {t}: {len(by_title[t].split())} words")
    return out


def _load_code_prompt(min_chars: int = 16000) -> dict:
    print("Loading openai_humaneval (concatenated) ...")
    he = load_dataset("openai/openai_humaneval", split="test")
    parts = []
    total = 0
    for ex in he:
        parts.append(ex["prompt"])
        total += len(ex["prompt"])
        if total >= min_chars:
            break
    text = "\n\n".join(parts)
    print(f"  - {len(parts)} examples, {len(text)} chars")
    return {"id": "code_humaneval", "kind": "code", "title": "HumanEval-concat", "text": text}


def _load_qa_prompt(min_chars: int = 16000) -> dict:
    print("Loading squad_v2 validation (context+question concat) ...")
    sq = load_dataset("lighteval/squad_v2", split="validation")
    parts = []
    total = 0
    seen_titles: set[str] = set()
    for ex in sq:
        title = ex.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        block = (
            f"Title: {title}\nContext: {ex['context']}\nQuestion: {ex['question']}\n"
        )
        parts.append(block)
        total += len(block)
        if total >= min_chars:
            break
    text = "\n".join(parts)
    print(f"  - {len(parts)} contexts, {len(text)} chars")
    return {"id": "qa_squad", "kind": "qa", "title": "SQuAD-v2 concat", "text": text}


# ---- per-head W_k / W_v eff rank (parameter cap, prompt-independent) -------

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


# ---- per-prompt run --------------------------------------------------------

def _run_prompt(model, info, prompt, tokenizer, weight_ranks):
    """Runs a single (model, prompt) and returns per-(layer, head) rows."""
    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_heads
    if info["family"] == "delta_net":
        D_k = cfg.hidden_size // H
        D_v = cfg.hidden_size // H
    else:
        D_k = cfg.head_dim
        D_v = cfg.head_dim

    captures_x: dict[int, torch.Tensor] = {}
    captures_k: dict[int, torch.Tensor] = {}
    captures_v: dict[int, torch.Tensor] = {}
    handles = []

    def _attn_pre_hook(li):
        def hook(module, inputs, kwargs):
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

    ids_full = tokenizer(prompt["text"], return_tensors="pt").input_ids[0]
    if ids_full.shape[0] < SEQ_LEN:
        for hh in handles:
            hh.remove()
        raise RuntimeError(
            f"prompt {prompt['id']} has {ids_full.shape[0]} tokens, need {SEQ_LEN}"
        )
    ids = ids_full[:SEQ_LEN].unsqueeze(0).to(DEVICE)
    t0 = perf_counter()
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    print(f"    [{prompt['id']}] prefill {perf_counter()-t0:.2f}s; computing SVDs ...")
    pkv = out.past_key_values

    for hh in handles:
        hh.remove()

    layer_x_metrics: dict[int, dict] = {}
    for li in range(L):
        x = captures_x[li]
        if x.dim() == 3:
            x = x.squeeze(0)
        sv_x = _singular_values(x)
        layer_x_metrics[li] = {
            "eff_rank_x": _eff_rank(sv_x),
            "num_rank99_x": _num_rank_at(sv_x, 0.99),
            "num_rank95_x": _num_rank_at(sv_x, 0.95),
        }

    rows = []
    for li in range(L):
        k_full = captures_k[li]
        v_full = captures_v[li]
        K = k_full.view(k_full.shape[0], H, D_k)
        V = v_full.view(v_full.shape[0], H, D_v)
        state = pkv.layers[li].state.get("recurrent_state")
        if state is None:
            continue
        s_cpu = state.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        for h in range(H):
            K_h = K[:, h, :]
            V_h = V[:, h, :]
            K_h_l2 = K_h / K_h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sv_k = _singular_values(K_h)
            sv_v = _singular_values(V_h)
            sv_kl2 = _singular_values(K_h_l2)
            sv_S = _singular_values(s_cpu[h])

            wr = weight_ranks[(li, h)]
            xm = layer_x_metrics[li]
            rows.append({
                "model": info["label"],
                "family": info["family"],
                "prompt_id": prompt["id"],
                "prompt_kind": prompt["kind"],
                "layer": li,
                "head": h,
                "head_dim_k": D_k,
                "head_dim_v": D_v,
                "T": int(K_h.shape[0]),
                "max_rank_state": min(D_k, D_v),
                "eff_rank_Wk": wr["eff_rank_Wk"],
                "eff_rank_Wv": wr["eff_rank_Wv"],
                "num_rank99_Wk": wr["num_rank99_Wk"],
                "num_rank99_Wv": wr["num_rank99_Wv"],
                "eff_rank_x": xm["eff_rank_x"],
                "num_rank99_x": xm["num_rank99_x"],
                "num_rank95_x": xm["num_rank95_x"],
                "eff_rank_K": _eff_rank(sv_k),
                "num_rank99_K": _num_rank_at(sv_k, 0.99),
                "eff_rank_K_l2": _eff_rank(sv_kl2),
                "num_rank99_K_l2": _num_rank_at(sv_kl2, 0.99),
                "eff_rank_V": _eff_rank(sv_v),
                "num_rank99_V": _num_rank_at(sv_v, 0.99),
                "eff_rank_state": _eff_rank(sv_S),
                "num_rank99_state": _num_rank_at(sv_S, 0.99),
            })

    del out, pkv, captures_x, captures_k, captures_v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _run_for_model(info: dict, prompts: list[dict]) -> list[dict]:
    label = info["label"]
    print(f"\n=== {label} ({info['model_id']}) ===")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    model = info["cls"].from_pretrained(info["model_id"], torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    print("  computing per-head parameter SVDs ...")
    weight_ranks = _per_head_weight_eff_ranks(model, info["family"])
    all_rows = []
    for p in prompts:
        rows = _run_prompt(model, info, p, tokenizer, weight_ranks)
        all_rows.extend(rows)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_rows


# ---- aggregation ------------------------------------------------------------

def _isnum(x):
    try:
        f = float(x)
        return f == f and f != float("inf") and f != float("-inf")
    except (TypeError, ValueError):
        return False


def _summarize_per_prompt(rows):
    """For each (model, prompt), return: mean/median over layers x heads of
    eff_rank_x, eff_rank_K, eff_rank_V, eff_rank_state, plus the two-stage
    factors."""
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r["model"], r["prompt_id"], r["prompt_kind"])
        by_key.setdefault(k, []).append(r)

    out = []
    for (model, prompt_id, kind), rs in by_key.items():
        def col(name, finite=True):
            vals = [float(r[name]) for r in rs if (not finite or _isnum(r[name]))]
            return torch.tensor(vals, dtype=torch.float64) if vals else torch.tensor([], dtype=torch.float64)

        eff_x = col("eff_rank_x")
        eff_Wk = col("eff_rank_Wk")
        eff_Wv = col("eff_rank_Wv")
        eff_K = col("eff_rank_K")
        eff_V = col("eff_rank_V")
        eff_state = col("eff_rank_state")

        # Two-stage decomposition factors per head.
        def per_head_factor(num_col, den_col):
            num_t = col(num_col)
            den_t = col(den_col)
            n = min(num_t.numel(), den_t.numel())
            if n == 0:
                return float("nan"), float("nan")
            ratio = num_t[:n] / den_t[:n].clamp_min(1e-12)
            return float(ratio.mean().item()), float(ratio.median().item())

        cap_over_K_mean, cap_over_K_med = per_head_factor("eff_rank_Wk", "eff_rank_K")
        K_over_state_mean, K_over_state_med = per_head_factor("eff_rank_K", "eff_rank_state")

        out.append({
            "model": model,
            "prompt_id": prompt_id,
            "prompt_kind": kind,
            "eff_rank_x_mean": float(eff_x.mean().item()) if eff_x.numel() else float("nan"),
            "eff_rank_x_median": float(eff_x.median().item()) if eff_x.numel() else float("nan"),
            "eff_rank_Wk_mean": float(eff_Wk.mean().item()) if eff_Wk.numel() else float("nan"),
            "eff_rank_Wv_mean": float(eff_Wv.mean().item()) if eff_Wv.numel() else float("nan"),
            "eff_rank_K_mean": float(eff_K.mean().item()) if eff_K.numel() else float("nan"),
            "eff_rank_K_median": float(eff_K.median().item()) if eff_K.numel() else float("nan"),
            "eff_rank_V_mean": float(eff_V.mean().item()) if eff_V.numel() else float("nan"),
            "eff_rank_V_median": float(eff_V.median().item()) if eff_V.numel() else float("nan"),
            "eff_rank_state_mean": float(eff_state.mean().item()) if eff_state.numel() else float("nan"),
            "eff_rank_state_median": float(eff_state.median().item()) if eff_state.numel() else float("nan"),
            "factor_cap_over_K_mean": cap_over_K_mean,
            "factor_cap_over_K_median": cap_over_K_med,
            "factor_K_over_state_mean": K_over_state_mean,
            "factor_K_over_state_median": K_over_state_med,
            "n_rows": len(rs),
        })
    out.sort(key=lambda r: (r["model"], r["prompt_id"]))
    return out


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("kv_act_multiprompt_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")

    prompts = []
    prompts.extend(_load_wiki_articles(WIKI_TARGETS))
    prompts.append(_load_code_prompt())
    prompts.append(_load_qa_prompt())

    all_rows: list[dict] = []
    for info in MODELS:
        rows = _run_for_model(info, prompts)
        all_rows.extend(rows)
        # Per-model CSV
        keys = list(rows[0].keys())
        with open(run_dir / f"{info['family']}_kv_act_multiprompt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {info['family']}_kv_act_multiprompt.csv ({len(rows)} rows)")

    # Combined long CSV
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(run_dir / "kv_act_multiprompt_combined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in keys})

    # Aggregate per-(model, prompt) summary
    summary_rows = _summarize_per_prompt(all_rows)
    with open(run_dir / "per_model_per_prompt_summary.csv", "w", newline="") as f:
        keys = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"  wrote per_model_per_prompt_summary.csv ({len(summary_rows)} rows)")

    # Pretty print
    print()
    print("=" * 96)
    print("Per-(model, prompt) summary  -- means over layers x heads at seq=2048")
    print("=" * 96)
    h = (
        f"{'model':<22}{'prompt':<24}"
        f"{'x_eff':>9}{'Wk_eff':>9}{'K_eff':>9}{'V_eff':>9}{'state':>9}"
        f"{'cap/K':>9}{'K/state':>10}"
    )
    print(h)
    for s in summary_rows:
        print(
            f"{s['model']:<22}{s['prompt_id']:<24}"
            f"{s['eff_rank_x_mean']:>9.2f}{s['eff_rank_Wk_mean']:>9.2f}"
            f"{s['eff_rank_K_mean']:>9.2f}{s['eff_rank_V_mean']:>9.2f}"
            f"{s['eff_rank_state_mean']:>9.2f}"
            f"{s['factor_cap_over_K_mean']:>9.2f}"
            f"{s['factor_K_over_state_mean']:>10.2f}"
        )

    # Cross-prompt stability per model: std of factor_cap_over_K and factor_K_over_state
    print()
    print("=" * 96)
    print("Cross-prompt stability of decomposition factors (per model)")
    print("=" * 96)
    print(f"{'model':<22}{'cap/K range':<20}{'cap/K std':>10}"
          f"{'K/state range':<22}{'K/state std':>12}")
    by_model = {}
    for s in summary_rows:
        by_model.setdefault(s["model"], []).append(s)
    for model, ss in by_model.items():
        a = torch.tensor([s["factor_cap_over_K_mean"] for s in ss], dtype=torch.float64)
        b = torch.tensor([s["factor_K_over_state_mean"] for s in ss], dtype=torch.float64)
        print(
            f"{model:<22}"
            f"{a.min().item():.2f} – {a.max().item():.2f}".ljust(20)
            + f"{a.std(unbiased=False).item():>10.3f}"
            + f"  {b.min().item():.2f} – {b.max().item():.2f}".ljust(22)
            + f"{b.std(unbiased=False).item():>12.3f}"
        )

    print(f"\nOutputs in {run_dir}")


if __name__ == "__main__":
    main()

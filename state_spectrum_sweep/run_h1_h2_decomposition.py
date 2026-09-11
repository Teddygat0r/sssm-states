"""
Decompose H1 and H2 further on DN-1.3B and GDN-1.3B.

For one prompt (Bob Dylan, seq=2048) we re-derive K_t, V_t, x at each layer
via hooks, then build several counterfactuals to pinpoint where the rank
deficit comes from.

H1 — activation anisotropy. We split it into:
   - Passive inheritance: applying a *random* Gaussian matrix of matched
     scale (instead of trained W_k) to the captured residual stream x_t,
     then computing eff rank of the resulting random-K stream.
   - Active concentration: the gap between the random-K eff rank and the
     actual K_t eff rank.
   If they're equal, W_k is passively passing x's anisotropy through.
   If actual K_eff << random K_eff, W_k is actively further-concentrating.

H2 — recurrence cancellation. We replace the recurrence with the simplest
possible counterfactual:
   - S_plain = Σ_t k_l2_t v_t^T   (no delta rule, no decay; just sum of
     outer products with L2-normed keys, the form the kernel actually uses).
   - eff_rank(S_plain) tells us what the state would look like with no
     cancellation. Compared to the actual state eff rank, the gap is the
     pure recurrence contribution.
   - For GDN we also compute S_decay = Σ_t exp(Σ_{s>t} g_s) k_l2_t v_t^T
     to isolate the gated-decay contribution from the delta rule. (g_s is
     the per-head, per-token decay drift produced by A_log + softplus(a + dt_bias).)
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
TORCH_GEN = torch.Generator(device="cpu").manual_seed(0)

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


# -- rank helpers -------------------------------------------------------------

def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.detach().to(device="cpu", dtype=torch.float32))


def _eff_rank(sv: torch.Tensor, eps: float = 1e-12) -> float:
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


# -- prompt -------------------------------------------------------------------

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
        raise KeyError(title)
    return by_title[title]


# -- per-(model) run ----------------------------------------------------------

def _run_model(info: dict) -> list[dict]:
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

    # -- Capture x, post-conv K, post-conv V, plus beta and decay (GDN) ------
    captures_x: dict[int, torch.Tensor] = {}
    captures_k: dict[int, torch.Tensor] = {}
    captures_v: dict[int, torch.Tensor] = {}
    captures_b: dict[int, torch.Tensor] = {}
    captures_a: dict[int, torch.Tensor] = {}
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

    def _b_hook(li):
        def hook(module, inputs, output):
            captures_b[li] = output.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    def _a_hook(li):
        def hook(module, inputs, output):
            captures_a[li] = output.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    for li in range(L):
        attn = model.model.layers[li].attn
        handles.append(attn.register_forward_pre_hook(_attn_pre_hook(li), with_kwargs=True))
        handles.append(attn.k_conv1d.register_forward_hook(_kconv_hook(li)))
        handles.append(attn.v_conv1d.register_forward_hook(_vconv_hook(li)))
        handles.append(attn.b_proj.register_forward_hook(_b_hook(li)))
        if info["has_decay"]:
            handles.append(attn.a_proj.register_forward_hook(_a_hook(li)))

    article = _load_wiki_article(PROMPT_TITLE)
    ids_full = tokenizer(article, return_tensors="pt").input_ids[0]
    if ids_full.shape[0] < SEQ_LEN:
        for hh in handles:
            hh.remove()
        raise RuntimeError(f"need >= {SEQ_LEN}, got {ids_full.shape[0]}")
    ids = ids_full[:SEQ_LEN].unsqueeze(0).to(DEVICE)
    print(f"  prefilling {SEQ_LEN} tokens ...")
    t0 = perf_counter()
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    print(f"  prefill in {perf_counter()-t0:.2f}s")
    pkv = out.past_key_values
    for hh in handles:
        hh.remove()

    # -- For each layer, compute counterfactuals ----------------------------
    rows = []
    for li in range(L):
        x = captures_x[li]                            # [T, hidden]
        if x.dim() == 3:
            x = x.squeeze(0)
        k_full = captures_k[li]                       # [T, hidden]
        v_full = captures_v[li]                       # [T, hidden]
        K = k_full.view(k_full.shape[0], H, D_k)      # [T, H, D_k]
        V = v_full.view(v_full.shape[0], H, D_v)
        K_l2 = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # GDN decay parameters: g_t,h = -exp(A_log_h) * softplus(a_proj(x)_t,h + dt_bias_h)
        # The state at end of seq has each token's contribution multiplied by
        # exp(Σ_{s>t} g_s,h). We use that to build S_decay.
        attn = model.model.layers[li].attn
        if info["has_decay"]:
            A_log = attn.A_log.detach().to(device="cpu", dtype=torch.float32)   # [H]
            dt_bias = attn.dt_bias.detach().to(device="cpu", dtype=torch.float32)
            a_proj_out = captures_a[li]               # [T, H]
            # g_t,h = -exp(A_log_h) * softplus(a_t,h + dt_bias_h)   (per FLA forward)
            g = -A_log.exp().unsqueeze(0) * F.softplus(a_proj_out + dt_bias.unsqueeze(0))
            # Reverse-cumulative sum of g, exclusive of t.
            #   For token t, decay factor = exp(Σ_{s>t} g_s)  (so latest tokens are barely decayed)
            #   We use dim=0 reverse cumsum then shift.
            # cum_inc[t] = Σ_{s<=t} g_s  ; reverse via flip.
            cum_inc = g.cumsum(dim=0)                 # [T, H]
            total = cum_inc[-1:, :]                   # [1, H]
            decay_factor = (total - cum_inc).exp()    # exp(Σ_{s>t} g_s) = exp(total - cum_inc[t])
        else:
            decay_factor = None

        # -- Random-W_k control ------------------------------------------------
        # Random matrix with same row-wise scale as W_k. We compare per-head
        # (D_k, hidden) blocks. Build W_k_random at the same shape, then apply
        # to x to get a random-K stream.
        Wk = attn.k_proj.weight.detach().to(device="cpu", dtype=torch.float32)   # [hidden, hidden] for DN/GDN
        Wv = attn.v_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        # Per-head std of W_k:
        Wk_per_head = Wk.view(H, D_k, x.shape[-1])
        Wv_per_head = Wv.view(H, D_v, x.shape[-1])

        # -- Recurrent state at end of prefill ---------------------------------
        state = pkv.layers[li].state.get("recurrent_state")
        if state is None:
            continue
        s_cpu = state.detach().to(device="cpu", dtype=torch.float32).squeeze(0)  # [H, D_k, D_v]

        for h in range(H):
            K_h = K[:, h, :]                          # [T, D_k]
            V_h = V[:, h, :]                          # [T, D_v]
            K_l2_h = K_l2[:, h, :]

            # -- Random-W_k baseline ----------------------------------------
            # Match per-element std of W_k for this head.
            wk_std = float(Wk_per_head[h].std().item())
            wv_std = float(Wv_per_head[h].std().item())
            Wk_random = torch.randn(
                Wk_per_head[h].shape, generator=TORCH_GEN, dtype=torch.float32
            ) * wk_std
            Wv_random = torch.randn(
                Wv_per_head[h].shape, generator=TORCH_GEN, dtype=torch.float32
            ) * wv_std
            K_random = x @ Wk_random.t()              # [T, D_k]
            V_random = x @ Wv_random.t()

            # -- Plain-sum counterfactual (no recurrence) -------------------
            # S_plain = Σ_t k_l2_t v_t^T  (uses L2-normed K)
            S_plain = K_l2_h.t() @ V_h                # [D_k, D_v]
            # S_plain_raw uses post-conv K (no L2): for completeness.
            S_plain_raw = K_h.t() @ V_h
            # -- Decay-only counterfactual (GDN only) ------------------------
            if decay_factor is not None:
                d = decay_factor[:, h].unsqueeze(-1)  # [T, 1]
                S_decay = (K_l2_h * d).t() @ V_h      # [D_k, D_v]
            else:
                S_decay = None

            sv_K = _singular_values(K_h)
            sv_K_l2 = _singular_values(K_l2_h)
            sv_V = _singular_values(V_h)
            sv_K_random = _singular_values(K_random)
            sv_V_random = _singular_values(V_random)
            sv_S_plain = _singular_values(S_plain)
            sv_S_plain_raw = _singular_values(S_plain_raw)
            sv_S_decay = _singular_values(S_decay) if S_decay is not None else None
            sv_S_actual = _singular_values(s_cpu[h])

            row = {
                "model": label,
                "family": info["family"],
                "layer": li,
                "head": h,
                "D_k": D_k, "D_v": D_v,
                "eff_rank_K": _eff_rank(sv_K),
                "eff_rank_K_l2": _eff_rank(sv_K_l2),
                "eff_rank_V": _eff_rank(sv_V),
                # H1 split:
                "eff_rank_K_random": _eff_rank(sv_K_random),
                "eff_rank_V_random": _eff_rank(sv_V_random),
                # H2 split:
                "eff_rank_S_plain": _eff_rank(sv_S_plain),
                "eff_rank_S_plain_raw": _eff_rank(sv_S_plain_raw),
                "eff_rank_S_actual": _eff_rank(sv_S_actual),
                "num_rank99_S_plain": _num_rank_at(sv_S_plain, 0.99),
                "num_rank99_S_actual": _num_rank_at(sv_S_actual, 0.99),
            }
            if sv_S_decay is not None:
                row["eff_rank_S_decay"] = _eff_rank(sv_S_decay)
                row["num_rank99_S_decay"] = _num_rank_at(sv_S_decay, 0.99)
            else:
                row["eff_rank_S_decay"] = ""
                row["num_rank99_S_decay"] = ""
            rows.append(row)

    del out, pkv, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# -- aggregation --------------------------------------------------------------

def _stat(values):
    t = torch.tensor([v for v in values if isinstance(v, (int, float)) and v == v],
                     dtype=torch.float64)
    if t.numel() == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _summarize(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for model, rs in by_model.items():
        cols = [
            "eff_rank_K", "eff_rank_K_l2", "eff_rank_V",
            "eff_rank_K_random", "eff_rank_V_random",
            "eff_rank_S_plain", "eff_rank_S_plain_raw",
            "eff_rank_S_decay", "eff_rank_S_actual",
        ]
        s = {"model": model, "n_rows": len(rs)}
        for c in cols:
            s[c] = _stat([r[c] for r in rs])
        # Compute reduction factors
        def ratio(num, den):
            num_t = torch.tensor([float(r[num]) for r in rs if r[num] != ""], dtype=torch.float64)
            den_t = torch.tensor([float(r[den]) for r in rs if r[den] != ""], dtype=torch.float64)
            n = min(num_t.numel(), den_t.numel())
            if n == 0:
                return float("nan")
            return float((num_t[:n] / den_t[:n].clamp_min(1e-12)).mean().item())
        s["ratio_K_random_over_K"] = ratio("eff_rank_K_random", "eff_rank_K")
        s["ratio_S_plain_over_S_actual"] = ratio("eff_rank_S_plain", "eff_rank_S_actual")
        if any(r["eff_rank_S_decay"] != "" for r in rs):
            s["ratio_S_plain_over_S_decay"] = ratio("eff_rank_S_plain", "eff_rank_S_decay")
            s["ratio_S_decay_over_S_actual"] = ratio("eff_rank_S_decay", "eff_rank_S_actual")
        out.append(s)
    return out


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("h1_h2_decomp_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")

    all_rows = []
    for info in MODELS:
        rows = _run_model(info)
        keys = list(rows[0].keys())
        with open(run_dir / f"{info['family']}_h1_h2.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {info['family']}_h1_h2.csv ({len(rows)} rows)")
        all_rows.extend(rows)

    summary = _summarize(all_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 84)
    print("H1 split — random-W_k vs trained W_k applied to actual residual stream")
    print("=" * 84)
    print(f"{'model':<22}{'K_eff (trained)':>18}{'K_eff (random)':>18}{'ratio rand/trained':>22}")
    for s in summary:
        print(f"{s['model']:<22}"
              f"{s['eff_rank_K']['mean']:>18.2f}"
              f"{s['eff_rank_K_random']['mean']:>18.2f}"
              f"{s['ratio_K_random_over_K']:>22.2f}")
    print()
    print("Interpretation:")
    print("  - ratio ≈ 1: W_k is passive (random matrix gives same eff rank)")
    print("  - ratio > 1: trained W_k actively concentrates beyond x's anisotropy")
    print()
    print("=" * 84)
    print("H2 split — plain Σ k_l2 v^T vs decay-only vs actual recurrent state")
    print("=" * 84)
    print(f"{'model':<22}{'S_plain':>10}{'S_decay':>10}{'S_actual':>10}{'plain/actual':>14}{'plain/decay':>14}{'decay/actual':>14}")
    for s in summary:
        spl = s["eff_rank_S_plain"]["mean"]
        sde = s["eff_rank_S_decay"]["mean"]
        sac = s["eff_rank_S_actual"]["mean"]
        r1 = s.get("ratio_S_plain_over_S_actual", float("nan"))
        r2 = s.get("ratio_S_plain_over_S_decay", float("nan"))
        r3 = s.get("ratio_S_decay_over_S_actual", float("nan"))
        print(
            f"{s['model']:<22}"
            f"{spl:>10.2f}"
            f"{sde:>10.2f}" if isinstance(sde, float) and sde == sde else f"{'-':>10}"
        )
    # Print again cleanly since the inline-if collapsed the columns
    print()
    print(f"{'model':<22}{'S_plain':>10}{'S_decay':>10}{'S_actual':>10}"
          f"{'plain/actual':>14}{'plain/decay':>14}{'decay/actual':>14}")
    for s in summary:
        spl = s["eff_rank_S_plain"]["mean"]
        sde = s["eff_rank_S_decay"]["mean"] if isinstance(s["eff_rank_S_decay"], dict) else float("nan")
        sac = s["eff_rank_S_actual"]["mean"]
        r1 = s.get("ratio_S_plain_over_S_actual", float("nan"))
        r2 = s.get("ratio_S_plain_over_S_decay", float("nan"))
        r3 = s.get("ratio_S_decay_over_S_actual", float("nan"))
        sde_s = f"{sde:>10.2f}" if sde == sde else f"{'-':>10}"
        r2_s = f"{r2:>14.2f}" if r2 == r2 else f"{'-':>14}"
        r3_s = f"{r3:>14.2f}" if r3 == r3 else f"{'-':>14}"
        print(
            f"{s['model']:<22}{spl:>10.2f}{sde_s}{sac:>10.2f}{r1:>14.2f}{r2_s}{r3_s}"
        )
    print()
    print("Interpretation:")
    print("  - plain/actual > 1: recurrence (delta + decay) lowers state rank below the unconstrained sum of outer products")
    print("  - plain/decay > 1: decay alone lowers rank below the unconstrained sum")
    print("  - decay/actual > 1: delta rule lowers rank below the decay-only counterfactual")

    print(f"\nOutputs in {run_dir}")


if __name__ == "__main__":
    main()

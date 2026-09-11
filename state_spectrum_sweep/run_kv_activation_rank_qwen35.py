"""
Activation-rank ablation on Qwen3.5-4B (hybrid linear + full attention).

Architecture (from text_config):
  - 32 hidden layers
  - layer_types pattern: [linear, linear, linear, full] x 8 -> 24 linear, 8 full
  - linear_num_key_heads = 16
  - linear_num_value_heads = 32  (so K is repeat-interleaved by 2 to match V)
  - linear_key_head_dim = linear_value_head_dim = 128
  - hidden_size = 2560
  - recurrent state per linear layer: [B, num_v_heads, head_k_dim, head_v_dim]
                                    = [B, 32, 128, 128]

We hook every Qwen3_5GatedDeltaNet (linear-attention) layer to capture:
  - x: residual stream into the layer block ([T, 2560])
  - mixed_qkv post-conv1d (silu applied) ([T, conv_dim=8192]); split into
      Q [T, 16, 128], K [T, 16, 128], V [T, 32, 128]
  - K is then repeat-interleaved by 2 to match V's head count (matches forward)

We compare:
  - eff_rank(W_k), eff_rank(W_v): per-head SVD on the K/V slice of in_proj_qkv
  - eff_rank(x): residual stream at each linear layer
  - eff_rank(K_t), eff_rank(V_t): empirical activation span per head, post-conv
  - eff_rank(state): per-head state at end of prefill

Run on the same six prompts at seq_len=2048.
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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset


SEQ_LEN = 2048
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

MODEL_ID = "Qwen/Qwen3.5-4B"

WIKI_TARGETS = [
    "Bob Dylan",
    "Battle of Romani",
    "Missouri River",
    "Roger Federer",
]


# ---- rank helpers ----------------------------------------------------------

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


# ---- locate linear-attn layers --------------------------------------------

def _find_linear_layers(model) -> list[int]:
    """Return indices of layers whose attention is Qwen3_5GatedDeltaNet."""
    out = []
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            out.append(li)
    return out


def _get_attn_module(layer):
    """Returns the linear_attn module on a Qwen3.5 decoder layer."""
    return layer.linear_attn


def _get_text_layers(model):
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return text_model.layers


# ---- per-head W_k / W_v from in_proj_qkv ----------------------------------

def _per_head_weight_eff_ranks(model, linear_layer_indices: list[int],
                                num_k_heads: int, num_v_heads: int,
                                head_k_dim: int, head_v_dim: int,
                                hidden_size: int) -> dict[tuple[int, int], dict]:
    """For each (linear_layer_idx, value_head_idx), compute the eff rank of the
    K-projection (after K's repeat-interleave to match V) and the V-projection.

    in_proj_qkv: hidden -> 2*key_dim + value_dim with row order [Q | K | V].
    K row block: rows key_dim..2*key_dim, viewed per-K-head [num_k_heads, head_k_dim, hidden],
    then repeat-interleaved by num_v_heads // num_k_heads to match V's head count.
    """
    out = {}
    layers = _get_text_layers(model)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    repeat = num_v_heads // num_k_heads
    for li in linear_layer_indices:
        attn = _get_attn_module(layers[li])
        W = attn.in_proj_qkv.weight.detach().to(device="cpu", dtype=torch.float32)
        # W shape [out, hidden] = [key_dim*2 + value_dim, hidden]
        W_k = W[key_dim:2 * key_dim, :]                              # [key_dim, hidden]
        W_v = W[2 * key_dim:2 * key_dim + value_dim, :]              # [value_dim, hidden]
        Wk_per_head = W_k.view(num_k_heads, head_k_dim, hidden_size)
        Wv_per_head = W_v.view(num_v_heads, head_v_dim, hidden_size)
        # K is repeat-interleaved by `repeat` over the head axis to match V.
        Wk_repeated = Wk_per_head.repeat_interleave(repeat, dim=0)   # [num_v_heads, head_k_dim, hidden]
        for h in range(num_v_heads):
            sv_k = _singular_values(Wk_repeated[h])
            sv_v = _singular_values(Wv_per_head[h])
            out[(li, h)] = {
                "eff_rank_Wk": _eff_rank(sv_k),
                "eff_rank_Wv": _eff_rank(sv_v),
                "num_rank99_Wk": _num_rank_at(sv_k, 0.99),
                "num_rank99_Wv": _num_rank_at(sv_v, 0.99),
            }
    return out


# ---- per-prompt run --------------------------------------------------------

def _run_prompt(model, prompt, tokenizer, weight_ranks, linear_layers,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                key_dim, value_dim, conv_dim, hidden_size):
    captures_x: dict[int, torch.Tensor] = {}
    captures_conv: dict[int, torch.Tensor] = {}
    handles = []
    layers = _get_text_layers(model)

    def _attn_pre_hook(li):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            captures_x[li] = x.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    def _conv_hook(li):
        def hook(module, inputs, output):
            # conv1d output shape: [B, conv_dim, T] (channel-first)
            captures_conv[li] = output.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        return hook

    for li in linear_layers:
        attn = _get_attn_module(layers[li])
        handles.append(attn.register_forward_pre_hook(_attn_pre_hook(li), with_kwargs=True))
        handles.append(attn.conv1d.register_forward_hook(_conv_hook(li)))

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
    cache = out.past_key_values

    for hh in handles:
        hh.remove()

    rows = []
    repeat = num_v_heads // num_k_heads
    for li in linear_layers:
        x = captures_x[li]                                  # [T, hidden]
        if x.dim() == 3:
            x = x.squeeze(0)
        sv_x = _singular_values(x)
        eff_x = _eff_rank(sv_x)
        nr99_x = _num_rank_at(sv_x, 0.99)
        nr95_x = _num_rank_at(sv_x, 0.95)

        # mixed_qkv post-conv (with silu, mirroring forward)
        conv = captures_conv[li]                            # [conv_dim, T] after squeeze
        # The forward applies F.silu(conv(mixed_qkv)[:, :, :seq_len])
        T = conv.shape[-1]
        if T > SEQ_LEN:
            conv = conv[:, :SEQ_LEN]
        conv = F.silu(conv)
        # Transpose to [T, conv_dim] then split
        mixed = conv.transpose(0, 1)                        # [T, conv_dim]
        Q = mixed[:, :key_dim]                              # [T, key_dim]
        K = mixed[:, key_dim:2 * key_dim]                   # [T, key_dim]
        V = mixed[:, 2 * key_dim:2 * key_dim + value_dim]   # [T, value_dim]
        # Reshape per-head, then repeat K to match V's head count.
        Q = Q.view(Q.shape[0], num_k_heads, head_k_dim)
        K = K.view(K.shape[0], num_k_heads, head_k_dim)
        V = V.view(V.shape[0], num_v_heads, head_v_dim)
        K_rep = K.repeat_interleave(repeat, dim=1)          # [T, num_v_heads, head_k_dim]

        # Recurrent state for this layer
        rec_state = cache.recurrent_states[li]              # [B, num_v_heads, D_k, D_v]
        s_cpu = rec_state.detach().to(device="cpu", dtype=torch.float32).squeeze(0)

        for h in range(num_v_heads):
            K_h = K_rep[:, h, :]
            V_h = V[:, h, :]
            K_h_l2 = K_h / K_h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sv_k = _singular_values(K_h)
            sv_v = _singular_values(V_h)
            sv_kl2 = _singular_values(K_h_l2)
            sv_S = _singular_values(s_cpu[h])
            wr = weight_ranks[(li, h)]
            rows.append({
                "model": "Qwen3.5-4B",
                "family": "qwen3_5_gated_deltanet",
                "prompt_id": prompt["id"],
                "prompt_kind": prompt["kind"],
                "layer": li,
                "head": h,
                "head_dim_k": head_k_dim,
                "head_dim_v": head_v_dim,
                "T": int(K_h.shape[0]),
                "max_rank_state": min(head_k_dim, head_v_dim),
                "eff_rank_Wk": wr["eff_rank_Wk"],
                "eff_rank_Wv": wr["eff_rank_Wv"],
                "num_rank99_Wk": wr["num_rank99_Wk"],
                "num_rank99_Wv": wr["num_rank99_Wv"],
                "eff_rank_x": eff_x,
                "num_rank99_x": nr99_x,
                "num_rank95_x": nr95_x,
                "eff_rank_K": _eff_rank(sv_k),
                "num_rank99_K": _num_rank_at(sv_k, 0.99),
                "eff_rank_K_l2": _eff_rank(sv_kl2),
                "num_rank99_K_l2": _num_rank_at(sv_kl2, 0.99),
                "eff_rank_V": _eff_rank(sv_v),
                "num_rank99_V": _num_rank_at(sv_v, 0.99),
                "eff_rank_state": _eff_rank(sv_S),
                "num_rank99_state": _num_rank_at(sv_S, 0.99),
            })

    del out, cache, captures_x, captures_conv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ---- aggregation -----------------------------------------------------------

def _isnum(x):
    try:
        f = float(x)
        return f == f and f != float("inf") and f != float("-inf")
    except (TypeError, ValueError):
        return False


def _summarize_per_prompt(rows):
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r["model"], r["prompt_id"], r["prompt_kind"])
        by_key.setdefault(k, []).append(r)
    out = []
    for (model, prompt_id, kind), rs in by_key.items():
        def col(name):
            vals = [float(r[name]) for r in rs if _isnum(r[name])]
            return torch.tensor(vals, dtype=torch.float64) if vals else torch.tensor([], dtype=torch.float64)
        eff_x = col("eff_rank_x")
        eff_Wk = col("eff_rank_Wk")
        eff_Wv = col("eff_rank_Wv")
        eff_K = col("eff_rank_K")
        eff_V = col("eff_rank_V")
        eff_state = col("eff_rank_state")

        def per_head_factor(num_col, den_col):
            num_t = col(num_col)
            den_t = col(den_col)
            n = min(num_t.numel(), den_t.numel())
            if n == 0:
                return float("nan"), float("nan")
            ratio = num_t[:n] / den_t[:n].clamp_min(1e-12)
            return float(ratio.mean().item()), float(ratio.median().item())

        cap_K_mean, cap_K_med = per_head_factor("eff_rank_Wk", "eff_rank_K")
        K_state_mean, K_state_med = per_head_factor("eff_rank_K", "eff_rank_state")

        out.append({
            "model": model,
            "prompt_id": prompt_id,
            "prompt_kind": kind,
            "eff_rank_x_mean": float(eff_x.mean().item()) if eff_x.numel() else float("nan"),
            "eff_rank_Wk_mean": float(eff_Wk.mean().item()) if eff_Wk.numel() else float("nan"),
            "eff_rank_Wv_mean": float(eff_Wv.mean().item()) if eff_Wv.numel() else float("nan"),
            "eff_rank_K_mean": float(eff_K.mean().item()) if eff_K.numel() else float("nan"),
            "eff_rank_V_mean": float(eff_V.mean().item()) if eff_V.numel() else float("nan"),
            "eff_rank_state_mean": float(eff_state.mean().item()) if eff_state.numel() else float("nan"),
            "factor_cap_over_K_mean": cap_K_mean,
            "factor_cap_over_K_median": cap_K_med,
            "factor_K_over_state_mean": K_state_mean,
            "factor_K_over_state_median": K_state_med,
            "n_rows": len(rs),
        })
    out.sort(key=lambda r: (r["model"], r["prompt_id"]))
    return out


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("kv_act_qwen35_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")

    cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
    num_k_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    head_k_dim = cfg.linear_key_head_dim
    head_v_dim = cfg.linear_value_head_dim
    hidden_size = cfg.hidden_size
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim

    print(f"Architecture: {cfg.num_hidden_layers} layers")
    print(f"  num_k_heads={num_k_heads}  num_v_heads={num_v_heads}  "
          f"head_k_dim={head_k_dim}  head_v_dim={head_v_dim}  hidden={hidden_size}")

    prompts = []
    prompts.extend(_load_wiki_articles(WIKI_TARGETS))
    prompts.append(_load_code_prompt())
    prompts.append(_load_qa_prompt())

    print(f"\nLoading {MODEL_ID} (this is the slow step) ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    linear_layers = _find_linear_layers(model)
    print(f"  linear-attention layers: {linear_layers}")
    print(f"  ({len(linear_layers)} of {cfg.num_hidden_layers})")

    # Disable the optimized causal-conv1d kernel so that the standard nn.Conv1d
    # forward runs and our hook on .conv1d fires. The standard path produces
    # numerically identical output: F.silu(self.conv1d(x)[:, :, :seq_len]).
    layers = _get_text_layers(model)
    for li in linear_layers:
        attn = _get_attn_module(layers[li])
        attn.causal_conv1d_fn = None
        # We don't disable causal_conv1d_update because it only fires for
        # single-token (use_precomputed_states) inference; we only do prefill.
    print("  disabled causal_conv1d_fn on all linear-attn layers")

    print("  computing per-(linear-layer, V-head) parameter SVDs ...")
    weight_ranks = _per_head_weight_eff_ranks(
        model, linear_layers, num_k_heads, num_v_heads, head_k_dim, head_v_dim, hidden_size
    )
    print(f"  done ({len(weight_ranks)} entries)")

    all_rows = []
    for p in prompts:
        rows = _run_prompt(
            model, p, tokenizer, weight_ranks, linear_layers,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            key_dim, value_dim, conv_dim, hidden_size,
        )
        all_rows.extend(rows)

    keys = list(all_rows[0].keys())
    with open(run_dir / "qwen35_kv_activation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"  wrote qwen35_kv_activation.csv ({len(all_rows)} rows)")

    summary_rows = _summarize_per_prompt(all_rows)
    with open(run_dir / "per_prompt_summary.csv", "w", newline="") as f:
        keys = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print()
    print("=" * 96)
    print("Qwen3.5-4B per-prompt summary  -- means over linear-attn layers x V-heads")
    print("=" * 96)
    h = (
        f"{'prompt':<24}{'x_eff':>10}{'Wk_eff':>10}{'K_eff':>10}{'V_eff':>10}"
        f"{'state':>10}{'cap/K':>10}{'K/state':>10}"
    )
    print(h)
    for s in summary_rows:
        print(
            f"{s['prompt_id']:<24}"
            f"{s['eff_rank_x_mean']:>10.2f}{s['eff_rank_Wk_mean']:>10.2f}"
            f"{s['eff_rank_K_mean']:>10.2f}{s['eff_rank_V_mean']:>10.2f}"
            f"{s['eff_rank_state_mean']:>10.2f}"
            f"{s['factor_cap_over_K_mean']:>10.2f}"
            f"{s['factor_K_over_state_mean']:>10.2f}"
        )

    a = torch.tensor([s["factor_cap_over_K_mean"] for s in summary_rows], dtype=torch.float64)
    b = torch.tensor([s["factor_K_over_state_mean"] for s in summary_rows], dtype=torch.float64)
    print()
    print("Cross-prompt stability:")
    print(f"  cap/K range  : {a.min().item():.2f} – {a.max().item():.2f}  (std {a.std(unbiased=False).item():.3f})")
    print(f"  K/state range: {b.min().item():.2f} – {b.max().item():.2f}  (std {b.std(unbiased=False).item():.3f})")

    print(f"\nOutputs in {run_dir}")


if __name__ == "__main__":
    main()

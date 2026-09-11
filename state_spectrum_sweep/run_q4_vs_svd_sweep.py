"""
Side-by-side ablation: rank-16 SVD vs Q4 fake-quantization of the recurrent state.

Reuses the same models, prompts, and seq-length sweep as run_svd_ablation_sweep.py.
For each captured recurrent_state of shape [B, H, D_k, D_v] we compute three
reconstructions and the same error metrics for each:

    1. svd_r16          : rank-16 truncated SVD on the last two dims (per head, per batch)
    2. q4_per_tensor    : 4-bit uniform affine fake-quant with one scale/zp for the
                          whole layer's state tensor (matches run_experiment_quant.py)
    3. q4_per_head      : 4-bit uniform affine fake-quant with one scale/zp per head
                          (matches the granularity used by SVD)

Storage cost reference (per [B=1, H, D_k, D_v] state, ignoring scale/zp overhead):
    fp32 dense          : H * D_k * D_v * 4 bytes
    SVD r=r (factored)  : H * r * (D_k + D_v) * 4 bytes      (= 1/(D/r/2) over dense for square heads)
    Q4 dense            : H * D_k * D_v * 0.5 bytes          (8x over fp32 dense)
    For DN  (D_k=D_v=128, r=16): SVD = 0.250 of fp32 dense; Q4 = 0.125
    For GDN (D_k=D_v=256, r=16): SVD = 0.125 of fp32 dense; Q4 = 0.125
"""

from __future__ import annotations

import gc
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

import fla  # noqa: F401  -- registers FLA model classes
from fla.models import DeltaNetForCausalLM, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


SEQ_LENS = [64, 256, 1024, 2048]
SVD_RANK = 16
QUANT_BITS = 4
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


def _load_wikitext_articles(targets: list[str]) -> list[dict]:
    print(f"Loading wikitext-2-raw-v1 train; matching {len(targets)} target articles ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full = "".join(x["text"] for x in ds)
    parts = re.split(r"\n( = [^=].*? = )\n", full)
    by_title: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip().strip("=").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        by_title[title] = body
    out = []
    for t in targets:
        if t not in by_title:
            raise KeyError(f"could not find article '{t}' in wikitext-2 train")
        out.append({"id": f"wiki_{t.lower().replace(' ', '_')}", "kind": "wiki", "title": t, "text": by_title[t]})
        print(f"  - {t}: {len(by_title[t].split())} words")
    return out


def _load_code_prompt(min_chars: int = 12000) -> dict:
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


def _load_qa_prompt(min_chars: int = 12000) -> dict:
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


# --- Reconstruction methods ---------------------------------------------------

def _low_rank_svd(tensor: torch.Tensor, n: int, oversample: int = 4, niter: int = 2) -> torch.Tensor:
    """Per-matrix randomized truncated SVD on last two dims, on CPU/fp32."""
    if tensor.dim() < 2:
        raise ValueError(f"expected rank>=2, got {tuple(tensor.shape)}")
    cpu_t = tensor.detach().to(device="cpu", dtype=torch.float32)
    q = min(n + oversample, min(cpu_t.shape[-2:]))
    u, s, v = torch.svd_lowrank(cpu_t, q=q, niter=niter)
    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    return (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)


def _fake_quant_per_tensor(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Uniform affine fake-quant with one scale/zp for the whole tensor."""
    qmin = 0
    qmax = (1 << n_bits) - 1
    f = tensor.detach().to(dtype=torch.float32)
    t_min = f.min()
    t_max = f.max()
    scale = ((t_max - t_min) / qmax).clamp_min(1e-12)
    zero_point = torch.round(-t_min / scale).clamp(qmin, qmax)
    quantized = torch.round(f / scale + zero_point).clamp(qmin, qmax)
    return (quantized - zero_point) * scale


def _fake_quant_per_head(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Uniform affine fake-quant with a per-(B,H) scale/zp; matches SVD granularity."""
    qmin = 0
    qmax = (1 << n_bits) - 1
    f = tensor.detach().to(dtype=torch.float32)
    # tensor: [B, H, D_k, D_v]; reduce over the last two dims.
    flat = f.flatten(start_dim=-2)                                 # [B, H, D_k*D_v]
    t_min = flat.min(dim=-1, keepdim=True).values                  # [B, H, 1]
    t_max = flat.max(dim=-1, keepdim=True).values                  # [B, H, 1]
    scale = ((t_max - t_min) / qmax).clamp_min(1e-12)              # [B, H, 1]
    zero_point = torch.round(-t_min / scale).clamp(qmin, qmax)     # [B, H, 1]
    q = torch.round(flat / scale + zero_point).clamp(qmin, qmax)
    deq = (q - zero_point) * scale
    return deq.reshape_as(f)


# --- Metrics ------------------------------------------------------------------

def _stats(t: torch.Tensor) -> dict:
    t = t.float().flatten().cpu()
    if t.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(t.mean().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "median": float(t.median().item()),
    }


def _reconstruction_metrics(state_f32: torch.Tensor, approx_f32: torch.Tensor) -> dict:
    err = state_f32 - approx_f32
    head_state_sq = state_f32.pow(2).sum(dim=(-1, -2)).clamp_min(1e-30)
    head_err_sq = err.pow(2).sum(dim=(-1, -2))
    rel_frob = head_err_sq.sqrt() / head_state_sq.sqrt().clamp_min(1e-12)
    retained_energy = 1.0 - (head_err_sq / head_state_sq)
    cos = F.cosine_similarity(
        state_f32.flatten(start_dim=-2), approx_f32.flatten(start_dim=-2), dim=-1
    )
    return {
        "rel_frob": _stats(rel_frob),
        "retained_energy": _stats(retained_energy),
        "cosine_similarity": _stats(cos),
        "rel_frob_per_head": rel_frob.float().cpu().tolist(),
        "retained_energy_per_head": retained_energy.float().cpu().tolist(),
    }


def _compute_state_methods(state: torch.Tensor) -> dict:
    """Compute reconstruction metrics for each method on a single layer's state."""
    state = state.detach()
    if state.dim() != 4:
        raise ValueError(f"expected [B,H,D_k,D_v], got {tuple(state.shape)}")
    B, H, D_k, D_v = state.shape
    # All reconstruction methods run on CPU/fp32 for numerical comparability and
    # because the SVD helper is CPU-only.
    f32 = state.to(device="cpu", dtype=torch.float32)

    svd_approx = _low_rank_svd(f32, n=SVD_RANK).to(dtype=torch.float32)
    q4_pt = _fake_quant_per_tensor(f32, n_bits=QUANT_BITS)
    q4_ph = _fake_quant_per_head(f32, n_bits=QUANT_BITS)

    return {
        "shape": [B, H, D_k, D_v],
        "svd_r16": _reconstruction_metrics(f32, svd_approx),
        "q4_per_tensor": _reconstruction_metrics(f32, q4_pt),
        "q4_per_head": _reconstruction_metrics(f32, q4_ph),
    }


def _aggregate_layers(layer_metrics: list[dict], method: str) -> dict:
    rel_frob = []
    retained = []
    for lm in layer_metrics:
        for batch in lm[method]["rel_frob_per_head"]:
            for v in batch:
                rel_frob.append(v)
        for batch in lm[method]["retained_energy_per_head"]:
            for v in batch:
                retained.append(v)
    return {
        "rel_frob_overall": _stats(torch.tensor(rel_frob)),
        "retained_energy_overall": _stats(torch.tensor(retained)),
    }


def _run_for_model(model_info: dict, prompts: list[dict]) -> dict:
    print(f"\n=== {model_info['label']} ({model_info['model_id']}) ===")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_id"])
    model = model_info["cls"].from_pretrained(model_info["model_id"], torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    by_prompt: dict[str, dict] = {}
    for p in prompts:
        ids_full = tokenizer(p["text"], return_tensors="pt").input_ids[0]
        prompt_total_len = int(ids_full.shape[0])
        per_seqlen: dict[str, dict] = {}
        for seq_len in SEQ_LENS:
            if prompt_total_len < seq_len:
                continue
            ids = ids_full[:seq_len].unsqueeze(0).to(DEVICE)
            t0 = perf_counter()
            with torch.inference_mode():
                out = model(ids, use_cache=True)
            prefill_s = perf_counter() - t0
            pkv = out.past_key_values
            t0 = perf_counter()
            layer_metrics = []
            for li, layer in enumerate(pkv.layers):
                state = layer.state.get("recurrent_state")
                if state is None:
                    continue
                metrics = _compute_state_methods(state)
                layer_metrics.append({"layer": li, **metrics})
            recon_s = perf_counter() - t0
            agg = {m: _aggregate_layers(layer_metrics, m)
                   for m in ("svd_r16", "q4_per_tensor", "q4_per_head")}
            per_seqlen[str(seq_len)] = {
                "seq_len": seq_len,
                "prefill_seconds": prefill_s,
                "reconstruction_seconds": recon_s,
                "aggregate": agg,
                "layers": layer_metrics,
            }
            print(
                f"  [{p['id']:<22}] seq={seq_len:<5} prefill={prefill_s:.2f}s recon={recon_s:.2f}s "
                f"svd_relF={agg['svd_r16']['rel_frob_overall']['mean']:.4f}  "
                f"Q4pt_relF={agg['q4_per_tensor']['rel_frob_overall']['mean']:.4f}  "
                f"Q4ph_relF={agg['q4_per_head']['rel_frob_overall']['mean']:.4f}"
            )
            del out, pkv
            torch.cuda.empty_cache()
        by_prompt[p["id"]] = {
            "prompt_id": p["id"],
            "prompt_kind": p["kind"],
            "prompt_title": p["title"],
            "prompt_total_tokens": prompt_total_len,
            "by_seq_len": per_seqlen,
        }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": model_info["label"],
        "model_id": model_info["model_id"],
        "family": model_info["family"],
        "by_prompt": by_prompt,
    }


def _print_pivots(summaries: list[dict]) -> None:
    methods = [
        ("svd_r16",       "SVD r=16"),
        ("q4_per_tensor", "Q4 per-tensor"),
        ("q4_per_head",   "Q4 per-head"),
    ]
    for stat, stat_label in [
        ("rel_frob_overall", "rel-Frob mean"),
        ("retained_energy_overall", "retained-energy mean"),
    ]:
        for method_key, method_label in methods:
            print(f"\n=== {stat_label}  |  method = {method_label} ===")
            header = f"{'model':<22}{'prompt':<28}"
            for s in SEQ_LENS:
                header += f"{'seq=' + str(s):>10}"
            print(header)
            for s in summaries:
                for prompt_id, prec in s["by_prompt"].items():
                    row = f"{s['label']:<22}{prompt_id:<28}"
                    for sl in SEQ_LENS:
                        v = prec["by_seq_len"].get(str(sl))
                        if v:
                            val = v["aggregate"][method_key][stat]["mean"]
                            row += f"{val:>10.4f}"
                        else:
                            row += f"{'-':>10}"
                    print(row)


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("q4_vs_svd_sweep_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")
    print(f"Seq lens: {SEQ_LENS}  |  SVD rank: {SVD_RANK}  |  Quant bits: {QUANT_BITS}")

    prompts: list[dict] = []
    prompts.extend(_load_wikitext_articles(WIKI_TARGETS))
    prompts.append(_load_code_prompt())
    prompts.append(_load_qa_prompt())

    summaries = []
    for info in MODELS:
        model_summary = _run_for_model(info, prompts)
        summaries.append(model_summary)
        out_path = run_dir / f"{info['family']}_summary.json"
        out_path.write_text(json.dumps(model_summary, indent=2))
        print(f"  saved {out_path.name}")

    combined = {
        "svd_rank": SVD_RANK,
        "quant_bits": QUANT_BITS,
        "seq_lens": SEQ_LENS,
        "prompts": [
            {"id": p["id"], "kind": p["kind"], "title": p["title"]}
            for p in prompts
        ],
        "models": summaries,
    }
    (run_dir / "combined.json").write_text(json.dumps(combined, indent=2))

    _print_pivots(summaries)
    print(f"\nWrote outputs to {run_dir}")


if __name__ == "__main__":
    main()

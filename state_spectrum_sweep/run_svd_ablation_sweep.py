"""
Seq-len x prompt sweep of the rank-16 SVD ablation on multi-head linear-attention models.

Models compared:
  - DeltaNet:        fla-hub/delta_net-1.3B-100B   (24 layers, 16 heads, 128x128)
  - GatedDeltaNet:   m-a-p/1.3B-100B-GatedDeltaNet-pure (24 layers, 8 heads, 256x256)

For each (model, prompt, seq_len) combination:
  1. Tokenize prompt and truncate to seq_len.
  2. Run prefill, capture recurrent_state per layer.
  3. Apply rank-r truncated SVD per (layer, head, batch). Compute reconstruction
     metrics vs the original state.
  4. Record per-layer + aggregate stats.

Sequence-length sweep: 64, 256, 1024, 2048.
Prompt set: 4 distinct Wikipedia articles + concatenated HumanEval (code) + concatenated SQuAD-v2 (Q&A).
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
LOW_RANK_RANK = 16
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
    "Bob Dylan",            # music biography
    "Battle of Romani",     # military history
    "Missouri River",       # natural geography
    "Roger Federer",        # modern sports
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


def low_rank_svd(tensor: torch.Tensor, n: int, oversample: int = 4, niter: int = 2) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"expected rank>=2, got {tuple(tensor.shape)}")
    orig_device = tensor.device
    orig_dtype = tensor.dtype
    cpu_t = tensor.detach().to(device="cpu", dtype=torch.float32)
    q = min(n + oversample, min(cpu_t.shape[-2:]))
    u, s, v = torch.svd_lowrank(cpu_t, q=q, niter=niter)
    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    approx = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
    return approx.to(device=orig_device, dtype=orig_dtype)


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


def _compute_state_metrics(state: torch.Tensor, rank: int) -> dict:
    state = state.detach()
    if state.dim() != 4:
        raise ValueError(f"expected [B,H,D_k,D_v], got {tuple(state.shape)}")
    B, H, D_k, D_v = state.shape

    f32 = state.to(dtype=torch.float32)
    approx = low_rank_svd(f32, n=rank).to(dtype=torch.float32)
    err = f32 - approx

    head_state_sq = f32.pow(2).sum(dim=(-1, -2)).clamp_min(1e-30)
    head_err_sq = err.pow(2).sum(dim=(-1, -2))
    rel_frob = (head_err_sq.sqrt() / head_state_sq.sqrt().clamp_min(1e-12))
    retained_energy = 1.0 - (head_err_sq / head_state_sq)
    cos = F.cosine_similarity(f32.flatten(start_dim=-2), approx.flatten(start_dim=-2), dim=-1)

    return {
        "shape": [B, H, D_k, D_v],
        "rel_frob": _stats(rel_frob),
        "retained_energy": _stats(retained_energy),
        "cosine_similarity": _stats(cos),
        "rel_frob_per_head": rel_frob.float().cpu().tolist(),
        "retained_energy_per_head": retained_energy.float().cpu().tolist(),
    }


def _aggregate_layers(layer_metrics: list[dict]) -> dict:
    rel_frob = []
    retained = []
    for lm in layer_metrics:
        for batch in lm["rel_frob_per_head"]:
            for v in batch:
                rel_frob.append(v)
        for batch in lm["retained_energy_per_head"]:
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
                metrics = _compute_state_metrics(state, rank=LOW_RANK_RANK)
                layer_metrics.append({"layer": li, **metrics})
            svd_s = perf_counter() - t0
            agg = _aggregate_layers(layer_metrics)
            per_seqlen[str(seq_len)] = {
                "seq_len": seq_len,
                "prefill_seconds": prefill_s,
                "svd_seconds": svd_s,
                "rel_frob_overall": agg["rel_frob_overall"],
                "retained_energy_overall": agg["retained_energy_overall"],
                "layers": layer_metrics,
            }
            print(
                f"  [{p['id']:<25}] seq={seq_len:<5} prefill={prefill_s:.2f}s svd={svd_s:.2f}s "
                f"rel_F={agg['rel_frob_overall']['mean']:.4f} retained={agg['retained_energy_overall']['mean']:.4f}"
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


def _print_pivot(summaries: list[dict]) -> None:
    print("\n=== Pivot: rel-Frob mean (rank=16) by (model, prompt, seq_len) ===")
    header = f"{'model':<22}{'prompt':<28}" + "".join([f"{'seq=' + str(s):>10}" for s in SEQ_LENS])
    print(header)
    for s in summaries:
        for prompt_id, prec in s["by_prompt"].items():
            row = f"{s['label']:<22}{prompt_id:<28}"
            for sl in SEQ_LENS:
                v = prec["by_seq_len"].get(str(sl))
                row += f"{v['rel_frob_overall']['mean']:>10.4f}" if v else f"{'-':>10}"
            print(row)
    print("\n=== Pivot: retained-energy mean by (model, prompt, seq_len) ===")
    print(header)
    for s in summaries:
        for prompt_id, prec in s["by_prompt"].items():
            row = f"{s['label']:<22}{prompt_id:<28}"
            for sl in SEQ_LENS:
                v = prec["by_seq_len"].get(str(sl))
                row += f"{v['retained_energy_overall']['mean']:>10.4f}" if v else f"{'-':>10}"
            print(row)


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("svd_sweep_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")
    print(f"Seq lens: {SEQ_LENS}")
    print(f"Rank: {LOW_RANK_RANK}")

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
        "rank": LOW_RANK_RANK,
        "seq_lens": SEQ_LENS,
        "prompts": [
            {"id": p["id"], "kind": p["kind"], "title": p["title"]}
            for p in prompts
        ],
        "models": summaries,
    }
    (run_dir / "combined.json").write_text(json.dumps(combined, indent=2))

    _print_pivot(summaries)
    print(f"\nWrote outputs to {run_dir}")


if __name__ == "__main__":
    main()

"""
SVD ablation: how much does rank-16 truncated SVD distort the recurrent state
of multi-head linear-attention models?

Compares:
  - DeltaNet:        fla-hub/delta_net-1.3B-100B    (24 layers, 16 heads, 128x128)
  - GatedDeltaNet:   m-a-p/1.3B-100B-GatedDeltaNet-pure (24 layers, 8 heads, 256x256)

For each layer's recurrent state S of shape [B, H, D_k, D_v] we:
  1. Compute the rank-r truncated SVD reconstruction S_hat (per-head, per-batch).
  2. Compute reconstruction-error metrics vs S:
       - relative Frobenius error  ||S - S_hat||_F / ||S||_F
       - MSE
       - cosine similarity (flattened)
       - retained energy 1 - ||err||^2 / ||S||^2
       - effective rank (entropy of singular spectrum)
  3. Aggregate per-layer (mean over heads) and per-model.

Storage ratio reported separately: a rank-r factored form costs r*(D_k+D_v),
so compression vs the dense D_k*D_v matrix is r*(D_k+D_v) / (D_k*D_v).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

import fla  # noqa: F401  (registers FLA model classes)
from fla.models import DeltaNetForCausalLM, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer


PROMPT = (
    "The history of computing is long and varied. From the abacus to the modern "
    "supercomputer, humans have continually invented machines to help us reason, "
    "calculate, and communicate. Charles Babbage designed the Analytical Engine "
    "in the 1830s, anticipating concepts that would not be implemented until a "
    "century later. Ada Lovelace wrote what is sometimes called the first computer "
    "program for that machine. World War II accelerated electronic computation: "
    "Colossus broke German cipher traffic, while the ENIAC simulated artillery "
    "trajectories. The transistor and then the integrated circuit shrunk room-"
    "sized machines into devices smaller than a postage stamp. Personal computers "
    "in the 1970s and 1980s put computation onto every desk, and the internet of "
    "the 1990s wove those desks together. Mobile phones brought the network into "
    "every pocket. Today, large language models trained on much of the world's "
    "written text have begun to act like fluent collaborators on tasks that were "
    "thought to require human cognition. Each step has rested on the prior one. "
    "Each has also altered what people thought a computer was, and what it was "
    "for, and who might use it, and toward what ends."
)
TARGET_PREFILL_TOKENS = 300  # 'a few hundred' as per the task description
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


def low_rank_svd(tensor: torch.Tensor, n: int, oversample: int = 4, niter: int = 2) -> torch.Tensor:
    """Per-matrix randomized truncated SVD. Operates on the last two dims."""
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


def _full_singular_values(tensor: torch.Tensor) -> torch.Tensor:
    """Full singular spectrum on the last two dims (used for effective rank)."""
    cpu_t = tensor.detach().to(device="cpu", dtype=torch.float32)
    return torch.linalg.svdvals(cpu_t)


def _effective_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy-based effective rank, per matrix (last dim is singular vals)."""
    s = singular_values.clamp_min(0.0)
    s_sum = s.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = s / s_sum
    safe = p.clamp_min(eps)
    entropy = -(p * safe.log()).sum(dim=-1)
    return entropy.exp()


def _compute_state_metrics(state: torch.Tensor, rank: int) -> dict:
    """Reconstruction-error metrics for a single layer's state.

    state shape: [B, H, D_k, D_v]
    """
    state = state.detach()
    if state.dim() != 4:
        raise ValueError(f"expected [B,H,D_k,D_v], got {tuple(state.shape)}")
    B, H, D_k, D_v = state.shape

    f32 = state.to(dtype=torch.float32)
    approx = low_rank_svd(f32, n=rank).to(dtype=torch.float32)
    err = f32 - approx

    head_state_sq = f32.pow(2).sum(dim=(-1, -2)).clamp_min(1e-30)  # [B,H]
    head_err_sq = err.pow(2).sum(dim=(-1, -2))                       # [B,H]
    head_state_norm = head_state_sq.sqrt()
    head_err_norm = head_err_sq.sqrt()

    rel_frob = (head_err_norm / head_state_norm.clamp_min(1e-12))
    retained_energy = 1.0 - (head_err_sq / head_state_sq)
    mse = err.pow(2).mean(dim=(-1, -2))                             # per-head MSE
    cos_per_head = F.cosine_similarity(
        f32.flatten(start_dim=-2), approx.flatten(start_dim=-2), dim=-1
    )                                                                # [B,H]

    # Effective rank of the original state (per head).
    sv_full = _full_singular_values(f32)        # [B, H, min(D_k, D_v)]
    eff_rank = _effective_rank(sv_full)         # [B, H]

    # Energy carried by the top-r singular values.
    energy_top_r = (sv_full[..., :rank].pow(2).sum(dim=-1)
                    / sv_full.pow(2).sum(dim=-1).clamp_min(1e-30))   # [B,H]

    # Aggregate over batch & heads.
    def _stats(t: torch.Tensor) -> dict:
        t = t.float().flatten().cpu()
        return {
            "mean": float(t.mean().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        }

    return {
        "shape": [B, H, D_k, D_v],
        "rel_frob": _stats(rel_frob),
        "retained_energy": _stats(retained_energy),
        "mse": _stats(mse),
        "cosine_similarity": _stats(cos_per_head),
        "effective_rank_full": _stats(eff_rank),
        "energy_top_r": _stats(energy_top_r),
        "rank": rank,
        "rel_frob_per_head": rel_frob.float().cpu().tolist(),
        "retained_energy_per_head": retained_energy.float().cpu().tolist(),
    }


def _run_one_model(model_info: dict, prompt_ids: torch.Tensor, rank: int) -> dict:
    print(f"\n=== {model_info['label']}  ({model_info['model_id']}) ===")
    t0 = perf_counter()
    model = model_info["cls"].from_pretrained(
        model_info["model_id"], torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    t0 = perf_counter()
    with torch.inference_mode():
        out = model(prompt_ids, use_cache=True)
    pkv = out.past_key_values
    print(f"  prefill ({prompt_ids.shape[1]} tokens) in {perf_counter()-t0:.2f}s")

    layer_metrics = []
    for li, layer in enumerate(pkv.layers):
        state = layer.state.get("recurrent_state")
        if state is None:
            continue
        metrics = _compute_state_metrics(state, rank=rank)
        layer_metrics.append({"layer": li, **metrics})

    # Whole-model aggregates over layers x heads.
    def _flat(metric_key: str) -> torch.Tensor:
        vals = []
        for lm in layer_metrics:
            for v in lm[metric_key]:
                if isinstance(v, list):
                    for vv in v:
                        vals.append(vv)
                else:
                    vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    rel_frob_all = _flat("rel_frob_per_head")
    retained_all = _flat("retained_energy_per_head")

    # Per-element storage analysis (the 'state object size' question).
    if layer_metrics:
        b, h, d_k, d_v = layer_metrics[0]["shape"]
    else:
        b, h, d_k, d_v = (None, None, None, None)
    elements_per_layer_dense = (h or 0) * (d_k or 0) * (d_v or 0)
    elements_per_layer_low_rank = (h or 0) * rank * ((d_k or 0) + (d_v or 0))
    bytes_per_elt = 4  # state is fp32 in FLA cache
    dense_bytes_total = elements_per_layer_dense * len(layer_metrics) * bytes_per_elt
    low_rank_bytes_total = elements_per_layer_low_rank * len(layer_metrics) * bytes_per_elt

    summary = {
        "label": model_info["label"],
        "model_id": model_info["model_id"],
        "family": model_info["family"],
        "num_layers_with_state": len(layer_metrics),
        "rank": rank,
        "per_head_state_shape": [d_k, d_v],
        "num_heads_per_layer": h,
        "rel_frob_overall": {
            "mean": float(rel_frob_all.mean().item()),
            "min": float(rel_frob_all.min().item()),
            "max": float(rel_frob_all.max().item()),
            "median": float(rel_frob_all.median().item()),
        },
        "retained_energy_overall": {
            "mean": float(retained_all.mean().item()),
            "min": float(retained_all.min().item()),
            "max": float(retained_all.max().item()),
            "median": float(retained_all.median().item()),
        },
        "dense_state_bytes_total_fp32": dense_bytes_total,
        "low_rank_state_bytes_total_fp32": low_rank_bytes_total,
        "compression_ratio_low_rank_over_dense": (
            low_rank_bytes_total / dense_bytes_total if dense_bytes_total else None
        ),
        "elements_dense_per_layer": elements_per_layer_dense,
        "elements_low_rank_per_layer": elements_per_layer_low_rank,
        "layers": layer_metrics,
    }

    # Free the model.
    del model, pkv, out
    torch.cuda.empty_cache()
    return summary


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("svd_ablation_gdn_vs_dn_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {run_dir}")
    print(f"Rank: {LOW_RANK_RANK} | Target prefill tokens: {TARGET_PREFILL_TOKENS}")

    # Tokenize once with each model's tokenizer; both fla-hub models actually use
    # the same Mistral-based tokenizer, but we tokenize per-model to be safe.
    summaries = []
    for info in MODELS:
        tok = AutoTokenizer.from_pretrained(info["model_id"])
        ids = tok(PROMPT, return_tensors="pt").input_ids
        # Trim/extend to ~TARGET_PREFILL_TOKENS by repetition if needed.
        if ids.shape[1] < TARGET_PREFILL_TOKENS:
            repeats = (TARGET_PREFILL_TOKENS // ids.shape[1]) + 1
            ids = ids.repeat(1, repeats)
        ids = ids[:, :TARGET_PREFILL_TOKENS].to(DEVICE)
        info_summary = _run_one_model(info, ids, rank=LOW_RANK_RANK)
        info_summary["actual_prefill_tokens"] = int(ids.shape[1])
        summaries.append(info_summary)

        out_path = run_dir / f"{info['family']}_summary.json"
        out_path.write_text(json.dumps(info_summary, indent=2))
        print(f"  saved {out_path.name}")

    # Compact comparison block.
    print("\n=== Summary (rank-{} truncated SVD reconstruction) ===".format(LOW_RANK_RANK))
    header = (
        f"{'model':<25}{'heads x D':<14}{'rel_F mean':>12}{'rel_F max':>12}"
        f"{'retained mean':>16}{'retained min':>15}{'compression':>13}"
    )
    print(header)
    for s in summaries:
        d_k, d_v = s["per_head_state_shape"]
        shape_str = f"{s['num_heads_per_layer']}x{d_k}x{d_v}"
        print(
            f"{s['label']:<25}{shape_str:<14}"
            f"{s['rel_frob_overall']['mean']:>12.4f}"
            f"{s['rel_frob_overall']['max']:>12.4f}"
            f"{s['retained_energy_overall']['mean']:>16.4f}"
            f"{s['retained_energy_overall']['min']:>15.4f}"
            f"{s['compression_ratio_low_rank_over_dense']:>13.4f}"
        )

    combined = {
        "rank": LOW_RANK_RANK,
        "target_prefill_tokens": TARGET_PREFILL_TOKENS,
        "prompt_text": PROMPT,
        "models": summaries,
    }
    (run_dir / "combined.json").write_text(json.dumps(combined, indent=2))
    print(f"\nWrote combined.json to {run_dir}")


if __name__ == "__main__":
    main()

"""
State-level reconstruction-fidelity sweep across five recurrent / SSM models.

For each (model, input distribution, sequence position) this runs a single
prefill, captures the recurrent state at every (layer, head), and computes one
batched SVD per task. From the SVD we report, per (layer, head, position,
prompt):

  * cumulative singular-value energy at each rank in K_GRID
  * effective-rank thresholds (99 / 99.5 / 99.9 % energy)  + random control
  * truncated-SVD reconstruction error at each rank:
        - relative Frobenius MSE   (rel_fro_mse_k*)
        - per-element max-abs / mean-abs error
        - flat cosine similarity   (cos_flat_k*)

All of the heavy lifting (SVD, metrics, prompt loading, the thread pool) is the
shared `svd_sweep/_helpers.py` module so the methodology is bit-for-bit the same
as the original svd_sweep runs. The only per-model code is the `state_fn`
adapter that turns one forward pass into {layer_idx: state[H, D_k, D_v]}.

Prompts: ShareGPT (chat) + codeparrot/The-Stack (code) via `load_prompts()`.
Control the count with SWEEP_N_PROMPTS (default 1; the .sh uses 200).

Usage:
    python run_reconstruction.py --model qwen35
    python run_reconstruction.py --model mamba2
    python run_reconstruction.py --model nemotron
    python run_reconstruction.py --model deltanet
    python run_reconstruction.py --model gated_deltanet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Reuse the exact sweep infrastructure from svd_sweep/_helpers.py so the metric
# definitions, batched SVD, prompt loaders and parallel runner are identical.
HERE = Path(__file__).resolve().parent
SVD_SWEEP_DIR = HERE.parent / "svd_sweep"
sys.path.insert(0, str(SVD_SWEEP_DIR))

from _helpers import (  # noqa: E402
    NUM_WORKERS,
    POSITIONS_FULL,
    build_tasks,
    free_memory,
    load_prompts,
    make_run_dir,
    run_tasks_parallel,
    save_spectra,
    write_csv,
)
from _lowrank_metrics import (  # noqa: E402
    NITER,
    OVERSAMPLE,
    RECON_DTYPE,
    compute_task_metrics_batched_lowrank,
)

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_ROOT = HERE / "results"

# One position sweep shared by every model so the cross-model comparison is
# apples-to-apples. These are the fixed prefill lengths; build_tasks() also
# appends a "full" (-1) task for every prompt longer than the largest position,
# so the effective grid is {256, 512, 1024, 2048, 4096, full}. Override with
# SWEEP_POSITIONS="256,1024,4096" (full is always added on top).
POSITIONS = POSITIONS_FULL
_env_positions = os.getenv("SWEEP_POSITIONS")
if _env_positions:
    POSITIONS = tuple(int(x) for x in _env_positions.replace(" ", "").split(",") if x)


# --------------------------------------------------------------------------- #
# Per-model state extraction. Each state_fn returns
#   {layer_idx: state_tensor of shape [H, D_k, D_v]} on CPU/float32.
# --------------------------------------------------------------------------- #

def _squeeze_state(s: torch.Tensor) -> torch.Tensor:
    return s.detach().to(device="cpu", dtype=torch.float32).squeeze(0)


def _builtin_recurrent_state_fn(model, layer_indices, cache_attr):
    """transformers built-in models (mamba2 / nemotron_h / qwen3_5_gdn):
    states live at `cache.layers[li].recurrent_states`, shape [B, H, D_k, D_v]."""
    def state_fn(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.inference_mode():
            out = model(ids.unsqueeze(0).to(DEVICE), use_cache=True)
        cache = getattr(out, cache_attr)
        layers = layer_indices if layer_indices is not None else range(len(cache.layers))
        states: dict[int, torch.Tensor] = {}
        for li in layers:
            s = getattr(cache.layers[li], "recurrent_states", None)
            if s is None or s.dim() < 4:
                continue
            states[li] = _squeeze_state(s)
        del out, cache
        free_memory()
        return states
    return state_fn


def _fla_recurrent_state_fn(model):
    """FLA models (DeltaNet / GatedDeltaNet): every layer has a recurrent state
    at `cache.layers[li].state['recurrent_state']`, shape [B, H, D_k, D_v]."""
    def state_fn(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.inference_mode():
            out = model(ids.unsqueeze(0).to(DEVICE), use_cache=True)
        cache = out.past_key_values
        states: dict[int, torch.Tensor] = {}
        for li, layer in enumerate(cache.layers):
            st = getattr(layer, "state", None)
            s = st.get("recurrent_state") if isinstance(st, dict) else None
            if s is None or s.dim() < 4:
                continue
            states[li] = _squeeze_state(s)
        del out, cache
        free_memory()
        return states
    return state_fn


# --------------------------------------------------------------------------- #
# Per-model loaders.  Each returns (model, tokenizer, state_fn, positions, info)
# --------------------------------------------------------------------------- #

def load_qwen35():
    model_id = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE).eval()
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    linear_layers = [li for li, lyr in enumerate(text_model.layers) if hasattr(lyr, "linear_attn")]
    for li in linear_layers:                      # force the slow conv so states are exact
        text_model.layers[li].linear_attn.causal_conv1d_fn = None
    sfn = _builtin_recurrent_state_fn(model, linear_layers, "past_key_values")
    return model, tok, sfn, f"linear layers={linear_layers}"


def load_mamba2():
    model_id = "AntonV/mamba2-1.3b-hf"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE).eval()
    sfn = _builtin_recurrent_state_fn(model, None, "cache_params")
    return model, tok, sfn, "all layers"


def load_nemotron():
    model_id = "unsloth/NVIDIA-Nemotron-3-Nano-4B"
    cfg = AutoConfig.from_pretrained(model_id)
    mamba_layers = [i for i, c in enumerate(cfg.hybrid_override_pattern) if c == "M"]
    tok = AutoTokenizer.from_pretrained(model_id)
    # sdpa: attention dispatches to F.scaled_dot_product_attention (flash kernels)
    # instead of materialising a full [seq, seq] score matrix -> avoids OOM on
    # long-context prefills. Uses transformers' bug-free built-in nemotron_h.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map=DEVICE, attn_implementation="sdpa",
    ).eval()
    sfn = _builtin_recurrent_state_fn(model, mamba_layers, "past_key_values")
    return model, tok, sfn, f"mamba layers={mamba_layers}"


def load_deltanet():
    import fla  # noqa: F401  registers FLA classes
    from fla.models import DeltaNetForCausalLM
    model_id = "fla-hub/delta_net-1.3B-100B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = DeltaNetForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE).eval()
    sfn = _fla_recurrent_state_fn(model)
    return model, tok, sfn, "all layers (FLA)"


def load_gated_deltanet():
    import fla  # noqa: F401
    from fla.models import GatedDeltaNetForCausalLM
    model_id = "m-a-p/1.3B-100B-GatedDeltaNet-pure"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = GatedDeltaNetForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE).eval()
    sfn = _fla_recurrent_state_fn(model)
    return model, tok, sfn, "all layers (FLA)"


MODELS = {
    "qwen35":         dict(label="Qwen3.5-4B",          family="qwen3_5_gated_deltanet", loader=load_qwen35),
    "mamba2":         dict(label="Mamba2-1.3B",         family="mamba2",                 loader=load_mamba2),
    "nemotron":       dict(label="Nemotron-3-Nano-4B",  family="nemotron_h",             loader=load_nemotron),
    "deltanet":       dict(label="DeltaNet-1.3B",       family="delta_net",              loader=load_deltanet),
    "gated_deltanet": dict(label="GatedDeltaNet-1.3B",  family="gated_deltanet",         loader=load_gated_deltanet),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="which model to sweep")
    args = ap.parse_args()
    spec = MODELS[args.model]

    run_dir = make_run_dir(RESULTS_ROOT, f"recon_{args.model}")
    print(f"Output dir: {run_dir}")

    prompts = load_prompts()

    print(f"\nLoading {spec['label']} ...")
    t0 = perf_counter()
    model, tokenizer, state_fn, info = spec["loader"]()
    print(f"  loaded in {perf_counter()-t0:.1f}s   ({info})")

    tasks = build_tasks(prompts, tokenizer, POSITIONS)
    print(f"\nBuilt {len(tasks)} tasks across {len(prompts)} prompts; "
          f"positions={POSITIONS} (+ full)")
    print(f"Reconstruction: randomized torch.svd_lowrank "
          f"(oversample={OVERSAMPLE}, niter={NITER}, dtype={RECON_DTYPE}); "
          f"spectrum/effective-rank/random-control use exact full SVD.")

    rows, sv_state_all, sv_random_all, sv_index = run_tasks_parallel(
        tasks,
        model_label=spec["label"],
        family=spec["family"],
        state_fn=state_fn,
        num_workers=NUM_WORKERS,
        metrics_fn=compute_task_metrics_batched_lowrank,
    )

    print(f"\nWriting {len(rows)} rows to metrics.csv ...")
    write_csv(run_dir / "metrics.csv", rows)
    save_spectra(run_dir / "spectra.pt", sv_state_all, sv_random_all, sv_index)
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()

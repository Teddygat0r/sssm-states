"""
Section 1.1 + 1.2 -- Qwen3.5-4B (GDN hybrid, full grid).

Sweep axes (full grid):
  * 24 linear-attention layers (every layer that has a Qwen3_5GatedDeltaNet)
  * 32 V-heads per layer (state per (layer, head) is [128, 128])
  * positions {256, 512, 1024, 2048, 4096, full}
  * input distributions {ShareGPT, code from The Stack}

Parallel pattern follows state_spectrum_sweep/run_experiment_kl_parallel.py:
  * model loaded once
  * each (input, position) becomes one task
  * ThreadPoolExecutor with SVD_PARALLEL_WORKERS workers shares the model
  * forward passes are serialized via a lock; CPU SVDs run truly in parallel
    (PyTorch BLAS releases the GIL)

Per-(layer, head) we do exactly ONE full SVD on the state matrix (cached
factors reused for every k in K_GRID) plus one svdvals on the matched random
control.  Both are wrapped in try/except: on failure we fall back to the
original state (zero reconstruction error) and flag svd_ok=False.

Outputs go to state_spectrum_sweep/experiments/svd_sweep_qwen35_<ts>/.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from _helpers import (
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


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_LABEL = "Qwen3.5-4B"
FAMILY = "qwen3_5_gated_deltanet"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "state_spectrum_sweep" / "experiments"


def _find_linear_layers(model) -> list[int]:
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [li for li, layer in enumerate(text_model.layers) if hasattr(layer, "linear_attn")]


def _disable_fast_conv(model, linear_layers):
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li in linear_layers:
        text_model.layers[li].linear_attn.causal_conv1d_fn = None


def _make_state_fn(model, linear_layers):
    """transformers >= 5.7 stores per-linear-attention-layer states as
    `cache.layers[li].recurrent_states` of shape
    `[batch=1, num_v_heads, D_k, D_v]`."""
    def state_fn(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.inference_mode():
            out = model(ids.unsqueeze(0).to(DEVICE), use_cache=True)
        cache = out.past_key_values
        states: dict[int, torch.Tensor] = {}
        for li in linear_layers:
            layer = cache.layers[li]
            s = getattr(layer, "recurrent_states", None)
            if s is None or s.dim() < 4:
                continue
            states[li] = s.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        del out, cache
        free_memory()
        return states
    return state_fn


def main():
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "svd_sweep_qwen35")
    print(f"Output dir: {run_dir}")

    cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
    print(f"  layers={cfg.num_hidden_layers}  num_v_heads={cfg.linear_num_value_heads}"
          f"  D_k={cfg.linear_key_head_dim}  D_v={cfg.linear_value_head_dim}")

    prompts = load_prompts()

    print(f"\nLoading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    linear_layers = _find_linear_layers(model)
    print(f"  linear-attn layers: {linear_layers}")
    _disable_fast_conv(model, linear_layers)

    tasks = build_tasks(prompts, tokenizer, POSITIONS_FULL)
    print(f"\nBuilt {len(tasks)} tasks across {len(prompts)} prompts.")

    state_fn = _make_state_fn(model, linear_layers)
    rows, sv_state_all, sv_random_all, sv_index = run_tasks_parallel(
        tasks,
        model_label=MODEL_LABEL,
        family=FAMILY,
        state_fn=state_fn,
        num_workers=NUM_WORKERS,
    )

    print(f"\nWriting {len(rows)} rows to metrics.csv ...")
    write_csv(run_dir / "metrics.csv", rows)
    save_spectra(run_dir / "spectra.pt", sv_state_all, sv_random_all, sv_index)
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()

"""
Section 1.1 + 1.2 -- AntonV/mamba2-1.3b-hf (pure Mamba2 control).

Pure-SSM control: confirms the spectrum-concentration property comes from
the SSM recurrence itself, not the hybrid / attention mixing.

Sweep axes (subset):
  * all 48 layers, all 64 heads (state per (layer, head) is [head_dim=64, state_size=128])
  * positions {256, 1024, 4096, full}
  * input distributions {ShareGPT, code from The Stack}

Parallel pattern follows state_spectrum_sweep/run_experiment_kl_parallel.py:
  * model loaded once
  * each (input, position) becomes one task
  * ThreadPoolExecutor with SVD_PARALLEL_WORKERS workers shares the model
  * forward passes serialize via lock; CPU SVDs run truly in parallel.

Outputs go to state_spectrum_sweep/experiments/svd_sweep_mamba2_<ts>/.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from _helpers import (
    NUM_WORKERS,
    POSITIONS_SUBSET,
    build_tasks,
    free_memory,
    load_prompts,
    make_run_dir,
    run_tasks_parallel,
    save_spectra,
    write_csv,
)


MODEL_ID = "AntonV/mamba2-1.3b-hf"
MODEL_LABEL = "Mamba2-1.3B"
FAMILY = "mamba2"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "state_spectrum_sweep" / "experiments"


def _make_state_fn(model):
    """transformers >= 5.7 stores per-Mamba-layer states as
    `cache.layers[li].recurrent_states` of shape
    `[batch=1, num_heads, head_dim, ssm_state_size]`."""
    def state_fn(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.inference_mode():
            out = model(ids.unsqueeze(0).to(DEVICE), use_cache=True)
        cache = out.cache_params
        states: dict[int, torch.Tensor] = {}
        for li, layer in enumerate(cache.layers):
            s = getattr(layer, "recurrent_states", None)
            if s is None or s.dim() < 4:
                continue
            states[li] = s.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        del out, cache
        free_memory()
        return states
    return state_fn


def main():
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "svd_sweep_mamba2")
    print(f"Output dir: {run_dir}")

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print(f"  layers={cfg.num_hidden_layers}  num_heads={cfg.num_heads}"
          f"  head_dim={cfg.head_dim}  state_size={cfg.state_size}")

    prompts = load_prompts()

    print(f"\nLoading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")

    tasks = build_tasks(prompts, tokenizer, POSITIONS_SUBSET)
    print(f"\nBuilt {len(tasks)} tasks across {len(prompts)} prompts.")

    state_fn = _make_state_fn(model)
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

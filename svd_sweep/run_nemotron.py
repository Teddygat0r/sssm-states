"""
Section 1.1 + 1.2 -- Nemotron-3-Nano-4B (Mamba2 hybrid, secondary sweep).

Architecture (NemotronH):
  * 42 layers; hybrid_override_pattern mixes Mamba ('M'), attention ('*')
    and MLP ('-') layers.  We only sweep the Mamba layers.
  * mamba_num_heads=96, mamba_head_dim=80, ssm_state_size=128, so per-head
    state is [80, 128].

We use unsloth/NVIDIA-Nemotron-3-Nano-4B because nvidia/Nemotron-3-Nano-4B
itself only ships GGUF; this is the matching PyTorch checkpoint.  The weights
are loaded into transformers' built-in `nemotron_h` model (no
`trust_remote_code` needed in transformers >= 5.7).  This avoids two real
upstream bugs in the unsloth/NVIDIA remote modeling code (`_supports_sdpa`
not declared, and an `attn_output.view` shape mismatch in `NemotronHSdpaAttention`).

Sweep axes (subset):
  * all Mamba layers, all 96 heads
  * positions {256, 1024, 4096, full}
  * input distributions {ShareGPT, code from The Stack}

Outputs go to state_spectrum_sweep/experiments/svd_sweep_nemotron_<ts>/.
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


MODEL_ID = "unsloth/NVIDIA-Nemotron-3-Nano-4B"
MODEL_LABEL = "Nemotron-3-Nano-4B"
FAMILY = "nemotron_h"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "state_spectrum_sweep" / "experiments"


def _mamba_layer_indices(cfg) -> list[int]:
    return [i for i, c in enumerate(cfg.hybrid_override_pattern) if c == "M"]


def _make_state_fn(model, mamba_layers):
    """Built-in nemotron_h stores per-Mamba-layer states as
    `cache.layers[li].recurrent_states` of shape
    `[batch=1, num_heads=96, head_dim=80, ssm_state_size=128]`. No reshape
    needed."""
    def state_fn(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.inference_mode():
            out = model(ids.unsqueeze(0).to(DEVICE), use_cache=True)
        cache = out.past_key_values
        states: dict[int, torch.Tensor] = {}
        for li in mamba_layers:
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
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "svd_sweep_nemotron")
    print(f"Output dir: {run_dir}")

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    mamba_layers = _mamba_layer_indices(cfg)
    print(f"  layers={cfg.num_hidden_layers}  mamba_layers={mamba_layers}")
    print(f"  mamba_num_heads={cfg.mamba_num_heads}  mamba_head_dim={cfg.mamba_head_dim}"
          f"  ssm_state_size={cfg.ssm_state_size}")

    prompts = load_prompts()

    print(f"\nLoading {MODEL_ID} (built-in nemotron_h, attn_implementation=sdpa) ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # SDPA dispatches attention through F.scaled_dot_product_attention so we
    # never materialise the full [seq, seq] attention-scores matrix. The default
    # eager path would OOM on long-context prefills (~450 GB for a 96k forward).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s; attn={model.config._attn_implementation}")

    tasks = build_tasks(prompts, tokenizer, POSITIONS_SUBSET)
    print(f"\nBuilt {len(tasks)} tasks across {len(prompts)} prompts.")

    state_fn = _make_state_fn(model, mamba_layers)
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

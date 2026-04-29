"""
Dedicated low-rank recurrent-state experiment runner for Qwen3.6-27B.

This keeps the existing Qwen3.5 scripts untouched while reusing the same
core SVD/KL workflow that already exists in this repo.

By default, this script loads the standard Hugging Face checkpoint:

    python state_spectrum_sweep/run_experiment_qwen36_27b.py

You can also point it at a GGUF checkpoint source:

    MODEL_NAME=ggml-org/Qwen3.6-27B-GGUF \
    GGUF_FILE=Qwen3.6-27B.Q4_K_M.gguf \
    python state_spectrum_sweep/run_experiment_qwen36_27b.py

Important note:
Transformers GGUF loading dequantizes weights back into PyTorch weights at
load time. That means Q4_K_M is useful as a checkpoint source, but it does
not necessarily preserve llama.cpp-style runtime memory savings inside this
experiment script.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from state_spectrum_sweep.run_experiment_kl import (
    PROMPT_SUITE,
    _summary_stats,
    run_single_prompt,
)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.6-27B")
GGUF_FILE = os.getenv("GGUF_FILE")
LOW_RANK_RANK = int(os.getenv("LOW_RANK_RANK", "64"))
SEED = int(os.getenv("SEED", "1234"))
SVD_NITER = int(os.getenv("SVD_NITER", "4"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))
PROMPTS_LIMIT = int(os.getenv("PROMPTS_LIMIT", "10"))
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"


def _resolve_dtype() -> torch.dtype:
    dtype_name = os.getenv(
        "TORCH_DTYPE",
        "bfloat16" if torch.cuda.is_available() else "float32",
    ).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported TORCH_DTYPE={dtype_name!r}. "
            "Use one of: float32, float16, bfloat16."
        )
    return mapping[dtype_name]


DTYPE = _resolve_dtype()

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_load_kwargs() -> dict:
    load_kwargs = {
        "torch_dtype": DTYPE,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if GGUF_FILE:
        load_kwargs["gguf_file"] = GGUF_FILE
    return load_kwargs


def _build_tokenizer_kwargs() -> dict:
    tokenizer_kwargs = {"trust_remote_code": True}
    if GGUF_FILE:
        tokenizer_kwargs["gguf_file"] = GGUF_FILE
    return tokenizer_kwargs


def _run_name() -> str:
    source = "gguf" if GGUF_FILE else "hf"
    quant = Path(GGUF_FILE).stem if GGUF_FILE else "full"
    quant = quant.replace(".", "_").replace("-", "_")
    return (
        f"suite_qwen36_27b_rank{LOW_RANK_RANK}_{source}_{quant}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def main():
    _set_seed(SEED)
    run_dir = EXPERIMENTS_ROOT / _run_name()
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **_build_tokenizer_kwargs())
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_build_load_kwargs())

    prompts = PROMPT_SUITE[:PROMPTS_LIMIT] if PROMPTS_LIMIT > 0 else PROMPT_SUITE

    print(f"Running Qwen3.6-27B KL suite with {len(prompts)} prompts")
    print(f"Model source: {MODEL_NAME}")
    if GGUF_FILE:
        print(f"GGUF file: {GGUF_FILE}")
        print(
            "Note: Transformers GGUF loading dequantizes weights into PyTorch "
            "weights during load."
        )
    print(f"Torch dtype: {DTYPE}")
    print(f"Seed: {SEED}")
    print(f"Low-rank rank: {LOW_RANK_RANK}")
    print(f"SVD niter: {SVD_NITER}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Saving outputs to: {run_dir}")

    run_config = {
        "model_name": MODEL_NAME,
        "gguf_file": GGUF_FILE,
        "rank": LOW_RANK_RANK,
        "low_rank_rank": LOW_RANK_RANK,
        "seed": SEED,
        "svd_niter": SVD_NITER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts_limit": PROMPTS_LIMIT,
        "torch_dtype": str(DTYPE),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    for i, prompt in enumerate(prompts, start=1):
        prompt_id = f"prompt_{i:02d}"
        print(f"\n[{i:02d}/{len(prompts)}] {prompt_id}")
        try:
            kl_tensor, metric_tensors, response, _prompt_len, meta = run_single_prompt(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                low_rank_n=LOW_RANK_RANK,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            kl_path = run_dir / f"{prompt_id}_kl.pt"
            mse_path = run_dir / f"{prompt_id}_mse.pt"
            rel_frob_path = run_dir / f"{prompt_id}_rel_frob.pt"
            cosine_path = run_dir / f"{prompt_id}_cosine_similarity.pt"
            retained_energy_path = run_dir / f"{prompt_id}_retained_energy.pt"
            summary_path = run_dir / f"{prompt_id}_metrics_summary.json"
            text_path = run_dir / f"{prompt_id}_prompt_response.txt"

            torch.save(kl_tensor, kl_path)
            torch.save(metric_tensors["mse"], mse_path)
            torch.save(metric_tensors["rel_frob"], rel_frob_path)
            torch.save(metric_tensors["cosine_similarity"], cosine_path)
            torch.save(metric_tensors["retained_energy"], retained_energy_path)

            summary = _summary_stats(kl_tensor, metric_tensors, meta)
            summary["prompt_id"] = prompt_id
            summary["model_name"] = MODEL_NAME
            summary["gguf_file"] = GGUF_FILE
            summary["low_rank_rank"] = LOW_RANK_RANK
            summary["max_new_tokens_cap"] = MAX_NEW_TOKENS
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            text_path.write_text(
                f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n",
                encoding="utf-8",
            )

            print(
                f"  metric steps: {meta['num_metric_steps']}, "
                f"generated tokens: {meta['num_generated_tokens']}"
            )
            print(f"  stopped: {meta['stopped_reason']}")
            print(f"  elapsed: {meta['elapsed_seconds']:.3f}s")
            if summary.get("kl"):
                print(f"  KL mean: {summary['kl']['mean']:.6f}")
            print(
                "  Saved: "
                f"{kl_path.name}, {mse_path.name}, {rel_frob_path.name}, "
                f"{cosine_path.name}, {retained_energy_path.name}, "
                f"{summary_path.name}, {text_path.name}"
            )
        except Exception as exc:
            print(f"Failed for {prompt_id}: {exc}")

    print("\nDone.")
    print(f"Experiment artifacts are in: {run_dir}")


if __name__ == "__main__":
    main()

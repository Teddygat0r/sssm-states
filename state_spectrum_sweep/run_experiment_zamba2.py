"""
Starter experiment for Zamba2-family models.

This mirrors `run_experiment_jamba.py`, but pins the workflow to Zamba2
checkpoints and defaults to the smaller 2.7B variant so it is easy to compare
cache structure and usefulness across sizes.

Examples
--------
Inspect cache fields after prompt prefill on the default 2.7B model:
    python state_spectrum_sweep/run_experiment_zamba2.py --inspect-cache

Switch to the 7B variant:
    python state_spectrum_sweep/run_experiment_zamba2.py --variant 7b

Run low-rank experiments on the recurrent state:
    python state_spectrum_sweep/run_experiment_zamba2.py \
        --experiment low_rank \
        --subset ssm

Run fake-quantization experiments on the attention cache:
    python state_spectrum_sweep/run_experiment_zamba2.py \
        --experiment quant \
        --subset attn \
        --quant-bits 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from state_spectrum_sweep.run_experiment_mamba import (
    DTYPE,
    EXPERIMENTS_ROOT,
    LOW_RANK_RANK,
    MAX_NEW_TOKENS,
    PROMPT_SUITE,
    QUANT_BITS,
    create_backend,
    inspect_cache,
    run_single_prompt,
)


MODEL_VARIANTS = {
    "2.7b": "Zyphra/Zamba2-2.7B",
    "7b": "Zyphra/Zamba2-7B",
}
DEFAULT_VARIANT = "2.7b"


def _default_model_name(variant: str) -> str:
    try:
        return MODEL_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Zamba2 variant {variant!r}. "
            f"Choose from: {', '.join(sorted(MODEL_VARIANTS))}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Zamba2 cache perturbation experiments.")
    parser.add_argument(
        "--variant",
        choices=tuple(MODEL_VARIANTS),
        default=DEFAULT_VARIANT,
        help="Convenience preset for the Zamba2 checkpoint to load.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional full Hugging Face model id. Overrides --variant when provided.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer id. Defaults to the resolved model id.",
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--experiment", choices=("low_rank", "quant"), default="low_rank")
    parser.add_argument("--subset", choices=("auto", "all", "ssm", "conv", "attn"), default="auto")
    parser.add_argument(
        "--target-layer",
        type=int,
        default=-1,
        help="Only perturb cache tensors from this layer index. Default: all matched layers.",
    )
    parser.add_argument(
        "--generate-from",
        choices=("baseline", "perturbed"),
        default="baseline",
        help="Which branch to use for autoregressive generation after the first token.",
    )
    parser.add_argument(
        "--allow-loose-ssm-match",
        action="store_true",
        help="Use the broader state matcher instead of strict Zamba2/Jamba-style cache-field matching.",
    )
    parser.add_argument("--low-rank-rank", type=int, default=LOW_RANK_RANK)
    parser.add_argument("--quant-bits", type=int, default=QUANT_BITS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--inspect-cache", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def _resolved_model_and_tokenizer(args: argparse.Namespace) -> tuple[str, str]:
    model_name = args.model or _default_model_name(args.variant)
    tokenizer_name = args.tokenizer or model_name
    return model_name, tokenizer_name


def _run_name(args: argparse.Namespace, target_layer: int | None) -> str:
    layer_suffix = "all_layers" if target_layer is None else f"layer_{target_layer:02d}"
    source_suffix = f"gen_{args.generate_from}"
    model_label = args.variant.replace(".", "p")
    return (
        f"zamba2_{model_label}_{args.experiment}_{layer_suffix}_{source_suffix}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def main() -> None:
    args = build_parser().parse_args()
    strict_ssm = not args.allow_loose_ssm_match
    target_layer = None if args.target_layer < 0 else args.target_layer
    model_name, tokenizer_name = _resolved_model_and_tokenizer(args)

    backend = create_backend(
        backend_name="hf",
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        dtype=DTYPE,
    )

    if args.inspect_cache:
        inspect_prompt = args.prompt or PROMPT_SUITE[0]
        inspect_cache(
            inspect_prompt,
            backend,
            max_new_tokens=args.max_new_tokens,
            subset=args.subset,
            strict_ssm=strict_ssm,
            target_layer=target_layer,
        )
        return

    prompts = [args.prompt] if args.prompt else PROMPT_SUITE
    if args.prompt_limit > 0:
        prompts = prompts[: args.prompt_limit]

    run_name = _run_name(args, target_layer)
    run_dir = Path(args.output_dir) if args.output_dir else EXPERIMENTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "variant": args.variant,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "experiment": args.experiment,
        "subset": args.subset,
        "strict_ssm": strict_ssm,
        "target_layer": target_layer,
        "generate_from": args.generate_from,
        "low_rank_rank": args.low_rank_rank,
        "quant_bits": args.quant_bits,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": str(DTYPE),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Running {len(prompts)} prompts")
    print(f"Variant: {args.variant}")
    print(f"Model: {model_name}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Generate from: {args.generate_from}")
    print(f"Strict SSM match: {strict_ssm}")
    print(f"Target layer: {'all matched layers' if target_layer is None else target_layer}")
    print(f"Saving outputs to: {run_dir}")

    for idx, prompt in enumerate(prompts, start=1):
        prompt_id = f"prompt_{idx:02d}"
        print(f"\n[{idx:02d}/{len(prompts):02d}] {prompt_id}")
        try:
            kl, metrics, response, prompt_len, meta = run_single_prompt(
                prompt,
                backend,
                experiment=args.experiment,
                subset=args.subset,
                strict_ssm=strict_ssm,
                target_layer=target_layer,
                generate_from=args.generate_from,
                low_rank_rank=args.low_rank_rank,
                quant_bits=args.quant_bits,
                max_new_tokens=args.max_new_tokens,
            )
            torch.save(kl.cpu(), run_dir / f"{prompt_id}_kl.pt")
            for metric_name, tensor in metrics.items():
                torch.save(tensor.cpu(), run_dir / f"{prompt_id}_{metric_name}.pt")

            (run_dir / f"{prompt_id}_prompt_response.txt").write_text(
                f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n",
                encoding="utf-8",
            )
            (run_dir / f"{prompt_id}_meta.json").write_text(
                json.dumps(
                    {
                        **meta,
                        "prompt_id": prompt_id,
                        "prompt_len": prompt_len,
                        "backend": "hf",
                        "variant": args.variant,
                        "model": model_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"prompt_len: {prompt_len}")
            print(f"num_generated_tokens: {meta['num_generated_tokens']}")
            print(f"num_state_tensors: {meta['num_state_tensors']}")
            print(f"stopped_reason: {meta['stopped_reason']}")
            print(f"Response: {response}")
        except Exception as exc:
            print(f"Failed for {prompt_id}: {exc}")

    print("\nDone.")
    print(f"Experiment artifacts are in: {run_dir}")


if __name__ == "__main__":
    main()

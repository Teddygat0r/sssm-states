"""
Standalone Zamba2 cache perturbation experiments.

This entrypoint replaces the Hugging Face Zamba2 path with Zyphra's standalone
PyTorch implementation because the temporary HF implementation has been failing
to load the released checkpoints reliably in the environments we tested.

Prerequisites
-------------
Clone the source repos next to this workspace. The runner will add them to
`sys.path` directly, which helps on platforms where the pip wheels/sdists are
broken or incomplete:

    git clone https://github.com/Zyphra/Zamba2.git
    git clone https://github.com/state-spaces/mamba.git mamba-ssm
    git clone https://github.com/Dao-AILab/causal-conv1d.git

Compiled extensions are optional for a first pass. If they are unavailable, we
may still be able to run the slower fallback path, depending on what the
standalone implementation imports at runtime.

Examples
--------
Inspect cache fields after prompt prefill on the default 2.7B model:
    python state_spectrum_sweep/run_experiment_zamba2.py --inspect-cache

Switch to the 7B variant:
    python state_spectrum_sweep/run_experiment_zamba2.py --variant 7b --inspect-cache

Run low-rank experiments on recurrent state tensors:
    python state_spectrum_sweep/run_experiment_zamba2.py \
        --experiment low_rank \
        --subset ssm
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_spectrum_sweep.run_experiment_mamba import (
    BaseBackend,
    DTYPE,
    EXPERIMENTS_ROOT,
    LOW_RANK_RANK,
    SVD_NITER,
    MAX_NEW_TOKENS,
    PROMPT_SUITE,
    QUANT_BITS,
    _load_tokenizer,
    _resolve_device,
    _resolve_eos_token_id,
    _tokenizer_chat_or_plain,
    inspect_cache,
    run_single_prompt,
)


MODEL_VARIANTS = {
    "2.7b": "Zyphra/Zamba2-2.7B",
    "7b": "Zyphra/Zamba2-7B",
}
MODEL_WEIGHT_FILENAMES = {
    "2.7b": "Zamba2_2p7b_direct_from_pytorch.pt",
    "7b": "Zamba2_7b_direct_from_pytorch.pt",
}
MODEL_MAX_SEQUENCE_LENGTH = {
    "2.7b": 4096,
    "7b": 4096,
}
DEFAULT_VARIANT = "2.7b"
DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"


class _FallbackDotProductAttention(nn.Module):
    def __init__(
        self,
        *,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layer_number: int,
        attn_mask_type: str,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.dropout_p = attention_dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Zyphra attention tensors are shaped [seq, batch, heads, dim].
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        is_causal = self.attn_mask_type == "causal"
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.permute(2, 0, 1, 3).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)


def _ensure_transformer_engine_fallback() -> None:
    try:
        import transformer_engine  # noqa: F401
        return
    except Exception:
        pass

    te_module = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.DotProductAttention = _FallbackDotProductAttention
    te_module.pytorch = te_pytorch
    sys.modules["transformer_engine"] = te_module
    sys.modules["transformer_engine.pytorch"] = te_pytorch

def _ensure_triton_compat() -> None:
    try:
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:
        return

    math_module = getattr(tl, "math", None)
    if math_module is None:
        return

    if not hasattr(math_module, "log1p") and hasattr(libdevice, "log1p"):
        math_module.log1p = libdevice.log1p

def _patch_zamba2_source_compat(candidate_roots: list[Path]) -> None:
    old = "tl.math.log1p(tl.exp(dt))"
    new = "tl.log(1 + tl.exp(dt))"

    for root in candidate_roots:
        triton_dir = root / "ops" / "triton"
        if not triton_dir.exists():
            continue
        for path in triton_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if old not in text or new in text:
                continue
            path.write_text(text.replace(old, new), encoding="utf-8")

def _require_zamba2_standalone():
    repo_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        repo_root / "causal-conv1d",
        repo_root.parent / "causal-conv1d",
        repo_root / "causal_conv1d",
        repo_root.parent / "causal_conv1d",
        repo_root / "mamba-ssm",
        repo_root.parent / "mamba-ssm",
        repo_root / "mamba",
        repo_root.parent / "mamba",
        repo_root / "Zamba2",
        repo_root.parent / "Zamba2",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
    zamba_roots = [path for path in candidate_paths if path.name == "Zamba2" and path.exists()]
    _patch_zamba2_source_compat(zamba_roots)
    _ensure_transformer_engine_fallback()
    _ensure_triton_compat()
    try:
        from mamba_model import MambaModel
        from mamba_config import MambaConfig
    except ImportError as exc:
        raise RuntimeError(
            "Zamba2 standalone experiments require Zyphra's standalone source tree "
            "or package plus its Mamba dependencies. The runner first looks for "
            "local checkouts at `<repo>/Zamba2`, `<repo>/mamba-ssm` (or `<repo>/mamba`), "
            "and `<repo>/causal-conv1d`. If those are unavailable, install them in this "
            "environment, for example:\n"
            "  git clone https://github.com/Zyphra/Zamba2.git\n"
            "  git clone https://github.com/state-spaces/mamba.git mamba-ssm\n"
            "  git clone https://github.com/Dao-AILab/causal-conv1d.git\n"
            "  cd Zamba2\n"
            "  # optional: install dependencies if your platform supports them"
        ) from exc
    return MambaModel, MambaConfig


@dataclass
class ZambaInferenceParams:
    max_sequence_length: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[Any, Any] = field(default_factory=dict)
    key_value_memory_dict_mamba: dict[Any, Any] = field(default_factory=dict)
    key_value_memory_dict_attn: dict[Any, Any] = field(default_factory=dict)
    lengths_per_sample: Any = None

    @property
    def max_seqlen(self) -> int:
        return self.max_sequence_length

    @max_seqlen.setter
    def max_seqlen(self, value: int) -> None:
        self.max_sequence_length = int(value)

    @property
    def seqlen_offset(self) -> int:
        return self.sequence_len_offset

    @seqlen_offset.setter
    def seqlen_offset(self, value: int) -> None:
        self.sequence_len_offset = int(value)


class StandaloneZamba2Backend(BaseBackend):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        dtype: torch.dtype,
        *,
        variant: str,
    ) -> None:
        super().__init__(model_name, tokenizer_name, dtype)
        self.variant = variant

    def load(self) -> None:
        MambaModel, MambaConfig = _require_zamba2_standalone()
        resolved_tokenizer = self.tokenizer_name or DEFAULT_TOKENIZER
        self.tokenizer = _load_tokenizer(resolved_tokenizer)
        self.model = self._load_model_from_checkpoint(MambaModel, MambaConfig)
        self.device = _resolve_device()
        if self.device == "cuda":
            self.model = self.model.cuda()
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.model = self.model.to(device=self.device, dtype=torch.float32)
            self.dtype = torch.float32
        self.model.eval()

    def _load_model_from_checkpoint(self, MambaModel, MambaConfig):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Standalone Zamba2 loading requires `huggingface_hub` to download the "
                "checkpoint weights."
            ) from exc

        repo_root = Path(__file__).resolve().parents[1]
        local_config_path = repo_root / "Zamba2" / "config.json"
        if not local_config_path.exists():
            local_config_path = repo_root.parent / "Zamba2" / "config.json"
        if not local_config_path.exists():
            raise RuntimeError(
                "Could not find the standalone Zamba2 `config.json` in the local "
                "checkout. Expected `<repo>/Zamba2/config.json`."
            )

        with local_config_path.open("r", encoding="utf-8") as handle:
            raw_config_data = json.load(handle)
        config_data = self._adapt_config_kwargs(raw_config_data, MambaConfig)
        config = MambaConfig(**config_data)
        model = self._construct_model(MambaModel, config, raw_config_data)

        weight_filename = MODEL_WEIGHT_FILENAMES.get(self.variant)
        if not weight_filename:
            raise RuntimeError(
                f"No standalone checkpoint filename mapping is configured for variant {self.variant!r}."
            )
        weight_path = hf_hub_download(repo_id=self.model_name, filename=weight_filename)
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            missing_preview = ", ".join(missing[:6]) if missing else "none"
            unexpected_preview = ", ".join(unexpected[:6]) if unexpected else "none"
            raise RuntimeError(
                "Standalone Zamba2 checkpoint did not match the constructed model. "
                f"Missing keys: {missing_preview}. Unexpected keys: {unexpected_preview}."
            )
        return model
    
    def _adapt_config_kwargs(self, raw_config: dict[str, Any], MambaConfig) -> dict[str, Any]:
        config_data = dict(raw_config)
        signature = inspect.signature(MambaConfig)
        valid_keys = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        # Common alias used by the HF-side config but not always by the standalone class.
        if "padded_vocab_size" in config_data and "vocab_size" in valid_keys and "vocab_size" not in config_data:
            config_data["vocab_size"] = config_data["padded_vocab_size"]

        # Preserve only constructor-supported fields so small schema drifts do not break loading.
        return {key: value for key, value in config_data.items() if key in valid_keys}

    def _construct_model(self, MambaModel, config, config_data: dict[str, Any]):
        signature = inspect.signature(MambaModel)
        kwargs: dict[str, Any] = {}
        if "max_sequence_length" in signature.parameters:
            max_sequence_length = (
                config_data.get("max_sequence_length")
                or config_data.get("max_position_embeddings")
                or getattr(config, "max_position_embeddings", None)
                or MODEL_MAX_SEQUENCE_LENGTH.get(self.variant)
            )
            if max_sequence_length is None:
                raise RuntimeError(
                    "Could not determine `max_sequence_length` for the standalone "
                    "Zamba2 model constructor."
                )
            kwargs["max_sequence_length"] = int(max_sequence_length)
        return MambaModel(config, **kwargs)

    def prepare_prompt(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer/model not loaded.")
        text = _tokenizer_chat_or_plain(self.tokenizer, prompt)
        encoded = self.tokenizer([text], return_tensors="pt")
        input_ids = encoded["input_ids"].transpose(0, 1).contiguous()
        return input_ids.to(self.device)

    def eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            return None
        return _resolve_eos_token_id(self.tokenizer, self.model)

    def _new_cache(self, input_ids: torch.Tensor, max_new_tokens: int) -> ZambaInferenceParams:
        return ZambaInferenceParams(
            max_sequence_length=int(input_ids.shape[0] + max_new_tokens),
            max_batch_size=int(input_ids.shape[1]),
        )

    def prefill(self, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, Any]:
        cache_obj = self._new_cache(input_ids, max_new_tokens=max_new_tokens)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, inference_params=cache_obj)
        cache_obj.sequence_len_offset = int(input_ids.shape[0])
        return logits, cache_obj

    def step(self, input_ids: torch.Tensor, cache_obj: Any) -> tuple[torch.Tensor, Any]:
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, inference_params=cache_obj)
        cache_obj.sequence_len_offset += int(input_ids.shape[0])
        return logits, cache_obj


def _default_model_name(variant: str) -> str:
    try:
        return MODEL_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Zamba2 variant {variant!r}. "
            f"Choose from: {', '.join(sorted(MODEL_VARIANTS))}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Zamba2 cache perturbation experiments using Zyphra's standalone implementation."
    )
    parser.add_argument(
        "--variant",
        choices=tuple(MODEL_VARIANTS),
        default=DEFAULT_VARIANT,
        help="Convenience preset for the Zamba2 checkpoint to load.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional full model id/path. Overrides --variant when provided.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer id. Defaults to Mistral v0.1, per the Zamba2 model card.",
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
        help="Use the broader state matcher instead of strict cache-field matching.",
    )
    parser.add_argument("--low-rank-rank", type=int, default=LOW_RANK_RANK)
    parser.add_argument("--svd-niter", type=int, default=SVD_NITER)
    parser.add_argument("--quant-bits", type=int, default=QUANT_BITS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--inspect-cache", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def _resolved_model_and_tokenizer(args: argparse.Namespace) -> tuple[str, str]:
    model_name = args.model or _default_model_name(args.variant)
    tokenizer_name = args.tokenizer or DEFAULT_TOKENIZER
    return model_name, tokenizer_name


def _run_name(args: argparse.Namespace, target_layer: int | None) -> str:
    layer_suffix = "all_layers" if target_layer is None else f"layer_{target_layer:02d}"
    source_suffix = f"gen_{args.generate_from}"
    model_label = args.variant.replace(".", "p")
    return (
        f"zamba2_standalone_{model_label}_{args.experiment}_{layer_suffix}_{source_suffix}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def main() -> None:
    args = build_parser().parse_args()
    strict_ssm = not args.allow_loose_ssm_match
    target_layer = None if args.target_layer < 0 else args.target_layer
    model_name, tokenizer_name = _resolved_model_and_tokenizer(args)

    backend = StandaloneZamba2Backend(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        dtype=DTYPE,
        variant=args.variant,
    )
    backend.load()

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
        "backend": "zamba2_standalone",
        "experiment": args.experiment,
        "subset": args.subset,
        "strict_ssm": strict_ssm,
        "target_layer": target_layer,
        "generate_from": args.generate_from,
        "low_rank_rank": args.low_rank_rank,
        "svd_niter": args.svd_niter,
        "quant_bits": args.quant_bits,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": str(backend.dtype),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"Running {len(prompts)} prompts")
    print("Backend: Zyphra standalone Zamba2")
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
                svd_niter=args.svd_niter,
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
                        "backend": "zamba2_standalone",
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

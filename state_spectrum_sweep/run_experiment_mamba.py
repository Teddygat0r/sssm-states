"""
Starter experiment for Mamba-family models.

This script mirrors the Qwen recurrent-state experiments in this repo, but
targets models whose decode-time memory is exposed via either:

1. Hugging Face `cache_params` / `past_key_values` style caches
2. Official `mamba_ssm` `inference_params` objects that are mutated in-place

Supported use cases:
- Hugging Face Mamba / Mamba-2 style causal LM checkpoints
- Official `mamba_ssm` Mamba / Mamba-2 checkpoints via `MambaLMHeadModel`
- Mamba-3 style models if you can provide a model object whose recurrent
  decode state lives in an externally mutable cache object

The main workflow is:
- run a prompt and capture the decode cache
- extract persistent state tensors from that cache
- perturb them with low-rank SVD or fake quantization
- compare original logits vs perturbed logits with KL / MSE / cosine metrics

Examples
--------
Inspect cache fields after prompt prefill:
    python state_spectrum_sweep/run_experiment_mamba.py \
        --backend hf \
        --model state-spaces/mamba-130m \
        --inspect-cache

Run low-rank experiments on Hugging Face Mamba:
    python state_spectrum_sweep/run_experiment_mamba.py \
        --backend hf \
        --model state-spaces/mamba-130m \
        --experiment low_rank

Run quantization experiments on official `mamba_ssm`:
    python state_spectrum_sweep/run_experiment_mamba.py \
        --backend mamba_ssm \
        --model state-spaces/mamba2-130m \
        --tokenizer EleutherAI/gpt-neox-20b \
        --experiment quant \
        --quant-bits 8
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
import torch.nn.functional as F


EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
DEFAULT_MODEL = "state-spaces/mamba-130m"
DEFAULT_TOKENIZER = "EleutherAI/gpt-neox-20b"
HF_MODEL_REWRITES = {
    "state-spaces/mamba-130m": "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m": "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m": "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b": "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b": "state-spaces/mamba-2.8b-hf",
}
LOW_RANK_RANK = 16
QUANT_BITS = 8
MAX_NEW_TOKENS = 64
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

PROMPT_SUITE = [
    "Hi there! Who are you?",
    "Explain photosynthesis in one paragraph.",
    "Write three creative names for a coffee shop.",
    "What is the capital of Japan and one famous landmark there?",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Translate this to Spanish: I enjoy learning new things every day.",
    "Give me a short bedtime story about a robot and a cat.",
    "List five practical ways to reduce household energy usage.",
    "What are the differences between lists and tuples in Python?",
    "Write a simple Python function to check if a number is prime.",
]


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `transformers` to load tokenizer/model objects. "
            "Install the repo dependencies first."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer

def _load_tokenizer(tokenizer_name: str):
    _, AutoTokenizer = _require_transformers()
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except (ImportError, ValueError) as exc:
        message = str(exc)
        fallback_markers = (
            "backend tokenizer",
            "sentencepiece",
            "tiktoken",
            "protobuf",
        )
        if not any(marker in message.lower() for marker in fallback_markers):
            raise
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

def _require_mamba_ssm():
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            "The `mamba_ssm` backend requires the official `mamba-ssm` package."
        ) from exc
    return MambaLMHeadModel
  
def _default_tokenizer_for_model(model_name: str) -> str:
    lowered = model_name.lower()
    # if "state-spaces/mamba" in lowered:
    #     return DEFAULT_TOKENIZER
    return model_name

def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _resolve_model_name_for_backend(backend_name: str, model_name: str) -> str:
    if backend_name != "hf":
        return model_name
    return HF_MODEL_REWRITES.get(model_name, model_name)

def _resolve_eos_token_id(tokenizer, model) -> int | None:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos, list):
        return eos[0] if eos else None
    return eos


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if logits.dim() == 3:
        logits = logits[:, -1, :]

    scaled_logits = logits / temperature
    vocab_size = scaled_logits.shape[-1]
    k = min(top_k, vocab_size)

    topk_vals, topk_idx = torch.topk(scaled_logits, k=k, dim=-1)
    topk_probs = torch.softmax(topk_vals, dim=-1)
    cumulative_probs = torch.cumsum(topk_probs, dim=-1)

    remove_mask = cumulative_probs > top_p
    remove_mask[..., 0] = False
    filtered_probs = topk_probs.masked_fill(remove_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )

    sampled_in_topk = torch.multinomial(filtered_probs, num_samples=1)
    next_token = torch.gather(topk_idx, dim=-1, index=sampled_in_topk)
    return next_token


def low_rank_svd(tensor: torch.Tensor, n: int = 16, oversample: int = 4, niter: int = 1):
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    work_tensor = tensor.float()
    q = min(n + oversample, min(work_tensor.shape[-2:]))
    u, s, v = torch.svd_lowrank(work_tensor, q=q, niter=niter)
    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    approx = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
    return approx.to(dtype=tensor.dtype)


def fake_quantize(tensor: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    work_tensor = tensor.float()
    qmin = 0
    qmax = (1 << n_bits) - 1
    t_min = work_tensor.min()
    t_max = work_tensor.max()
    scale = (t_max - t_min) / qmax
    scale = scale.clamp_min(1e-12)
    zero_point = torch.round(-t_min / scale).clamp(qmin, qmax)
    quantized = torch.round(work_tensor / scale + zero_point).clamp(qmin, qmax)
    return ((quantized - zero_point) * scale).to(dtype=tensor.dtype)


@dataclass(frozen=True)
class TensorPath:
    path: tuple[Any, ...]
    name: str


@dataclass
class CacheTensorView:
    path: TensorPath
    tensor: torch.Tensor


def _iter_object_members(obj: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(obj, dict):
        yield from obj.items()
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield idx, value
        return

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            yield key, value


def _walk_tensor_leaves(
    obj: Any,
    *,
    prefix: tuple[Any, ...] = (),
    max_depth: int = 6,
) -> list[CacheTensorView]:
    if max_depth < 0:
        return []

    if torch.is_tensor(obj):
        name = ".".join(str(part) for part in prefix) if prefix else "<root>"
        return [CacheTensorView(path=TensorPath(prefix, name), tensor=obj)]

    views: list[CacheTensorView] = []
    for key, value in _iter_object_members(obj):
        views.extend(_walk_tensor_leaves(value, prefix=prefix + (key,), max_depth=max_depth - 1))
    return views


def _get_by_path(obj: Any, path: tuple[Any, ...]) -> Any:
    current = obj
    for key in path:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, (list, tuple)):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def _set_by_path(obj: Any, path: tuple[Any, ...], value: Any) -> None:
    if not path:
        raise ValueError("Cannot assign to an empty path.")

    parent = _get_by_path(obj, path[:-1]) if len(path) > 1 else obj
    key = path[-1]

    if isinstance(parent, dict):
        parent[key] = value
    elif isinstance(parent, list):
        parent[key] = value
    elif isinstance(parent, tuple):
        raise TypeError(f"Cannot mutate tuple-backed cache path: {path}")
    else:
        setattr(parent, key, value)


def _matches_subset(name: str, subset: str) -> bool:
    lowered = name.lower()
    if subset == "all":
        return True
    if subset == "ssm":
        return "ssm" in lowered or "state" in lowered
    if subset == "conv":
        return "conv" in lowered
    if subset == "auto":
        return any(token in lowered for token in ("ssm", "conv", "recurrent", "state"))
    raise ValueError(f"Unsupported subset: {subset}")


def extract_cache_tensors(cache_obj: Any, subset: str) -> list[CacheTensorView]:
    views = _walk_tensor_leaves(cache_obj)
    filtered = [view for view in views if _matches_subset(view.path.name, subset)]
    if filtered:
        return filtered
    if subset == "auto":
        return views
    return filtered

def _clone_cache_obj(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().clone()

    if isinstance(obj, dict):
        return {key: _clone_cache_obj(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_clone_cache_obj(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(_clone_cache_obj(value) for value in obj)

    if hasattr(obj, "__dict__"):
        cloned = copy.copy(obj)
        for key, value in vars(obj).items():
            setattr(cloned, key, _clone_cache_obj(value))
        return cloned

    return copy.copy(obj)

def clone_cache(cache_obj: Any) -> Any:
    return _clone_cache_obj(cache_obj)


def _tokenizer_chat_or_plain(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


class BaseBackend:
    def __init__(self, model_name: str, tokenizer_name: str, dtype: torch.dtype) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.dtype = dtype
        self.device = _resolve_device()
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        raise NotImplementedError

    def prepare_prompt(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        text = _tokenizer_chat_or_plain(self.tokenizer, prompt)
        encoded = self.tokenizer([text], return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def prefill(self, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def step(self, input_ids: torch.Tensor, cache_obj: Any) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def eos_token_id(self) -> int | None:
        if self.tokenizer is None or self.model is None:
            return None
        return _resolve_eos_token_id(self.tokenizer, self.model)


class HuggingFaceBackend(BaseBackend):
    def load(self) -> None:
        AutoModelForCausalLM, _ = _require_transformers()
        self.tokenizer = _load_tokenizer(self.tokenizer_name or self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)

    def _extract_cache(self, output: Any) -> Any:
        cache_obj = getattr(output, "cache_params", None)
        if cache_obj is None:
            cache_obj = getattr(output, "past_key_values", None)
        if cache_obj is None:
            raise ValueError("Model output exposed neither `cache_params` nor `past_key_values`.")
        return cache_obj

    def prefill(self, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, Any]:
        output = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        return output.logits, self._extract_cache(output)

    def step(self, input_ids: torch.Tensor, cache_obj: Any) -> tuple[torch.Tensor, Any]:
        output = self.model(
            input_ids=input_ids,
            cache_params=cache_obj,
            use_cache=True,
            return_dict=True,
        )
        return output.logits, self._extract_cache(output)


class NativeMambaBackend(BaseBackend):
    def load(self) -> None:
        MambaLMHeadModel = _require_mamba_ssm()
        self.tokenizer = _load_tokenizer(self.tokenizer_name or DEFAULT_TOKENIZER)
        self.model = MambaLMHeadModel.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )

    def _allocate_cache(self, input_ids: torch.Tensor, max_new_tokens: int) -> Any:
        total_len = int(input_ids.shape[1] + max_new_tokens)
        return self.model.allocate_inference_cache(
            batch_size=int(input_ids.shape[0]),
            max_seqlen=total_len,
            dtype=self.dtype,
        )

    def prefill(self, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, Any]:
        inference_params = self._allocate_cache(input_ids, max_new_tokens=max_new_tokens)
        output = self.model(input_ids=input_ids, inference_params=inference_params)
        logits = getattr(output, "logits", output)
        return logits, inference_params

    def step(self, input_ids: torch.Tensor, cache_obj: Any) -> tuple[torch.Tensor, Any]:
        output = self.model(
            input_ids=input_ids,
            inference_params=cache_obj,
            num_last_tokens=1,
        )
        logits = getattr(output, "logits", output)
        return logits, cache_obj


def create_backend(backend_name: str, model_name: str, tokenizer_name: str, dtype: torch.dtype):
    resolved_model_name = _resolve_model_name_for_backend(backend_name, model_name)
    if backend_name == "hf":
        backend = HuggingFaceBackend(
            resolved_model_name,
            tokenizer_name or _default_tokenizer_for_model(resolved_model_name),
            dtype,
        )
    elif backend_name == "mamba_ssm":
        backend = NativeMambaBackend(resolved_model_name, tokenizer_name or DEFAULT_TOKENIZER, dtype)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")
    backend.load()
    return backend


def _metric_dict_from_rows(rows: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, values in rows.items():
        if values:
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = torch.empty(0, 0, dtype=torch.float32)
    return out


def _make_perturbed_cache(
    cache_obj: Any,
    base_tensors: dict[tuple[Any, ...], torch.Tensor],
    current_views: list[CacheTensorView],
    *,
    experiment: str,
    low_rank_rank: int,
    quant_bits: int,
) -> tuple[Any, dict[str, torch.Tensor]]:
    approx_cache = clone_cache(cache_obj)
    mse_rows: list[torch.Tensor] = []
    rel_frob_rows: list[torch.Tensor] = []
    cosine_rows: list[torch.Tensor] = []
    retained_energy_rows: list[torch.Tensor] = []

    for view in current_views:
        current = view.tensor
        base = base_tensors[view.path.path]
        if experiment == "low_rank":
            if current.dim() < 2:
                approx = current.detach().clone()
            else:
                delta = current - base
                approx = base + low_rank_svd(delta, n=low_rank_rank)
        elif experiment == "quant":
            approx = fake_quantize(current, n_bits=quant_bits)
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        approx = approx.to(device=current.device, dtype=current.dtype)
        _set_by_path(approx_cache, view.path.path, approx)

        err = current - approx
        current_norm = current.norm().clamp_min(1e-12)
        mse_rows.append(F.mse_loss(current, approx).view(1).detach().float().cpu())
        rel_frob_rows.append((err.norm() / current_norm).view(1).detach().float().cpu())
        cosine_rows.append(
            F.cosine_similarity(current.flatten(), approx.flatten(), dim=0)
            .view(1)
            .detach()
            .float()
            .cpu()
        )
        retained_energy_rows.append(
            (
                1.0
                - (err.pow(2).sum() / current.pow(2).sum().clamp_min(1e-12))
            )
            .view(1)
            .detach()
            .float()
            .cpu()
        )

    metrics = {
        "mse": torch.cat(mse_rows, dim=0) if mse_rows else torch.empty(0, dtype=torch.float32),
        "rel_frob": (
            torch.cat(rel_frob_rows, dim=0)
            if rel_frob_rows
            else torch.empty(0, dtype=torch.float32)
        ),
        "cosine_similarity": (
            torch.cat(cosine_rows, dim=0) if cosine_rows else torch.empty(0, dtype=torch.float32)
        ),
        "retained_energy": (
            torch.cat(retained_energy_rows, dim=0)
            if retained_energy_rows
            else torch.empty(0, dtype=torch.float32)
        ),
    }
    return approx_cache, metrics


def run_single_prompt(
    prompt: str,
    backend: BaseBackend,
    *,
    experiment: str,
    subset: str,
    low_rank_rank: int,
    quant_bits: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, int, dict]:
    input_ids = backend.prepare_prompt(prompt)
    prompt_len = int(input_ids.shape[1])
    eos_token_id = backend.eos_token_id()

    prefill_start = perf_counter()
    logits, cache_obj = backend.prefill(input_ids, max_new_tokens=max_new_tokens)
    prefill_elapsed = perf_counter() - prefill_start

    current_views = extract_cache_tensors(cache_obj, subset=subset)
    if not current_views:
        raise ValueError(
            "No cache tensors matched the requested subset. Try `--inspect-cache` or `--subset all`."
        )

    base_tensors = {
        view.path.path: view.tensor.detach().clone()
        for view in current_views
    }

    next_token = _sample_next_token(
        logits,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        top_k=SAMPLING_TOP_K,
    )

    if eos_token_id is not None and next_token.item() == eos_token_id:
        empty_kl = torch.tensor([], dtype=DTYPE, device="cpu")
        empty_metrics = {
            "mse": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
            "rel_frob": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
            "cosine_similarity": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
            "retained_energy": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
        }
        meta = {
            "prompt_len": prompt_len,
            "num_generated_tokens": 0,
            "num_metric_steps": 0,
            "stopped_reason": "eos_first_token",
            "prefill_seconds": prefill_elapsed,
            "num_state_tensors": len(current_views),
        }
        return empty_kl, empty_metrics, "", prompt_len, meta

    kl_list: list[float] = []
    metric_rows: dict[str, list[torch.Tensor]] = {
        "mse": [],
        "rel_frob": [],
        "cosine_similarity": [],
        "retained_energy": [],
    }
    generated_ids: list[int] = []
    step = 0
    stopped_reason = "max_new_tokens"
    average_transform_seconds = 0.0

    with torch.inference_mode():
        while step < max_new_tokens:
            token_in = next_token
            if eos_token_id is not None and token_in.item() == eos_token_id:
                stopped_reason = "eos"
                break

            generated_ids.append(int(token_in.item()))
            current_views = extract_cache_tensors(cache_obj, subset=subset)
            transform_start = perf_counter()
            approx_cache, step_metrics = _make_perturbed_cache(
                cache_obj,
                base_tensors,
                current_views,
                experiment=experiment,
                low_rank_rank=low_rank_rank,
                quant_bits=quant_bits,
            )
            transform_elapsed = perf_counter() - transform_start
            average_transform_seconds += transform_elapsed
            print(f"    token {step + 1} state transform: {transform_elapsed:.3f}s")

            forward_start = perf_counter()
            out_logits, updated_cache = backend.step(token_in, cache_obj)
            approx_logits, _ = backend.step(token_in, approx_cache)
            forward_elapsed = perf_counter() - forward_start
            print(f"    token {step + 1} model forward(s): {forward_elapsed:.3f}s")

            out_log_logits = torch.log_softmax(out_logits, dim=-1)
            approx_log_logits = torch.log_softmax(approx_logits, dim=-1)
            kl = F.kl_div(
                approx_log_logits,
                out_log_logits,
                reduction="batchmean",
                log_target=True,
            ).item()
            kl_list.append(kl)

            for key in metric_rows:
                metric_rows[key].append(step_metrics[key].detach().float().cpu())

            next_token = _sample_next_token(
                out_logits,
                temperature=SAMPLING_TEMPERATURE,
                top_p=SAMPLING_TOP_P,
                top_k=SAMPLING_TOP_K,
            )
            cache_obj = updated_cache
            step += 1

    response = ""
    if generated_ids and backend.tokenizer is not None:
        response = backend.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    metrics = _metric_dict_from_rows(metric_rows)
    meta = {
        "prompt_len": prompt_len,
        "num_generated_tokens": len(generated_ids),
        "num_metric_steps": len(kl_list),
        "stopped_reason": stopped_reason,
        "prefill_seconds": prefill_elapsed,
        "average_transform_seconds": average_transform_seconds / max(len(kl_list), 1),
        "num_state_tensors": len(base_tensors),
        "state_tensor_names": [".".join(str(p) for p in path) for path in base_tensors],
        "experiment": experiment,
        "subset": subset,
    }
    return torch.tensor(kl_list, dtype=torch.float32), metrics, response, prompt_len, meta


def inspect_cache(prompt: str, backend: BaseBackend, max_new_tokens: int, subset: str) -> None:
    input_ids = backend.prepare_prompt(prompt)
    _, cache_obj = backend.prefill(input_ids, max_new_tokens=max_new_tokens)
    all_views = extract_cache_tensors(cache_obj, subset="all")
    filtered_views = extract_cache_tensors(cache_obj, subset=subset)
    print(f"All tensor leaves found: {len(all_views)}")
    for view in all_views:
        print(f"  {view.path.name}: shape={tuple(view.tensor.shape)} dtype={view.tensor.dtype}")
    print()
    print(f"Subset `{subset}` matched: {len(filtered_views)}")
    for view in filtered_views:
        print(f"  {view.path.name}: shape={tuple(view.tensor.shape)} dtype={view.tensor.dtype}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mamba cache perturbation experiments.")
    parser.add_argument("--backend", choices=("hf", "mamba_ssm"), default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--experiment", choices=("low_rank", "quant"), default="low_rank")
    parser.add_argument("--subset", choices=("auto", "all", "ssm", "conv"), default="auto")
    parser.add_argument("--low-rank-rank", type=int, default=LOW_RANK_RANK)
    parser.add_argument("--quant-bits", type=int, default=QUANT_BITS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--inspect-cache", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = create_backend(
        backend_name=args.backend,
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        dtype=DTYPE,
    )

    if args.inspect_cache:
        inspect_prompt = args.prompt or PROMPT_SUITE[0]
        inspect_cache(
            inspect_prompt,
            backend,
            max_new_tokens=args.max_new_tokens,
            subset=args.subset,
        )
        return

    prompts = [args.prompt] if args.prompt else PROMPT_SUITE
    if args.prompt_limit > 0:
        prompts = prompts[: args.prompt_limit]

    run_name = f"{args.backend}_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) if args.output_dir else EXPERIMENTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Running {len(prompts)} prompts")
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

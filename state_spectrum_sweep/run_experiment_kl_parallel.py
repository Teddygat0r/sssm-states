from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


MODEL_NAME = "Qwen/Qwen3.5-4B"
LOW_RANK_RANK = int(os.getenv("LOW_RANK_RANK", "16"))
MAX_NEW_TOKENS = 200
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20
NUM_WORKERS = max(1, int(os.getenv("KL_PARALLEL_WORKERS", "2")))
PROMPT_LIMIT = int(os.getenv("KL_PROMPT_LIMIT", "0"))
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DATASET = os.getenv("DATASET", "suite")
MMLU_DATASET_NAME = "cais/mmlu"
MMLU_DATASET_CONFIG = "all"
MMLU_SPLITS = ("auxiliary_train", "dev", "validation", "test")

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
    "Write a haiku about rain in a city.",
    "Create a two-day itinerary for visiting New York City.",
    "Explain Newton's second law with a simple example.",
    "Suggest a healthy breakfast under 400 calories.",
    "Draft a polite email asking for a project deadline extension.",
    "What are the main ideas behind gradient descent?",
    "Give me three interview questions for a junior data scientist role.",
    "Describe the plot of Romeo and Juliet in four sentences.",
    "Write a short dialogue between a teacher and a curious student.",
    "Provide a regex for validating a basic email format.",
    "What is overfitting in machine learning, and how can we reduce it?",
    "Generate a list of ten random words and use each in a sentence.",
    "Explain recursion to a 10-year-old.",
    "Compare REST and GraphQL in a concise table-style format.",
    "Write a simple Python function to check if a number is prime.",
    "What are three ethical concerns with large language models?",
    "Give me a 7-day beginner workout plan with light equipment.",
    "Describe how rainbows form using simple physics.",
    "Write a motivational message for someone learning to code.",
    "Explain the difference between precision and recall.",
    "Create a short sci-fi scene set on a lunar research station.",
]


def _clone_recurrent_state(recurrent_state):
    if recurrent_state is None:
        return None
    if isinstance(recurrent_state, torch.Tensor):
        return recurrent_state.detach().clone()
    if isinstance(recurrent_state, (list, tuple)):
        return [state.detach().clone() if state is not None else None for state in recurrent_state]
    raise TypeError(f"Unsupported recurrent_state type: {type(recurrent_state)}")


def _extract_layer_index_from_name(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def _resolve_recurrent_update_call(args, kwargs):
    layer_idx = kwargs.get("layer_idx")
    recurrent_state = None
    for key in ("recurrent_state", "recurrent_states", "state", "new_state", "new_recurrent_state"):
        if key in kwargs and kwargs[key] is not None:
            recurrent_state = kwargs[key]
            break
    for arg in args:
        if layer_idx is None and isinstance(arg, int):
            layer_idx = arg
        if recurrent_state is None and isinstance(arg, (torch.Tensor, list, tuple)):
            recurrent_state = arg
    return layer_idx, recurrent_state


def _capture_recurrent_state_updates(cache_cls_or_instance, num_layers: int, fn):
    cache_cls = cache_cls_or_instance if isinstance(cache_cls_or_instance, type) else type(cache_cls_or_instance)
    update_recurrent_state = getattr(cache_cls, "update_recurrent_state", None)
    if update_recurrent_state is None:
        raise ValueError("Cache object has no update_recurrent_state method; cannot capture recurrent states.")

    captured = [None] * num_layers

    def wrapped_update_recurrent_state(self, *args, **kwargs):
        result = update_recurrent_state(self, *args, **kwargs)
        layer_idx, recurrent_state = _resolve_recurrent_update_call(args, kwargs)
        if layer_idx is not None and recurrent_state is not None and 0 <= layer_idx < num_layers:
            captured[layer_idx] = _clone_recurrent_state(recurrent_state)
        return result

    cache_cls.update_recurrent_state = wrapped_update_recurrent_state
    try:
        output = fn()
    finally:
        cache_cls.update_recurrent_state = update_recurrent_state

    return output, captured


def _set_recurrent_state(cache, layer_idx: int, recurrent_state):
    update_recurrent_state = getattr(cache, "update_recurrent_state", None)
    if update_recurrent_state is None:
        raise ValueError("Cache object has no update_recurrent_state method; cannot inject recurrent states.")

    attempts = (
        ((), {"layer_idx": layer_idx, "recurrent_state": recurrent_state}),
        ((), {"layer_idx": layer_idx, "new_recurrent_state": recurrent_state}),
        ((), {"layer_idx": layer_idx, "state": recurrent_state}),
        ((layer_idx, recurrent_state), {}),
        ((recurrent_state, layer_idx), {}),
    )
    last_error = None
    for args, kwargs in attempts:
        try:
            update_recurrent_state(*args, **kwargs)
            return
        except TypeError as exc:
            last_error = exc
    raise TypeError(f"Could not call update_recurrent_state for layer {layer_idx}.") from last_error


def _inject_recurrent_states(cache, recurrent_states):
    for layer_idx, recurrent_state in enumerate(recurrent_states):
        if recurrent_state is not None:
            _set_recurrent_state(cache, layer_idx, recurrent_state)


def generate_delta(state, state_ref):
    if isinstance(state, torch.Tensor):
        return state - state_ref
    if isinstance(state, list):
        return [state[i] - state_ref[i] for i in range(len(state)) if state[i] is not None and state_ref[i] is not None]
    raise TypeError(f"Unsupported state type for delta: {type(state)}")


def low_rank_svd_cpu(tensor: torch.Tensor, n: int = 16, oversample: int = 4, niter: int = 1):
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    orig_device = tensor.device
    orig_dtype = tensor.dtype
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    q = min(n + oversample, min(cpu_tensor.shape[-2:]))
    u, s, v = torch.svd_lowrank(cpu_tensor, q=q, niter=niter)
    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    approx_cpu = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
    return approx_cpu.to(device=orig_device, dtype=orig_dtype)


def low_rank_svd_list_cpu(lst: list, n: int = 16) -> list:
    batched_svd = torch.stack(lst, dim=0)
    batched_svd = low_rank_svd_cpu(batched_svd, n=n)
    return list(batched_svd.unbind(dim=0))


def _format_mmlu_prompt(example):
    choices = "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(example["choices"]))
    return (
        f"Subject: {example['subject']}\n\n"
        f"Question: {example['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with the single best option."
    )


def load_mmlu_prompts():
    from datasets import load_dataset
    
    dataset = load_dataset(MMLU_DATASET_NAME, MMLU_DATASET_CONFIG)
    prompts = []
    for split in MMLU_SPLITS:
        if split not in dataset:
            continue
        for idx, example in enumerate(dataset[split]):
            prompts.append({"id": f"{split}_{idx:05d}", "prompt": _format_mmlu_prompt(example), "split": split})
    if not prompts:
        raise RuntimeError("No prompts found in MMLU dataset.")
    return prompts


def _resolve_eos_token_id(tokenizer, model) -> int | None:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, list):
        return eos[0] if eos else None
    return eos


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
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
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled_in_topk = torch.multinomial(filtered_probs, num_samples=1)
    return torch.gather(topk_idx, dim=-1, index=sampled_in_topk)


def _forward_with_state_capture(model, cache_cls_or_instance, num_layers: int, **forward_kwargs):
    return _capture_recurrent_state_updates(
        cache_cls_or_instance,
        num_layers,
        lambda: model(**forward_kwargs, use_cache=True, return_dict=True),
    )


def run_single_prompt(prompt: str, model, tokenizer, *, low_rank_n: int = LOW_RANK_RANK, max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, int, dict]:
    model_device = next(model.parameters()).device
    gated_delta_layer_indices = sorted(
        idx
        for idx in (_extract_layer_index_from_name(name) for name, module in model.named_modules() if isinstance(module, Qwen3_5GatedDeltaNet))
        if idx is not None
    )
    if not gated_delta_layer_indices:
        raise RuntimeError("No Qwen3_5GatedDeltaNet layers found in the model.")
    num_layers = max(gated_delta_layer_indices) + 1

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
    prompt_len = int(model_inputs.input_ids.shape[1])
    eos_token_id = _resolve_eos_token_id(tokenizer, model)

    kl_list: list[float] = []
    mse_rows: list[torch.Tensor] = []
    rel_frob_rows: list[torch.Tensor] = []
    cosine_rows: list[torch.Tensor] = []
    retained_energy_rows: list[torch.Tensor] = []
    prompt_start_time = perf_counter()
    average_svd_time = 0.0

    with torch.inference_mode():
        out = model(**model_inputs, use_cache=True, return_dict=True)
        past_key_values = out.past_key_values
        if past_key_values is None:
            raise ValueError("Model returned no past_key_values; cannot compute recurrent deltas.")
        cache_cls = type(past_key_values)

        out, original_state = _forward_with_state_capture(model, cache_cls, num_layers, **model_inputs)
        past_key_values = out.past_key_values
        if past_key_values is None:
            raise ValueError("Model returned no past_key_values on captured prompt forward.")
        if not any(state is not None for state in original_state):
            raise ValueError("Could not capture recurrent states from DynamicCache.update_recurrent_state.")

        current_state = [_clone_recurrent_state(state) for state in original_state]
        next_token = _sample_next_token(out.logits, temperature=SAMPLING_TEMPERATURE, top_p=SAMPLING_TOP_P, top_k=SAMPLING_TOP_K)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            empty_kl = torch.tensor([], dtype=DTYPE, device="cpu")
            empty_metrics = {
                "mse": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
                "rel_frob": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
                "cosine_similarity": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
                "retained_energy": torch.empty(0, 0, dtype=DTYPE, device="cpu"),
            }
            meta = {"prompt_len": prompt_len, "num_generated_tokens": 0, "num_metric_steps": 0, "stopped_reason": "eos_first_token"}
            return empty_kl, empty_metrics, "", prompt_len, meta

        generated_ids: list[int] = []
        step = 0
        stopped_reason = "max_new_tokens"

        while step < max_new_tokens:
            token_start_time = perf_counter()
            token_in = next_token
            if eos_token_id is not None and token_in.item() == eos_token_id:
                break

            generated_ids.append(int(token_in.item()))
            cache_position = torch.tensor([prompt_len + step], device=model_device, dtype=torch.long)

            deltas = generate_delta(current_state, original_state)
            svd_start_time = perf_counter()
            low_rank_deltas = low_rank_svd_list_cpu(deltas, n=low_rank_n)
            average_svd_time += perf_counter() - svd_start_time

            approximated_state = [_clone_recurrent_state(state) for state in original_state]
            mse_error: list[torch.Tensor] = []
            rel_frob_error: list[torch.Tensor] = []
            cosine_similarity_vals: list[torch.Tensor] = []
            retained_energy_vals: list[torch.Tensor] = []
            ssm_state_idx = 0
            for layer_idx in range(num_layers):
                if current_state[layer_idx] is None:
                    continue
                approx_state = original_state[layer_idx] + low_rank_deltas[ssm_state_idx]
                approximated_state[layer_idx] = approx_state
                current_layer_state = current_state[layer_idx]
                mse_error.append(F.mse_loss(current_layer_state, approx_state))
                err = current_layer_state - approx_state
                current_norm = current_layer_state.norm().clamp_min(1e-12)
                rel_frob_error.append(err.norm() / current_norm)
                cosine_similarity_vals.append(F.cosine_similarity(current_layer_state.flatten(), approx_state.flatten(), dim=0))
                retained_energy_vals.append(1.0 - (err.pow(2).sum() / current_layer_state.pow(2).sum().clamp_min(1e-12)))
                ssm_state_idx += 1

            if not mse_error:
                raise ValueError("No recurrent states were available to compute approximation metrics.")

            mse_rows.append(torch.stack(mse_error, dim=0).detach().float().cpu())
            rel_frob_rows.append(torch.stack(rel_frob_error, dim=0).detach().float().cpu())
            cosine_rows.append(torch.stack(cosine_similarity_vals, dim=0).detach().float().cpu())
            retained_energy_rows.append(torch.stack(retained_energy_vals, dim=0).detach().float().cpu())

            approximated_cache = copy.deepcopy(past_key_values)
            _inject_recurrent_states(approximated_cache, approximated_state)

            out, next_state = _forward_with_state_capture(
                model,
                cache_cls,
                num_layers,
                input_ids=token_in,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            out_approx, _ = _forward_with_state_capture(
                model,
                cache_cls,
                num_layers,
                input_ids=token_in,
                past_key_values=approximated_cache,
                cache_position=cache_position,
            )

            out_log_logits = torch.log_softmax(out.logits, dim=-1)
            approx_log_logits = torch.log_softmax(out_approx.logits, dim=-1)
            kl = F.kl_div(approx_log_logits, out_log_logits, reduction="batchmean", log_target=True).item()
            kl_list.append(kl)

            next_token = _sample_next_token(out.logits, temperature=SAMPLING_TEMPERATURE, top_p=SAMPLING_TOP_P, top_k=SAMPLING_TOP_K)
            past_key_values = out.past_key_values
            current_state = next_state
            step += 1

            if eos_token_id is not None and next_token.item() == eos_token_id:
                generated_ids.append(int(next_token.item()))
                stopped_reason = "eos"
                break

        print(f"Total SVD time: {average_svd_time:.3f}s")

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if not kl_list:
        kl_tensor = torch.tensor([], dtype=DTYPE)
        metric_tensors = {
            "mse": torch.empty(0, 0, dtype=DTYPE),
            "rel_frob": torch.empty(0, 0, dtype=DTYPE),
            "cosine_similarity": torch.empty(0, 0, dtype=DTYPE),
            "retained_energy": torch.empty(0, 0, dtype=DTYPE),
        }
    else:
        kl_tensor = torch.tensor(kl_list, dtype=DTYPE)
        metric_tensors = {
            "mse": torch.stack(mse_rows, dim=0),
            "rel_frob": torch.stack(rel_frob_rows, dim=0),
            "cosine_similarity": torch.stack(cosine_rows, dim=0),
            "retained_energy": torch.stack(retained_energy_rows, dim=0),
        }

    meta = {
        "prompt_len": prompt_len,
        "num_generated_tokens": len(generated_ids),
        "num_metric_steps": len(kl_list),
        "stopped_reason": stopped_reason,
        "elapsed_seconds": perf_counter() - prompt_start_time,
    }
    return kl_tensor, metric_tensors, response.strip(), prompt_len, meta


def _summary_stats(kl: torch.Tensor, metrics: dict[str, torch.Tensor], meta: dict) -> dict:
    out: dict = {
        "prompt_len": meta.get("prompt_len"),
        "num_generated_tokens": meta.get("num_generated_tokens"),
        "num_metric_steps": meta.get("num_metric_steps"),
        "stopped_reason": meta.get("stopped_reason"),
    }
    if kl.numel() == 0:
        out["kl"] = None
    else:
        out["kl"] = {
            "mean": float(kl.mean().item()),
            "min": float(kl.min().item()),
            "max": float(kl.max().item()),
        }
    mse = metrics["mse"]
    rel_frob = metrics["rel_frob"]
    cosine_similarity = metrics["cosine_similarity"]
    retained_energy = metrics["retained_energy"]
    if mse.numel() == 0:
        out["mse"] = None
        out["relative_frobenius_error"] = None
        out["cosine_similarity"] = None
        out["retained_energy"] = None
        out["per_layer"] = None
    else:
        per_step_mean = mse.mean(dim=-1)
        out["mse"] = {
            "mean_over_steps_and_layers": float(mse.mean().item()),
            "mean_over_steps_of_layer_mean": float(per_step_mean.mean().item()),
            "min_over_steps_of_layer_mean": float(per_step_mean.min().item()),
            "max_over_steps_of_layer_mean": float(per_step_mean.max().item()),
        }
        out["relative_frobenius_error"] = {
            "mean_over_steps_and_layers": float(rel_frob.mean().item()),
            "min": float(rel_frob.min().item()),
            "max": float(rel_frob.max().item()),
        }
        out["cosine_similarity"] = {
            "mean_over_steps_and_layers": float(cosine_similarity.mean().item()),
            "min": float(cosine_similarity.min().item()),
            "max": float(cosine_similarity.max().item()),
        }
        out["retained_energy"] = {
            "mean_over_steps_and_layers": float(retained_energy.mean().item()),
            "min": float(retained_energy.min().item()),
            "max": float(retained_energy.max().item()),
        }
        out["per_layer"] = {
            "mean_mse": [float(x) for x in mse.mean(dim=0).tolist()],
            "max_mse": [float(x) for x in mse.max(dim=0).values.tolist()],
            "mean_relative_frobenius_error": [float(x) for x in rel_frob.mean(dim=0).tolist()],
            "mean_cosine_similarity": [float(x) for x in cosine_similarity.mean(dim=0).tolist()],
            "mean_retained_energy": [float(x) for x in retained_energy.mean(dim=0).tolist()],
        }
    return out


def _run_and_save_prompt(*, prompt: str, prompt_id: str, run_dir: Path, model, tokenizer) -> dict:
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
    summary["low_rank_rank"] = LOW_RANK_RANK
    summary["max_new_tokens_cap"] = MAX_NEW_TOKENS
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text_path.write_text(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n", encoding="utf-8")

    return {
        "meta": meta,
        "summary": summary,
        "files": [
            kl_path.name,
            mse_path.name,
            rel_frob_path.name,
            cosine_path.name,
            retained_energy_path.name,
            summary_path.name,
            text_path.name,
        ],
    }


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("suite_kl_parallel_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    start = perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="auto")

    if DATASET == "mmlu":
        prompts = [p["prompt"] for p in load_mmlu_prompts()]
    else:
        prompts = PROMPT_SUITE
    if PROMPT_LIMIT > 0:
        prompts = prompts[:PROMPT_LIMIT]

    workers = min(NUM_WORKERS, len(prompts))
    prompt_id_width = max(2, len(str(len(prompts))))
    print(f"Running KL suite in parallel with {len(prompts)} prompts")
    print(f"Model loaded once: {MODEL_NAME}")
    print(f"Workers: {workers}")
    print(f"Saving outputs to: {run_dir}")
    if not prompts:
        raise ValueError("No prompts to run.")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, prompt in enumerate(prompts, start=1):
            prompt_id = f"prompt_{i:0{prompt_id_width}d}"
            fut = executor.submit(_run_and_save_prompt, prompt=prompt, prompt_id=prompt_id, run_dir=run_dir, model=model, tokenizer=tokenizer)
            futures[fut] = (i, prompt_id)

        completed = 0
        for fut in as_completed(futures):
            i, prompt_id = futures[fut]
            completed += 1
            print(f"\n[{completed:0{prompt_id_width}d}/{len(prompts):0{prompt_id_width}d}] finished {prompt_id}")
            try:
                result = fut.result()
                meta = result["meta"]
                summary = result["summary"]
                print(f"  metric steps: {meta['num_metric_steps']}, generated tokens: {meta['num_generated_tokens']}")
                print(f"  stopped: {meta['stopped_reason']}")
                print(f"  elapsed: {meta['elapsed_seconds']:.3f}s")
                if summary.get("kl"):
                    print(f"  KL mean: {summary['kl']['mean']:.6f}")
                print("  Saved: " + ", ".join(result["files"]))
            except Exception as exc:
                print(f"Failed for {prompt_id} (index {i}): {exc}")

    print("\nDone.")
    print(f"Experiment artifacts are in: {run_dir}")
    print(f"Total time: {perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

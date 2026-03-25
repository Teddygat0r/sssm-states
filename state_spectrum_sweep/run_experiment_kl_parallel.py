"""
Prompt-suite experiment: per-token KL divergence and MSE between original logits and
logits from a low-rank approximation of recurrent-state deltas.

Parallel variant:
- Loads model/tokenizer once.
- Runs multiple prompts concurrently with a thread pool on this node.
- Moves delta tensors to CPU before low-rank SVD.
- Computes original/approx logits in one batched forward pass (batch=2) while
  preserving per-branch cache state isolation.
"""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from datasets import load_dataset

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-4B"
LOW_RANK_RANK = 16
MAX_NEW_TOKENS = 200
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20
NUM_WORKERS = max(1, int(os.getenv("KL_PARALLEL_WORKERS", "2")))
PROMPT_LIMIT = int(os.getenv("KL_PROMPT_LIMIT", "0"))  # 0 means no limit
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


def generate_delta(state, state_ref):
    if isinstance(state, torch.Tensor):
        return state - state_ref

    if isinstance(state, list):
        return [
            state[i] - state_ref[i]
            for i in range(len(state))
            if state[i] is not None and state_ref[i] is not None
        ]

    raise TypeError(f"Unsupported state type for delta: {type(state)}")

# I think this is optimal... CPU >>> GPU
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
    choices = "\n".join(
        f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(example["choices"])
    )
    return (
        f"Subject: {example['subject']}\n\n"
        f"Question: {example['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with the single best option."
    )

def load_mmlu_prompts():
    dataset = load_dataset(MMLU_DATASET_NAME, MMLU_DATASET_CONFIG)
    prompts = []
    for split in MMLU_SPLITS:
        if split not in dataset:
            continue
        for idx, example in enumerate(dataset[split]):
            prompts.append(
                {
                    "id": f"{split}_{idx:05d}",
                    "prompt": _format_mmlu_prompt(example),
                    "split": split,
                }
            )
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
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    sampled_in_topk = torch.multinomial(filtered_probs, num_samples=1)
    next_token = torch.gather(topk_idx, dim=-1, index=sampled_in_topk)
    return next_token


def _cat_branch_tensor(orig_t: torch.Tensor, approx_t: torch.Tensor, *, field: str, layer_idx: int) -> torch.Tensor:
    if orig_t.shape[0] != 1 or approx_t.shape[0] != 1:
        raise ValueError(
            f"Expected batch=1 caches before concat for {field}[{layer_idx}], "
            f"got {orig_t.shape} and {approx_t.shape}"
        )
    return torch.cat([orig_t, approx_t], dim=0)


def _batch_two_caches(orig_cache, approx_cache):
    batched = copy.deepcopy(orig_cache)
    for field in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        orig_list = getattr(orig_cache, field)
        approx_list = getattr(approx_cache, field)
        merged = []
        for layer_idx, (orig_t, approx_t) in enumerate(zip(orig_list, approx_list)):
            if orig_t is None and approx_t is None:
                merged.append(None)
                continue
            if orig_t is None or approx_t is None:
                raise ValueError(f"Mismatched None states in {field}[{layer_idx}] during cache batching.")
            merged.append(_cat_branch_tensor(orig_t, approx_t, field=field, layer_idx=layer_idx))
        setattr(batched, field, merged)
    return batched


def _assert_cache_batch_size(cache_obj, expected_batch: int):
    for field in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        src_list = getattr(cache_obj, field)
        for layer_idx, tensor in enumerate(src_list):
            if tensor is None:
                continue
            if tensor.shape[0] != expected_batch:
                raise ValueError(
                    f"Expected batch={expected_batch} for {field}[{layer_idx}], got shape {tuple(tensor.shape)}"
                )


def _select_cache_batch_index(cache_obj, index: int):
    selected = copy.deepcopy(cache_obj)
    for field in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        src_list = getattr(cache_obj, field)
        out_list = []
        for layer_idx, tensor in enumerate(src_list):
            if tensor is None:
                out_list.append(None)
                continue
            if tensor.shape[0] <= index:
                raise ValueError(
                    f"Cannot select batch index {index} for {field}[{layer_idx}] with shape {tensor.shape}"
                )
            out_list.append(tensor[index : index + 1].contiguous())
        setattr(selected, field, out_list)
    return selected


def run_single_prompt(
    prompt: str,
    model,
    tokenizer,
    *,
    low_rank_n: int = LOW_RANK_RANK,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, int, dict]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
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
        out = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        if past_key_values is None:
            raise ValueError("Model returned no past_key_values; cannot compute recurrent deltas.")

        recurrent = getattr(past_key_values, "recurrent_states", None)
        if recurrent is None:
            raise ValueError(
                "past_key_values has no recurrent_states; this script expects Qwen3.5-style cache."
            )

        original_state = [
            s.detach().clone() if s is not None else None
            for s in past_key_values.recurrent_states
        ]
        next_token = _sample_next_token(
            out.logits,
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
            }
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
            cache_position = torch.tensor(
                [prompt_len + step], device=model.device, dtype=torch.long
            )

            deltas = generate_delta(
                past_key_values.recurrent_states,
                original_state,
            )
            svd_start_time = perf_counter()
            low_rank_deltas = low_rank_svd_list_cpu(deltas, n=low_rank_n)
            svd_elapsed_s = perf_counter() - svd_start_time
            average_svd_time += svd_elapsed_s

            approximated_states = copy.deepcopy(past_key_values)
            approximated_states.recurrent_states = []
            mse_error: list[torch.Tensor] = []
            rel_frob_error: list[torch.Tensor] = []
            cosine_similarity_vals: list[torch.Tensor] = []
            retained_energy_vals: list[torch.Tensor] = []
            ssm_states = 0
            for i in range(len(original_state)):
                if original_state[i] is not None:
                    approx_state = original_state[i] + low_rank_deltas[ssm_states]
                    approximated_states.recurrent_states.append(approx_state)
                    current_state = past_key_values.recurrent_states[i]
                    mse_error.append(
                        F.mse_loss(
                            current_state,
                            approx_state,
                        )
                    )
                    err = current_state - approx_state
                    current_norm = current_state.norm().clamp_min(1e-12)
                    rel_frob_error.append(err.norm() / current_norm)
                    cosine_similarity_vals.append(
                        F.cosine_similarity(
                            current_state.flatten(), approx_state.flatten(), dim=0
                        )
                    )
                    retained_energy_vals.append(
                        1.0 - (err.pow(2).sum() / current_state.pow(2).sum().clamp_min(1e-12))
                    )
                    ssm_states += 1
                else:
                    approximated_states.recurrent_states.append(None)

            if not mse_error:
                raise ValueError("No non-None recurrent states; cannot compute MSE.")

            mse_rows.append(torch.stack(mse_error, dim=0).detach().float().cpu())
            rel_frob_rows.append(torch.stack(rel_frob_error, dim=0).detach().float().cpu())
            cosine_rows.append(torch.stack(cosine_similarity_vals, dim=0).detach().float().cpu())
            retained_energy_rows.append(
                torch.stack(retained_energy_vals, dim=0).detach().float().cpu()
            )

            batched_cache = _batch_two_caches(past_key_values, approximated_states)
            _assert_cache_batch_size(batched_cache, expected_batch=2)
            batched_token_in = token_in.repeat(2, 1)
            out_batched = model(
                input_ids=batched_token_in,
                past_key_values=batched_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            _assert_cache_batch_size(out_batched.past_key_values, expected_batch=2)

            logits_batched = out_batched.logits
            if logits_batched.shape[0] != 2:
                raise ValueError(f"Expected batched logits leading dim 2, got {tuple(logits_batched.shape)}")
            orig_logits = logits_batched[0:1]
            approx_logits = logits_batched[1:2]

            out_log_logits = torch.log_softmax(orig_logits, dim=-1)
            approx_log_logits = torch.log_softmax(approx_logits, dim=-1)
            kl = F.kl_div(
                approx_log_logits,
                out_log_logits,
                reduction="batchmean",
                log_target=True,
            ).item()
            kl_list.append(kl)

            next_token = _sample_next_token(
                orig_logits,
                temperature=SAMPLING_TEMPERATURE,
                top_p=SAMPLING_TOP_P,
                top_k=SAMPLING_TOP_K,
            )

            past_key_values = _select_cache_batch_index(out_batched.past_key_values, 0)
            _assert_cache_batch_size(past_key_values, expected_batch=1)
            step += 1

            if eos_token_id is not None and next_token.item() == eos_token_id:
                generated_ids.append(int(next_token.item()))
                stopped_reason = "eos"
                break

        print(f"Total SVD time: {average_svd_time}s")

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
            "mean_relative_frobenius_error": [
                float(x) for x in rel_frob.mean(dim=0).tolist()
            ],
            "mean_cosine_similarity": [
                float(x) for x in cosine_similarity.mean(dim=0).tolist()
            ],
            "mean_retained_energy": [
                float(x) for x in retained_energy.mean(dim=0).tolist()
            ],
        }
    return out


def _run_and_save_prompt(
    *,
    prompt: str,
    prompt_id: str,
    run_dir: Path,
    model,
    tokenizer,
) -> dict:
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
    text_path.write_text(
        f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n",
        encoding="utf-8",
    )

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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    if DATASET == "mmlu":
        prompts = load_mmlu_prompts()
        prompts = [p["prompt"] for p in prompts]
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
            fut = executor.submit(
                _run_and_save_prompt,
                prompt=prompt,
                prompt_id=prompt_id,
                run_dir=run_dir,
                model=model,
                tokenizer=tokenizer,
            )
            futures[fut] = (i, prompt_id)

        completed = 0
        for fut in as_completed(futures):
            i, prompt_id = futures[fut]
            completed += 1
            print(
                f"\n[{completed:0{prompt_id_width}d}/{len(prompts):0{prompt_id_width}d}] finished {prompt_id}"
            )
            try:
                result = fut.result()
                meta = result["meta"]
                summary = result["summary"]
                print(
                    f"  metric steps: {meta['num_metric_steps']}, generated tokens: {meta['num_generated_tokens']}"
                )
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

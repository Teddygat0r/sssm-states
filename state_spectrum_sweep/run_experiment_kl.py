"""
Prompt-suite experiment: per-token KL divergence and MSE between original logits and
logits from a low-rank approximation of recurrent-state deltas (see notebook flow).

Stops generation on EOS (tokenizer/model), with MAX_NEW_TOKENS as a safety cap.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


MODEL_NAME = "Qwen/Qwen3.5-4B"
LOW_RANK_RANK = 16
MAX_NEW_TOKENS = 200
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
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


def low_rank_svd(tensor: torch.Tensor, n: int = 16) -> torch.Tensor:
    u, s, v = torch.linalg.svd(tensor, full_matrices=False, driver='gesvdj')
    u, s, v = u[..., :n], s[..., :n], v[..., :n, :]
    return u @ torch.diag_embed(s) @ v

# def low_rank_svd(tensor: torch.Tensor, n: int = 16, oversample: int = 4, niter: int = 1):
#     # randomized/truncated SVD; much cheaper when n << min(m, n)
#     q = min(n + oversample, min(tensor.shape[-2:]))
#     u, s, v = torch.svd_lowrank(tensor, q=q, niter=niter)
#     u, s, v = u[..., :n], s[..., :n], v[..., :n]
#     return (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)

def low_rank_svd_list(lst: list, n: int = 16) -> list:
    return [low_rank_svd(state, n) for state in lst]


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
    remove_mask[..., 0] = False  # Keep at least one token.
    filtered_probs = topk_probs.masked_fill(remove_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    sampled_in_topk = torch.multinomial(filtered_probs, num_samples=1)
    next_token = torch.gather(topk_idx, dim=-1, index=sampled_in_topk)
    return next_token


def run_single_prompt(
    prompt: str,
    model,
    tokenizer,
    *,
    low_rank_n: int = LOW_RANK_RANK,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, int, dict]:
    """
    Returns:
        kl_per_step: float tensor [T]
        metric tensors with shape [T, L] for:
            mse, rel_frob, cosine_similarity, retained_energy
        response text, prompt length in tokens, metadata dict
    """
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

        # No generated tokens if first greedy token is already EOS
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

            model_forward_start_time = perf_counter()
            out = model(
                input_ids=token_in,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            model_forward_elapsed_s = perf_counter() - model_forward_start_time
            print(
                f"    token {step + 1} model forward: "
                f"{model_forward_elapsed_s:.3f}s"
            )

            # After forward, cache is updated (same reference semantics as notebook).
            deltas = generate_delta(
                out.past_key_values.recurrent_states,
                original_state,
            )
            svd_start_time = perf_counter()
            low_rank_deltas = low_rank_svd_list(deltas, n=low_rank_n)
            svd_elapsed_s = perf_counter() - svd_start_time
            print(f"    token {step + 1} low-rank svd: {svd_elapsed_s:.3f}s")

            approximated_states = copy.deepcopy(out.past_key_values)
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
                    current_state = out.past_key_values.recurrent_states[i]
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

            approx_forward_start_time = perf_counter()
            out_approx = model(
                input_ids=token_in,
                past_key_values=approximated_states,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            approx_forward_elapsed_s = perf_counter() - approx_forward_start_time
            print(
                f"    token {step + 1} approx model forward: "
                f"{approx_forward_elapsed_s:.3f}s"
            )

            out_log_logits = torch.log_softmax(out.logits, dim=-1)
            approx_log_logits = torch.log_softmax(out_approx.logits, dim=-1)
            kl = F.kl_div(
                approx_log_logits,
                out_log_logits,
                reduction="batchmean",
                log_target=True,
            ).item()
            kl_list.append(kl)

            next_token = _sample_next_token(
                out.logits,
                temperature=SAMPLING_TEMPERATURE,
                top_p=SAMPLING_TOP_P,
                top_k=SAMPLING_TOP_K,
            )
            past_key_values = out.past_key_values
            token_elapsed_s = perf_counter() - token_start_time
            print(
                f"  token {step + 1} finished "
                f"({token_elapsed_s:.3f}s)"
            )
            step += 1

            if eos_token_id is not None and next_token.item() == eos_token_id:
                generated_ids.append(int(next_token.item()))
                stopped_reason = "eos"
                break

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


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("suite_kl_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    print(f"Running KL suite with {len(PROMPT_SUITE)} prompts")
    print(f"Saving outputs to: {run_dir}")

    for i, prompt in enumerate(PROMPT_SUITE, start=1):
        prompt_id = f"prompt_{i:02d}"
        print(f"\n[{i:02d}/{len(PROMPT_SUITE)}] {prompt_id}")
        try:
            kl_tensor, metric_tensors, response, prompt_len, meta = run_single_prompt(
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

            print(f"  metric steps: {meta['num_metric_steps']}, generated tokens: {meta['num_generated_tokens']}")
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

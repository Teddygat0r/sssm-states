import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet as Qwen3_5GatedDeltaNet


def low_rank_svd(tensor: torch.Tensor, n: int = 16, oversample: int = 4, niter: int = 1):
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    orig_device = tensor.device
    orig_dtype = tensor.dtype
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    q = min(n + oversample, min(cpu_tensor.shape[-2:]))
    try:
        u, s, v = torch.svd_lowrank(cpu_tensor, q=q, niter=niter)
    except RuntimeError:
        return None
    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    approx_cpu = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
    return approx_cpu.to(device=orig_device, dtype=orig_dtype)


MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MAX_NEW_TOKENS = 150
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

PROMPT_SUITE = [
    "Hi there! Who are you?",
    "Explain photosynthesis in one paragraph.",
    "Write a simple Python function to check if a number is prime.",
    "Describe the plot of Romeo and Juliet in four sentences.",
    "Write a haiku about rain in a city.",
]

SVD_CONFIGS = [
    {"rank": None, "interval": 0, "label": "baseline"},
    {"rank": 64, "interval": 32, "label": "rank64_every32"},
    {"rank": 16, "interval": 32, "label": "rank16_every32"},
    {"rank": 4,  "interval": 32, "label": "rank4_every32"},
]


def make_svd_hook(rank: int, interval: int, state: dict):
    def hook(module, args, kwargs, output):
        cache = kwargs.get("cache_params", None)
        if cache is None:
            return
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or layer_idx >= len(cache.layers):
            return
        layer_cache = cache.layers[layer_idx]
        rs = getattr(layer_cache, "recurrent_states", None)
        if rs is None:
            return

        hidden = args[0] if args else kwargs.get("hidden_states")
        seq_len = hidden.shape[1] if hidden is not None else 1

        counters = state.setdefault("counters", {})
        metrics = state.setdefault("metrics", {})

        fire = False
        if seq_len > 1:
            fire = True
            counters[layer_idx] = 0
        else:
            counters[layer_idx] = counters.get(layer_idx, 0) + 1
            if counters[layer_idx] >= interval:
                fire = True
                counters[layer_idx] = 0

        if not fire:
            return

        compressed = low_rank_svd(rs, n=rank)
        if compressed is None:
            return

        orig_f = rs.detach().float()
        comp_f = compressed.float()
        err = orig_f - comp_f
        orig_norm_sq = orig_f.pow(2).sum().clamp_min(1e-12)
        mse_val = F.mse_loss(comp_f, orig_f).item()
        rel_frob_val = (err.norm() / orig_f.norm().clamp_min(1e-12)).item()
        cos_val = F.cosine_similarity(orig_f.flatten(), comp_f.flatten(), dim=0).item()
        energy_val = (1.0 - err.pow(2).sum() / orig_norm_sq).item()

        metrics.setdefault(layer_idx, []).append(
            {
                "mse": mse_val,
                "rel_frob": rel_frob_val,
                "cosine_similarity": cos_val,
                "retained_energy": energy_val,
            }
        )

        layer_cache.recurrent_states.copy_(compressed)

    return hook


def register_svd_hooks(model, rank, interval):
    hook_state = {"counters": {}, "metrics": {}}
    handles = []
    if rank is None:
        return handles, hook_state
    for _, module in model.named_modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            h = module.register_forward_hook(
                make_svd_hook(rank, interval, hook_state),
                with_kwargs=True,
            )
            handles.append(h)
    return handles, hook_state


def _resolve_eos_token_id(tokenizer, model):
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, list):
        return eos[0] if eos else None
    return eos


def _log_softmax_cpu(logits_last):
    return torch.log_softmax(logits_last.float(), dim=-1).detach().cpu()


def run_baseline(prompt_inputs, model, max_new_tokens, eos_token_id):
    tokens = []
    logprobs_list = []
    with torch.inference_mode():
        out = model(
            input_ids=prompt_inputs.input_ids,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        for _step in range(max_new_tokens):
            logp = _log_softmax_cpu(out.logits[0, -1, :])
            next_tok = int(logp.argmax().item())
            tokens.append(next_tok)
            logprobs_list.append(logp)
            if eos_token_id is not None and next_tok == eos_token_id:
                break
            token_in = torch.tensor([[next_tok]], device=model.device)
            out = model(
                input_ids=token_in,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
    return tokens, logprobs_list


def run_compressed(prompt_inputs, force_tokens, model, base_logprobs, rank, interval):
    handles, hook_state = register_svd_hooks(model, rank, interval)
    kl_list = []
    pred_tokens = []
    try:
        with torch.inference_mode():
            out = model(
                input_ids=prompt_inputs.input_ids,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            for step, tok in enumerate(force_tokens):
                logp = _log_softmax_cpu(out.logits[0, -1, :])
                pred_tokens.append(int(logp.argmax().item()))
                p_base = base_logprobs[step].exp()
                kl = (p_base * (base_logprobs[step] - logp)).sum().item()
                kl_list.append(kl)
                if step == len(force_tokens) - 1:
                    break
                token_in = torch.tensor([[tok]], device=model.device)
                out = model(
                    input_ids=token_in,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
    finally:
        for h in handles:
            h.remove()
    return kl_list, pred_tokens, hook_state["metrics"]


def _stack_layer_metrics(metrics):
    if not metrics:
        empty = torch.empty(0, 0, dtype=torch.float32)
        return [], empty, empty, empty, empty
    linear_indices = sorted(metrics.keys())
    trigger_counts = [len(metrics[l]) for l in linear_indices]
    T = min(trigger_counts) if trigger_counts else 0
    if T == 0:
        empty = torch.empty(0, len(linear_indices), dtype=torch.float32)
        return linear_indices, empty, empty, empty, empty

    def gather(key):
        return torch.tensor(
            [[metrics[l][t][key] for l in linear_indices] for t in range(T)],
            dtype=torch.float32,
        )

    return (
        linear_indices,
        gather("mse"),
        gather("rel_frob"),
        gather("cosine_similarity"),
        gather("retained_energy"),
    )


def _summary_from_metrics(kl, mse, rel_frob, cosine, energy, prompt_len, n_generated):
    summary = {
        "prompt_len": prompt_len,
        "num_generated_tokens": n_generated,
        "num_kl_steps": int(kl.numel()),
        "num_triggers": int(mse.shape[0]) if mse.numel() else 0,
        "num_linear_layers": int(mse.shape[1]) if mse.numel() else 0,
    }
    if kl.numel():
        summary["kl"] = {
            "mean": float(kl.mean()),
            "min": float(kl.min()),
            "max": float(kl.max()),
        }
    else:
        summary["kl"] = None

    if mse.numel():
        per_step_mse_mean = mse.mean(dim=-1)
        summary["mse"] = {
            "mean_over_steps_and_layers": float(mse.mean()),
            "mean_over_steps_of_layer_mean": float(per_step_mse_mean.mean()),
            "min_over_steps_of_layer_mean": float(per_step_mse_mean.min()),
            "max_over_steps_of_layer_mean": float(per_step_mse_mean.max()),
        }
        summary["relative_frobenius_error"] = {
            "mean_over_steps_and_layers": float(rel_frob.mean()),
            "min": float(rel_frob.min()),
            "max": float(rel_frob.max()),
        }
        summary["cosine_similarity"] = {
            "mean_over_steps_and_layers": float(cosine.mean()),
            "min": float(cosine.min()),
            "max": float(cosine.max()),
        }
        summary["retained_energy"] = {
            "mean_over_steps_and_layers": float(energy.mean()),
            "min": float(energy.min()),
            "max": float(energy.max()),
        }
        summary["per_layer"] = {
            "mean_mse": [float(x) for x in mse.mean(dim=0).tolist()],
            "max_mse": [float(x) for x in mse.max(dim=0).values.tolist()],
            "mean_relative_frobenius_error": [float(x) for x in rel_frob.mean(dim=0).tolist()],
            "mean_cosine_similarity": [float(x) for x in cosine.mean(dim=0).tolist()],
            "mean_retained_energy": [float(x) for x in energy.mean(dim=0).tolist()],
        }
    else:
        for k in ("mse", "relative_frobenius_error", "cosine_similarity", "retained_energy", "per_layer"):
            summary[k] = None
    return summary


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("svd_sanity_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()

    eos_token_id = _resolve_eos_token_id(tokenizer, model)
    n_gdn = sum(1 for _, m in model.named_modules() if isinstance(m, Qwen3_5GatedDeltaNet))
    print(f"Model loaded. {n_gdn} Qwen3_5MoeGatedDeltaNet layers found.")
    print(f"Saving to: {run_dir}")

    for i, prompt in enumerate(PROMPT_SUITE, start=1):
        prompt_id = f"prompt_{i:02d}"
        print(f"\n========== [{i:02d}/{len(PROMPT_SUITE)}] {prompt_id} ==========")
        print(f"PROMPT: {prompt}")
        t0 = perf_counter()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_len = int(prompt_inputs.input_ids.shape[1])

        base_tokens, base_logprobs_list = run_baseline(prompt_inputs, model, MAX_NEW_TOKENS, eos_token_id)
        base_logprobs = torch.stack(base_logprobs_list, dim=0) if base_logprobs_list else torch.empty(0)
        base_response = tokenizer.decode(base_tokens, skip_special_tokens=True).strip()
        print(f"baseline tokens: {len(base_tokens)}, elapsed: {perf_counter() - t0:.1f}s")
        print(f"baseline response: {base_response}")

        responses = {"baseline": base_response}

        for cfg in SVD_CONFIGS:
            if cfg["rank"] is None:
                continue
            cfg_t0 = perf_counter()
            kl_list, pred_tokens, layer_metrics = run_compressed(
                prompt_inputs, base_tokens, model, base_logprobs,
                cfg["rank"], cfg["interval"],
            )
            pred_response = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            responses[cfg["label"]] = pred_response

            kl_t = torch.tensor(kl_list, dtype=torch.float32)
            linear_indices, mse_t, rel_t, cos_t, energy_t = _stack_layer_metrics(layer_metrics)

            prefix = f"{prompt_id}_{cfg['label']}"
            torch.save(kl_t, run_dir / f"{prefix}_kl.pt")
            torch.save(mse_t, run_dir / f"{prefix}_mse.pt")
            torch.save(rel_t, run_dir / f"{prefix}_rel_frob.pt")
            torch.save(cos_t, run_dir / f"{prefix}_cosine_similarity.pt")
            torch.save(energy_t, run_dir / f"{prefix}_retained_energy.pt")

            summary = _summary_from_metrics(
                kl_t, mse_t, rel_t, cos_t, energy_t,
                prompt_len=prompt_len, n_generated=len(base_tokens),
            )
            summary.update(
                {
                    "prompt_id": prompt_id,
                    "label": cfg["label"],
                    "rank": cfg["rank"],
                    "interval": cfg["interval"],
                    "linear_layer_indices": linear_indices,
                    "predicted_response": pred_response,
                    "elapsed_seconds": perf_counter() - cfg_t0,
                }
            )
            (run_dir / f"{prefix}_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

            kl_mean = f"{summary['kl']['mean']:.5f}" if summary["kl"] else "n/a"
            cos_mean = (
                f"{summary['cosine_similarity']['mean_over_steps_and_layers']:.4f}"
                if summary["cosine_similarity"] else "n/a"
            )
            print(
                f"[{cfg['label']}] kl_mean={kl_mean} cos_mean={cos_mean} "
                f"triggers={summary['num_triggers']} layers={summary['num_linear_layers']} "
                f"elapsed={summary['elapsed_seconds']:.1f}s"
            )

        (run_dir / f"{prompt_id}_responses.txt").write_text(
            f"PROMPT:\n{prompt}\n\n"
            + "\n\n".join(f"=== {k} ===\n{v}" for k, v in responses.items())
            + "\n",
            encoding="utf-8",
        )

    print(f"\nDone. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()

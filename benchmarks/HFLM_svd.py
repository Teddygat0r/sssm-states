from lm_eval.models.huggingface import HFLM
import torch
import transformers
from lm_eval.models.utils_hf import stop_sequences_criteria

# Run on CPU because its faster for some reason...
def low_rank_svd(tensor: torch.Tensor, n: int = 16, oversample: int = 4, niter: int = 1):
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    orig_device = tensor.device
    orig_dtype = tensor.dtype

    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    q = min(n + oversample, min(cpu_tensor.shape[-2:]))

    try:
        u, s, v = torch.svd_lowrank(cpu_tensor, q=q, niter=niter)
    except RuntimeError as e:
        # log the failed tensor
        tensor = cpu_tensor.reshape((-1, cpu_tensor.shape[-2], cpu_tensor.shape[-1]))
        for i in range(tensor.shape[0]):
            try:
                u, s, v = torch.svd_lowrank(tensor[i], q=q, niter=niter)
            except RuntimeError as e:
                with open("svd_error.log", "a") as f:
                    f.write(f"SVD error: {e}\n")
                    f.write(f"tensor[{i}] = {tensor[i]}\n")
        return None

    u, s, v = u[..., :n], s[..., :n], v[..., :n]
    approx_cpu = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
    return approx_cpu.to(device=orig_device, dtype=orig_dtype)

def low_rank_svd_list(lst: list, n: int = 16) -> list:
    filtered_lst = [x for x in lst if x is not None]
    batched_svd = torch.stack(filtered_lst, dim=0)
    batched_svd = low_rank_svd(batched_svd, n=n)

    if batched_svd is None:
        return None
    processed = iter(batched_svd.unbind(dim=0))
    return [None if x is None else next(processed) for x in lst]

### WORKS ONLY FOR QWEN3.5
class HFLM_svd(HFLM):
    def __init__(self, *args, rank_k=64, prefix_length=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_k = rank_k
        self.prefix_length = prefix_length
    
    def _low_rank_approx(self, state):
        if isinstance(state, torch.Tensor):
            return low_rank_svd(state, n=self.rank_k)

        if isinstance(state, list):
            approx = low_rank_svd_list(state, n=self.rank_k)
            return approx

        raise TypeError(f"Unsupported state type for low-rank approximation: {type(state)}")
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        if labels is not None:
            return super()._model_call(inps, attn_mask, labels)

        batch_size, total_len = inps.shape
        all_logits = []

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            for i in range(batch_size):
                seq = inps[i]  # [total_len]

                # Strip left-padding (HFLM pads on the left for causal models).
                if attn_mask is not None:
                    actual_len = int(attn_mask[i].sum().item())
                else:
                    actual_len = total_len
                pad_len = total_len - actual_len
                actual_seq = seq[pad_len:].unsqueeze(0)  # [1, actual_len]

                if actual_len < 2:
                    logits = self.model(actual_seq).logits
                else:
                    # Prefill everything except the last token.
                    prefix_out = self.model(
                        input_ids=actual_seq[:, :-1],
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = prefix_out.past_key_values

                    # SVD compress the recurrent state.
                    if (
                        hasattr(cache, "recurrent_states")
                        and cache.recurrent_states is not None
                    ):
                        compressed = self._low_rank_approx(cache.recurrent_states)
                        if compressed is not None:
                            cache.recurrent_states = compressed

                    # Score the last token using the damaged state.
                    full_mask = torch.ones(
                        1, actual_len, dtype=torch.long, device=self.device
                    )
                    cont_out = self.model(
                        input_ids=actual_seq[:, -1:],
                        attention_mask=full_mask,
                        past_key_values=cache,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = torch.cat(
                        [prefix_out.logits, cont_out.logits], dim=1
                    )

                # Re-insert left-padding to match the expected shape.
                if pad_len > 0:
                    pad_logits = torch.zeros(
                        1, pad_len, logits.shape[-1],
                        device=logits.device, dtype=logits.dtype,
                    )
                    logits = torch.cat([pad_logits, logits], dim=1)

                all_logits.append(logits)

        return torch.cat(all_logits, dim=0)

    # Returns self.model.generate(...) after calling the model once
    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ):
        # We run token-by-token decoding so we can modify recurrent state cache.
        # model.generate(...) does not provide a clean hook to low-rank transform
        # prompt cache recurrent states before the first generated token.
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if (temp := generation_kwargs.get("temperature")) == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and temp == 0.0:
            generation_kwargs.pop("temperature", None)

        temperature = float(generation_kwargs.pop("temperature", 0.7))
        top_k = int(generation_kwargs.pop("top_k", 20))
        top_p = float(generation_kwargs.pop("top_p", 0.8))
        attention_mask = generation_kwargs.pop("attention_mask", None)

        # Optional external cache/state injection hook:
        # - initial_past_key_values: full cache object
        # - initial_recurrent_states: list/tensor for cache.recurrent_states
        initial_past_key_values = generation_kwargs.pop("initial_past_key_values", None)
        initial_recurrent_states = generation_kwargs.pop("initial_recurrent_states", None)
        apply_low_rank_to_prompt_state = generation_kwargs.pop(
            "apply_low_rank_to_prompt_state", True
        )

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        max_new_tokens = max(0, max_length - context.shape[1])
        if max_new_tokens == 0:
            return context

        def _sample_next_token(logits: torch.Tensor) -> torch.Tensor:
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            if do_sample:
                if temperature <= 0.0:
                    raise ValueError("temperature must be > 0 when do_sample=True")
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.shape[-1])
                    topk_vals, _ = torch.topk(logits, k=k, dim=-1)
                    kth = topk_vals[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumprobs > top_p
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        dim=-1, index=sorted_indices, src=sorted_logits
                    )
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, num_samples=1)

            return torch.argmax(logits, dim=-1, keepdim=True)

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            if initial_past_key_values is None:
                first_out = self.model(
                    input_ids=context,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = first_out.past_key_values
                next_token_logits = first_out.logits[:, -1, :]
            else:
                past_key_values = initial_past_key_values
                last_token = context[:, -1:]
                first_out = self.model(
                    input_ids=last_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = first_out.past_key_values
                next_token_logits = first_out.logits[:, -1, :]

            if initial_recurrent_states is not None and hasattr(past_key_values, "recurrent_states"):
                past_key_values.recurrent_states = initial_recurrent_states
            elif (
                apply_low_rank_to_prompt_state
                and hasattr(past_key_values, "recurrent_states")
                and past_key_values.recurrent_states is not None
            ):
                recurrent_states = self._low_rank_approx(
                    past_key_values.recurrent_states
                )

                if recurrent_states is not None:
                    past_key_values.recurrent_states = recurrent_states

            generated = context
            if attention_mask is None:
                running_attention_mask = torch.ones_like(context, dtype=torch.long)
            else:
                running_attention_mask = attention_mask

            for _ in range(max_new_tokens):
                next_token = _sample_next_token(next_token_logits)
                generated = torch.cat([generated, next_token], dim=-1)

                stop_result = stopping_criteria(generated, next_token_logits)
                if isinstance(stop_result, torch.Tensor):
                    should_stop = bool(stop_result.all().item())
                else:
                    should_stop = bool(stop_result)
                if should_stop:
                    break

                running_attention_mask = torch.cat(
                    [
                        running_attention_mask,
                        torch.ones(
                            (running_attention_mask.shape[0], 1),
                            dtype=running_attention_mask.dtype,
                            device=running_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )

                step_out = self.model(
                    input_ids=next_token,
                    attention_mask=running_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = step_out.past_key_values
                next_token_logits = step_out.logits[:, -1, :]

            return generated
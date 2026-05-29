"""HFLM subclass that applies a pluggable compressor to cache.recurrent_states
between prefill and continuation.

Structure mirrors benchmarks/HFLM_svd.py but generalizes the compression hook
so a single class supports baseline / SVD / quant / Hadamard configurations.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria


CompressorFn = Callable[[Tensor], Tensor]


class HFLM_compress(HFLM):
    """Qwen3.5-4B-compatible HFLM wrapper with a pluggable recurrent-state
    compressor.

    `compressor` is invoked exactly once per scoring/generation call, on the
    full `cache.recurrent_states` tensor produced by prefill. If `compressor`
    is None, the cache is unchanged (baseline).
    """

    def __init__(
        self,
        *args,
        compressor: Optional[CompressorFn] = None,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compressor = compressor
        # torch.compile speeds up the per-token decode forward by reducing
        # python/launch overhead. Compression mutates layer.recurrent_states
        # by assignment (new tensor object, same shape/dtype), which the
        # compiled graph re-reads each step, so compile + compression compose
        # correctly.
        if compile_mode is not None:
            # HFLM exposes `model` as a property reading from `_model`; write
            # the compiled version into the backing attribute.
            self._model = torch.compile(self._model, mode=compile_mode)

    @staticmethod
    def _layer_recurrent_tensor(layer) -> Optional[Tensor]:
        """Mirror kl_generation_experiments/kl_core.py:_layer_recurrent_tensor.

        Qwen3.5 transformers-builtin layers expose `.recurrent_states`; FLA
        layers keep it in a `.state` dict under 'recurrent_state'. Either way,
        the tensor has shape [B, H, D_k, D_v].
        """
        t = getattr(layer, "recurrent_states", None)
        if torch.is_tensor(t) and t.dim() >= 4:
            return t
        st = getattr(layer, "state", None)
        if isinstance(st, dict):
            t = st.get("recurrent_state")
            if torch.is_tensor(t) and t.dim() >= 4:
                return t
        return None

    @staticmethod
    def _write_layer_recurrent(layer, new_t: Tensor) -> None:
        if torch.is_tensor(getattr(layer, "recurrent_states", None)):
            layer.recurrent_states = new_t
            return
        st = getattr(layer, "state", None)
        if isinstance(st, dict) and torch.is_tensor(st.get("recurrent_state")):
            st["recurrent_state"] = new_t
            return
        raise RuntimeError("could not locate recurrent_states slot on layer")

    def _maybe_compress(self, cache) -> None:
        if self.compressor is None or not hasattr(cache, "layers"):
            return

        layer_states: list[tuple[int, Tensor]] = []
        for li, layer in enumerate(cache.layers):
            t = self._layer_recurrent_tensor(layer)
            if t is not None:
                layer_states.append((li, t))
        if not layer_states:
            return

        # Stack along the batch axis so the compressor runs once over all
        # linear-attn layers: [L * B, H, D_k, D_v]. rsvd_eigh is much more
        # efficient on a single large batch than 24 per-layer calls.
        stacked = torch.cat([t for _, t in layer_states], dim=0)
        compressed = self.compressor(stacked)
        if compressed is None:
            return
        assert compressed.shape == stacked.shape, (
            f"compressor changed shape: {stacked.shape} -> {compressed.shape}"
        )
        assert compressed.dtype == stacked.dtype, (
            f"compressor changed dtype: {stacked.dtype} -> {compressed.dtype}"
        )

        # Scatter back to each layer's slot, preserving the original B.
        offset = 0
        for li, original in layer_states:
            b = original.shape[0]
            piece = compressed[offset : offset + b].contiguous()
            offset += b
            self._write_layer_recurrent(cache.layers[li], piece)

    # ---- scoring path (loglikelihood tasks) --------------------------------

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
                seq = inps[i]
                if attn_mask is not None:
                    actual_len = int(attn_mask[i].sum().item())
                else:
                    actual_len = total_len
                pad_len = total_len - actual_len
                actual_seq = seq[pad_len:].unsqueeze(0)

                if actual_len < 2:
                    logits = self.model(actual_seq).logits
                else:
                    prefix_out = self.model(
                        input_ids=actual_seq[:, :-1],
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = prefix_out.past_key_values
                    self._maybe_compress(cache)

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

                if pad_len > 0:
                    pad_logits = torch.zeros(
                        1, pad_len, logits.shape[-1],
                        device=logits.device, dtype=logits.dtype,
                    )
                    logits = torch.cat([pad_logits, logits], dim=1)

                all_logits.append(logits)

        return torch.cat(all_logits, dim=0)

    # ---- generation path (GSM8K) -------------------------------------------

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ):
        """Manual prefill → compress recurrent state → delegate decoding to
        HF model.generate.

        Delegating to model.generate gives us per-sequence early termination
        for free (otherwise the slowest sample in a batch keeps the whole
        batch decoding for thousands of wasted tokens).
        """
        temp = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if do_sample is None:
            do_sample = (temp is not None) and (temp > 0.0)

        attention_mask = generation_kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(context, dtype=torch.long)

        max_new_tokens = max(0, max_length - context.shape[1])
        if max_new_tokens == 0:
            return context

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            # Prefill everything EXCEPT the last token so we can hand the
            # compressed cache + the last token to generate as its first
            # decode step. This avoids HF generate's input/cache-length
            # disagreement when past covers the full input.
            prefix_ids = context[:, :-1]
            prefix_mask = attention_mask[:, :-1]
            prefill_out = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prefill_out.past_key_values
            self._maybe_compress(past_key_values)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "stopping_criteria": stopping_criteria,
                "pad_token_id": pad_token_id,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(
                    generation_kwargs.get("temperature", 0.7)
                )
                top_k = int(generation_kwargs.get("top_k", 0) or 0)
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k
                top_p = float(generation_kwargs.get("top_p", 1.0) or 1.0)
                if top_p < 1.0:
                    gen_kwargs["top_p"] = top_p

            # generate's first forward processes context[:, -1:] using the
            # (compressed) prefix cache, then continues decoding normally
            # with per-sequence early termination.
            generated = self.model.generate(
                input_ids=context,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            return generated

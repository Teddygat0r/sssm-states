"""Diagnose the per-token throughput bottleneck on Qwen3.5-4B at bs=8.

Reports:
  - prefill latency for a single ~1000-token prompt at bs=1, 4, 8
  - decode latency per token at bs=1, 4, 8 (greedy, 200 new tokens)
  - effective tokens/sec at each batch size
  - compression overhead (one rsvd_eigh call on the stacked recurrent state)

This bypasses lm-eval and our HFLM wrapper so we can see whether the
slowdown is intrinsic to the model on this GPU or sits in our scaffolding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsvd_eigh import randomized_svd_eigh  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
PROMPT_TOKENS = 1024
DECODE_TOKENS = 200


def main():
    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # Build a long-ish synthetic prompt of ~PROMPT_TOKENS tokens.
    base = "The cat sat on the mat. " * 200
    ids = tok.encode(base, return_tensors="pt").to(DEVICE)[:, :PROMPT_TOKENS]
    print(f"Prompt length: {ids.shape[1]} tokens")

    # Warm up.
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()

    for bs in (1, 4, 8, 16):
        batched_ids = ids.repeat(bs, 1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=batched_ids,
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        pkv = out.past_key_values

        # Compression overhead diagnostic at bs.
        if hasattr(pkv, "layers"):
            stacked = []
            for layer in pkv.layers:
                t = getattr(layer, "recurrent_states", None)
                if torch.is_tensor(t) and t.dim() >= 4:
                    stacked.append(t)
            if stacked:
                S = torch.cat(stacked, dim=0).float()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                U, Sv, Vh = randomized_svd_eigh(
                    S, rank=16, n_iter=1, oversample=4, power_dtype=torch.float32
                )
                torch.cuda.synchronize()
                svd_s = time.perf_counter() - t0
            else:
                svd_s = float("nan")
        else:
            svd_s = float("nan")

        # Decode DECODE_TOKENS tokens greedily.
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        running_mask = torch.ones(bs, batched_ids.shape[1], device=DEVICE, dtype=torch.long)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(DECODE_TOKENS):
                running_mask = torch.cat(
                    [running_mask, torch.ones(bs, 1, device=DEVICE, dtype=torch.long)],
                    dim=-1,
                )
                step = model(
                    input_ids=next_token,
                    attention_mask=running_mask,
                    past_key_values=pkv,
                    use_cache=True,
                    return_dict=True,
                )
                pkv = step.past_key_values
                next_token = step.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

        tok_per_s = (DECODE_TOKENS * bs) / decode_s
        print(
            f"bs={bs:2d}: prefill={prefill_s*1000:6.1f}ms  "
            f"decode={decode_s:5.1f}s ({DECODE_TOKENS} tok x {bs} seq)  "
            f"throughput={tok_per_s:6.1f} tok/s  "
            f"per-sample-200-tok={decode_s/1:.2f}s  "
            f"svd_r16_overhead={svd_s*1000:5.1f}ms"
        )


if __name__ == "__main__":
    main()

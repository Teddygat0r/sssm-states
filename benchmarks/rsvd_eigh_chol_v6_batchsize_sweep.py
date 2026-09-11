"""
Batch-size sweep for `chol_v6` fp32 on Qwen3.5-4B states.

Hypothesis from the per-state run: the raise is a batched-cuSOLVER eigh
convergence issue, not a property of individual states. At batch=1 the
raise rate is ~0.02%; at batch=768 it's ~50% of prompts.

This script measures raise rate at batch sizes {1, 8, 32, 64, 128, 256,
384, 768} on the same set of MMLU prompts. For each prompt we split the
768-state batch into chunks of size B, run chol_v6 fp32 on each chunk,
and count how many chunks raised. We also report mean runtime per chunk
and total runtime per prompt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsvd_eigh import randomized_svd_eigh  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16


def _find_linear_layers(model):
    out = []
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            out.append(li)
    return out


def _format_mmlu(ex):
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    return (f"The following is a multiple choice question about "
            f"{ex['subject'].replace('_', ' ')}.\n\n"
            f"Question: {ex['question']}\n\n{body}\n\nAnswer:")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "eval_results" / datetime.now().strftime(
        "rsvd_chol_v6_batchsize_%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    print(f"Loading {MODEL_ID} ...")
    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map=DEVICE)
    model.eval()
    print(f"  loaded in {perf_counter()-t0:.1f}s")
    linear_layers = _find_linear_layers(model)

    print(f"Sampling {args.n_prompts} MMLU prompts ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(ds), generator=g)[:args.n_prompts].tolist()

    batch_sizes = [1, 8, 32, 64, 128, 256, 384, 768]
    rows = []

    for pi, i in enumerate(idxs):
        ex = ds[i]
        prompt = _format_mmlu(ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.max_seq_len].to(DEVICE)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        cache = out.past_key_values
        states = torch.cat([cache.layers[li].recurrent_states.squeeze(0).detach()
                             for li in linear_layers], dim=0).float()
        del out, cache
        N = states.shape[0]

        print(f"\n[{pi+1}/{len(idxs)}] mmlu_{i} ({ex['subject']})  N={N}")
        for B in batch_sizes:
            n_chunks = (N + B - 1) // B
            n_raised = 0
            n_ok = 0
            n_states_raised = 0
            # warmup
            warm = states[:B]
            try:
                _ = randomized_svd_eigh(warm, rank=16, n_iter=2, oversample=8,
                                        orth="chol_v6", power_dtype=torch.float32)
            except Exception:
                pass
            _sync()
            t0 = perf_counter()
            for c in range(n_chunks):
                chunk = states[c * B : (c + 1) * B]
                if chunk.shape[0] == 0:
                    continue
                torch.manual_seed(0)
                try:
                    _ = randomized_svd_eigh(chunk, rank=16, n_iter=2, oversample=8,
                                            orth="chol_v6", power_dtype=torch.float32)
                    n_ok += 1
                except Exception:
                    n_raised += 1
                    n_states_raised += chunk.shape[0]
            _sync()
            rt = perf_counter() - t0
            print(f"  B={B:>4d}  chunks={n_chunks:>4d}  chunks_raised={n_raised:>3d}  "
                  f"states_in_raised_chunks={n_states_raised:>4d}  "
                  f"chunk_raise_rate={100*n_raised/n_chunks:>5.1f}%  "
                  f"prompt_total={rt*1000:>7.1f}ms  ({rt*1000/n_chunks:.2f}ms/chunk)")
            rows.append({
                "prompt_id": f"mmlu_{i}",
                "subject": ex["subject"],
                "batch_size": B,
                "n_chunks": n_chunks,
                "chunks_raised": n_raised,
                "chunks_ok": n_ok,
                "states_in_raised_chunks": n_states_raised,
                "chunk_raise_rate_pct": 100 * n_raised / n_chunks,
                "prompt_total_ms": rt * 1000,
                "per_chunk_ms": rt * 1000 / n_chunks,
            })

        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "batchsize_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Aggregate.
    from collections import defaultdict
    agg = defaultdict(lambda: dict(prompts=0, prompts_w_raise=0, tot_chunks=0,
                                     tot_raised=0, tot_states_raised=0,
                                     prompt_totals=[]))
    for r in rows:
        a = agg[r["batch_size"]]
        a["prompts"] += 1
        if r["chunks_raised"] > 0:
            a["prompts_w_raise"] += 1
        a["tot_chunks"] += r["n_chunks"]
        a["tot_raised"] += r["chunks_raised"]
        a["tot_states_raised"] += r["states_in_raised_chunks"]
        a["prompt_totals"].append(r["prompt_total_ms"])

    print()
    print("=" * 110)
    print(f"chol_v6 fp32 raise rate vs batch size  (across {args.n_prompts} prompts)")
    print("=" * 110)
    print(f"{'batch_size':<12}{'prompts_w_raise':>18}{'chunks_raised':>16}{'chunks_total':>14}"
          f"{'chunk_raise_%':>16}{'states_lost':>14}{'prompt_ms_avg':>16}")
    for B in batch_sizes:
        a = agg[B]
        rate = 100 * a["tot_raised"] / a["tot_chunks"] if a["tot_chunks"] else 0
        avg = sum(a["prompt_totals"]) / len(a["prompt_totals"]) if a["prompt_totals"] else 0
        print(f"{B:<12d}{a['prompts_w_raise']:>10d}/{args.n_prompts:<7d}"
              f"{a['tot_raised']:>16d}{a['tot_chunks']:>14d}{rate:>15.1f}%"
              f"{a['tot_states_raised']:>14d}{avg:>16.1f}")


if __name__ == "__main__":
    main()

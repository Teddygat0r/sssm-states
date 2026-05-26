"""
KL-under-state-compression during generation, across five recurrent / SSM models.

For each (model, input distribution, prefill length) we prefill the prompt, then
greedily generate KL_GEN_TOKENS tokens. Every KL_EVERY steps we measure the
instantaneous cost of compressing the *entire* recurrent state to each rank in
KL_RANKS: one batched forward whose row 0 carries the true state and rows 1..R
carry rank-k reconstructions (attention KV / conv states left exact). We log
KL(P_true || P_rank-k) for the next-token distribution.

This reuses reconstruction_experiments' model loaders + svd_sweep's prompt
loaders / task builder so the five-model setup and the two input distributions
(ShareGPT chat, The-Stack code) are identical to the reconstruction sweep.

Output: results/kl_<model>_<ts>/
  - kl_metrics.csv   one row per (prompt, position, gen_step, rank)
  - run_config.json  all knobs + per-task timing summary

Usage:
    python run_kl_generation.py --model mamba2
    python run_kl_generation.py --model qwen35 --limit-prompts 3   # calibration
Knobs (env):
    KL_GEN_TOKENS=128  KL_EVERY=4  KL_RANKS=4,8,16
    SWEEP_POSITIONS=256        (prefill lengths; a "full" task is always added)
    SWEEP_N_PROMPTS=100        (per distribution; via svd_sweep/_helpers)
    KL_STOP_ON_EOS=0   RECON_OVERSAMPLE=4  RECON_NITER=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reconstruction_experiments"))
sys.path.insert(0, str(HERE.parent / "svd_sweep"))
sys.path.insert(0, str(HERE))

from run_reconstruction import MODELS, DEVICE  # noqa: E402
from _helpers import (  # noqa: E402
    build_tasks,
    free_memory,
    load_prompts,
    make_run_dir,
    write_csv,
)
import kl_core as kc  # noqa: E402

RESULTS_ROOT = HERE / "results"

GEN_TOKENS = int(os.getenv("KL_GEN_TOKENS", "128"))
EVERY = max(1, int(os.getenv("KL_EVERY", "4")))
RANKS = tuple(int(x) for x in os.getenv("KL_RANKS", "4,8,16").replace(" ", "").split(",") if x)
STOP_ON_EOS = os.getenv("KL_STOP_ON_EOS", "0") == "1"
OVERSAMPLE = int(os.getenv("KL_OVERSAMPLE", "8"))   # rsvd_eigh sketch oversample
NITER = int(os.getenv("KL_NITER", "2"))             # rsvd_eigh power iterations

_pos_env = os.getenv("SWEEP_POSITIONS", "256")
POSITIONS = tuple(int(x) for x in _pos_env.replace(" ", "").split(",") if x)


def _eos_id(tokenizer, model):
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    return eos[0] if isinstance(eos, list) and eos else eos


def run_task(model, tokenizer, cache_kwarg, layer_idx_cache, task, *, eos_id):
    """Generate from one prefilled prompt, measuring compressed-state KL every
    EVERY steps. Returns (rows, n_measured, stopped_reason)."""
    ids = task.ids.unsqueeze(0).to(DEVICE)        # [1, T]
    T = ids.shape[1]
    N = len(RANKS) + 1

    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    cache = getattr(out, cache_kwarg)
    pos = T
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)

    # recurrent layer indices are fixed per model; compute once and reuse.
    if layer_idx_cache["idx"] is None:
        layer_idx_cache["idx"] = kc.recurrent_layer_indices(cache)
    layer_idx = layer_idx_cache["idx"]

    rows: list[dict] = []
    stopped = "max_tokens"
    for step in range(GEN_TOKENS):
        if step % EVERY == 0:
            states = kc.read_recurrent_states(cache, layer_idx)
            recons, rel_fro = kc.lowrank_recon_all(states, RANKS,
                                                   oversample=OVERSAMPLE, niter=NITER)
            kc.cache_repeat_(cache, N)
            for ri, k in enumerate(RANKS, start=1):
                for li in layer_idx:
                    kc.write_recurrent_row(cache, li, ri, recons[k][li])
            out = kc.forward_step(model, nxt.repeat(N, 1), cache, pos, cache_kwarg)
            logits = out.logits[:, -1]            # [N, V]
            for ri, k in enumerate(RANKS, start=1):
                rows.append({
                    "model": task.prompt.get("_model", ""),
                    "family": task.prompt.get("_family", ""),
                    "input_id": task.prompt.get("input_id", task.prompt["id"]),
                    "input_kind": task.prompt["kind"],
                    "prompt_id": task.prompt["id"],
                    "position": task.position,
                    "prefill_tokens": T,
                    "gen_step": step,
                    "context_len": T + step,
                    "rank": k,
                    "kl": kc.kl_full_vs_approx(logits[0], logits[ri]),
                    "state_rel_fro": rel_fro[k],
                })
            cache = getattr(out, cache_kwarg)
            kc.cache_select_row_(cache, 0)
            nxt = logits[0:1].argmax(-1, keepdim=True)
        else:
            out = kc.forward_step(model, nxt, cache, pos, cache_kwarg)
            cache = getattr(out, cache_kwarg)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
        if STOP_ON_EOS and eos_id is not None and int(nxt.item()) == eos_id:
            stopped = "eos"
            break

    n_measured = len(rows) // max(1, len(RANKS))
    return rows, n_measured, stopped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--limit-prompts", type=int, default=0,
                    help="cap the number of prompts (0 = all); for calibration")
    args = ap.parse_args()
    spec = MODELS[args.model]

    run_dir = make_run_dir(RESULTS_ROOT, f"kl_{args.model}")
    print(f"Output dir: {run_dir}")
    print(f"config: gen_tokens={GEN_TOKENS} every={EVERY} ranks={RANKS} "
          f"positions={POSITIONS}(+full) stop_on_eos={STOP_ON_EOS} "
          f"oversample={OVERSAMPLE} niter={NITER}")

    prompts = load_prompts()
    if args.limit_prompts > 0:
        # keep a balanced head of each distribution
        by_kind: dict[str, list] = {}
        for p in prompts:
            by_kind.setdefault(p["kind"], []).append(p)
        per = max(1, args.limit_prompts // max(1, len(by_kind)))
        prompts = [p for ps in by_kind.values() for p in ps[:per]]
        print(f"  limited to {len(prompts)} prompts ({per}/distribution)")

    print(f"\nLoading {spec['label']} ...")
    t0 = perf_counter()
    model, tokenizer, _state_fn, info = spec["loader"]()
    cache_kwarg = kc.detect_cache_kwarg(model)
    eos_id = _eos_id(tokenizer, model)
    print(f"  loaded in {perf_counter()-t0:.1f}s   ({info}); cache_kwarg={cache_kwarg}")

    for p in prompts:                              # tag rows with model identity
        p["_model"] = spec["label"]
        p["_family"] = spec["family"]

    tasks = build_tasks(prompts, tokenizer, POSITIONS)
    print(f"Built {len(tasks)} tasks across {len(prompts)} prompts; "
          f"positions={POSITIONS}(+full)")

    rows: list[dict] = []
    timings: list[dict] = []
    layer_idx_cache = {"idx": None}
    for i, task in enumerate(tasks, start=1):
        t1 = perf_counter()
        try:
            r, n_meas, stopped = run_task(
                model, tokenizer, cache_kwarg, layer_idx_cache, task, eos_id=eos_id)
        except Exception as e:
            print(f"  [{i:>4}/{len(tasks)}] {task.prompt['id']} pos={task.position} "
                  f"FAILED: {type(e).__name__}: {e}")
            free_memory()
            continue
        dt = perf_counter() - t1
        rows.extend(r)
        timings.append({"prompt_id": task.prompt["id"], "position": task.position,
                        "prefill_tokens": task.actual_tokens, "seconds": dt,
                        "n_measured": n_meas, "stopped": stopped})
        print(f"  [{i:>4}/{len(tasks)}] {task.prompt['id']} pos={task.position} "
              f"T={task.actual_tokens}  {dt:.1f}s  measured={n_meas} ({stopped})")
        free_memory()

    write_csv(run_dir / "kl_metrics.csv", rows)
    total_s = sum(t["seconds"] for t in timings)
    cfg = {
        "model": spec["label"], "family": spec["family"], "model_key": args.model,
        "gen_tokens": GEN_TOKENS, "every": EVERY, "ranks": list(RANKS),
        "positions": list(POSITIONS), "stop_on_eos": STOP_ON_EOS,
        "oversample": OVERSAMPLE, "niter": NITER,
        "n_prompts": len(prompts), "n_tasks": len(tasks),
        "n_tasks_ok": len(timings), "n_rows": len(rows),
        "total_seconds": total_s,
        "mean_seconds_per_task": (total_s / len(timings)) if timings else 0.0,
        "timings": timings,
    }
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nWrote {len(rows)} rows; {len(timings)}/{len(tasks)} tasks in "
          f"{total_s:.0f}s (mean {cfg['mean_seconds_per_task']:.1f}s/task).")
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()

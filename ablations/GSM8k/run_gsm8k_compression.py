"""Sweep Qwen3.5-4B GSM8K accuracy under 12 state-compression configs.

Streaming results land in `ablations/GSM8k/gsm8k_results.csv`. Per-config
lm-eval sample dumps land in `ablations/GSM8k/raw/{config_name}.json`.

Usage:
    source .venv/bin/activate
    python ablations/GSM8k/run_gsm8k_compression.py                 # full sweep
    python ablations/GSM8k/run_gsm8k_compression.py --limit 8       # smoke test
    python ablations/GSM8k/run_gsm8k_compression.py --only baseline svd_r16
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compression_ops import (  # noqa: E402
    make_hadamard_quant_compressor,
    make_quant_compressor,
    make_svd_compressor,
)
from hflm_compress import HFLM_compress  # noqa: E402


SVD_RANKS = [1, 2, 4, 8, 12, 16, 24]

CONFIGS: list[dict] = (
    [
        {"name": "baseline", "method": "baseline", "param": 0, "factory": lambda: None},
        {"name": "q4_per_head", "method": "quant", "param": 4, "factory": lambda: make_quant_compressor(4)},
        {"name": "q8_per_head", "method": "quant", "param": 8, "factory": lambda: make_quant_compressor(8)},
    ]
    + [
        {
            "name": f"svd_r{r}",
            "method": "svd",
            "param": r,
            "factory": (lambda r=r: make_svd_compressor(r)),
        }
        for r in SVD_RANKS
    ]
    + [
        {
            "name": "hadamard_q4",
            "method": "hadamard_quant",
            "param": 4,
            "factory": lambda: make_hadamard_quant_compressor(4),
        }
    ]
)


CSV_COLUMNS = [
    "name",
    "method",
    "param",
    "exact_match_strict",
    "exact_match_flex",
    "n_samples",
    "seconds",
    "timestamp",
]


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _ensure_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def _append_csv_row(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def _resolve_gsm8k(task_manager: TaskManager) -> str:
    matches = task_manager.match_tasks(["gsm8k"])
    if not matches:
        raise RuntimeError("lm-eval has no 'gsm8k' task installed")
    return matches[0]


def _extract_metrics(results: dict, task_name: str) -> tuple[Optional[float], Optional[float], int]:
    task_results = results.get("results", {}).get(task_name, {})
    strict = task_results.get("exact_match,strict-match")
    flex = task_results.get("exact_match,flexible-extract")
    n_samples = 0
    samples = results.get("samples", {}).get(task_name)
    if isinstance(samples, list):
        n_samples = len(samples)
    return (
        float(strict) if strict is not None else None,
        float(flex) if flex is not None else None,
        n_samples,
    )


def _run_one(
    cfg: dict,
    args: argparse.Namespace,
    task_name: str,
    task_manager: TaskManager,
    csv_path: Path,
    raw_dir: Path,
) -> None:
    factory: Callable[[], Optional[Callable]] = cfg["factory"]
    compressor = factory()

    print(f"\n=== {cfg['name']} (method={cfg['method']}, param={cfg['param']}) ===")
    t0 = time.perf_counter()

    lm = HFLM_compress(
        pretrained=args.model,
        backend="causal",
        trust_remote_code=True,
        device=args.device,
        batch_size=args.batch_size,
        compressor=compressor,
        compile_mode=args.compile_mode,
    )

    gen_kwargs = {
        "max_gen_toks": args.max_gen_toks,
        "temperature": 0.0,
        "do_sample": False,
    }

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        task_manager=task_manager,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=0,
        gen_kwargs=gen_kwargs,
        log_samples=True,
    )

    elapsed = time.perf_counter() - t0
    strict, flex, n_samples = _extract_metrics(results, task_name)

    print(f"  strict={strict}  flex={flex}  n={n_samples}  ({elapsed:.1f}s)")

    _append_csv_row(
        csv_path,
        {
            "name": cfg["name"],
            "method": cfg["method"],
            "param": cfg["param"],
            "exact_match_strict": strict if strict is not None else "",
            "exact_match_flex": flex if flex is not None else "",
            "n_samples": n_samples,
            "seconds": f"{elapsed:.2f}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )

    raw_path = raw_dir / f"{cfg['name']}.json"
    raw_path.write_text(json.dumps(_json_safe(results), indent=2))

    # Release VRAM before next config.
    del lm, results, compressor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--max-gen-toks", type=int, default=768)
    p.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help="If set, passes mode to torch.compile on the model.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap for quick smoke tests.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to a subset of config names (default: all).",
    )
    p.add_argument(
        "--results-csv",
        default=str(_HERE / "gsm8k_results.csv"),
    )
    p.add_argument(
        "--raw-dir",
        default=str(_HERE / "raw"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.results_csv)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _ensure_csv_header(csv_path)

    task_manager = TaskManager()
    task_name = _resolve_gsm8k(task_manager)
    print(f"Resolved task: {task_name}")
    print(f"Model:          {args.model}")
    print(f"Sample limit:   {args.limit if args.limit is not None else 'full test set'}")
    print(f"max_gen_toks:   {args.max_gen_toks}")
    print(f"Results CSV:    {csv_path}")
    print(f"Raw JSON dir:   {raw_dir}")

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c["name"] for c in CONFIGS}
        if unknown:
            raise SystemExit(f"Unknown config names: {sorted(unknown)}")
        configs = [c for c in CONFIGS if c["name"] in wanted]
    else:
        configs = CONFIGS

    for cfg in configs:
        try:
            _run_one(cfg, args, task_name, task_manager, csv_path, raw_dir)
        except Exception as e:
            print(f"  !! {cfg['name']} failed: {type(e).__name__}: {e}")
            _append_csv_row(
                csv_path,
                {
                    "name": cfg["name"],
                    "method": cfg["method"],
                    "param": cfg["param"],
                    "exact_match_strict": "",
                    "exact_match_flex": "",
                    "n_samples": 0,
                    "seconds": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()

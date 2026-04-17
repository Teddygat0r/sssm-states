from __future__ import annotations

"""
Aggregate saved Qwen KL experiment runs and plot KL summary percentiles by setting.

This script scans low-rank and quant experiment directories under
`state_spectrum_sweep/experiments`, loads every saved `*_kl.pt` tensor for each run,
and computes four aggregate statistics over all collected KL values:
median, 90th percentile, 99th percentile, and 99.9th percentile.

The x-axis is inferred from each run's saved summary JSON:
- `low_rank_rank` for low-rank runs
- `quant_bits` for quant runs

It then saves a side-by-side figure for:
- low-rank KL vs rank
- quant KL vs bits
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - user-facing import guard
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
LOW_RANK_PREFIXES = ("suite_kl_", "suite_kl_parallel_")
QUANT_PREFIX = "suite_quant_"
PERCENTILES = {
    "median": 50.0,
    "p90": 90.0,
    "p99": 99.0,
    "p99.9": 99.9,
}


@dataclass
class RunStats:
    method: str
    x_value: float
    run_dir: Path
    prompt_count: int
    values: dict[str, float]


def _iter_summary_paths(run_dir: Path) -> Iterable[Path]:
    return sorted(run_dir.glob("*_metrics_summary.json"))


def _load_run_parameter(run_dir: Path, method: str) -> float | None:
    for summary_path in _iter_summary_paths(run_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if method == "low_rank" and "low_rank_rank" in summary:
            return float(summary["low_rank_rank"])
        if method == "quant" and "quant_bits" in summary:
            return float(summary["quant_bits"])
    return None


def _load_kl_values(run_dir: Path) -> np.ndarray:
    values: list[np.ndarray] = []
    for kl_path in sorted(run_dir.glob("*_kl.pt")):
        tensor = torch.load(kl_path, map_location="cpu")
        array = tensor.detach().float().cpu().numpy().reshape(-1)
        if array.size:
            values.append(array)
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values).astype(np.float64, copy=False)


def _collect_method_runs(root: Path, method: str) -> list[RunStats]:
    if not root.exists():
        return []

    runs: list[RunStats] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = run_dir.name
        if method == "low_rank":
            if not any(name.startswith(prefix) for prefix in LOW_RANK_PREFIXES):
                continue
        else:
            if not name.startswith(QUANT_PREFIX):
                continue

        x_value = _load_run_parameter(run_dir, method)
        if x_value is None:
            continue

        kl_values = _load_kl_values(run_dir)
        if kl_values.size == 0:
            continue

        runs.append(
            RunStats(
                method=method,
                x_value=x_value,
                run_dir=run_dir,
                prompt_count=len(list(run_dir.glob("*_kl.pt"))),
                values={
                    label: float(np.percentile(kl_values, percentile))
                    for label, percentile in PERCENTILES.items()
                },
            )
        )

    runs.sort(key=lambda run: (run.x_value, run.run_dir.name))
    deduped: dict[float, RunStats] = {}
    for run in runs:
        current = deduped.get(run.x_value)
        if current is None or run.prompt_count > current.prompt_count:
            deduped[run.x_value] = run
    return [deduped[key] for key in sorted(deduped)]


def _plot_method(ax, runs: list[RunStats], title: str, color_map: dict[str, str]) -> None:
    if not runs:
        ax.set_title(f"{title} (no runs found)")
        ax.set_xlabel("Rank / parameter value")
        ax.set_ylabel("KL")
        ax.grid(alpha=0.25)
        return

    x = [run.x_value for run in runs]
    for label in PERCENTILES:
        y = [run.values[label] for run in runs]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=label,
            color=color_map[label],
        )

    ax.set_title(title)
    ax.set_xlabel("Rank / parameter value")
    ax.set_ylabel("KL")
    ax.grid(alpha=0.25)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot KL percentile summaries across low-rank and quant experiment runs. "
            "The script scans experiment folders, reads saved KL tensors, and plots "
            "median / p90 / p99 / p99.9 for each recorded rank or quant-bit value."
        )
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=EXPERIMENTS_ROOT,
        help=f"Experiment root directory (default: {EXPERIMENTS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <experiments-root>/qwen_kl_rank_summary.png",
    )
    args = parser.parse_args()

    output_path = args.output or (args.experiments_root / "qwen_kl_rank_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    low_rank_runs = _collect_method_runs(args.experiments_root, "low_rank")
    quant_runs = _collect_method_runs(args.experiments_root, "quant")

    if not low_rank_runs and not quant_runs:
        raise SystemExit(
            f"No matching experiment runs were found under {args.experiments_root}. "
            "Expected run folders like `suite_kl_*` or `suite_quant_*` with saved "
            "`*_kl.pt` tensors and `*_metrics_summary.json` summaries."
        )

    colors = {
        "median": "#1b9e77",
        "p90": "#d95f02",
        "p99": "#7570b3",
        "p99.9": "#e7298a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_method(axes[0], low_rank_runs, "Qwen Low-Rank KL vs Rank", colors)
    _plot_method(axes[1], quant_runs, "Qwen Quant KL vs Bits", colors)
    fig.suptitle("Qwen KL Percentiles Across Experiment Settings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved plot to {output_path}")
    for method, runs in (("low_rank", low_rank_runs), ("quant", quant_runs)):
        if not runs:
            print(f"{method}: no runs found")
            continue
        print(f"{method}:")
        for run in runs:
            metrics = ", ".join(f"{k}={v:.6g}" for k, v in run.values.items())
            print(
                f"  x={run.x_value:g} prompts={run.prompt_count} "
                f"dir={run.run_dir.name} :: {metrics}"
            )


if __name__ == "__main__":
    main()

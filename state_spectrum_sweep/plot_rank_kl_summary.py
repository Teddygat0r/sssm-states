from __future__ import annotations

"""
Plot KL summary statistics across low-rank experiment runs.

This script supports two workflows:

1. Legacy auto-scan mode for Qwen-style runs that saved `*_metrics_summary.json`
   with `low_rank_rank`.
2. Explicit run mapping mode for Jamba-style runs that only saved `*_meta.json`
   and `*_kl.pt`. In this mode, pass `--run <rank>=<run_dir>` for each rank.

The plot aggregates all saved `*_kl.pt` tensors in each run directory and shows
KL percentiles vs rank.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
LOW_RANK_PREFIXES = (
    "suite_kl_",
    "suite_kl_parallel_",
    "suite_qwen36_27b_",
    "jamba_low_rank_",
)
PERCENTILES = {
    "mean": None,
    "median": 50.0,
    "p90": 90.0,
    "p99": 99.0,
    "p99.9": 99.9,
}


@dataclass
class RunStats:
    x_value: float
    run_dir: Path
    prompt_count: int
    values: dict[str, float]
    label: str
    model_name: str | None = None


def _iter_summary_paths(run_dir: Path) -> Iterable[Path]:
    return sorted(run_dir.glob("*_metrics_summary.json"))


def _iter_meta_paths(run_dir: Path) -> Iterable[Path]:
    return sorted(run_dir.glob("*_meta.json"))


def _load_run_parameter(run_dir: Path) -> float | None:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = None
        if isinstance(config, dict):
            for key in ("low_rank_rank", "rank"):
                value = config.get(key)
                if isinstance(value, (int, float)):
                    return float(value)

    for summary_path in _iter_summary_paths(run_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "low_rank_rank" in summary:
            return float(summary["low_rank_rank"])
    
    match = re.search(r"rank(\d+(?:\.\d+)?)", run_dir.name)
    if match:
        return float(match.group(1))
    return None

def _load_model_name(run_dir: Path) -> str | None:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = None
        if isinstance(config, dict):
            model_name = config.get("model_name")
            if isinstance(model_name, str) and model_name:
                return model_name

    for summary_path in _iter_summary_paths(run_dir):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        model_name = summary.get("model_name")
        if isinstance(model_name, str) and model_name:
            return model_name

    name = run_dir.name.lower()
    if name.startswith("suite_qwen36_27b_") or "qwen36_27b" in name:
        return "Qwen3.6-27B"
    if name.startswith("jamba_low_rank_") or "jamba" in name:
        return "Jamba"
    if name.startswith("suite_kl_") or name.startswith("suite_kl_parallel_"):
        return "Qwen3.5-4B"
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


def _summarize_kl_values(kl_values: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"mean": float(np.mean(kl_values))}
    for label, percentile in PERCENTILES.items():
        if percentile is None:
            continue
        out[label] = float(np.percentile(kl_values, percentile))
    return out


def _parse_run_mapping(value: str) -> tuple[float, Path]:
    rank_text, sep, run_dir_text = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value {value!r}. Expected format <rank>=<run_dir>."
        )
    try:
        rank = float(rank_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid rank {rank_text!r} in --run {value!r}."
        ) from exc
    return rank, Path(run_dir_text).expanduser()


def _load_jamba_label(run_dir: Path) -> str:
    for meta_path in _iter_meta_paths(run_dir):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        subset = meta.get("subset")
        target_layer = meta.get("target_layer")
        generate_from = meta.get("generate_from")
        pieces = []
        if subset:
            pieces.append(f"subset={subset}")
        if target_layer is not None:
            pieces.append(f"layer={target_layer}")
        if generate_from:
            pieces.append(f"gen={generate_from}")
        if pieces:
            return ", ".join(pieces)
    return run_dir.name


def _collect_explicit_runs(run_mappings: list[tuple[float, Path]]) -> list[RunStats]:
    runs: list[RunStats] = []
    for rank, run_dir in run_mappings:
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")

        kl_values = _load_kl_values(run_dir)
        if kl_values.size == 0:
            raise SystemExit(f"No non-empty `*_kl.pt` tensors found in: {run_dir}")

        runs.append(
            RunStats(
                x_value=rank,
                run_dir=run_dir,
                prompt_count=len(list(run_dir.glob("*_kl.pt"))),
                values=_summarize_kl_values(kl_values),
                label=_load_jamba_label(run_dir),
                model_name=_load_model_name(run_dir),
            )
        )
    runs.sort(key=lambda run: (run.x_value, run.run_dir.name))
    return runs

def collect_run_stats_for_directories(run_dirs: list[Path]) -> list[RunStats]:
    runs: list[RunStats] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")

        x_value = _load_run_parameter(run_dir)
        if x_value is None:
            raise SystemExit(
                f"Could not infer low-rank rank for run directory: {run_dir}. "
                "Add `low_rank_rank` to run metadata or include `rank<N>` in the folder name."
            )

        kl_values = _load_kl_values(run_dir)
        if kl_values.size == 0:
            raise SystemExit(f"No non-empty `*_kl.pt` tensors found in: {run_dir}")

        runs.append(
            RunStats(
                x_value=x_value,
                run_dir=run_dir,
                prompt_count=len(list(run_dir.glob("*_kl.pt"))),
                values=_summarize_kl_values(kl_values),
                label=run_dir.name,
                model_name=_load_model_name(run_dir),
            )
        )
    runs.sort(key=lambda run: (run.x_value, run.run_dir.name))
    return runs

def _collect_auto_runs(root: Path) -> list[RunStats]:
     return _collect_auto_runs_filtered(root, include_substrings=[])


def _collect_auto_runs_filtered(root: Path, include_substrings: list[str]) -> list[RunStats]:
    if not root.exists():
        return []

    runs: list[RunStats] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(run_dir.name.startswith(prefix) for prefix in LOW_RANK_PREFIXES):
            continue
        if include_substrings and not all(substr in run_dir.name for substr in include_substrings):
            continue

        x_value = _load_run_parameter(run_dir)
        if x_value is None:
            continue

        kl_values = _load_kl_values(run_dir)
        if kl_values.size == 0:
            continue

        runs.append(
            RunStats(
                x_value=x_value,
                run_dir=run_dir,
                prompt_count=len(list(run_dir.glob("*_kl.pt"))),
                values=_summarize_kl_values(kl_values),
                label=run_dir.name,
                model_name=_load_model_name(run_dir),
            )
        )

    runs.sort(key=lambda run: (run.x_value, run.run_dir.name))
    deduped: dict[float, RunStats] = {}
    for run in runs:
        current = deduped.get(run.x_value)
        if current is None or run.prompt_count > current.prompt_count:
            deduped[run.x_value] = run
    return [deduped[key] for key in sorted(deduped)]


def _plot_runs(ax, runs: list[RunStats], title: str, color_map: dict[str, str]) -> None:
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
    ax.set_xlabel("Low-rank rank")
    ax.set_ylabel("KL")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

def _resolve_plot_title(
    base_title: str,
    runs: list[RunStats],
    explicit_model_label: str | None,
) -> str:
    if explicit_model_label:
        return f"{base_title} ({explicit_model_label})"
    model_names = sorted({run.model_name for run in runs if run.model_name})
    if not model_names:
        return base_title
    if len(model_names) == 1:
        return f"{base_title} ({model_names[0]})"
    return f"{base_title} ({', '.join(model_names)})"

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot KL percentile summaries across low-rank runs. "
            "Use --run <rank>=<run_dir> for Jamba runs that only saved `*_meta.json`."
        )
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=EXPERIMENTS_ROOT,
        help=f"Experiment root directory for auto-scan mode (default: {EXPERIMENTS_ROOT})",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=_parse_run_mapping,
        default=[],
        help="Explicit run mapping in the form <rank>=<run_dir>. Repeat once per rank.",
    )
    parser.add_argument(
        "--title",
        default="Low-Rank KL vs Rank",
        help="Plot title.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help=(
            "Optional explicit model label to append to the title, for example "
            "`--model-label Qwen3.6-27B`."
        ),
    )
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help=(
            "Only include auto-scanned run directories whose names contain this substring. "
            "Repeat to require multiple substrings, for example "
            "`--name-contains qwen36_27b --name-contains gguf`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <experiments-root>/low_rank_kl_rank_summary.png",
    )
    args = parser.parse_args()

    output_path = args.output or (args.experiments_root / "low_rank_kl_rank_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = (
        _collect_explicit_runs(args.runs)
        if args.runs
        else _collect_auto_runs_filtered(args.experiments_root, args.name_contains)
    )
    if not runs:
        raise SystemExit(
            "No matching low-rank runs were found. "
            "For Jamba runs, pass explicit mappings like "
            "`--run 16=/path/to/run16 --run 8=/path/to/run8 --run 4=/path/to/run4`. "
            "For auto-scan mode, you can narrow the selection with "
            "`--name-contains qwen36_27b`."
        )

    colors = {
        "mean": "#1b9e77",
        "median": "#d95f02",
        "p90": "#7570b3",
        "p99": "#e7298a",
        "p99.9": "#66a61e",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    plot_title = _resolve_plot_title(args.title, runs, args.model_label)
    _plot_runs(ax, runs, plot_title, colors)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved plot to {output_path}")
    for run in runs:
        metrics = ", ".join(f"{k}={v:.6g}" for k, v in run.values.items())
        print(
            f"rank={run.x_value:g} prompts={run.prompt_count} "
            f"dir={run.run_dir} label={run.label} :: {metrics}"
        )


if __name__ == "__main__":
    main()

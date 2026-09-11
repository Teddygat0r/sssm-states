"""
Plot the deliverables for Sections 1.1 + 1.2 from per-model run dirs.

Usage:
    python plot_results.py \
        --qwen35  state_spectrum_sweep/experiments/svd_sweep_qwen35_<ts> \
        --mamba2  state_spectrum_sweep/experiments/svd_sweep_mamba2_<ts> \
        --nemotron state_spectrum_sweep/experiments/svd_sweep_nemotron_<ts> \
        [--out plots]

Produces, per model and aggregate:
  Section 1.1
    1) per-layer effective-rank heatmap  (rows=layer, cols=position; one per input)
    2) cross-layer / cross-input violin of effective rank
    3) SSM-vs-random spectrum overlay (mean over (layer, head) at fixed input/pos)
  Section 1.2
    4) reconstruction error vs rank curves with error bars (mean +/- IQR)
    5) CDF of per-head relative-Frobenius error at rank 16  (p99 / max annotated)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


K_GRID = (1, 2, 4, 8, 16, 32, 64)


# -----------------------------------------------------------------------------
# loaders
# -----------------------------------------------------------------------------

def _load_metrics_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_run(run_dir: Path) -> dict:
    rows = _load_metrics_csv(run_dir / "metrics.csv")
    spectra = torch.load(run_dir / "spectra.pt", map_location="cpu", weights_only=False)
    return {"rows": rows, "spectra": spectra, "dir": run_dir}


def _filter_pos(rows: list[dict], position: int | None = None,
                input_id: str | None = None) -> list[dict]:
    out = rows
    if position is not None:
        out = [r for r in out if int(r["position"]) == position]
    if input_id is not None:
        out = [r for r in out if r["input_id"] == input_id]
    return out


# -----------------------------------------------------------------------------
# 1.1 (a) -- per-layer heatmap
# -----------------------------------------------------------------------------

def plot_effrank_heatmap(rows: list[dict], model_label: str, out: Path) -> None:
    inputs = sorted({r["input_id"] for r in rows})
    positions = sorted({int(r["position"]) for r in rows}, key=lambda p: (p == -1, p))
    layers = sorted({int(r["layer"]) for r in rows})
    fig, axes = plt.subplots(1, len(inputs), figsize=(5.6 * len(inputs), 6.5),
                             sharey=True, squeeze=False)
    for ax, inp in zip(axes[0], inputs):
        grid = np.full((len(layers), len(positions)), np.nan)
        for i, li in enumerate(layers):
            for j, pos in enumerate(positions):
                vals = [_to_float(r["num_rank990"]) for r in rows
                        if r["input_id"] == inp and int(r["layer"]) == li
                        and int(r["position"]) == pos]
                if vals:
                    grid[i, j] = float(np.mean(vals))
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(["full" if p == -1 else str(p) for p in positions])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_title(f"{model_label}  ({inp})")
        ax.set_xlabel("position")
        if ax is axes[0, 0]:
            ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, label="effective rank @ 99% energy (mean over heads)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1.1 (b) -- violin plot of effective rank
# -----------------------------------------------------------------------------

def plot_effrank_violin(rows_by_model: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = []
    labels = []
    for model_label, rows in rows_by_model.items():
        for inp in sorted({r["input_id"] for r in rows}):
            sub = [_to_float(r["num_rank990"])
                   for r in rows if r["input_id"] == inp]
            sub = [v for v in sub if v == v]
            if sub:
                data.append(sub)
                labels.append(f"{model_label}\n{inp}")
    if not data:
        plt.close(fig); return
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("effective rank @ 99% energy (per (layer, head))")
    ax.set_title("Cross-layer / cross-input distribution of effective state rank")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1.1 (c) -- SSM-vs-random spectrum overlay
# -----------------------------------------------------------------------------

def plot_spectrum_overlay(spectra: dict, model_label: str, out: Path) -> None:
    """Plot mean +/- band of (sv / sv_max) curves for state vs random control."""
    state = spectra["state"]   # [N, max_rank]
    rand = spectra["random"]
    if state.numel() == 0:
        return
    # Normalize by per-row max (top singular value).
    def _norm(M):
        m = M[:, :1].clamp_min(1e-30)
        return (M / m)
    Sn = _norm(state).to(torch.float64)
    Rn = _norm(rand).to(torch.float64)
    ranks = np.arange(1, Sn.shape[1] + 1)
    Smean = Sn.nanmean(dim=0).numpy()
    Rmean = Rn.nanmean(dim=0).numpy()
    Sq25 = torch.nanquantile(Sn, 0.25, dim=0).numpy()
    Sq75 = torch.nanquantile(Sn, 0.75, dim=0).numpy()
    Rq25 = torch.nanquantile(Rn, 0.25, dim=0).numpy()
    Rq75 = torch.nanquantile(Rn, 0.75, dim=0).numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(ranks, Smean, color="C0", lw=2, label="SSM state (mean)")
    ax.fill_between(ranks, Sq25, Sq75, color="C0", alpha=0.2, label="SSM state IQR")
    ax.plot(ranks, Rmean, color="C3", lw=2, ls="--", label="random control (mean)")
    ax.fill_between(ranks, Rq25, Rq75, color="C3", alpha=0.2, label="random IQR")
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma_k / sigma_1")
    ax.set_title(f"{model_label}: SSM state vs random-matrix spectrum")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1.2 (a) -- reconstruction error vs rank
# -----------------------------------------------------------------------------

def plot_error_vs_rank(rows_by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ks = list(K_GRID)
    for ax, metric, ylabel, log in [
        (axes[0], "rel_fro_mse", "relative Frobenius MSE", True),
        (axes[1], "max_abs",      "per-element max |error|", True),
    ]:
        for model_label, rows in rows_by_model.items():
            means = []
            q25 = []
            q75 = []
            for k in ks:
                vals = np.array([_to_float(r[f"{metric}_k{k}"]) for r in rows], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    means.append(np.nan); q25.append(np.nan); q75.append(np.nan); continue
                means.append(float(np.mean(vals)))
                q25.append(float(np.quantile(vals, 0.25)))
                q75.append(float(np.quantile(vals, 0.75)))
            ax.plot(ks, means, marker="o", label=model_label)
            ax.fill_between(ks, q25, q75, alpha=0.2)
        ax.set_xscale("log", base=2)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("truncation rank k")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Truncated-SVD reconstruction error vs rank (mean +/- IQR over (model, input, pos, layer, head))")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1.2 (b) -- CDF at rank 16
# -----------------------------------------------------------------------------

def plot_cdf_at_k16(rows_by_model: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_label, rows in rows_by_model.items():
        vals = np.array([_to_float(r["rel_fro_mse_k16"]) for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        s = np.sort(vals)
        cdf = np.arange(1, s.size + 1) / s.size
        ax.plot(s, cdf, lw=1.6, label=model_label)
        for q, lbl in [(0.99, "p99"), (1.0, "max")]:
            v = float(np.quantile(s, q)) if q < 1.0 else float(s[-1])
            ax.axvline(v, ls=":", color=ax.lines[-1].get_color(), alpha=0.6)
            ax.text(v, 0.02, f"{model_label} {lbl}={v:.2e}",
                    rotation=90, fontsize=7, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("relative Frobenius MSE at rank 16  (per (layer, head))")
    ax.set_ylabel("CDF")
    ax.set_title("Per-head reconstruction-error CDF at rank 16")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen35", type=Path, default=None)
    ap.add_argument("--mamba2", type=Path, default=None)
    ap.add_argument("--nemotron", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "plots")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    for label, p in [("Qwen3.5-4B", args.qwen35),
                      ("Mamba2-1.3B", args.mamba2),
                      ("Nemotron-3-Nano-4B", args.nemotron)]:
        if p is None:
            continue
        runs[label] = _load_run(p)
        print(f"loaded {label}: {len(runs[label]['rows'])} rows from {p}")

    if not runs:
        raise SystemExit("no runs provided -- pass at least one of --qwen35/--mamba2/--nemotron")

    rows_by_model = {label: r["rows"] for label, r in runs.items()}

    # 1.1 (a)
    for label, r in runs.items():
        plot_effrank_heatmap(r["rows"], label,
                             args.out / f"effrank_heatmap_{label.replace('/', '_')}.png")
    # 1.1 (b)
    plot_effrank_violin(rows_by_model, args.out / "effrank_violin.png")
    # 1.1 (c)
    for label, r in runs.items():
        plot_spectrum_overlay(r["spectra"], label,
                              args.out / f"spectrum_overlay_{label.replace('/', '_')}.png")
    # 1.2 (a)
    plot_error_vs_rank(rows_by_model, args.out / "error_vs_rank.png")
    # 1.2 (b)
    plot_cdf_at_k16(rows_by_model, args.out / "cdf_at_k16.png")

    print(f"plots written to {args.out}")


if __name__ == "__main__":
    main()

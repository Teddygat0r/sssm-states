"""Overlay mean cumulative unsquared singular-value mass from saved spectra.

Run from the repository root:
    .venv/bin/python reconstruction_experiments/plot_cumulative_overlay.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from make_paper_figures import COLORS, LABELS, MODELS, OUT, latest_dir


def mean_cumulative_mass(mat, chunk_size=4096):
    """Match fig2_cumulative while bounding temporary memory usage."""
    total = torch.zeros(mat.shape[1], dtype=torch.float64)
    count = 0
    for chunk in mat.split(chunk_size):
        valid = ~torch.isnan(chunk[:, 0]) & (chunk[:, 0] > 0)
        values = chunk[valid].to(torch.float64)
        cumulative = values.cumsum(dim=1)
        fractions = cumulative / cumulative[:, -1:].clamp_min(1e-30)
        total += fractions.sum(dim=0)
        count += len(values)
    if count == 0:
        raise ValueError("No valid spectra")
    mean = (total / count).numpy()
    assert np.isfinite(mean).all()
    assert np.all(np.diff(mean) >= -1e-12)
    assert np.isclose(mean[-1], 1.0)
    return mean, count


def main():
    torch.set_num_threads(4)
    fig, ax = plt.subplots(figsize=(10, 6))
    rows = []
    handles = []
    for model in MODELS:
        path = latest_dir(model) / "spectra.pt"
        print(f"Reading {path}", flush=True)
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for key, style in [("state", "-"), ("random", "--")]:
            mean, count = mean_cumulative_mass(blob[key])
            ranks = np.arange(1, len(mean) + 1)
            ax.plot(ranks, mean, color=COLORS[model], linestyle=style,
                    linewidth=2.1 if key == "state" else 1.6,
                    alpha=1.0 if key == "state" else 0.8)
            rows.extend((model, key, int(k), float(y), count, str(path))
                        for k, y in zip(ranks, mean))
            print(f"  {key}: {count} spectra, n={len(mean)}; checks passed", flush=True)
        handles.append(Line2D([], [], color=COLORS[model], lw=2,
                              label=f"{LABELS[model]} (n={len(mean)})"))
        del blob

    handles.extend([
        Line2D([], [], color="black", lw=2, linestyle="-", label="SSM state"),
        Line2D([], [], color="black", lw=1.6, linestyle="--", label="Random control"),
    ])
    ax.set(xlabel="Number of singular values kept, $k$",
           ylabel=r"Mean cumulative singular-value mass  $\sum_{i=1}^{k}\sigma_i / \sum_{i=1}^{n}\sigma_i$",
           title="Cumulative singular-value mass across models",
           xlim=(1, max(row[2] for row in rows)), ylim=(0, 1.025))
    ax.grid(alpha=0.22)
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = OUT / f"fig2_cumulative_overlay.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Wrote {path}", flush=True)
    plt.close(fig)
    with (OUT / "fig2_cumulative_overlay.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "kind", "rank", "mean_cumulative_mass", "num_spectra", "source"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()

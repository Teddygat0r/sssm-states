"""Aggregate per-layer KL from a kl_layer_sweep run and plot for ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

RUN_DIR = Path(
    "state_spectrum_sweep/experiments/suite_kl_layer_sweep_20260325_103013"
)
OUT_DIR = Path("ablations/layerwise-kl")


def collect() -> dict[int, dict[str, np.ndarray]]:
    """Return {layer_idx: {'per_prompt_mean': [...], 'all_tokens': [...]}}."""
    per_layer: dict[int, dict[str, np.ndarray]] = {}
    for layer_dir in sorted(RUN_DIR.glob("layer_*")):
        layer_idx = int(layer_dir.name.split("_")[1])
        means: list[float] = []
        all_tokens: list[np.ndarray] = []
        for summary_path in sorted(layer_dir.glob("prompt_*_metrics_summary.json")):
            with open(summary_path) as f:
                s = json.load(f)
            if not s.get("kl"):
                continue
            kl_path = summary_path.with_name(summary_path.name.replace("_metrics_summary.json", "_kl.pt"))
            kl = torch.load(kl_path, weights_only=True, map_location="cpu").float()
            # bfloat16 noise can make true-zero KL slightly negative; clamp at 0.
            kl = kl.clamp_min(0.0).numpy()
            means.append(float(kl.mean()))
            all_tokens.append(kl)
        per_layer[layer_idx] = {
            "per_prompt_mean": np.array(means),
            "all_tokens": np.concatenate(all_tokens),
        }
    return per_layer


def main() -> None:
    data = collect()
    layers = np.array(sorted(data.keys()))

    mean_of_means = np.array([data[l]["per_prompt_mean"].mean() for l in layers])
    p25 = np.array([np.quantile(data[l]["per_prompt_mean"], 0.25) for l in layers])
    p75 = np.array([np.quantile(data[l]["per_prompt_mean"], 0.75) for l in layers])
    p99_token = np.array([np.quantile(data[l]["all_tokens"], 0.99) for l in layers])

    with open(RUN_DIR / "run_config.json") as f:
        cfg = json.load(f)
    rank = cfg["low_rank_rank"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(
        layers, p25, p75,
        alpha=0.25, color="C0", linewidth=0, label="IQR across prompts",
    )
    ax.plot(layers, mean_of_means, marker="o", color="C0", label="mean KL (avg over prompts & tokens)")
    ax.plot(layers, p99_token, marker="x", linestyle="--", color="C3", label="p99 across tokens")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e-2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(2.0, 5.0), numticks=10)
    )
    ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_xlabel("Layer index (compressed in isolation)")
    ax.set_ylabel(r"KL$(P_{\mathrm{orig}}\,\|\,P_{\mathrm{approx}})$")
    ax.set_title(f"Per-layer KL under rank-{rank} compression")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()

    png_path = OUT_DIR / "kl_per_layer_ablation.png"
    pdf_path = OUT_DIR / "kl_per_layer_ablation.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")

    csv_path = OUT_DIR / "kl_per_layer_summary.csv"
    with open(csv_path, "w") as f:
        f.write("layer,mean_kl,p25_kl,p75_kl,p99_token_kl\n")
        for i, l in enumerate(layers):
            f.write(f"{l},{mean_of_means[i]:.3e},{p25[i]:.3e},{p75[i]:.3e},{p99_token[i]:.3e}\n")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

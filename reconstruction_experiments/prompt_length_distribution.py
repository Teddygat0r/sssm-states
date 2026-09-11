"""Extract the per-prompt token-length distribution from each model's run.

Each (prompt, position=-1) row carries the prompt's full token count after the
MAX_FULL_TOKENS=65536 cap, tokenized by that model's own tokenizer (so counts
differ across models for the same source text). Prompts shorter than the
largest fixed position (4096) don't get a `-1` task; for those we fall back to
the longest actual_tokens seen for that prompt_id.
"""
from __future__ import annotations
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["mamba2", "nemotron", "qwen35", "deltanet", "gated_deltanet"]
LABELS = {
    "mamba2": "Mamba2-1.3B",
    "nemotron": "Nemotron-3-Nano-4B",
    "qwen35": "Qwen3.5-4B",
    "deltanet": "DeltaNet-1.3B",
    "gated_deltanet": "GatedDeltaNet-1.3B",
}
COLORS = {
    "mamba2": "#d62728", "nemotron": "#2ca02c", "qwen35": "#1f77b4",
    "deltanet": "#9467bd", "gated_deltanet": "#ff7f0e",
}
CAP = 65536


def latest(model):
    return Path(sorted(glob.glob(str(RESULTS / f"recon_{model}_*")))[-1])


def lengths(model):
    """Return {prompt_id: (input_kind, max_actual_tokens)}."""
    df = pd.read_csv(latest(model) / "metrics.csv",
                     usecols=["prompt_id", "position", "actual_tokens", "input_kind"])
    # for each prompt, the longest actual_tokens seen (= full-cap if -1 row present, else the largest fixed position that fit)
    g = df.groupby("prompt_id").agg(actual_tokens=("actual_tokens", "max"),
                                    input_kind=("input_kind", "first"))
    return g.reset_index()


def stats(s, label):
    if len(s) == 0:
        return
    pctl = lambda q: int(np.quantile(s, q))
    print(f"  {label:8s}  n={len(s):>4}  "
          f"min={s.min():>5}  p10={pctl(0.1):>5}  med={int(np.median(s)):>5}  "
          f"mean={int(s.mean()):>5}  p90={pctl(0.9):>5}  max={s.max():>6}  "
          f"hit_cap({CAP})={int((s>=CAP).sum())}")


def main():
    print("=" * 90)
    print("Per-prompt token-length distribution (longest captured position per prompt)")
    print("=" * 90)
    per_model = {}
    for m in MODELS:
        print(f"\n{LABELS[m]}:")
        g = lengths(m)
        per_model[m] = g
        stats(g["actual_tokens"].values, "all")
        for k in sorted(g["input_kind"].unique()):
            stats(g.loc[g["input_kind"] == k, "actual_tokens"].values, k)

    # histogram: 2 panels (chat vs code), 5 models overlaid
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, kind, title in zip(axes, ["chat", "code"],
                               ["ShareGPT (chat)", "Code (codeparrot)"]):
        for m in MODELS:
            g = per_model[m]
            vals = g.loc[g["input_kind"] == kind, "actual_tokens"].values
            bins = np.logspace(np.log10(max(vals.min(), 100)), np.log10(CAP * 1.05), 30)
            ax.hist(vals, bins=bins, color=COLORS[m], alpha=0.45,
                    label=f"{LABELS[m]}  (median {int(np.median(vals))})",
                    histtype="step", lw=2.0)
        ax.axvline(CAP, color="k", ls=":", lw=0.9, alpha=0.6)
        ax.text(CAP, 0.95, f"cap = {CAP}", transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=8, color="k", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Token Count (Full Prompt)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    axes[0].set_ylabel("Number of Prompts", fontsize=11)
    fig.suptitle("Per-Prompt Token-Length Distribution by Tokenizer", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prompt_lengths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT/'fig_prompt_lengths.png'}")


if __name__ == "__main__":
    main()

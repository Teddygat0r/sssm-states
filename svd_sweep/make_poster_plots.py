"""Generate two poster plots:

  1. poster_recon_error_vs_rank.png  — rel-Frob reconstruction error
     vs SVD rank, three SSM models (Qwen3.5-4B, Mamba2-1.3B,
     Nemotron-3-Nano-4B); mean and p99 curves from the n=200 sweep.
  2. poster_kl_vs_rank.png           — per-token KL(full || low-rank)
     vs SVD rank, Qwen3.5-4B; mean and p99 of per-prompt max-step KL,
     plus Q4 / Q8 reference lines.

Sources:
  * /sssm-states/svd_sweep/report_n200.md (numbers transcribed below)
  * /sssm-states/state_spectrum_sweep/experiments/suite_kl_parallel_rank*_*
  * /sssm-states/state_spectrum_sweep/experiments/suite_quant_20260328_*bit
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent / "state_spectrum_sweep" / "experiments"
OUT_DIR = HERE / "plots"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Plot 1: reconstruction error (rel-Frob) vs rank
# --------------------------------------------------------------------------- #
# Numbers come straight from report_n200.md (n=200 sweep, three models).
RANKS = np.array([4, 8, 16, 32])

# rel-Frob MSE p50 / p99 per model
recon = {
    "Qwen3.5-4B (max rank 128)": {
        "p50":  np.array([3.299e-02, 1.082e-02, 2.136e-03, 1.844e-04]),
        "mean": np.array([6.409e-02, 2.905e-02, 1.033e-02, 2.469e-03]),
        "p99":  np.array([3.747e-01, 2.269e-01, 1.060e-01, 3.022e-02]),
    },
    "Mamba2-1.3B (max rank 64)": {
        "p50":  np.array([9.570e-03, 1.676e-03, 1.237e-04, 2.226e-06]),
        "mean": np.array([2.716e-02, 1.032e-02, 3.123e-03, 5.187e-04]),
        "p99":  np.array([2.561e-01, 1.361e-01, 5.316e-02, 1.057e-02]),
    },
    "Nemotron-3-Nano-4B (max rank 80)": {
        "p50":  np.array([1.439e-03, 3.501e-04, 5.599e-05, 4.716e-06]),
        "mean": np.array([5.713e-03, 1.874e-03, 4.971e-04, 8.345e-05]),
        "p99":  np.array([5.897e-02, 2.186e-02, 6.198e-03, 1.062e-03]),
    },
}

colors = {
    "Qwen3.5-4B (max rank 128)":      "#1f77b4",
    "Mamba2-1.3B (max rank 64)":      "#d62728",
    "Nemotron-3-Nano-4B (max rank 80)": "#2ca02c",
}

fig, ax = plt.subplots(figsize=(7.5, 5.0))
for label, d in recon.items():
    c = colors[label]
    ax.plot(RANKS, d["mean"], "o-",  color=c, label=f"{label}  (mean)",  lw=2.2, ms=7)
    ax.plot(RANKS, d["p99"],  "s--", color=c, label=f"{label}  (p99)",   lw=1.6, ms=6, alpha=0.85)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(RANKS)
ax.set_xticklabels([str(r) for r in RANKS])
ax.set_xlabel("SVD truncation rank $k$", fontsize=12)
ax.set_ylabel(r"Relative Frobenius error  $\Vert S-\hat S\Vert_F / \Vert S\Vert_F$", fontsize=12)
ax.set_title("State reconstruction error vs. rank\n(200 ShareGPT + 200 code prompts; per (layer, head, position) record)", fontsize=12)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower left", fontsize=9, ncol=1, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT_DIR / "poster_recon_error_vs_rank.png", dpi=200)
plt.close(fig)
print(f"wrote {OUT_DIR/'poster_recon_error_vs_rank.png'}")


# --------------------------------------------------------------------------- #
# Plot 2: KL(full || low-rank) vs rank, Qwen3.5-4B, 30 prompts × 200 tokens
# --------------------------------------------------------------------------- #
RANK_DIRS = {
    1:  "suite_kl_parallel_rank1_20260417_023441",
    2:  "suite_kl_parallel_rank2_20260417_024459",
    4:  "suite_kl_parallel_rank4_20260417_025521",
    6:  "suite_kl_parallel_rank6_20260417_030538",
    8:  "suite_kl_parallel_rank8_20260417_031552",
    10: "suite_kl_parallel_rank10_20260417_033019",
    12: "suite_kl_parallel_rank12_20260417_034451",
    16: "suite_kl_parallel_rank16_20260417_035951",
    24: "suite_kl_parallel_rank24_20260417_041809",
    32: "suite_kl_parallel_rank32_20260417_043638",
}
QUANT_DIRS = {
    "Q4 (4-bit state quant)": "suite_quant_20260328_4bit",
    "Q8 (8-bit state quant)": "suite_quant_20260328_8bit",
}


def _read_kl(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-prompt mean-KL, per-prompt max-KL)."""
    means, maxes = [], []
    for f in sorted(run_dir.glob("prompt_*_metrics_summary.json")):
        j = json.loads(f.read_text())
        means.append(j["kl"]["mean"])
        maxes.append(j["kl"]["max"])
    return np.array(means), np.array(maxes)


ranks_sorted = sorted(RANK_DIRS)
mean_of_means = []
p99_of_maxes  = []
for r in ranks_sorted:
    means, maxes = _read_kl(EXP_ROOT / RANK_DIRS[r])
    mean_of_means.append(means.mean())
    p99_of_maxes.append(np.percentile(maxes, 99))
mean_of_means = np.array(mean_of_means)
p99_of_maxes  = np.array(p99_of_maxes)

quant_lines = {}
for label, d in QUANT_DIRS.items():
    means, maxes = _read_kl(EXP_ROOT / d)
    quant_lines[label] = (means.mean(), np.percentile(maxes, 99))


fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.plot(ranks_sorted, mean_of_means, "o-",  color="#1f77b4", lw=2.4, ms=7,
        label="SVD low-rank state  (mean per-token KL)")
ax.plot(ranks_sorted, p99_of_maxes,  "s--", color="#1f77b4", lw=1.6, ms=6, alpha=0.85,
        label="SVD low-rank state  (p99 of worst-step KL)")

# Quant baselines as horizontal reference lines
ref_colors = {"Q4 (4-bit state quant)": "#d62728", "Q8 (8-bit state quant)": "#ff7f0e"}
for label, (mean_kl, p99_kl) in quant_lines.items():
    c = ref_colors[label]
    ax.axhline(mean_kl, color=c, ls="-",  lw=1.6, alpha=0.9, label=f"{label}  (mean = {mean_kl:.2e})")
    ax.axhline(p99_kl,  color=c, ls=":",  lw=1.2, alpha=0.7)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ranks_sorted)
ax.set_xticklabels([str(r) for r in ranks_sorted])
ax.set_xlabel("SVD truncation rank $k$ on per-head recurrent state (max rank = 128)", fontsize=12)
ax.set_ylabel(r"Per-token KL", fontsize=12)
ax.set_title("Output-distribution fidelity vs. state compression rank\n"
             "Qwen3.5-4B, 30 prompts × 200 decode tokens", fontsize=12)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT_DIR / "poster_kl_vs_rank.png", dpi=200)
plt.close(fig)
print(f"wrote {OUT_DIR/'poster_kl_vs_rank.png'}")

print("\nSVD KL curve (mean / p99-of-max):")
for r, m, p in zip(ranks_sorted, mean_of_means, p99_of_maxes):
    print(f"  rank={r:>3}  mean_kl={m:.3e}  p99_max_kl={p:.3e}")
for label, (m, p) in quant_lines.items():
    print(f"  {label:25s}  mean_kl={m:.3e}  p99_max_kl={p:.3e}")

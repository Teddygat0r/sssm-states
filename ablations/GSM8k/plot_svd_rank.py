"""Square figure: SVD rank vs GSM8K accuracy for Qwen3.5-4B recurrent-state
compression. Sized square so it sits inline next to the results table.

Reads gsm8k_results.csv; writes gsm8k_svd_rank.{png,pdf}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "gsm8k_results.csv"
FIG_PNG = HERE / "gsm8k_svd_rank.png"
FIG_PDF = HERE / "gsm8k_svd_rank.pdf"


def main() -> None:
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["exact_match_strict"].notna() & (df["exact_match_strict"] != "")]
    df["exact_match_strict"] = df["exact_match_strict"].astype(float)
    df["param"] = df["param"].astype(int)
    df = df.sort_values("timestamp").drop_duplicates("name", keep="last")

    svd = df[df["method"] == "svd"].sort_values("param")
    quant = df[df["method"] == "quant"].sort_values("param")
    baseline = df[df["method"] == "baseline"]["exact_match_strict"]
    base_y = float(baseline.iloc[0]) if not baseline.empty else None

    # Place each b-bit quant point at its iso-storage equivalent SVD rank:
    # SVD rank r stores r*(D_k+D_v+1) values/head; b-bit quant stores the
    # equivalent of D_k*D_v*b/16 bf16 values/head. Equating -> r_eq = 3.98*b.
    D_K = D_V = 128
    NATIVE_BITS = 16  # bf16
    quant["r_eq"] = quant["param"] * (D_K * D_V) / (NATIVE_BITS * (D_K + D_V + 1))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    if base_y is not None:
        ax.axhline(
            base_y, color="grey", linestyle="--", linewidth=1.2,
            label=f"uncompressed (bf16) = {base_y:.3f}",
        )

    ax.plot(
        svd["param"], svd["exact_match_strict"],
        marker="o", markersize=6, color="tab:blue", linewidth=1.8,
        label="SVD low-rank state",
    )

    ax.scatter(
        quant["r_eq"], quant["exact_match_strict"],
        marker="s", s=70, color="tab:orange", zorder=4,
        label="naive quant (at equal-storage rank)",
    )
    for _, row in quant.iterrows():
        ax.annotate(
            f"q{int(row['param'])}",
            (row["r_eq"], row["exact_match_strict"]),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=9, color="tab:orange",
        )

    xticks = sorted(set(svd["param"].tolist() + [32]))
    ax.set_xlabel("SVD rank $r$  (quant at equal-storage rank)")
    ax.set_ylabel("GSM8K exact-match (strict)")
    ax.set_title("Recurrent-state rank vs. accuracy", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(0.70, 0.88)
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"wrote {FIG_PNG}\nwrote {FIG_PDF}")


if __name__ == "__main__":
    main()

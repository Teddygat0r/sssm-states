"""Plot GSM8K accuracy vs effective state-storage cost across compression methods.

Reads `gsm8k_results.csv` produced by run_gsm8k_compression.py.
Writes:
    gsm8k_summary.csv        — tidy table with derived bits-per-element column
    gsm8k_ablation.png/.pdf  — accuracy vs effective bits/element figure
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Qwen3.5-4B linear-attention head geometry.
D_K = 128
D_V = 128
# fp32 baseline reference (used to compute the relative compression ratio).
FP32_BITS = 32.0


HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "gsm8k_results.csv"
SUMMARY_CSV = HERE / "gsm8k_summary.csv"
FIG_PNG = HERE / "gsm8k_ablation.png"
FIG_PDF = HERE / "gsm8k_ablation.pdf"


def _effective_bits(method: str, param: int) -> float:
    """Effective bits per fp32 element after compression.

    SVD r=r stores U (D_k * r) + S (r) + Vh (r * D_v) fp32 values per head;
    we normalize by D_k * D_v elements.
    Quant stores n_bits per element plus negligible scale/zp overhead.
    Hadamard quant has the same storage cost as plain quant (rotation
    matrices are constants, not stored per state).
    """
    if method == "baseline":
        return FP32_BITS
    if method in ("quant", "hadamard_quant"):
        return float(param)
    if method == "svd":
        r = int(param)
        n_factored = r * (D_K + D_V + 1)
        n_dense = D_K * D_V
        return FP32_BITS * n_factored / n_dense
    raise ValueError(f"unknown method: {method}")


def _load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"missing {RESULTS_CSV} — run the sweep first")
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["exact_match_strict"].notna() & (df["exact_match_strict"] != "")]
    df["exact_match_strict"] = df["exact_match_strict"].astype(float)
    df["exact_match_flex"] = pd.to_numeric(df["exact_match_flex"], errors="coerce")
    df["param"] = df["param"].astype(int)
    # If a config was re-run, keep the most recent row.
    df = df.sort_values("timestamp").drop_duplicates("name", keep="last")
    df["effective_bits"] = df.apply(
        lambda r: _effective_bits(r["method"], r["param"]), axis=1
    )
    df["compression_ratio"] = FP32_BITS / df["effective_bits"]
    return df.sort_values(["method", "param"]).reset_index(drop=True)


def _write_summary(df: pd.DataFrame) -> None:
    cols = [
        "name",
        "method",
        "param",
        "exact_match_strict",
        "exact_match_flex",
        "effective_bits",
        "compression_ratio",
        "n_samples",
    ]
    df[cols].to_csv(SUMMARY_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {SUMMARY_CSV}")


def _plot(df: pd.DataFrame) -> None:
    baseline = df[df["method"] == "baseline"]
    svd = df[df["method"] == "svd"].sort_values("param")
    quant = df[df["method"] == "quant"].sort_values("param")
    hada = df[df["method"] == "hadamard_quant"].sort_values("param")

    fig, ax = plt.subplots(figsize=(8, 5.2))

    if not baseline.empty:
        y = float(baseline["exact_match_strict"].iloc[0])
        ax.axhline(
            y,
            color="grey",
            linestyle="--",
            linewidth=1.2,
            label=f"baseline (fp32 state) = {y:.3f}",
        )

    if not svd.empty:
        ax.plot(
            svd["effective_bits"], svd["exact_match_strict"],
            marker="o", color="tab:blue", label="SVD low-rank",
        )
        for _, row in svd.iterrows():
            ax.annotate(
                f"r={int(row['param'])}",
                (row["effective_bits"], row["exact_match_strict"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="tab:blue",
            )

    if not quant.empty:
        ax.scatter(
            quant["effective_bits"], quant["exact_match_strict"],
            marker="s", color="tab:orange", s=80, label="naive per-head quant", zorder=3,
        )
        for _, row in quant.iterrows():
            ax.annotate(
                f"{int(row['param'])}b",
                (row["effective_bits"], row["exact_match_strict"]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
                color="tab:orange",
            )

    if not hada.empty:
        ax.scatter(
            hada["effective_bits"], hada["exact_match_strict"],
            marker="D", color="tab:green", s=80, label="Hadamard-rotated quant", zorder=3,
        )
        for _, row in hada.iterrows():
            ax.annotate(
                f"H+{int(row['param'])}b",
                (row["effective_bits"], row["exact_match_strict"]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
                color="tab:green",
            )

    ax.set_xlabel("Effective bits per state element (fp32 = 32)")
    ax.set_ylabel("GSM8K exact-match (strict)")
    ax.set_title(
        "Qwen3.5-4B GSM8K accuracy vs recurrent-state compression\n"
        "(5-shot, 1319 samples, greedy)"
    )
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"wrote {FIG_PNG}\nwrote {FIG_PDF}")


def main() -> None:
    df = _load_results()
    print(df.to_string(index=False))
    _write_summary(df)
    _plot(df)


if __name__ == "__main__":
    main()

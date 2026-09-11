"""
Figure for §3.2 (Output Quality Preservation).

One panel:
  x = rank (log, 1..32)
  y = KL (log)
  solid  lines = median KL
  dashed lines = p99    KL
  colors = model (Nemotron, Qwen3.5)
  marker = context regime (256, full)

Input:   results/kl_summary.csv  (already has median + p99 columns)
Output:  figures/fig_3_2_quality_preservation.{pdf,png}

Usage:
    ../.venv/bin/python plot_figure_3_2.py
    ../.venv/bin/python plot_figure_3_2.py --include-pure   # adds faded pure-SSM lines
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results" / "kl_summary.csv"
OUTDIR = HERE / "figures"

HYBRIDS = ("Nemotron-3-Nano-4B", "Qwen3.5-4B")
PURE = ("Mamba2-1.3B", "DeltaNet-1.3B", "GatedDeltaNet-1.3B")
SHORT = {
    "Nemotron-3-Nano-4B": "Nemotron",
    "Qwen3.5-4B": "Qwen3.5",
    "Mamba2-1.3B": "Mamba2",
    "DeltaNet-1.3B": "DeltaNet",
    "GatedDeltaNet-1.3B": "GDN",
}
# distinct, colorblind-friendly: blue and orange for the two hybrids;
# pure-SSM (when shown) gets desaturated greys.
COLORS = {
    "Nemotron-3-Nano-4B": "#1f77b4",
    "Qwen3.5-4B":         "#d62728",
    "Mamba2-1.3B":        "#888888",
    "DeltaNet-1.3B":      "#aaaaaa",
    "GatedDeltaNet-1.3B": "#666666",
}
# 256-token prefix = circle marker; full prefix = square marker
MARKERS = {"256": "o", "-1": "s"}
CTX_LABEL = {"256": "256-tok ctx", "-1": "full ctx"}


def load_summary(path: Path):
    """Returns {(model, position): {rank: (median, p99)}}."""
    out: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["position"])
            out.setdefault(key, {})[int(r["rank"])] = (
                float(r["kl_median"]), float(r["kl_p99"]))
    return out


def _plot_series(ax, ranks, vals, *, color, marker, ls, lw, label,
                 alpha=1.0, markersize=5.5):
    ax.plot(ranks, vals, color=color, marker=marker, linestyle=ls,
            linewidth=lw, markersize=markersize, alpha=alpha, label=label,
            markerfacecolor=color, markeredgecolor=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-pure", action="store_true",
                    help="Overlay faded pure-SSM lines for context.")
    ap.add_argument("--no-p99", action="store_true",
                    help="Plot medians only (cleaner for talk slides).")
    ap.add_argument("--side-by-side", action="store_true",
                    help="Square-ish, median-only figure tuned for side-by-side "
                         "placement next to a table. Implies --no-p99.")
    args = ap.parse_args()
    if args.side_by_side:
        args.no_p99 = True

    data = load_summary(SUMMARY)
    OUTDIR.mkdir(exist_ok=True)

    if args.side_by_side:
        figsize = (4.2, 4.0)
        line_lw = 1.8
        marker_sz = 6.0
        font_axis = 10
        font_tick = 9
        font_leg = 8
        title = None
    else:
        figsize = (6.4, 4.4)
        line_lw = 2.0
        marker_sz = 5.5
        font_axis = 11
        font_tick = 10
        font_leg = 8.5
        title = "Output quality vs. compression rank (hybrids)"
        if args.include_pure:
            title += " — pure-SSM faded"

    fig, ax = plt.subplots(figsize=figsize)

    # ---- pure SSM in the background, if requested ----
    if args.include_pure:
        for m in PURE:
            for pos in ("256", "-1"):
                d = data.get((m, pos), {})
                if not d:
                    continue
                ranks = sorted(d)
                med = [d[r][0] for r in ranks]
                _plot_series(ax, ranks, med,
                             color=COLORS[m], marker=MARKERS[pos],
                             ls="-", lw=1.0, alpha=0.45,
                             markersize=marker_sz,
                             label=f"{SHORT[m]} ({CTX_LABEL[pos]})")

    # ---- hybrids in the foreground ----
    for m in HYBRIDS:
        for pos in ("256", "-1"):
            d = data.get((m, pos), {})
            if not d:
                continue
            ranks = sorted(d)
            med = [d[r][0] for r in ranks]
            p99 = [d[r][1] for r in ranks]
            label_med = f"{SHORT[m]} ({CTX_LABEL[pos]})"
            _plot_series(ax, ranks, med,
                         color=COLORS[m], marker=MARKERS[pos],
                         ls="-", lw=line_lw, markersize=marker_sz,
                         label=label_med)
            if not args.no_p99:
                _plot_series(ax, ranks, p99,
                             color=COLORS[m], marker=MARKERS[pos],
                             ls="--", lw=1.1, alpha=0.85, label=None,
                             markersize=marker_sz)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Compression rank $r$", fontsize=font_axis)
    ax.set_ylabel(r"Median KL$(P_{\mathrm{full}} \,\Vert\, P_{\mathrm{rank-}r})$"
                  if args.no_p99
                  else r"KL$(P_{\mathrm{full}} \,\Vert\, P_{\mathrm{rank-}r})$",
                  fontsize=font_axis)
    ax.tick_params(axis="both", labelsize=font_tick)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    # Two-part legend: model/ctx lines (handled above) + a tiny inset legend
    # describing solid=median, dashed=p99 (only if both are drawn).
    leg1 = ax.legend(loc="upper right", fontsize=font_leg, framealpha=0.92,
                     handlelength=2.4, borderpad=0.5)
    ax.add_artist(leg1)

    if not args.no_p99:
        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([0], [0], color="black", lw=2.0, ls="-",  label="median"),
            Line2D([0], [0], color="black", lw=1.1, ls="--", label="p99"),
        ]
        ax.legend(handles=style_handles, loc="lower left",
                  fontsize=font_leg, framealpha=0.92, handlelength=2.4)

    if title is not None:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    suffix = ""
    if args.include_pure:
        suffix += "_with_pure"
    if args.side_by_side:
        suffix += "_sbs"
    elif args.no_p99:
        suffix += "_medianonly"
    base = OUTDIR / f"fig_3_2_quality_preservation{suffix}"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()

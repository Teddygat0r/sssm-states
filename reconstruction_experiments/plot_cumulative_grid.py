"""Render three panels above two centered panels from the saved overlay curve data.

Run plot_cumulative_overlay.py first if its CSV is not available.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from make_paper_figures import COLORS, LABELS, MODELS, OUT


def main():
    data = pd.read_csv(OUT / "fig2_cumulative_overlay.csv")
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 6, left=0.075, right=0.985, bottom=0.09,
                           top=0.9, wspace=1.15, hspace=0.4)
    slots = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6],
             grid[1, 1:3], grid[1, 3:5]]
    axes = [fig.add_subplot(slot) for slot in slots]
    for ax, model in zip(axes, MODELS):
        for kind, style, color, label in [
            ("state", "-", COLORS[model], "SSM state"),
            ("random", "--", "#888888", "Random control"),
        ]:
            curve = data[(data.model == model) & (data.kind == kind)].sort_values("rank")
            assert len(curve) > 0, (model, kind)
            ax.plot(curve["rank"], curve.mean_cumulative_mass,
                    linestyle=style, color=color, linewidth=2, label=label)
        n = int(curve["rank"].max())
        ax.set(title=f"{LABELS[model]} (n={n})",
               xlabel="Number of singular values kept, $k$",
               xlim=(1, n), ylim=(0, 1.03))
        ax.axhline(1, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=9)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Mean cumulative singular-value mass\n"
                      r"$\sum_{i=1}^{k}\sigma_i / \sum_{i=1}^{n}\sigma_i$")
    fig.suptitle("Cumulative singular-value mass", fontsize=16)
    for extension in ("png", "svg"):
        output = OUT / f"fig2_cumulative_2rows_centered.{extension}"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Wrote {output}")
    assert len(axes) == 5
    top_center = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
    bottom_center = (axes[3].get_position().x0 + axes[4].get_position().x1) / 2
    assert abs(top_center - bottom_center) < 1e-10
    print("PASS: five panels; bottom pair centered under top three")
    plt.close(fig)


if __name__ == "__main__":
    main()

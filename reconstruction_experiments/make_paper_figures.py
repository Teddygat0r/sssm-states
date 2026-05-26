"""
Paper figures for the state-level reconstruction-fidelity sweep (5 models).

Reads the latest results/recon_<model>_*/{metrics.csv, spectra.pt} for each
model and emits:

  fig1_reconstruction_and_effrank.png  (main, 2 panels)
      (a) relative-Frobenius error vs SVD rank k   (median + p99)
      (b) effective rank @99% energy, normalized by D, state vs random control
  fig2_spectra.png                     (appendix) per-model singular-spectrum overlay
  fig3_context_length.png              (appendix) rel-Frob @k=16 vs sequence position
  fig4_depth.png                       (appendix) effective rank vs relative depth
  table1_summary.{md,csv}              (main) headline numbers per model

Reconstruction columns come from randomized svd_lowrank (the deployed algo);
spectrum / effective-rank / random-control columns are exact SVD.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

K_GRID = (1, 2, 4, 8, 16, 32, 64)
K_RECON = (1, 2, 4, 8, 16, 32)   # ranks shown in fig1(a); rank 64 dropped per request

# model key -> display label; ordered by state dim for clean comparison
MODELS = ["mamba2", "nemotron", "qwen35", "deltanet", "gated_deltanet"]
LABELS = {
    "mamba2": "Mamba2-1.3B",
    "nemotron": "Nemotron-3-Nano-4B",
    "qwen35": "Qwen3.5-4B",
    "deltanet": "DeltaNet-1.3B",
    "gated_deltanet": "GatedDeltaNet-1.3B",
}
COLORS = {
    "mamba2": "#d62728",
    "nemotron": "#2ca02c",
    "qwen35": "#1f77b4",
    "deltanet": "#9467bd",
    "gated_deltanet": "#ff7f0e",
}


def latest_dir(model: str) -> Path:
    ds = sorted(glob.glob(str(RESULTS / f"recon_{model}_*")))
    if not ds:
        raise FileNotFoundError(f"no results dir for {model}")
    return Path(ds[-1])


def load_model_summary(model: str) -> dict:
    """One pass over a model's metrics.csv -> compact summary for all figures."""
    d = latest_dir(model)
    cols = (["position", "layer", "max_rank", "D_k", "D_v",
             "num_rank990", "num_rank990_random"]
            + [f"rel_fro_mse_k{k}" for k in K_GRID]
            + [f"cos_flat_k{k}" for k in K_GRID])
    print(f"  reading {d.name} ...", flush=True)
    df = pd.read_csv(d / "metrics.csv", usecols=cols)
    D = int(df["max_rank"].iloc[0])
    D_k, D_v = int(df["D_k"].iloc[0]), int(df["D_v"].iloc[0])

    # (a) reconstruction error percentiles per rank
    rel_med = {k: float(df[f"rel_fro_mse_k{k}"].median()) for k in K_GRID}
    rel_p99 = {k: float(df[f"rel_fro_mse_k{k}"].quantile(0.99)) for k in K_GRID}
    cos_med = {k: float(df[f"cos_flat_k{k}"].median()) for k in K_GRID}

    # (b) normalized effective rank, state vs control (subsample for violins)
    eff_state = (df["num_rank990"].to_numpy() / D)
    eff_rand = (df["num_rank990_random"].to_numpy() / D)
    rng = np.random.default_rng(0)
    n_sub = min(20000, len(eff_state))
    idx = rng.choice(len(eff_state), size=n_sub, replace=False)

    # fig3: rel-Frob @k=16 by position
    pos_group = df.groupby("position")["rel_fro_mse_k16"].median()

    # fig4: normalized eff rank by layer
    lay = df.groupby("layer")["num_rank990"].median() / D

    summary = dict(
        model=model, label=LABELS[model], color=COLORS[model],
        D=D, D_k=D_k, D_v=D_v,
        rel_med=rel_med, rel_p99=rel_p99, cos_med=cos_med,
        eff_state_sub=eff_state[idx], eff_rand_sub=eff_rand[idx],
        eff_state_med=float(np.median(eff_state)),
        eff_state_p99=float(np.quantile(eff_state, 0.99)),
        eff_rand_med=float(np.median(eff_rand)),
        num_rank990_med=int(np.median(df["num_rank990"])),
        num_rank990_p99=int(np.quantile(df["num_rank990"], 0.99)),
        num_rank990_rand_med=int(np.median(df["num_rank990_random"])),
        cos16_mean=float(df["cos_flat_k16"].mean()),
        rel16_med=float(df["rel_fro_mse_k16"].median()),
        rel16_p99=float(df["rel_fro_mse_k16"].quantile(0.99)),
        pos_group=pos_group, lay_group=lay,
    )
    del df
    return summary


def fig1(summ: dict):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- (a) reconstruction error vs rank ---
    ks = np.array(K_RECON)
    for m in MODELS:
        s = summ[m]
        c = s["color"]
        axA.plot(ks, [s["rel_med"][k] for k in K_RECON], "o-", color=c, lw=2.0, ms=6,
                 label=s["label"])
        axA.plot(ks, [s["rel_p99"][k] for k in K_RECON], "--", color=c, lw=1.2, alpha=0.7)
    axA.set_xscale("log", base=2); axA.set_yscale("log")
    axA.set_xticks(ks); axA.set_xticklabels([str(k) for k in ks])
    axA.set_xlabel("SVD Rank $k$", fontsize=12)
    axA.set_ylabel(r"Relative Frobenius Error  $\Vert S-\hat S\Vert_F^2/\Vert S\Vert_F^2$", fontsize=12)
    axA.set_title("(a) Reconstruction Error vs. Rank", fontsize=12)
    axA.grid(True, which="both", alpha=0.25)
    # legend: models + linestyle key
    handles, labels = axA.get_legend_handles_labels()
    style_med = plt.Line2D([], [], color="k", ls="-", marker="o", label="Median")
    style_p99 = plt.Line2D([], [], color="k", ls="--", label="p99")
    axA.legend(handles + [style_med, style_p99], labels + ["Median", "p99"],
               fontsize=8.5, loc="lower left", ncol=1, framealpha=0.92)

    # --- (b) effective rank normalized, state vs control ---
    positions = []
    state_data = []
    rand_data = []
    xticklabels = []
    for i, m in enumerate(MODELS):
        s = summ[m]
        positions.append(2 * i + 1 - 0.32)   # state
        positions.append(2 * i + 1 + 0.32)   # control
        state_data.append(s["eff_state_sub"])
        rand_data.append(s["eff_rand_sub"])
        xticklabels.append(f"{s['label'].split('-')[0]}\n(D={s['D']})")

    # interleave for plotting
    all_data, all_pos, all_colors = [], [], []
    for i, m in enumerate(MODELS):
        s = summ[m]
        all_data.append(s["eff_state_sub"]); all_pos.append(2 * i + 1 - 0.32); all_colors.append(s["color"])
        all_data.append(s["eff_rand_sub"]);  all_pos.append(2 * i + 1 + 0.32); all_colors.append("#888888")
    vp = axB.violinplot(all_data, positions=all_pos, widths=0.55,
                        showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], all_colors):
        body.set_facecolor(col); body.set_edgecolor("black"); body.set_alpha(0.75)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.2)
    axB.set_xticks([2 * i + 1 for i in range(len(MODELS))])
    axB.set_xticklabels(xticklabels, fontsize=8.5)
    axB.set_ylabel("Effective Rank @ 99% Energy  /  $D$", fontsize=12)
    axB.set_ylim(0, 1.05)
    axB.set_title("(b) State Spectrum Concentration vs. Random Control", fontsize=12)
    axB.grid(True, axis="y", alpha=0.25)
    axB.legend([plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", alpha=0.75),
                plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.75)],
               ["SSM State", "Random Control"], fontsize=9, loc="center left")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_reconstruction_and_effrank.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig1_reconstruction_and_effrank.png'}")


def fig2_cumulative():
    """Cumulative fraction of the singular-value sum vs. number of components n,
    as a horizontal row of 5 small multiples (one per model). Each curve grows
    to 1.0 at n=D. Fast-rising = low-rank (state); slow = high-rank (control)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 4.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = latest_dir(m)
        blob = torch.load(d / "spectra.pt")
        D = None
        for key, ls, lw, col, lab in [("state", "-", 2.2, COLORS[m], "SSM State"),
                                      ("random", "--", 1.5, "#888888", "Random Control")]:
            mat = blob[key].to(torch.float64)               # [N, D]
            valid = ~torch.isnan(mat[:, 0]) & (mat[:, 0] > 0)
            mv = mat[valid]
            csum = torch.cumsum(mv, dim=1)
            frac = csum / csum[:, -1:].clamp_min(1e-30)      # -> 1.0 at n=D for every row
            mean_frac = frac.mean(dim=0).numpy()
            D = len(mean_frac)
            n = np.arange(1, D + 1)
            ax.plot(n, mean_frac, ls, color=col, lw=lw, label=lab)
        ax.set_ylim(0, 1.03)
        ax.set_xlim(left=1)
        ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{LABELS[m]}\n(D={D})", fontsize=11)
        ax.set_xlabel("Number of Singular Values Kept, $n$", fontsize=10.5)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
        del blob, mat, mv
    axes[0].set_ylabel(r"Cumulative Singular-Value Mass" "\n" r"$\sum_{i\leq n}\sigma_i \,/\, \sum_i \sigma_i$",
                       fontsize=11)
    fig.suptitle("Cumulative Singular-Value Mass vs. Rank (Solid = SSM State, Dashed = Random Control)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cumulative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig2_cumulative.png'}")


def fig3_context_length(summ: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fixed = [256, 512, 1024, 2048, 4096]
    for m in MODELS:
        s = summ[m]
        pg = s["pos_group"]
        xs = [p for p in fixed if p in pg.index]
        ys = [pg[p] for p in xs]
        ax.plot(xs, ys, "o-", color=s["color"], lw=2.0, ms=6, label=s["label"])
        if -1 in pg.index:   # "full" position, plot as a trailing marker
            ax.plot([8192], [pg[-1]], "*", color=s["color"], ms=13, mec="black", mew=0.4)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(fixed + [8192]); ax.set_xticklabels([str(p) for p in fixed] + ["Full"])
    ax.set_xlabel("Sequence Position (Prefill Length, Tokens)", fontsize=12)
    ax.set_ylabel(r"Relative Frobenius Error @ Rank 16 (Median)", fontsize=12)
    ax.set_title("Reconstruction Error vs. Context Length (Rank Fixed at 16)", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_context_length.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig3_context_length.png'}")


def fig4_depth(summ: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m in MODELS:
        s = summ[m]
        lg = s["lay_group"].sort_index()
        layers = lg.index.to_numpy().astype(float)
        rel_depth = (layers - layers.min()) / max(1.0, (layers.max() - layers.min()))
        ax.plot(rel_depth, lg.to_numpy(), "o-", color=s["color"], lw=1.8, ms=5, label=s["label"])
    ax.set_xlabel("Relative Depth (Layer Index, Normalized)", fontsize=12)
    ax.set_ylabel("Effective Rank @ 99% / $D$ (Median per Layer)", fontsize=12)
    ax.set_title("Where the Rank Lives: Effective Rank vs. Depth", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper center", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_depth.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig4_depth.png'}")


def table1(summ: dict):
    rows = []
    for m in MODELS:
        s = summ[m]
        mem_red = 100.0 * (1.0 - 16.0 * (s["D_k"] + s["D_v"]) / (s["D_k"] * s["D_v"]))
        rows.append({
            "Model": s["label"],
            "State (D_k x D_v)": f"{s['D_k']}x{s['D_v']}",
            "Eff.rank@99% (med)": s["num_rank990_med"],
            "Eff.rank@99% (p99)": s["num_rank990_p99"],
            "Control eff.rank (med)": s["num_rank990_rand_med"],
            "cos@k16 (mean)": round(s["cos16_mean"], 4),
            "relFro@k16 (med)": f"{s['rel16_med']:.2e}",
            "relFro@k16 (p99)": f"{s['rel16_p99']:.2e}",
            "Mem.reduction@k16": f"{mem_red:.1f}%",
        })
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT / "table1_summary.csv", index=False)
    md = "| " + " | ".join(tdf.columns) + " |\n"
    md += "|" + "|".join(["---"] * len(tdf.columns)) + "|\n"
    for _, r in tdf.iterrows():
        md += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    (OUT / "table1_summary.md").write_text(md)
    print(f"wrote {OUT/'table1_summary.md'} and .csv\n")
    print(md)


def main():
    print("Loading per-model summaries (one pass over each metrics.csv) ...")
    summ = {m: load_model_summary(m) for m in MODELS}
    print("\nRendering figures ...")
    fig1(summ)
    fig3_context_length(summ)
    fig4_depth(summ)
    table1(summ)
    fig2_cumulative()   # last: heaviest (loads spectra.pt)
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()

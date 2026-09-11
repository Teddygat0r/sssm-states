"""Generate a poster-ready table of baseline vs. truncated-SVD eval scores,
grouped by model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "plots" / "poster_eval_table.png"


# (task, baseline_pct, svd_pct).  All values shown as percentages.
rows_by_model: dict[str, list[tuple[str, float, float]]] = {
    "Qwen3.5-4B": [
        ("GSM8k",            84.38, 83.85),
    ],
    "Qwen3.5-35B": [
        ("MMLU-Pro",         84.67, 85.01),
        ("GPQA Diamond",     81.82, 81.19),
        ("LongBench v2",     55.50, 57.50),
        ("SWE-Bench",        56.00, 56.20),
    ],
    "Nemotron-3-Nano-4B": [
        ("MMLU-Pro",         76.27, 76.48),
        ("GPQA Diamond",     67.68, 66.67),
        ("LongBench v2",     42.90, 42.90),
    ],
}


# Flatten into table rows. Insert a model header row before each group.
header = ["Model", "Task", "Baseline (%)", "Truncated SVD (%)"]
table_rows: list[list[str]] = []
for model, items in rows_by_model.items():
    for i, (task, base, svd) in enumerate(items):
        table_rows.append([
            model if i == 0 else "",
            task,
            f"{base:.2f}",
            f"{svd:.2f}",
        ])


n_rows = len(table_rows)
n_cols = len(header)

fig_h = 0.55 + 0.42 * (n_rows + 1)
fig, ax = plt.subplots(figsize=(9.0, fig_h))
ax.axis("off")

table = ax.table(
    cellText=table_rows,
    colLabels=header,
    cellLoc="center",
    loc="center",
    colWidths=[0.26, 0.22, 0.20, 0.24],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.55)

# --- styling: monochrome ---
# Header row
for c in range(n_cols):
    cell = table[0, c]
    cell.set_facecolor("white")
    cell.set_edgecolor("black")
    cell.get_text().set_weight("bold")

# Body rows
for r in range(n_rows):
    is_model_header = table_rows[r][0] != ""
    for c in range(n_cols):
        cell = table[r + 1, c]
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        if c == 0 and is_model_header:
            cell.get_text().set_weight("bold")

ax.set_title(
    "Downstream eval scores: baseline vs. truncated-SVD recurrent-state compression",
    fontsize=13, weight="bold", pad=12,
)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")


# Also emit a markdown version for inline use
md_lines = [
    "| Model | Task | Baseline (%) | Truncated SVD (%) |",
    "|---|---|---:|---:|",
]
for row in table_rows:
    md_lines.append("| " + " | ".join(row) + " |")
print("\n" + "\n".join(md_lines))

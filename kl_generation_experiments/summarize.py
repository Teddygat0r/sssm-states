"""
Aggregate kl_metrics.csv files into the headline tables.

Reads every results/kl_*/kl_metrics.csv (or paths passed on argv), and reports
KL distribution stats per (model, context position, rank). Writes
results/kl_summary.csv and prints a readable table.

Usage:
    python summarize.py                 # all results/kl_*/
    python summarize.py results/kl_mamba2_*/kl_metrics.csv
"""
from __future__ import annotations

import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(math.ceil(p / 100.0 * len(s))) - 1))
    return s[i]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    args = sys.argv[1:]
    if args:
        paths = []
        for a in args:
            paths.extend(glob.glob(a))
    else:
        paths = glob.glob(str(RESULTS / "kl_*" / "kl_metrics.csv"))
    if not paths:
        print("no kl_metrics.csv found")
        return

    kl = defaultdict(list)       # (model, position, rank) -> [kl]
    rel = defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            key = (r["model"], r["position"], int(r["rank"]))
            kl[key].append(float(r["kl"]))
            rel[key].append(float(r["state_rel_fro"]))

    cols = ["model", "position", "rank", "n", "kl_mean", "kl_median",
            "kl_p90", "kl_p99", "kl_max", "rel_fro_mean"]
    rows = []
    for key in sorted(kl, key=lambda k: (k[0], k[1], k[2])):
        v = kl[key]
        rows.append({
            "model": key[0], "position": key[1], "rank": key[2], "n": len(v),
            "kl_mean": _mean(v), "kl_median": _pct(v, 50),
            "kl_p90": _pct(v, 90), "kl_p99": _pct(v, 99), "kl_max": max(v),
            "rel_fro_mean": _mean(rel[key]),
        })

    out = RESULTS / "kl_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':<18}{'pos':>6}{'rank':>5}{'n':>7}{'kl_mean':>12}"
          f"{'kl_median':>12}{'kl_p99':>12}{'rel_fro':>12}")
    for r in rows:
        pos = "full" if r["position"] in ("-1", -1) else r["position"]
        print(f"{r['model']:<18}{pos:>6}{r['rank']:>5}{r['n']:>7}"
              f"{r['kl_mean']:>12.3e}{r['kl_median']:>12.3e}"
              f"{r['kl_p99']:>12.3e}{r['rel_fro_mean']:>12.3e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

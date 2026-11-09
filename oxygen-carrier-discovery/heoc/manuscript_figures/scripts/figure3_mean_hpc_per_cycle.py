"""Figure 3: mean HPC of the compounds selected by active learning, all seed runs.

Reads data/compounds.csv. Compounds: charge neutral, Bartel tolerance <= 4.18, HPC > 0
(as plot_multiseed.py). Per run and cycle (or 5-cycle block) the mean over the run's compounds is taken; plotted is
the mean over the runs:
  black line  mean over the runs for each AL cycle (points); band = 95 % CI over the runs (t distribution)
  cycle 0     the starting set (shared by all runs)
  thin lines  each run's own per-cycle means (background), labelled at their last cycle (r1..r7 = seeds 1-5, 42, 100)
Cycles that not all runs have (empty cycles, unfinished runs) are averaged over the runs that have them.
Output: output/manuscript/Figure_3.png/pdf, Figure_3_per_cycle.csv
Usage: python figure3_mean_hpc_per_cycle.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import DATA, MANUSCRIPT, label  # noqa: E402

MANUSCRIPT.mkdir(parents=True, exist_ok=True)
BARTEL_MAX = 4.18
STEM = "Figure_3"
INK, INK2, GREY, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e4e3df"
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # r1..r7
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False})


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


df = pd.read_csv(DATA / "compounds.csv")
n_start_all = (df.run == "start").sum()
df = df[(df.charge_neutral == True) & (df.bartel_tolerance <= BARTEL_MAX) & (df.hpc > 0)]  # noqa: E712
start, al = df[df.run == "start"], df[df.run != "start"].copy()
runs = sorted(al.run.unique(), key=lambda r: int(r[1:]))           # r1..r7
n_runs = len(runs)

per_run_cycle = al.groupby(["run", "cycle"]).hpc.mean()
cyc = per_run_cycle.groupby("cycle").agg(mean="mean", ci95=ci95, n_runs="size")
cyc.to_csv(MANUSCRIPT / "Figure_3_per_cycle.csv")

fig, ax = plt.subplots(figsize=(4.2, 3.3))
ends = {}
for r, col in zip(runs, RUN_COLORS):                     # single runs in the background, thin and light
    rb = per_run_cycle.loc[r]
    x = rb.index.values.astype(float)
    ax.plot(x, rb.values, color=col, lw=0.6, alpha=0.45, zorder=1)
    ends.setdefault(x[-1] if x[-1] < 45 else 50, []).append((rb.values[-1], r, col))   # runs ending near 50: one group
for b, lst in ends.items():                              # end labels, spread apart where close
    ys = []
    for y, r, col in sorted(lst):
        ys.append(y if not ys else max(y, ys[-1] + 0.09))
        x_end = b
        ax.text(x_end + 0.8 if x_end < 45 else 50.8, ys[-1], r, color=col, fontsize=5.5,
                va="center")
ax.fill_between(cyc.index, cyc["mean"] - cyc.ci95, cyc["mean"] + cyc.ci95, color=INK, alpha=0.12, lw=0, zorder=3,
                label="95 % CI over the runs")
ax.plot(np.r_[0, cyc.index], np.r_[start.hpc.mean(), cyc["mean"]], color=INK, lw=1.6, marker="o", ms=3, zorder=4,
        label="mean of the runs, each AL cycle")
ax.annotate(f"starting set\n(cycle 0): {start.hpc.mean():.2f}", (0, start.hpc.mean()), xytext=(3, 0.25),
            fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
last = cyc.index.max()
ax.annotate(f"cycle 1: {cyc.loc[1, 'mean']:.2f}", (1, cyc.loc[1, "mean"]), xytext=(4.5, 1.15), ha="left",
            va="center", fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate(f"cycle {last}: {cyc.loc[last, 'mean']:.2f}", (last, cyc.loc[last, "mean"]), xytext=(49, 1.45),
            ha="center", va="center", fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xticks(range(0, 51, 5))
ax.set_xticks(range(0, 51), minor=True)
ax.set_xlim(-1, 54)
ax.set_ylim(0, 3.5)
ax.set_yticks(np.arange(0, 3.01, 0.5))
ax.set_xlabel("AL cycle")
ax.set_ylabel("mean HPC [mmol H$_2$ g$^{-1}$]")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.plot([], [], color=INK2, lw=0.8, alpha=0.55, label="single runs, each AL cycle")
ax.legend(frameon=False, loc="upper left", fontsize=6, labelspacing=0.3)
fig.tight_layout()
fig.savefig(MANUSCRIPT / f"{STEM}.png", dpi=300, bbox_inches="tight")
fig.savefig(MANUSCRIPT / f"{STEM}.pdf", bbox_inches="tight")

print(f"runs: {', '.join(runs)}; last cycle per run: {al.groupby('run').cycle.max().to_dict()}")
print(f"starting set: {len(start)} of {n_start_all} compounds (charge neutral, Bartel <= {BARTEL_MAX}, HPC > 0), "
      f"mean {start.hpc.mean():.3f}")
print(f"cycle 1: mean {cyc.loc[1, 'mean']:.3f} ({cyc.loc[1, 'n_runs']} runs); cycle {last}: mean "
      f"{cyc.loc[last, 'mean']:.3f} ({cyc.loc[last, 'n_runs']} runs)")
print(f"AL compounds: {len(al)}")
print(cyc.round(3).to_string())

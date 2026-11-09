"""Figure 5: A- and B-site element enrichment of the compounds selected by active learning, per run (r1-r7).

Enrichment = mean site fraction of an element in a run's AL compounds (cycles 1-50) / its mean site fraction in the
starting set (ratio > 1: enriched). Compounds: brownmillerite rows with an HPC (quadratic fit), charge neutral,
Bartel tolerance <= 4.18. Cycle 0 is the shared starting set.
Output: output/manuscript/Figure_5.png/pdf; tables in <TABLES>/figure5/: enrichment_A.csv, enrichment_B.csv
(site fractions and ratios), top_candidates.csv (the TOP_N compounds with the highest HPC per run).
Usage: python figure5_element_enrichment.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import MANUSCRIPT, TABLES, connect, label, run_list  # noqa: E402
from cycle_fix import fix_cycle  # noqa: E402
RUNS = run_list()
OUT = TABLES / "figure5"
OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT.mkdir(parents=True, exist_ok=True)
HPC_KEY = "hpp_quadratic_fit"
BARTEL_MAX = 4.18
TOP_N = 10                 # top candidates per seed
A_EL = ["Ca", "Ce", "K", "Mg", "Y", "Ba", "La", "Sm", "Sr"]
B_EL = ["Al", "Cr", "Co", "Cu", "Fe", "Mn", "Ni", "Sn", "Ti", "V", "W", "Zr", "Zn"]
N_SITE = 16                # cations per sublattice in the 80-atom cell
RUN_COLORS = dict(zip(RUNS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
})


def load_run(run):
    con = connect(run)
    recs = []
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        d = json.loads(kvp)
        d["cycle"] = fix_cycle(run, d["cycle"])
        if d["cycle"] is None:
            continue
        formula, _, struct = d["tag"].partition("_")
        # the tag carries the reduced brownmillerite formula (O40); the row is the oxidised perovskite (O48)
        formula = re.sub(r"O40(?!\d)", "O48", formula)
        if (struct != "brownmillerite" or d.get(HPC_KEY) is None
                or d.get("delta_charge_neutrality") is not True or d["bartel_tolerance"] > BARTEL_MAX):
            continue
        counts = {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
        rec = {"run": run, "cycle": int(d["cycle"]), "formula": formula, "hpc": d[HPC_KEY],
               "hpc_raw": d.get("hpp"), "bartel_tolerance": d["bartel_tolerance"],
               "goldschmidt_tolerance": d["goldschmidt_tolerance"], "price": d.get("price")}
        rec.update({f"A_{e}": counts.get(e, 0) / N_SITE for e in A_EL})
        rec.update({f"B_{e}": counts.get(e, 0) / N_SITE for e in B_EL})
        recs.append(rec)
    con.close()
    return pd.DataFrame(recs)


runs = {r: load_run(r) for r in RUNS}
base = runs[RUNS[0]].query("cycle == 0")
al = pd.concat([df[df.cycle > 0] for df in runs.values()], ignore_index=True)
for r, df in runs.items():
    print(f"{label(r)}: {(df.cycle > 0).sum()} AL compounds, last cycle {df.cycle.max()}")


# ---------------------------------------------------------------- 2. top candidates
def site_string(row, els, prefix):
    parts = [(e, round(row[f"{prefix}_{e}"] * N_SITE)) for e in els if row[f"{prefix}_{e}"] > 0]
    return " ".join(f"{e}{n}" for e, n in sorted(parts, key=lambda x: -x[1]))


top = []
for r in RUNS:
    g = runs[r][runs[r].cycle > 0].sort_values("hpc", ascending=False).head(TOP_N).copy()
    g["rank"] = range(1, len(g) + 1)
    g["A_site"] = g.apply(site_string, axis=1, els=A_EL, prefix="A")
    g["B_site"] = g.apply(site_string, axis=1, els=B_EL, prefix="B")
    top.append(g)
top = pd.concat(top)[["run", "rank", "formula", "hpc", "hpc_raw", "cycle", "A_site", "B_site",
                      "bartel_tolerance", "goldschmidt_tolerance", "price"]]
top.to_csv(OUT / "top_candidates.csv", index=False)
print(f"\nTop {TOP_N} per seed")
print(top[["run", "rank", "formula", "hpc", "hpc_raw", "cycle"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- 3. enrichment
# Column groups (labelled above the columns, separated by white gaps); within a group, columns are
# ordered by pooled enrichment.
GROUPS_A = {"alkali": ["K"], "alkaline earth": ["Ca", "Mg", "Sr", "Ba"], "rare earth": ["Ce", "La", "Sm", "Y"]}
# B-site: a group may be a list of sub-blocks; columns stay in sub-block order (no gap between sub-blocks):
# transition metals = redox-active 3d | d0 (Ti, Zr, W) | Cr; main-group metals = Al, Zn (group 12), Sn
GROUPS_B = {"transition metals": [["Mn", "Fe", "Co", "Ni", "Cu", "V"], ["Ti", "Zr", "W"], ["Cr"]],
            "post-transition\nmetals": [["Al", "Zn", "Sn"]]}
ENR_LIM = 10.0             # colour scale clipped at 1/ENR_LIM .. ENR_LIM (log scale)


def enrichment(cols, sub, ref_set=None, pooled=None):
    """Mean site fraction in each run's AL compounds (sub) / mean site fraction in the start set
    (or in ref_set). The "all runs" row is taken from pooled if given, otherwise from all of sub.
    An element absent from the reference gives inf if selected and nan if not."""
    ref = (base if ref_set is None else ref_set)[cols].mean()
    rows = {r: sub[sub.run == r][cols].mean() for r in RUNS}
    rows["all runs"] = (sub if pooled is None else pooled)[cols].mean()
    frac = pd.DataFrame(rows).T
    frac.loc["start set"] = ref
    ratio = frac.drop(index="start set") / ref
    frac.columns = ratio.columns = [c.split("_", 1)[1] for c in cols]
    return frac, ratio


def fmt_ratio(v):
    if np.isnan(v):
        return "–"          # absent from both the selection and the reference
    if np.isinf(v):
        return "new"        # absent from the reference only
    if v == 0:
        return "0"          # element never selected
    if v < 0.1:
        return "<0.1"
    return f"{v:.1f}"


def ordered_columns(groups, enr):
    cols, bounds = [], []
    for name, els in groups.items():
        blocks = els if isinstance(els[0], list) else [els]
        els = [e for blk in blocks for e in sorted(blk, key=lambda e: -enr.loc["all runs", e])]
        bounds.append((name, len(cols), len(cols) + len(els)))
        cols += els
    return cols, bounds


# column order fixed from all AL compounds so every version of the figure lines up
ORDER_A = enrichment([f"A_{e}" for e in A_EL], al)[1]
ORDER_B = enrichment([f"B_{e}" for e in B_EL], al)[1]


def plot_enrichment(sub, stem, ref_set=None, pooled=None, ref_label="start set"):
    frac_A, enr_A = enrichment([f"A_{e}" for e in A_EL], sub, ref_set, pooled)
    frac_B, enr_B = enrichment([f"B_{e}" for e in B_EL], sub, ref_set, pooled)
    n = {r: (sub.run == r).sum() for r in RUNS}
    n["all runs"] = len(sub if pooled is None else pooled)
    n_ref = len(base if ref_set is None else ref_set)
    labels = [label(r) for r in enr_A.index] + [f"{ref_label}, %"]
    norm = LogNorm(vmin=1 / ENR_LIM, vmax=ENR_LIM)
    # truncated so full depletion is a mid blue, not the darkest step (keeps the eye on enrichment)
    cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.15, 0.95, 256)))
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [len(A_EL), len(B_EL)]})
    for ax, enr, frac, groups, title in [(axes[0], enr_A, frac_A, GROUPS_A, "(a) A-site"),
                                         (axes[1], enr_B, frac_B, GROUPS_B, "(b) B-site")]:
        cols, bounds = ordered_columns(groups, ORDER_A if groups is GROUPS_A else ORDER_B)
        e = enr[cols]
        vals = np.vstack([e.clip(lower=1 / ENR_LIM, upper=ENR_LIM).values, np.full(len(cols), np.nan)])
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
        for i in range(e.shape[0]):
            for j in range(e.shape[1]):
                v = e.values[i, j]
                dark = np.log(np.clip(v, 1 / ENR_LIM, ENR_LIM)) > 0.55 * np.log(ENR_LIM)
                ax.text(j, i, fmt_ratio(v), ha="center", va="center", fontsize=5.5, color="white" if dark else INK)
        for j, el in enumerate(cols):
            ax.text(j, e.shape[0], f"{100 * frac.loc['start set', el]:.1f}", ha="center", va="center",
                    fontsize=5.5, color=INK2)
        ax.axhline(len(RUNS) - 0.5, color="white", lw=2)
        ax.axhline(e.shape[0] - 0.5, color=INK2, lw=0.6)
        for name, a, b in bounds:
            if a > 0:
                ax.axvline(a - 0.5, color="white", lw=2.5)
            ax.annotate(name, ((a + b - 1) / 2, -0.75), ha="center", va="bottom", fontsize=6, color=INK2,
                        annotation_clip=False, linespacing=1.0)
            ax.plot([a - 0.4, b - 0.6], [-0.7, -0.7], color=INK2, lw=0.6, clip_on=False)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels))
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_title(title, loc="left", pad=24)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.035, ticks=[0.1, 0.5, 1, 2, 10])
    cb.ax.set_yticklabels(["≤0.1", "0.5", "1", "2", "≥10"])
    cb.ax.minorticks_off()
    cb.set_label(f"enrichment vs {ref_label} (ratio)")
    fig.subplots_adjust(left=0.17 if ref_set is None else 0.2, right=0.86, bottom=0.1, top=0.84, wspace=0.06)
    fig.savefig(MANUSCRIPT / f"{stem}.png", dpi=300)
    fig.savefig(MANUSCRIPT / f"{stem}.pdf")
    plt.close(fig)
    return frac_A, enr_A, frac_B, enr_B


frac_A, enr_A, frac_B, enr_B = plot_enrichment(al, "Figure_5")
for name, frac, enr in [("A", frac_A, enr_A), ("B", frac_B, enr_B)]:
    pd.concat({"site_fraction": frac, "enrichment_vs_start": enr}).to_csv(OUT / f"enrichment_{name}.csv")
    print(f"\n{name}-site enrichment, all AL compounds (AL fraction / start-set fraction)")
    print(enr.round(2).to_string())
print(f"\nSaved {MANUSCRIPT}/Figure_5.png/pdf and tables in {OUT}/")

"""Figure S1: which A- and B-site elements are favoured as the steam-reactor H2O/H2 ratio changes.

The mean HPC of every element rises with the ratio (a more oxidising steam reactor increases the swing for all
compounds), so element means hide the relative changes. Here, at each ratio, the compounds are ranked by HPC
(quadratic fit) and the top TOP_FRAC are taken; enrichment of an element = (share of the top compounds in which
the element fills >= RICH of its site) / (share among all compounds). log2 enrichment > 0: favoured at that ratio.

Input: <TABLES>/figureS1/hpc_vs_ratio_all.csv (figureS1_sr_ratio_tables.py: HPC of every compound at every ratio).
Main: the starting set only (random sample; the AL compounds were selected at H2O/H2 = 4 and would bias every ratio
towards what is good at 4). Check: all compounds (start + AL).
Output: output/manuscript/Figure_S1.png/pdf (default call: starting set, ratios FIGURE_ROWS); other options write
element_enrichment_vs_sr_ratio_<set>[_sel]<suffix>.png/pdf to <TABLES>/figureS1/; the enrichment table
(all ratios) goes to <TABLES>/figureS1/element_enrichment_vs_sr_ratio_<set>.csv.
Usage: python figureS1_element_enrichment_vs_sr_ratio.py [start|all] [--rows 0.1,0.2,...|all] [--suffix S]
       (default start; --rows selects the ratios shown, default PLOT_ROWS; the CSV always has all ratios)
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import InterpolatedUnivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import MANUSCRIPT, METAL_REPO, TABLES  # noqa: E402
from paths import label  # noqa: E402
METAL = METAL_REPO / "metal" / "applications" / "crystals"
T_SR = 800 + 273.15


def log10_pO2_sr(ratio, T=T_SR):
    """Equilibrium pO2 (bar) of H2O/H2 at T, as get_mu_oxygen_sr: ln pO2 = 2 [ln(H2O/H2) - ln Kf(H2O)]."""
    d = np.loadtxt(METAL / "data/H2O_ref.txt")
    ok = np.isfinite(d[:, 7])
    lnK = float(InterpolatedUnivariateSpline(d[ok, 0], d[ok, 7] * np.log(10), k=3)(T))
    return 2.0 * (np.log(ratio) - lnK) / np.log(10)

OUT = TABLES / "figureS1"
OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT.mkdir(parents=True, exist_ok=True)
_args = sys.argv[1:]
PUBLISHED = not _args                         # default call = Figure S1
FIGURE_ROWS = [0.1, 0.2, 0.5, 1, 2, 4, 10, 200, 1000, 10000, 100000, 1000000]
SET = next((a for a in _args if a in ("start", "all")), "start")
PLOT_ROWS = [0.1, 0.2, 0.5, 1, 2, 4, 10, 200]
_rows = _args[_args.index("--rows") + 1] if "--rows" in _args else None
if PUBLISHED:
    PLOT_ROWS = FIGURE_ROWS
elif _rows == "all":
    PLOT_ROWS = None
elif _rows:
    PLOT_ROWS = [float(x) for x in _rows.split(",")]
STEM = f"element_enrichment_vs_sr_ratio_{SET}" + ("" if PLOT_ROWS is None else "_sel")
if "--suffix" in _args:
    STEM += _args[_args.index("--suffix") + 1]
TOP_FRAC, RICH, RATIO_0 = 0.10, 4, 4
A_EL = ["Ca", "Ce", "K", "Mg", "Y", "Ba", "La", "Sm", "Sr"]          # order as element_hpc_vs_sr_ratio
B_EL = ["Mn", "Fe", "Co", "Ni", "Cu", "V", "Ti", "Zr", "W", "Cr", "Al", "Zn", "Sn"]

res = pd.read_csv(OUT / "hpc_vs_ratio_all.csv")
if SET == "start":
    res = res[res.run == "start set"]
ratios = sorted(res.ratio.unique())
comp = res.drop_duplicates("formula").set_index("formula")
cnt = pd.DataFrame([{e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f)} for f in comp.index],
                   index=comp.index).reindex(columns=A_EL + B_EL).fillna(0)
has = cnt >= RICH                                                   # element fills >= 4 of 16 sites
base = has.mean()
n_comp = len(comp)
n_top = int(round(TOP_FRAC * n_comp))

rows, tops = [], {}
for r in ratios:
    h = res[res.ratio == r].set_index("formula").hpc_fit.reindex(comp.index)
    top = h.sort_values(ascending=False, kind="mergesort").index[:n_top]
    tops[r] = set(top)
    share = has.loc[top].mean()
    rows.append({"ratio": r, "hpc_cut": float(h.sort_values(ascending=False).iloc[n_top - 1]),
                 **{e: share[e] / base[e] if base[e] > 0 else np.nan for e in A_EL + B_EL}})
enr = pd.DataFrame(rows).set_index("ratio")
enr.to_csv(OUT / f"element_enrichment_vs_sr_ratio_{SET}.csv")
overlap = pd.Series({r: len(tops[r] & tops[RATIO_0]) / n_top for r in ratios})
pd.set_option("display.width", 250)
print(f"{SET}: {n_comp} compounds, top {TOP_FRAC:.0%} = {n_top}; element present if >= {RICH} of 16 sites")
print("overlap of the top set with the top set at H2O/H2 = 4:", overlap.round(2).to_dict())
print(np.log2(enr[A_EL + B_EL]).round(2).to_string())

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 7})
show = ratios if PLOT_ROWS is None else [r for r in ratios if any(abs(r - x) < 1e-9 for x in PLOT_ROWS)]
E = enr.loc[show, A_EL + B_EL]
L = np.log2(E.clip(lower=1 / 16))                                   # absent from the top -> floor at -4
vmax = 2.0
fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.6 + 0.16 * len(show)), gridspec_kw={"width_ratios": [len(A_EL), len(B_EL), 1.6]})
for ax, (site, els) in zip(axes, [("A", A_EL), ("B", B_EL)]):
    M = L[els].values
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = E[els].values[i, j]
            ax.text(j, i, "0" if v == 0 else f"{v:.1f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(M[i, j]) > 1.3 else "black")
    ax.set_xticks(range(len(els)), [f"{e}\n{int(has[e].sum())}" for e in els], fontsize=6)
    ax.set_yticks(range(len(show)), [(f"{r:.0f}" if r >= 1 else f"{r:g}") + f"  ({log10_pO2_sr(r):.1f})".replace("-", "−") for r in show] if site == "A" else [])
    ax.axhline(show.index(RATIO_0), color="black", lw=0.6, ls=":")
    ax.set_title(f"({'ab'[site == 'B']}) {site}-site", loc="left", fontsize=8)
    ax.tick_params(length=0)
axes[0].set_ylabel("H$_2$O/H$_2$ in steam reactor\n(log$_{10}$ $p_{\\mathrm{O_2}}$/bar at 800 °C)")
ax = axes[2]
ax.barh(range(len(show)), overlap[show].values, color="#888888", height=0.7)
ax.set_ylim(len(show) - 0.5, -0.5)
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_title("(c) overlap", loc="left", fontsize=8)
ax.set_xlabel(f"share of top set\nalso top at {RATIO_0}", fontsize=6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.16, aspect=50)
cb.set_label(f"enrichment in the top {TOP_FRAC:.0%} by HPC (log$_2$ scale)")
cb.set_ticks([-2, -1, 0, 1, 2], labels=["0.25", "0.5", "1", "2", "4"])
for ext in ("png", "pdf"):
    fig.savefig(MANUSCRIPT / f"Figure_S1.{ext}" if PUBLISHED else OUT / f"{STEM}.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved {'Figure_S1' if PUBLISHED else STEM}.png/.pdf and element_enrichment_vs_sr_ratio_{SET}.csv")

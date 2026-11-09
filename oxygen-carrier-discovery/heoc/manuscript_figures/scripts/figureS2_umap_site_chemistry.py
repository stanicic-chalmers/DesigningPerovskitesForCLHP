"""Figure S2: the Figure 4 UMAP map coloured by (a) HPC, (b) the dominant A-site and (c) the dominant B-site element.

Reads saved UMAP coordinates (start set + AL compounds of all runs) and writes umap_<tag>_site_chemistry.png/.pdf next
to them (<tag> from the file name).
Dominant element = the element with the most cations on that site (16 A and 16 B sites per 80-atom cell). Compounds
where two or more elements share the largest count are drawn as "tie" (light grey). At most 8 elements per site get a
colour (the 8 most frequent dominant elements among the AL compounds); the others are pooled as "other" (grey).
Usage: python figureS2_umap_site_chemistry.py [coords.csv]
       (default: output/UMAP/Projection/umap_r6_c50_7runs_frozen_coords.csv = the Figure 4 map)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import UMAP  # noqa: E402

COORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else UMAP / "Projection" / "umap_r6_c50_7runs_frozen_coords.csv"
TAG = COORDS.name.removeprefix("umap_").removesuffix("_frozen_coords.csv")
A_EL = ["Ca", "Ce", "K", "Mg", "Y", "Ba", "La", "Sm", "Sr"]
B_EL = ["Al", "Cr", "Co", "Cu", "Fe", "Mn", "Ni", "Sn", "Ti", "V", "W", "Zr", "Zn"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed order
OTHER, TIE, START_ALPHA = "#8c8c8c", "#d9d9d9", 0.55


def counts(formula):
    return {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}


df = pd.read_csv(COORDS)
comp = pd.DataFrame([counts(f) for f in df.formula]).fillna(0)
is_al = (df.run != "start").values


def dominant(elements):
    x = comp.reindex(columns=elements, fill_value=0)
    top = x.max(axis=1)
    dom = x.idxmax(axis=1).where(x.eq(top, axis=0).sum(axis=1) == 1, "tie")
    ranked = dom[is_al & (dom != "tie").values].value_counts().index[:len(PALETTE)]
    colours = dict(zip(ranked, PALETTE))
    pooled = sorted(set(elements) - set(colours))
    cat = dom.where(dom.isin(list(colours)) | (dom == "tie"), "other")
    colours.update({"other": OTHER, "tie": TIE})
    return cat, colours, pooled


plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.36, wspace=0.08)

# (a) HPC: all points, best on top
ax = axes[0]
o = df.assign(h=df.hpc.clip(lower=0)).sort_values("h")
sc = ax.scatter(o.umap1, o.umap2, s=4, c=o.h, cmap="viridis", vmin=0, vmax=df.hpc.max(), lw=0)
ax.set_title("(a) HPC", loc="left")
cax = ax.inset_axes([0.05, -0.36, 0.9, 0.05])   # below the axis, same width as panels (b) and (c)
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("HPC (mmol H$_2$ g$^{-1}$)")

for ax, (site, elements, letter) in zip(axes[1:], [("A", A_EL, "b"), ("B", B_EL, "c")]):
    cat, colours, pooled = dominant(elements)
    order = ["tie", "other"] + [k for k in colours if k not in ("tie", "other")][::-1]   # common ones drawn first
    for k in order:
        for al, alpha, s in ((False, START_ALPHA, 3), (True, 0.9, 5)):   # start set under the AL compounds
            m = (cat == k).values & (is_al == al)
            ax.scatter(df.umap1[m], df.umap2[m], s=s, color=colours[k], lw=0, alpha=alpha)
    ax.set_title(f"({letter}) dominant {site}-site element", loc="left")
    labels = [k for k in colours if k not in ("tie", "other")] + ["other", "tie"]
    names = {"other": "other (" + ", ".join(pooled) + ")" if pooled else "other", "tie": "tie"}
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=colours[k], label=names.get(k, k)) for k in labels]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
              handletextpad=0.1, columnspacing=0.6, fontsize=6)
    n = cat[is_al].value_counts()
    print(f"{site} site, AL compounds: " + ", ".join(f"{k} {n.get(k, 0)}" for k in labels) + f"; pooled: {pooled}")

for ax in axes:
    ax.set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
print(f"{len(df)} points ({(~is_al).sum()} start set, {is_al.sum()} AL): {df.run.value_counts().to_dict()}")
for ext in ("png", "pdf"):
    fig.savefig(COORDS.parent / f"umap_{TAG}_site_chemistry.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved {COORDS.parent}/umap_{TAG}_site_chemistry.png/.pdf")

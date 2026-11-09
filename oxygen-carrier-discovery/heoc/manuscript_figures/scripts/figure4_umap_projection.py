"""Figure 4: UMAP projection of the WAE latent space with the compounds of all runs, redrawn from saved coordinates.

The map (Section S2.4): every composition encoded with the WAE encoder of run r6 (the run with the highest HPC) as
trained in AL cycle 50; UMAP (15 neighbours, min_dist 0.1, Euclidean, random_state 42) fitted on the starting set +
r6 compounds and held fixed, the other runs placed with the fitted transform. Fitting needs the WAE checkpoints, which
are not part of this repository, so this script only redraws the map from the saved coordinates
(umap_<tag>_frozen_coords.csv; the fitted reducer is saved as umap_<tag>_reducer.pkl).
(a) all runs coloured by run, (b)-(h) one run each coloured by HPC, on the starting set (light) and the other runs (grey).
Output: umap_<tag>_frozen.png/pdf next to the coordinates; for the default map also output/manuscript/Figure_4.png/pdf.
Usage: python figure4_umap_projection.py [coords.csv]   (default: output/UMAP/Projection/umap_r6_c50_7runs_frozen_coords.csv)
"""
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import MANUSCRIPT, UMAP  # noqa: E402

DEFAULT = UMAP / "Projection" / "umap_r6_c50_7runs_frozen_coords.csv"
COORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
TAG = COORDS.name.removeprefix("umap_").removesuffix("_frozen_coords.csv")
m = re.search(r"_c(\d+)", TAG)
REF_CYCLE = int(m.group(1)) if m else None
RUN_COLORS = {"r1": "#2a78d6", "r2": "#eb6834", "r3": "#1baf7a", "r4": "#eda100", "r5": "#e87ba4",
              "r6": "#008300", "r7": "#6250d6"}

df = pd.read_csv(COORDS)
start, al = df[df.run == "start"], df[df.run != "start"]
SRC_COL = {r: c for r, c in RUN_COLORS.items() if r in set(al.run)}

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
HPC_MAX = al.hpc.max()                     # colour scale ends at the best HPC found by the runs
ncol = -(-(len(SRC_COL) + 1) // 2)
fig, axes = plt.subplots(2, ncol, figsize=(7.2, 4.4 if ncol == 3 else 3.9), sharex=True, sharey=True)
ax = axes.flat[0]
ax.scatter(start.umap1, start.umap2, s=3, c="#d0d0d0", lw=0, label="start set")
for r, c in SRC_COL.items():
    sub = al[al.run == r]
    ax.scatter(sub.umap1, sub.umap2, s=7, color=c, lw=0, alpha=0.85, label=r)
ax.set_title("(a) all runs", loc="left")
ax.legend(fontsize=5.5 if ncol == 3 else 5, frameon=True, framealpha=0.85, edgecolor="none", loc="lower right",
          markerscale=1.5, handletextpad=0.2, borderaxespad=0.1, ncol=1 if ncol == 3 else 2, columnspacing=0.6)
for k, (ax, r) in enumerate(zip(axes.flat[1:], SRC_COL)):
    ax.scatter(start.umap1, start.umap2, s=3, c="#e3e3e3", lw=0)
    ax.scatter(al[al.run != r].umap1, al[al.run != r].umap2, s=3, c="#bdbdbd", lw=0)
    sub = al[al.run == r].sort_values("hpc")
    sc = ax.scatter(sub.umap1, sub.umap2, s=9, c=sub.hpc.clip(lower=0), cmap="viridis", vmin=0, vmax=HPC_MAX,
                    edgecolor="black", linewidth=0.15)
    last = int(sub.cycle.max())
    ax.set_title(f"({'bcdefghij'[k]}) {r}" + (f" (to c{last})" if REF_CYCLE and last < REF_CYCLE else ""),
                 loc="left")
for ax in axes.flat[len(SRC_COL) + 1:]:
    ax.set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("UMAP 1")
for ax in axes[:, 0]:
    ax.set_ylabel("UMAP 2")
cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02 if ncol == 3 else 0.035, shrink=0.8)
cb.set_label("HPC (mmol H$_2$ g$^{-1}$)")
for ext in ("png", "pdf"):
    fig.savefig(COORDS.parent / f"umap_{TAG}_frozen.{ext}", dpi=300, bbox_inches="tight")
print(f"{len(df)} points {df.run.value_counts().to_dict()}; saved {COORDS.parent}/umap_{TAG}_frozen.png/.pdf")
if COORDS.resolve() == DEFAULT.resolve():
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copyfile(COORDS.parent / f"umap_{TAG}_frozen.{ext}", MANUSCRIPT / f"Figure_4.{ext}")
    print(f"copied to {MANUSCRIPT}/Figure_4.png/.pdf")

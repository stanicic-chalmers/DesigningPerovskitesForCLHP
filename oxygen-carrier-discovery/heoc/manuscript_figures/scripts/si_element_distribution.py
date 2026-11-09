"""SI: element distributions of each run (r1-r7) in its own WAE latent space as trained in cycle 50.

Uses the saved UMAP coordinates output/UMAP/c50/rN/umap_rN_c50_frozen_coords.csv: the encoder of run rN from the
iteration that ran cycle 50, UMAP fitted on the start set + rN compounds (cycles <= 50). Only those points
(the run's own training data) are plotted, coloured by element site fraction, with the plotting functions of
active-learning/src/plot_element_distributions.py (multi-panel heatmaps + dominant-element map).
Output: output/UMAP/c50/rN/element_distribution_heatmaps.png, dominant_element_distribution.png, individual_elements/.
Usage: python si_element_distribution.py [r1 r2 ...]
"""
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import AL_SRC, DATA, UMAP, label  # noqa: E402
sys.path.insert(0, str(AL_SRC))
from plot_element_distributions import (create_dominant_element_plots,  # noqa: E402
                                        create_multi_panel_heatmaps, setup_style)

args = sys.argv[1:]
CYCLE = 50
OUT = UMAP / f"c{CYCLE}"                       # -> output/UMAP/c50/rN/
REFS = [label(r) for r in args] or [f"r{i}" for i in range(1, 8)]
HPP_PERCENTILE = 8          # defaults of plot_element_distributions.py
MARKER_SIZE = 10.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(f"element_distribution_c{CYCLE}")
setup_style()

for ref in REFS:
    cfg = json.load(open(DATA / "run_config" / f"{ref}.json"))
    A, B = cfg["A_elements"], cfg["B_elements"]
    coords = pd.read_csv(OUT / ref / f"umap_{ref}_c{CYCLE}_frozen_coords.csv")
    df = coords[coords.run.isin(["start", ref])].reset_index(drop=True)
    comp = np.zeros((len(df), len(A) + len(B)))
    for i, formula in enumerate(df.formula):
        counts = {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
        tot = sum(counts.values())
        # as the WAE input: atom fraction / sublattice scaling (0.2) = site fraction
        comp[i] = [counts.get(e, 0) / tot / cfg["scaling_dictionary"][e] for e in A + B]
    out = OUT / ref
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"{ref}: {len(df)} points (start set + {ref}, cycles <= {df.cycle.max()}) -> {out}")
    xy, hpc = df[["umap1", "umap2"]].values, df.hpc.values
    create_multi_panel_heatmaps(xy, comp, A, B, str(out), logger, property_values=hpc,
                                hpp_percentile=HPP_PERCENTILE, cmap="magma", exclude_zeros=True,
                                marker_size=MARKER_SIZE)
    create_dominant_element_plots(xy, comp, A, B, str(out), logger, property_values=hpc,
                                  hpp_percentile=HPP_PERCENTILE, marker_size=MARKER_SIZE)

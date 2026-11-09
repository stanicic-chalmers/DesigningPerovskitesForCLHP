"""Figure S1, step 1: sensitivity of HPC to the steam-reactor H2O/H2 ratio, for all compounds of r1-r7 (tables only).

The stored SR Gibbs landscape is G_SR(v) = F(v, T_SR) - T S(v) + v mu_O(T_SR, ratio_0), relative to v = 0
(update_training_database.py, ratio_0 = 4). Changing only the ratio shifts it by v * [mu_O(ratio) - mu_O(ratio_0)],
so the landscape at any ratio follows exactly from the stored gibbs_SR_* values: no new phonon calculations,
and FR/AR and the configurational entropy are unchanged. Raw and quadratic-fit HPC are then recomputed with the
same rules as update_training_database.py.

Output: <TABLES>/figureS1/hpc_vs_ratio_all.csv (read by figureS1_element_enrichment_vs_sr_ratio.py) and summary tables.
Usage: python figureS1_sr_ratio_tables.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import METAL_REPO, TABLES, connect, run_list  # noqa: E402
sys.path.insert(0, str(METAL_REPO))
from metal.applications.crystals.tools import get_mu_oxygen_sr  # noqa: E402

from cycle_fix import fix_cycle  # noqa: E402
RUNS = run_list()
OUT = TABLES / "figureS1"
OUT.mkdir(parents=True, exist_ok=True)
T_SR = 800 + 273.15        # as in update_training_database.py
RATIO_0 = 4                # ratio the stored gibbs_SR_* were computed with
RATIOS = [0.1, 0.2, 0.5, 1, 2, 4, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 1000000]
N_VAC = 9                  # gibbs_*_0 .. gibbs_*_8
BARTEL_MAX = 4.18
TOP_N = 10
KEY = "hpc_fit"            # HPC used for the analysis and figures ("hpc_raw" for the discrete value)
KEY_LABEL = "quadratic-fit HPC"
A_EL = ["Ca", "Ce", "K", "Mg", "Y", "Ba", "La", "Sm", "Sr"]
B_EL = ["Mn", "Fe", "Co", "Ni", "Cu", "V", "Ti", "Zr", "W", "Cr", "Al", "Zn", "Sn"]   # Fig. 4 order
MAIN_GROUP_B = ["Al", "Zn", "Sn"]
RICH = 0.25                # element counted as "present" if it fills >= 25% of its site (4 of 16)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- load
recs = []
for run in RUNS:
    con = connect(run)
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        d = json.loads(kvp)
        d["cycle"] = fix_cycle(run, d["cycle"])
        if d["cycle"] is None:
            continue
        formula, _, struct = d["tag"].partition("_")
        # the tag carries the reduced brownmillerite formula (O40); the row is the oxidised perovskite (O48)
        formula = re.sub(r"O40(?!\d)", "O48", formula)
        cycle = int(d["cycle"])
        if (struct != "brownmillerite" or (cycle == 0 and run != RUNS[0]) or d.get("mass_") is None
                or d.get("delta_charge_neutrality") is not True or d["bartel_tolerance"] > BARTEL_MAX
                or any(f"gibbs_{r}_{v}" not in d for r in ("SR", "FR") for v in range(N_VAC))):
            continue
        counts = {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
        recs.append({"formula": formula, "run": "start set" if cycle == 0 else run, "cycle": cycle,
                     "mass": d["mass_"], "hpp_stored": d["hpp"], "hpp_fit_stored": d.get("hpp_quadratic_fit"),
                     "G_SR": np.array([d[f"gibbs_SR_{v}"] for v in range(N_VAC)]),
                     "G_FR": np.array([d[f"gibbs_FR_{v}"] for v in range(N_VAC)]),
                     **{f"A_{e}": counts.get(e, 0) / 16 for e in A_EL},
                     **{f"B_{e}": counts.get(e, 0) / 16 for e in B_EL}})
    con.close()
df = pd.DataFrame(recs).drop_duplicates("formula").reset_index(drop=True)
df["B_main_group"] = df[[f"B_{e}" for e in MAIN_GROUP_B]].sum(axis=1)
print(f"{len(df)} compounds ({(df.cycle > 0).sum()} AL), "
      f"charge-neutral, Bartel tau <= {BARTEL_MAX}")


# ---------------------------------------------------------------- HPC at a given ratio
VAC = np.arange(N_VAC)


def quad_min(coeffs, lo, hi):
    """Same rule as find_quadratic_minimum in update_training_database.py."""
    a, b = coeffs[0], coeffs[1]
    if abs(a) < 1e-10:
        return lo if b > 0 else hi
    if a < 0:
        return lo if np.polyval(coeffs, lo) < np.polyval(coeffs, hi) else hi
    return np.clip(-b / (2 * a), lo, hi)


fr_disc = np.array([VAC[np.argmin(g)] for g in df.G_FR])
fr_fit = np.array([quad_min(np.polyfit(VAC, g, 2), 0, N_VAC - 1) for g in df.G_FR])
# mu_O shift relative to RATIO_0; checked to be independent of add_u_dft (constant offset)
dmu = {r: float(get_mu_oxygen_sr(T_SR, r, add_u_dft=True)[0] - get_mu_oxygen_sr(T_SR, RATIO_0, add_u_dft=True)[0])
       for r in RATIOS}
dmu_check = max(abs(dmu[r] - float(get_mu_oxygen_sr(T_SR, r, add_u_dft=False)[0]
                                   - get_mu_oxygen_sr(T_SR, RATIO_0, add_u_dft=False)[0])) for r in RATIOS)
assert dmu_check < 1e-9, f"mu_O shift depends on add_u_dft ({dmu_check})"
lnK_pO2 = {r: 2.0 * (np.log(r) - np.log(RATIO_0)) for r in RATIOS}   # ln(pO2/pO2_0)

rows = []
for r in RATIOS:
    sr_disc, sr_fit = [], []
    for g in df.G_SR:
        g_new = g + VAC * dmu[r]
        sr_disc.append(VAC[np.argmin(g_new)])
        sr_fit.append(quad_min(np.polyfit(VAC, g_new, 2), 0, N_VAC - 1))
    sr_disc, sr_fit = np.array(sr_disc), np.array(sr_fit)
    rows.append(pd.DataFrame({"formula": df.formula, "ratio": r, "dmu_O_eV": dmu[r],
                              "v_SR": sr_disc, "v_FR": fr_disc, "v_SR_fit": sr_fit, "v_FR_fit": fr_fit,
                              "hpc_raw": 1000 * (fr_disc - sr_disc) / df.mass,
                              "hpc_fit": 1000 * (fr_fit - sr_fit) / df.mass}))
res = pd.concat(rows, ignore_index=True)
meta_cols = ["formula", "run", "cycle", "B_main_group"] + [f"A_{e}" for e in A_EL] + [f"B_{e}" for e in B_EL]
res = res.merge(df[meta_cols], on="formula")

# sanity check: the stored values must come back at RATIO_0
chk = res[res.ratio == RATIO_0].merge(df[["formula", "hpp_stored", "hpp_fit_stored"]], on="formula")
err_raw = (chk.hpc_raw - chk.hpp_stored).abs().max()
err_fit = (chk.hpc_fit - chk.hpp_fit_stored).abs().max()
print(f"Reproduction at ratio {RATIO_0}: max |raw - stored| = {err_raw:.2e}, max |fit - stored| = {err_fit:.2e}")
if err_raw > 1e-6 or err_fit > 1e-6:
    bad = chk[((chk.hpc_raw - chk.hpp_stored).abs() > 1e-6) | ((chk.hpc_fit - chk.hpp_fit_stored).abs() > 1e-6)]
    print(f"WARNING: {len(bad)} compounds not reproduced; see not_reproduced.csv")
    bad.to_csv(OUT / "not_reproduced.csv", index=False)
# at very low ratios the SR can be more reducing than the FR (v_SR > v_FR): no H2 is produced, so HPC = 0.
# The pipeline never clips (cannot happen at ratio 4); the unclipped values are kept in the CSV.
for k in ("hpc_raw", "hpc_fit"):
    neg = (res[k] < 0).groupby(res.ratio).sum()
    if neg.any():
        print(f"{k} < 0 (SR more reducing than FR), set to 0: " + ", ".join(f"ratio {r}: {n}" for r, n in neg[neg > 0].items()))
    res[f"{k}_unclipped"] = res[k]
    res[k] = res[k].clip(lower=0)
res.drop(columns=[f"A_{e}" for e in A_EL] + [f"B_{e}" for e in B_EL]).to_csv(OUT / "hpc_vs_ratio_all.csv",
                                                                              index=False)

# ---------------------------------------------------------------- summaries
al = res[res.cycle > 0]
summary = (res.assign(set=np.where(res.cycle > 0, "AL", "start set"))
           .groupby(["set", "ratio"])
           .agg(n=("formula", "size"), mean_raw=("hpc_raw", "mean"), median_raw=("hpc_raw", "median"),
                frac_raw0=("hpc_raw", lambda s: (s <= 0).mean()), mean_fit=("hpc_fit", "mean"),
                median_fit=("hpc_fit", "median"), max_raw=("hpc_raw", "max"))
           .reset_index())
summary["pO2_rel"] = summary.ratio.map(lambda r: np.exp(lnK_pO2[r]))
summary.to_csv(OUT / "summary_by_ratio.csv", index=False)
print("\nHPC vs H2O/H2 ratio (pO2_rel = pO2 / pO2 at ratio 4)")
print(summary.round(3).to_string(index=False))

# per element: mean HPC (KEY) of compounds where the element fills >= RICH of its site
elem = []
for site, els in [("A", A_EL), ("B", B_EL)]:
    for e in els:
        g = res[res[f"{site}_{e}"] >= RICH]
        if g.empty:
            continue
        m = g.groupby("ratio")[KEY].mean()
        elem.append({"site": site, "element": e, "n": g.formula.nunique(), **{r: m[r] for r in RATIOS}})
elem = pd.DataFrame(elem).set_index(["site", "element"])
elem.to_csv(OUT / f"mean_{KEY}_by_element_rich.csv")

# rank stability of the AL candidates (KEY) vs the reference ratio
ref_rank = al[al.ratio == RATIO_0].set_index("formula")[KEY]
stab = []
for r in RATIOS:
    cur = al[al.ratio == r].set_index("formula")[KEY]
    top_ref = set(ref_rank.nlargest(TOP_N).index)
    top_cur = set(cur.nlargest(TOP_N).index)
    stab.append({"ratio": r, "spearman_vs_ratio4": cur.corr(ref_rank, method="spearman"),
                 f"top{TOP_N}_overlap_with_ratio4": len(top_ref & top_cur)})
stab = pd.DataFrame(stab)
stab.to_csv(OUT / f"rank_stability_{KEY}.csv", index=False)
print(f"\nAL candidates, {KEY_LABEL}: rank stability vs ratio {RATIO_0}")
print(stab.round(2).to_string(index=False))

top = (res[res.cycle > 0].sort_values(["ratio", KEY], ascending=[True, False])
       .groupby("ratio").head(TOP_N)[["ratio", "formula", "run", "cycle", "hpc_raw", "hpc_fit", "v_SR", "v_FR", "v_SR_fit", "v_FR_fit",
                                      "B_main_group"]])
top.to_csv(OUT / f"top{TOP_N}_per_ratio_{KEY}.csv", index=False)

print(f"\nSaved to {OUT}/")

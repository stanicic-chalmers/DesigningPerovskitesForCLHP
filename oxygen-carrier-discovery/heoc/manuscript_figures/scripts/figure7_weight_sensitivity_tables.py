"""Figure 7, step 1: percentiles and weight sensitivity of the combined (multi-objective) rank (tables only).

Combined score = sum_k w_k * percentile_k (ranking.py), criteria HPC (quadratic fit), price, CRM, toxicity
(optionally energy above hull). Weight-space sampling (SMAA, Lahdelma & Salminen 2001): weight vectors are drawn
from Dirichlet distributions and every compound is re-ranked for each draw. Reported per compound: probability of
being in the top 10 / top 20, median and 5-95 % rank.

Scenarios
  uniform        no preference: uniform over the weight simplex (4 criteria)
  local          around the intuition weights W0 (Dirichlet, concentration CONC; sd of w_HPC ~ 0.1)
  local+ehull    as local, with energy above hull as fifth criterion (centre 0.85 * W0 + 0.15 e_hull)
  local+HPCunc   as local, HPC per draw drawn from {fit, raw} x {AL conditions, SR H2O/H2 = 1, FR pO2 = 1e-17}
  local,poolnorm as local, percentiles computed over the ranked pool (HPC > 0) instead of over all compounds

Pool: start set + AL compounds of r1-r7, charge neutral, Bartel tau <= 4.18, HPC_fit > 0, unique formulas.
EH score (--eh): base (default, manuscript) or precautionary, see common/ranking.py.
Output (read by figure7_candidates.py): <TABLES>/figure7_<case>/percentiles_pool.csv, weight_sensitivity_table.csv
Usage: python figure7_weight_sensitivity_tables.py [--eh base|precautionary]
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import METAL_REPO, RANKING_PY, TABLES, connect, label, run_list  # noqa: E402
sys.path.insert(0, str(METAL_REPO))
from metal.applications.crystals.tools import get_mu_oxygen, get_mu_oxygen_fr, get_mu_oxygen_sr  # noqa: E402

from cycle_fix import fix_cycle  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EH_CASE = arg("--eh", "base")   # EH scores: base (manuscript) or precautionary (sensitivity)
assert EH_CASE in ("precautionary", "base"), EH_CASE
CASE_NAME = "precautionary" if EH_CASE == "precautionary" else "basecase"
RUNS = run_list()
OUT = TABLES / f"figure7_{CASE_NAME}"
OUT.mkdir(parents=True, exist_ok=True)
T_FR, T_SR = 900 + 273.15, 800 + 273.15
FR_RATIO_0, SR_RATIO_0 = 1 / 9, 4
N_VAC, BARTEL_MAX = 9, 4.18
CRIT = ["hpp", "price", "crm", "toxicity"]
LABEL = {"hpp": "HPC", "price": "price", "crm": "Res", "toxicity": "toxicity", "ehull": "E$_{hull}$"}
W0 = np.array([0.6, 0.1, 0.15, 0.15])
CONC = 24              # Dirichlet concentration: alpha = CONC * W0
N_DRAW = 10000
TOP = (10, 20)
SEED = 0
B_EL = ["Al", "Cr", "Co", "Cu", "Fe", "Mn", "Ni", "Sn", "Ti", "V", "W", "Zr", "Zn"]

spec = importlib.util.spec_from_file_location("ranking", RANKING_PY)
ranking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranking)
ranking.TOXICITY_SCORE = ranking.TOXICITY_SCORE_PRECAUTIONARY if EH_CASE == "precautionary" else ranking.TOXICITY_SCORE_BASE
print(f"EH case: {EH_CASE} -> {OUT}")


# ---------------------------------------------------------------- data
recs = []
for run in RUNS:
    con = connect(run)
    for natoms, kvp in con.execute("select natoms, key_value_pairs from systems"):
        d = json.loads(kvp)
        d["cycle"] = fix_cycle(run, d["cycle"])
        if d["cycle"] is None:
            continue
        formula, _, struct = d["tag"].partition("_")
        # the tag carries the reduced brownmillerite formula (O40); the row is the oxidised perovskite (O48)
        formula = re.sub(r"O40(?!\d)", "O48", formula)
        cycle = int(d["cycle"])
        if (struct != "brownmillerite" or (cycle == 0 and run != RUNS[0]) or d.get("mass_") is None
                or d.get("hpp_quadratic_fit") is None
                or d.get("delta_charge_neutrality") is not True or d["bartel_tolerance"] > BARTEL_MAX
                or any(f"gibbs_{r}_{v}" not in d for r in ("SR", "FR") for v in range(N_VAC))):
            continue
        counts = {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
        eh = d.get("e_above_hull")
        recs.append({"formula": formula, "run": "start" if cycle == 0 else label(run),
                     "cycle": cycle, "mass": d["mass_"], "hpc_fit": d["hpp_quadratic_fit"], "hpc_raw": d["hpp"],
                     "price": d["price"], "ehull": np.nan if eh is None else 1000 * eh / natoms,
                     "B_main_group": sum(counts.get(e, 0) for e in ("Al", "Zn", "Sn")) / 16,
                     "B_site": " ".join(f"{e}{counts[e]}" for e in sorted(B_EL, key=lambda e: -counts.get(e, 0))
                                        if counts.get(e)),
                     "G_SR": [d[f"gibbs_SR_{v}"] for v in range(N_VAC)],
                     "G_FR": [d[f"gibbs_FR_{v}"] for v in range(N_VAC)]})
    con.close()
df = pd.DataFrame(recs).drop_duplicates("formula").reset_index(drop=True)
df[["crm", "toxicity"]] = df.formula.apply(ranking.chemistry_scores).apply(pd.Series)
df["hpp"] = df.hpc_fit

# HPC variants at alternative conditions (exact v * dmu_O shift; fit: b -> b + dmu_O)
VAC = np.arange(N_VAC)
G_SR, G_FR, mass = np.vstack(df.G_SR), np.vstack(df.G_FR), df.mass.values
dmu_sr1 = float(get_mu_oxygen_sr(T_SR, 1.0)[0] - get_mu_oxygen_sr(T_SR, SR_RATIO_0)[0])
dmu_fr17 = (float(np.atleast_1d(get_mu_oxygen(T_FR, np.log(1e-17), logarithmic=True))[0])
            - float(get_mu_oxygen_fr(T_FR, FR_RATIO_0)[0]))


def quad_min(a, b, c, lo=0.0, hi=N_VAC - 1.0):
    """Vectorised find_quadratic_minimum (update_training_database.py)."""
    out = np.clip(-b / np.where(a == 0, 1, 2 * a), lo, hi)
    v_lo, v_hi = a * lo ** 2 + b * lo + c, a * hi ** 2 + b * hi + c
    out = np.where(a < 0, np.where(v_lo < v_hi, lo, hi), out)
    return np.where(np.abs(a) < 1e-10, np.where(b > 0, lo, hi), out)


cf_SR, cf_FR = np.polyfit(VAC, G_SR.T, 2), np.polyfit(VAC, G_FR.T, 2)
hpc_var = {}
for cond, d_fr, d_sr in [("AL", 0, 0), ("SR1", 0, dmu_sr1), ("FR17", dmu_fr17, 0)]:
    raw = VAC[np.argmin(G_FR + VAC * d_fr, axis=1)] - VAC[np.argmin(G_SR + VAC * d_sr, axis=1)]
    fit = quad_min(cf_FR[0], cf_FR[1] + d_fr, cf_FR[2]) - quad_min(cf_SR[0], cf_SR[1] + d_sr, cf_SR[2])
    hpc_var[f"raw_{cond}"] = np.clip(1000 * raw / mass, 0, None)
    hpc_var[f"fit_{cond}"] = np.clip(1000 * fit / mass, 0, None)
assert np.allclose(hpc_var["fit_AL"], np.clip(df.hpc_fit.values, 0, None), atol=1e-6)
for k, v in hpc_var.items():
    df[f"hpc_{k}"] = v


def pct(frame):
    """Percentile of each criterion over `frame` (1 = best), as in ranking.py; ehull lower = better."""
    p = pd.DataFrame(index=frame.index)
    p["hpp"] = frame.hpp.rank(pct=True)
    p["price"] = 1 - frame.price.rank(pct=True)
    p["crm"] = frame.crm.rank(pct=True)
    p["toxicity"] = frame.toxicity.rank(pct=True)
    p["ehull"] = 1 - frame.ehull.rank(pct=True)
    for k in hpc_var:
        p[f"hpp_{k}"] = frame[f"hpc_{k}"].rank(pct=True)
    return p


P_all = pct(df)
pool_mask = (df.hpc_fit > 0).values
pool = df[pool_mask].reset_index(drop=True)
P = P_all[pool_mask].reset_index(drop=True)
P_pool = pct(pool)
n = len(pool)
n_eh = pool.ehull.isna().sum()
print(f"{len(df)} compounds, ranked pool (HPC_fit > 0) n = {n} "
      f"({(pool.cycle > 0).sum()} AL); e_hull missing for {n_eh} (ranked worst in local+ehull)")
P["ehull"] = P.ehull.fillna(0)
P_pool["ehull"] = P_pool.ehull.fillna(0)
P.assign(formula=pool.formula.values).to_csv(OUT / "percentiles_pool.csv", index=False)   # for diverse_selection.py


def ranks(S):
    """S: (n, N) scores -> (n, N) ranks, 1 = best."""
    order = np.argsort(-S, axis=0, kind="stable")
    R = np.empty_like(order)
    R[order, np.arange(S.shape[1])] = np.arange(1, S.shape[0] + 1)[:, None]
    return R


rng = np.random.default_rng(SEED)
M4 = P[CRIT].values
scen = {}
W_uni = rng.dirichlet(np.ones(4), N_DRAW)
scen["uniform"] = ranks(M4 @ W_uni.T)
W_loc = rng.dirichlet(CONC * W0, N_DRAW)
scen["local"] = ranks(M4 @ W_loc.T)
w0_eh = np.append(0.85 * W0, 0.15)
W_eh = rng.dirichlet(CONC * w0_eh, N_DRAW)
scen["local+ehull"] = ranks(P[CRIT + ["ehull"]].values @ W_eh.T)
# HPC uncertainty: each draw uses one of the six HPC variants for the HPC column
var_keys = list(hpc_var)
pick = rng.integers(len(var_keys), size=N_DRAW)
H = np.column_stack([P[f"hpp_{k}"].values for k in var_keys])            # (n, 6)
S = H[:, pick] * W_loc[:, 0] + M4[:, 1:] @ W_loc[:, 1:].T
scen["local+HPCunc"] = ranks(S)
scen["local,poolnorm"] = ranks(P_pool[CRIT].values @ W_loc.T)
base_rank = ranks((M4 @ W0)[:, None])[:, 0]
base_rank_pool = ranks((P_pool[CRIT].values @ W0)[:, None])[:, 0]

res = pool[["formula", "run", "cycle", "B_site", "hpc_fit", "hpc_raw", "price", "crm", "toxicity", "ehull",
            "hpc_raw_SR1", "hpc_fit_SR1", "hpc_raw_FR17", "hpc_fit_FR17"]].copy()
res["rank_W0"] = base_rank
res["rank_W0_poolnorm"] = base_rank_pool
for name, R in scen.items():
    for t in TOP:
        res[f"P_top{t}_{name}"] = (R <= t).mean(axis=1)
    res[f"P_first_{name}"] = (R == 1).mean(axis=1)
    res[f"rank_med_{name}"] = np.median(R, axis=1)
    res[f"rank_p5_{name}"] = np.percentile(R, 5, axis=1)
    res[f"rank_p95_{name}"] = np.percentile(R, 95, axis=1)


# ---------------------------------------------------------------- Pareto front (weight-free)
def pareto_mask(X):
    """X: (n, k), larger = better. True where no other row is >= in all and > in one."""
    keep = np.ones(len(X), bool)
    for i in range(len(X)):
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        keep[i] = not dom.any()
    return keep


X4 = np.column_stack([pool.hpc_fit, -pool.price, pool.crm, pool.toxicity])
X5 = np.column_stack([X4, -pool.ehull.fillna(np.inf)])
res["pareto_4"] = pareto_mask(X4)
res["pareto_5_ehull"] = pareto_mask(X5)
res = res.sort_values(["P_top10_local", "rank_W0"], ascending=[False, True]).reset_index(drop=True)
res.to_csv(OUT / "weight_sensitivity_table.csv", index=False)

# ---------------------------------------------------------------- summary printout
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
top10_W0 = set(res.nsmallest(10, "rank_W0").formula)
print(f"\nPareto-optimal (4 criteria): {res.pareto_4.sum()} of {n}; with e_hull: {res.pareto_5_ehull.sum()}")
print(f"Baseline top 10 on the 4-criteria Pareto front: {res[res.formula.isin(top10_W0)].pareto_4.sum()} / 10")
for name in scen:
    win = res[res[f"P_first_{name}"] > 0]
    print(f"{name:15s}: {len(win)} compounds ever rank 1; top-10 overlap with W0 at the median draw "
          f"{len(top10_W0 & set(res.nsmallest(10, f'rank_med_{name}').formula))}/10")
print(f"Top-10 overlap W0 all-compound vs pool normalisation: "
      f"{len(top10_W0 & set(res.nsmallest(10, 'rank_W0_poolnorm').formula))}/10")
show = ["formula", "run", "B_site", "hpc_fit", "hpc_raw", "price", "ehull", "rank_W0"] + \
       [f"P_top10_{s}" for s in scen] + ["rank_p5_local", "rank_p95_local", "pareto_4"]
print("\nTop 25 by P(top 10), local weights:")
print(res[show].head(25).round(2).to_string(index=False))

print(f"\nWrote percentiles_pool.csv and weight_sensitivity_table.csv to {OUT}")

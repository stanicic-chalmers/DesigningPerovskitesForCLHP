"""Figure 7: HPC, price, supply resilience (Res) and EH of the 20 candidates of Figure 6.

Published figure = the default call (python figure7_candidates.py): quadratic-fit HPC (--fit), the 20 Figure 6 picks
(--mmr 20 --picks <TABLES>/figure6_<case>/multiobjective_mmr_bsite_k20_l0.5.csv), percentiles from
figure7_weight_sensitivity_tables.py (run it first with the same --eh).
Output: output/manuscript/Figure_7.png/pdf (base-case EH) or Figure_7_precautionary.png/pdf (--eh precautionary); tables in
<TABLES>/figure7_<case>/.
Usage: python figure7_candidates.py [--eh base|precautionary]

Background (the general selection this script grew from):

The global combined rank is dominated by one Sn/Ti family. Here candidates are chosen
  1. per B-site family: within each family the compounds are re-ranked for weight vectors drawn around W0 with the
     HPC drawn from {fit, raw} x {AL conditions (SR H2O/H2 = 4, FR CO2/CO = 1/9), SR H2O/H2 = 1, FR pO2 = 1e-17} (the local+HPCunc scenario of
     weight_sensitivity.py); reported are P(best in family), P(top 3 in family) and the median family rank;
  2. across families by maximal marginal relevance (MMR): greedy picks that trade the expected combined score
     against the cation-composition similarity (cosine) to the compounds already picked.
Family = the redox-active transition metal (Fe, Mn, Cu, Co, Ni, Cr, V) with the largest B-site count if it holds
>= 4 of the 16 B sites, otherwise "Sn/Ti/Al" (B-site mainly main-group or d0 cations).
Only compounds whose HPC is real are eligible: raw and fit HPC >= MIN_HPC and |fit - raw| <= MAX_GAP.
Run weight_sensitivity.py first (writes weight_sensitivity_table.csv and percentiles_pool.csv).

--fit: base everything on the quadratic-fit HPC at the AL conditions (SR H2O/H2 = 4, FR CO2/CO = 1/9) only:
eligible if fit HPC >= MIN_HPC (no raw HPC or |fit - raw| criterion), and only the weights are drawn (no HPC
variants). Outputs get the suffix _fit.
--hpc: (without --fit) left panel shows HPC (fit bars, raw markers) instead of delta; figure saved as
diverse_top_per_family_hpc.{png,pdf}, nothing else written differently. With --fit the panel always shows HPC.
--mmr K: make K MMR picks (default K_MMR) and plot only those, grouped by family in pick order; the MMR csv, the
delta csv and the figure get the extra suffix _mmrK (e.g. diverse_mmr20_fit.png), the per-family csv is unchanged.
--picks CSV: (with --mmr) plot the picks listed in CSV (columns pick, formula, bgroup, n_members, as written by
multiobjective_mmr.py, e.g. ../multiobjective_ranking/multiobjective_mmr_bsite_k20_l0.5.csv: B-site Jaccard MMR,
lambda = 0.5) instead of this script's own MMR picks, in pick order and labelled by B-site chemistry; n = number of
compounds assigned to the pick. The suffix becomes _mmrK_<CSV stem>; no MMR csv is written.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import MANUSCRIPT, TABLES, connect, label, run_list  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EH_CASE = arg("--eh", "base")   # EH scores: base (manuscript) or precautionary (sensitivity)
assert EH_CASE in ("precautionary", "base"), EH_CASE
CASE_NAME = "precautionary" if EH_CASE == "precautionary" else "basecase"
OUT = TABLES / f"figure7_{CASE_NAME}"                 # tables of figure7_weight_sensitivity_tables.py
PICKS_CSV = TABLES / f"figure6_{CASE_NAME}" / "multiobjective_mmr_bsite_k20_l0.5.csv"
W0 = np.array([0.6, 0.1, 0.15, 0.15])
CONC, N_DRAW, SEED = 24, 10000, 1
MIN_HPC, MAX_GAP = 1.0, 0.5
TOP_PER_FAMILY = 3
K_MMR, LAMBDA = 12, 0.4         # MMR: LAMBDA * relevance - (1 - LAMBDA) * max similarity
REDOX = ["Fe", "Mn", "Cu", "Co", "Ni", "Cr", "V"]
B_EL = ["Al", "Cr", "Co", "Cu", "Fe", "Mn", "Ni", "Sn", "Ti", "V", "W", "Zr", "Zn"]
A_EL = ["Ba", "Ca", "Ce", "K", "La", "Mg", "Sm", "Sr", "Y"]
RUNS = run_list()
PUBLISHED = not any(a in sys.argv for a in ("--fit", "--hpc", "--mmr", "--picks"))   # default = Figure 7
FIT_ONLY = PUBLISHED or "--fit" in sys.argv
SUF = "_fit" if FIT_ONLY else ""
HPC_PANEL = FIT_ONLY or "--hpc" in sys.argv
FIG_SUF = SUF if FIT_ONLY else ("_hpc" if HPC_PANEL else "")
MMR_ONLY = PUBLISHED or "--mmr" in sys.argv
if MMR_ONLY:
    K_MMR = int(arg("--mmr", 20))
PICKS = None
if PUBLISHED or "--picks" in sys.argv:
    picks_path = PICKS_CSV if PUBLISHED else (Path.cwd() / arg("--picks", "")).resolve()
    PICKS = pd.read_csv(picks_path)
    PICKS.attrs["stem"] = picks_path.stem
MMR_SUF = (f"_mmr{K_MMR}" + (f"_{PICKS.attrs['stem']}" if PICKS is not None else "")) if MMR_ONLY else ""
FIG_NAME = f"diverse{MMR_SUF}{FIG_SUF}" if MMR_ONLY else f"diverse_top_per_family{FIG_SUF}"
VARIANTS = ["raw_AL", "fit_AL", "raw_SR1", "fit_SR1", "raw_FR17", "fit_FR17"]
if FIT_ONLY:
    VARIANTS = ["fit_AL"]

tab = pd.read_csv(OUT / "weight_sensitivity_table.csv")
pc = pd.read_csv(OUT / "percentiles_pool.csv")
df = pc.merge(tab, on="formula", suffixes=("_pct", ""))
assert len(df) == len(pc) == len(tab)
counts = pd.DataFrame([{e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f)} for f in df.formula]
                      ).reindex(columns=A_EL + B_EL).fillna(0)


def family(row):
    tm = row[REDOX]
    return tm.idxmax() if tm.max() >= 4 else "Sn/Ti/Al"


df["family"] = counts.apply(family, axis=1)
# percentiles for display only, over the ranked pool (HPC > 0; 1 = best); the scores use the ranking.py
# percentiles over all compounds (*_pct)
df["price_pct_pool"] = 1 - df.price.rank(pct=True)
df["crm_pct_pool"] = df.crm.rank(pct=True)
df["toxicity_pct_pool"] = df.toxicity.rank(pct=True)
df["hpc_raw_AL"], df["hpc_fit_AL"] = df.hpc_raw, df.hpc_fit.clip(lower=0)
df["hpc_min_var"] = df[[f"hpc_{v}" for v in VARIANTS]].min(axis=1)
df["hpc_max_var"] = df[[f"hpc_{v}" for v in VARIANTS]].max(axis=1)
if FIT_ONLY:
    ok = (df.hpc_fit >= MIN_HPC).values
    crit = f"fit HPC >= {MIN_HPC} at H$_2$O/H$_2$ = 4, CO$_2$/CO = 1/9"
else:
    ok = ((df.hpc_raw >= MIN_HPC) & (df.hpc_fit >= MIN_HPC) & ((df.hpc_fit - df.hpc_raw).abs() <= MAX_GAP)).values
    crit = f"raw & fit HPC >= {MIN_HPC}, |fit - raw| <= {MAX_GAP}"
el = df[ok].reset_index(drop=True)
X = counts[ok].values / 16                                  # cation fractions per site
X = X / np.linalg.norm(X, axis=1, keepdims=True)
print(f"{ok.sum()} of {len(df)} ranked compounds pass {crit}; HPC draws from {VARIANTS}")
print(df.assign(eligible=ok).groupby("family").agg(n=("formula", "size"), eligible=("eligible", "sum"),
                                                   hpc_fit_max=("hpc_fit", "max"), hpc_raw_max=("hpc_raw", "max")).to_string())

# ---------------------------------------------------------------- weight + HPC draws (local+HPCunc)
rng = np.random.default_rng(SEED)
W = rng.dirichlet(CONC * W0, N_DRAW)
pick = rng.integers(len(VARIANTS), size=N_DRAW)
H = el[[f"hpp_{v}" for v in VARIANTS]].values
S = H[:, pick] * W[:, 0] + el[["price_pct", "crm_pct", "toxicity_pct"]].values @ W[:, 1:].T   # (n, N)
el["score_mean"] = S.mean(axis=1)
el["score_W0_fit"] = el[["hpp", "price_pct", "crm_pct", "toxicity_pct"]].values @ W0

# 1. within-family robustness
N_FAMILY_POOL = df.groupby("family").size()   # df is the ranked pool (fit HPC > 0)
rows = []
for fam, idx in el.groupby("family").indices.items():
    Sf = S[idx]
    R = np.argsort(np.argsort(-Sf, axis=0), axis=0) + 1
    sub = el.iloc[idx].copy()
    sub["n_family_eligible"] = len(idx)
    sub["n_family_pool"] = N_FAMILY_POOL[fam]   # all family members with HPC > 0, as multiobjective_clusters.py --family
    sub["P_best_in_family"] = (R == 1).mean(axis=1)
    sub["P_top3_in_family"] = (R <= 3).mean(axis=1)
    sub["family_rank_med"] = np.median(R, axis=1)
    rows.append(sub)
fam_tab = pd.concat(rows).sort_values(["family", "score_mean"], ascending=[True, False])
cols = ["family", "formula", "run", "B_site", "hpc_fit", "hpc_raw", "hpc_min_var", "hpc_max_var", "hpc_raw_SR1",
        "hpc_raw_FR17", "ehull", "price", "crm", "toxicity", "rank_W0", "score_mean", "n_family_eligible", "n_family_pool",
        "P_best_in_family", "P_top3_in_family", "family_rank_med"]
fam_tab[cols].to_csv(OUT / f"diverse_per_family{SUF}.csv", index=False)
fam_order = (fam_tab.groupby("family").score_mean.max().sort_values(ascending=False).index.tolist())
best = fam_tab.groupby("family").head(TOP_PER_FAMILY)
best = pd.concat([best[best.family == f] for f in fam_order])
pd.set_option("display.width", 250)
print(f"\nTop {TOP_PER_FAMILY} per family (sorted by expected combined score incl. HPC uncertainty):")
print(best[["family", "formula", "run", "B_site", "hpc_fit", "hpc_raw", "hpc_min_var", "ehull", "price",
            "rank_W0", "P_best_in_family", "P_top3_in_family"]].round(2).to_string(index=False))

# 2. MMR across all eligible compounds
rel = (el.score_mean - el.score_mean.min()) / (el.score_mean.max() - el.score_mean.min())
chosen = []
for _ in range(K_MMR):
    sim = (X @ X[chosen].T).max(axis=1) if chosen else np.zeros(len(el))
    mmr = LAMBDA * rel.values - (1 - LAMBDA) * sim
    mmr[chosen] = -np.inf
    chosen.append(int(np.argmax(mmr)))
mmr_tab = el.iloc[chosen].copy()
mmr_tab.insert(0, "pick", range(1, K_MMR + 1))
mmr_tab["max_sim_to_earlier"] = [0.0] + [float((X[c] @ X[chosen[:i]].T).max()) for i, c in enumerate(chosen[1:], 1)]
if PICKS is None:
    mmr_tab[["pick"] + cols[:-5] + ["max_sim_to_earlier"]].to_csv(OUT / f"diverse_mmr_picks{MMR_SUF}{SUF}.csv",
                                                                   index=False)
print(f"\nMMR picks (lambda = {LAMBDA}):")
print(mmr_tab[["pick", "family", "formula", "run", "B_site", "hpc_fit", "hpc_raw", "hpc_min_var", "ehull", "price",
               "rank_W0", "max_sim_to_earlier"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- figure: top per family
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED = "#222222", "#888888"
mmr_set = set(mmr_tab.formula)
if MMR_ONLY:  # plot the MMR picks instead of the top per family, families in order of their first pick
    if PICKS is not None:
        best = PICKS[["pick", "formula", "bgroup", "n_members"]].merge(fam_tab, on="formula", how="left")
        assert best.family.notna().all(), f"picks not eligible here: {best.formula[best.family.isna()].tolist()}"
    else:
        best = mmr_tab.merge(fam_tab[["formula", "n_family_pool"]], on="formula")
        best = pd.concat([best[best.family == f] for f in best.family.unique()])
# ---------------------------------------------------------------- oxygen non-stoichiometry at the AL conditions
def v_eq(g):
    """Equilibrium vacancy count: clamped minimum of a quadratic fit of the stored G_v (as hpp_quadratic_fit)."""
    vs = np.arange(len(g))
    a, b, _ = np.polyfit(vs, g, 2)
    if a > 1e-10:
        return float(np.clip(-b / (2 * a), vs[0], vs[-1]))
    return float(vs[-1] if a * vs[-1] ** 2 + b * vs[-1] < 0 else vs[0])


want = {re.sub(r"O48(?!\d)", "O40", f): f for f in best.formula}
delta = {}
for run in RUNS:
    con = connect(run)
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        d = json.loads(kvp)
        formula, _, struct = d["tag"].partition("_")
        if struct != "brownmillerite" or formula not in want or want[formula] in delta:
            continue
        v = {r: v_eq([d[f"gibbs_{r}_{i}"] for i in range(9)]) for r in ("AR", "SR", "FR")}
        assert abs(1000 * (v["FR"] - v["SR"]) / d["mass_"] - d["hpp_quadratic_fit"]) < 1e-6
        delta[want[formula]] = {f"delta_{r}": x / 16 for r, x in v.items()}      # ABO3-delta: 48 O per 16 ABO3
    con.close()
best = best.join(pd.DataFrame.from_dict(delta, orient="index"), on="formula")
best[["family", "formula", "run", "hpc_fit", "delta_AR", "delta_SR", "delta_FR", "price", "crm", "toxicity"]].to_csv(
    OUT / f"diverse_top_per_family_delta{MMR_SUF}{SUF}.csv", index=False)

y = np.arange(len(best))
RED, BLUE, GREY = "#c0532a", "#1c5cab", "#555555"
fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(8.0, 0.2 * len(best) + 1.5), sharey=True,
                                     gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
if HPC_PANEL:  # HPC itself (AL conditions)
    a1.barh(y, best.hpc_fit, height=0.6, color=BLUE, zorder=2,
            label="HPC, quadratic fit (SR: H$_2$O/H$_2$ = 4, 800 °C; FR: CO$_2$/CO = 1/9, 900 °C)")
    x_lab = best.hpc_fit if FIT_ONLY else np.maximum(best.hpc_fit, best.hpc_raw)
    for yi, h, xl in zip(y, best.hpc_fit, x_lab):
        a1.text(xl + 0.06, yi, f"{h:.2f}", va="center", fontsize=5.5, color=INK)
    if not FIT_ONLY:
        a1.scatter(best.hpc_raw, y, marker="|", s=60, color=RED, lw=1.4, zorder=3,
                   label="HPC, raw (discrete vacancy minima)")
    a1.set_xlabel("HPC (mmol H$_2$ g$^{-1}$)")
    a1.set_xlim(0, np.ceil(best.hpc_fit.max() * 2 + 0.5) / 2)
else:
    a1.hlines(y, best.delta_SR, best.delta_FR, color="#bbbbbb", lw=2.2, zorder=1, label="SR–FR swing (∝ HPC)")
    a1.scatter(best.delta_AR, y, marker="s", facecolor="white", edgecolor=GREY, s=16, zorder=3,
               label="AR: air, 1000 °C, pO$_2$ = 0.05 bar")
    a1.scatter(best.delta_SR, y, facecolor="white", edgecolor=BLUE, s=18, zorder=4,
               label="SR: H$_2$O/H$_2$ = 4, 800 °C")
    a1.scatter(best.delta_FR, y, color=RED, s=16, zorder=5, label="FR: CO$_2$/CO = 1/9, 900 °C")
    a1.set_xlabel("δ in ABO$_{3-δ}$ (quadratic fit)")
    a1.set_xlim(-0.02, 0.52)
# secondary criteria as percentiles over the ranked pool (HPC > 0, 100 = best): the raw supply-resilience and EH
# scores are compressed (most compounds 0.8-0.97), so only their position relative to the other compounds is readable
for ax, c, lab in ((a2, "price", "price"), (a3, "crm", "Res"), (a4, "toxicity", "EH")):
    ax.axvline(50, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.scatter(100 * best[f"{c}_pct_pool"], y, color=INK, s=14, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(f"{lab}\n(percentile)")
for yi, pc, usd in zip(y, best.price_pct_pool, best.price):   # absolute price next to its dot
    a2.text(100 * pc + (-6 if pc > 0.6 else 6), yi, f"{usd:.1f} $/kg", va="center", fontsize=5,
            color=MUTED, ha="right" if pc > 0.6 else "left")
def site_formula(f):
    """(A)(B)O48 notation, elements of each site in alphabetical order, e.g. (Ca8CeK2Mg5)(Sn12Ti4)O48."""
    n = {e: int(c or 1) for e, c in re.findall(r"([A-Z][a-z]?)(\d*)", f)}
    site = lambda els: "".join(f"{e}{n[e] if n[e] > 1 else ''}" for e in sorted(els) if e in n)
    return f"({site(A_EL)})({site(B_EL)})O48"


if PICKS is not None:
    labels = [f"{r.pick}. {r.bgroup}:  {site_formula(r.formula)} ({r.run})"
              for r in best.itertuples()]
elif MMR_ONLY:
    labels = [f"{r.pick}. {site_formula(r.formula)} ({r.run})"
              for r in best.itertuples()]
else:
    labels = [f"{site_formula(r.formula)} ({r.run})"
              + (" ◆" if r.formula in mmr_set else "") for r in best.itertuples()]
a1.set_yticks(y)
a1.set_yticklabels(labels, fontsize=6)
a1.invert_yaxis()
# family separators and names
bounds = np.flatnonzero(best.family.values[1:] != best.family.values[:-1]) + 0.5
if PICKS is not None:  # pick order, one B-site chemistry per row: no family blocks, n = compounds assigned to the pick
    bounds = []
for ax in (a1, a2, a3, a4):
    for b in bounds:
        ax.axhline(b, color="#dddddd", lw=0.6)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
starts = np.r_[0, np.ceil(bounds).astype(int)] if PICKS is None else []
for yi, n_m in zip(y, best.n_members if PICKS is not None else []):
    a4.text(1.04, yi, f"n = {n_m}", transform=a4.get_yaxis_transform(), fontsize=6, color=MUTED, va="center")
for s, fam in zip(starts, best.family.values[starts] if PICKS is None else []):
    n_el = int(best.n_family_pool.values[s])
    a4.text(1.04, s - 0.35, f"{fam} B-site (n = {n_el})", transform=a4.get_yaxis_transform(), fontsize=6.5,
            color=INK, va="top", fontweight="bold")
handles, hl = a1.get_legend_handles_labels()
fig.legend(handles, hl, loc="upper left", bbox_to_anchor=(0.01, 1.0), fontsize=6.5, frameon=False,
           ncol=1 if HPC_PANEL else 2)
mmr_note = (f"MMR on B-site chemistry (λ = {re.search(r'_l([\d.]+)$', PICKS.attrs['stem']).group(1)}); "
            f"number = pick order; n = compounds assigned" if PICKS is not None else
            f"the {K_MMR} picks of the diversity-aware (MMR, λ = {LAMBDA}) selection; number = pick order"
            if MMR_ONLY else "◆ = picked by the diversity-aware (MMR) selection")
fig.text(0.99, 0.985, mmr_note, fontsize=6.5, color=MUTED, ha="right", va="top")
fig.text(0.99, 0.955, f"eligible: {crit}", fontsize=6.5, color=MUTED, ha="right", va="top")
fig.text(0.99, 0.925, f"price, Res (supply resilience), EH: percentile among the {len(df)} compounds with HPC > 0, 100 = best", fontsize=6.5,
         color=MUTED, ha="right", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.9))
if PUBLISHED:
    FIG_NAME = "Figure_7" if EH_CASE == "base" else "Figure_7_precautionary"
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig((MANUSCRIPT if PUBLISHED else OUT) / f"{FIG_NAME}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote diverse_per_family{SUF}.csv, " + ("" if PICKS is not None else f"diverse_mmr_picks{MMR_SUF}{SUF}.csv, ")
      + f"{FIG_NAME}.png/pdf, "
      f"diverse_top_per_family_delta{MMR_SUF}{SUF}.csv to {OUT}" + (f"; figure in {MANUSCRIPT}" if PUBLISHED else ""))

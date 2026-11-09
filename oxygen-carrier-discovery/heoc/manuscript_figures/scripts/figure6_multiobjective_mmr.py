"""Figure 6: multi-objective ranking of 20 candidates with B-site families chosen by maximal marginal relevance
(start set + AL compounds of r1-r7).

Ranking: ranking.py percentiles over all compounds, then rank positions over the HPC > 0 pool; combined weights 0.6 / 0.1 / 0.15 / 0.15; HPC = hpp_quadratic_fit; charge neutral, Bartel <= 4.18).

MMR (Carbonell & Goldstein 1998): K representatives are picked one at a time, each maximising
    LAMBDA * relevance - (1 - LAMBDA) * max similarity to the representatives already picked,
relevance = combined score scaled to [0, 1] over the eligible compounds; similarity = Jaccard overlap of the
B-site element sets, counting only elements on >= KMIN of the 16 B sites (single-site dopants ignored), e.g.
Sn-Ti vs Sn-Ti-V = 2/3, Sn-Ti vs Fe-Sn = 1/3. A compound whose B-site set equals that of an earlier pick
(similarity 1) cannot be picked, so each B-site chemistry is represented at most once.
Every compound is then assigned to its most similar representative (ties -> the better-ranked representative),
which defines the families; a formula found in several runs counts for each of them.
Only compounds with HPC >= MIN_HPC are picked and assigned (ranks are still over the whole HPC > 0 pool).
EH score (--eh): base (default, manuscript) or precautionary (sensitivity), see common/ranking.py.
Output: output/manuscript/Figure_6_<basecase|precautionary>.png/pdf (base case also as Figure_6.png/pdf);
tables in <TABLES>/figure6_<case>/: the picks (multiobjective_mmr_bsite_k20_l0.5.csv, read by Figure 7) and
all assigned compounds (..._members.csv); the lambda sensitivity of the picks is printed.
Usage: python figure6_multiobjective_mmr.py [--eh base|precautionary] [--k K] [--lambda LAMBDA]   (default 20, 0.5)
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from paths import MANUSCRIPT, RANKING_PY, TABLES, connect, label, run_list  # noqa: E402
from cycle_fix import fix_cycle  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EH_CASE = arg("--eh", "base")
assert EH_CASE in ("precautionary", "base"), EH_CASE
CASE_NAME = "precautionary" if EH_CASE == "precautionary" else "basecase"
RUNS = run_list()
OUT = TABLES / f"figure6_{CASE_NAME}"
OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT.mkdir(parents=True, exist_ok=True)
HPC_KEY = "hpp_quadratic_fit"
BARTEL_MAX = 4.18
COMBINED = "Combined"
CRITERIA = {
    "Performance (HPC)": {"hpp": 1.0},
    "Price": {"price": 1.0},
    "Res": {"crm": 1.0},
    "EH" if EH_CASE == "precautionary" else "EH (base)": {"toxicity": 1.0},
    COMBINED: {"hpp": 0.6, "price": 0.1, "crm": 0.15, "toxicity": 0.15},
}
A_EL = ["Ba", "Ca", "Ce", "K", "La", "Mg", "Sm", "Sr", "Y"]
B_EL = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Sn", "Ti", "V", "W", "Zn", "Zr"]
SOURCES = ["start"] + [label(r) for r in RUNS]

spec = importlib.util.spec_from_file_location("ranking", RANKING_PY)
ranking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranking)
ranking.TOXICITY_SCORE = ranking.TOXICITY_SCORE_PRECAUTIONARY if EH_CASE == "precautionary" else ranking.TOXICITY_SCORE_BASE
print(f"EH case: {EH_CASE}")

# ---------------------------------------------------------------- data
recs = []
for run in RUNS:
    con = connect(run)
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        d = json.loads(kvp)
        d["cycle"] = fix_cycle(run, d["cycle"])
        if d["cycle"] is None:
            continue
        formula, _, struct = d["tag"].partition("_")
        formula = re.sub(r"O40(?!\d)", "O48", formula)   # tag = reduced brownmillerite; row = oxidised perovskite
        cycle = int(d["cycle"])
        if (struct != "brownmillerite" or d.get(HPC_KEY) is None or (cycle == 0 and run != RUNS[0])
                or d.get("delta_charge_neutrality") is not True or d["bartel_tolerance"] > BARTEL_MAX):
            continue
        recs.append({"formula": formula, "source": "start" if cycle == 0 else label(run), "cycle": cycle,
                     "hpp": d[HPC_KEY], "hpp_raw": d.get("hpp"), "price": d["price"]})
    con.close()
allrec = pd.DataFrame(recs)
found_in = allrec.groupby("formula").source.agg(lambda s: sorted(set(s), key=SOURCES.index))
df = allrec.drop_duplicates("formula").reset_index(drop=True)
df["found_in"] = df.formula.map(found_in)
df[["crm", "toxicity"]] = df["formula"].apply(ranking.chemistry_scores).apply(pd.Series)

# percentiles over all compounds, rank positions over the HPC > 0 pool (as multiobjective_heatmap.py)
df["hpp_rank"] = df["hpp"].rank(pct=True)
df["price_rank"] = 1 - df["price"].rank(pct=True)
df["crm_rank"] = df["crm"].rank(pct=True)
df["toxicity_rank"] = df["toxicity"].rank(pct=True)
pool = df[df.hpp > 0].reset_index(drop=True)
for name, w in CRITERIA.items():
    pool[name] = ranking.rank_materials(pool, w)["score"].reindex(pool.index).rank(ascending=False, method="min").astype(int)
pool = pool.sort_values([COMBINED, "hpp"], ascending=[True, False]).reset_index(drop=True)
n_pool = len(pool)
print(f"{len(df)} compounds, ranked pool (HPC > 0) n = {n_pool}")


K = int(arg("--k", 20))
LAMBDA = float(arg("--lambda", 0.5))
MIN_HPC = 1.0
KMIN = 2
STEM = f"multiobjective_mmr_bsite_k{K}_l{LAMBDA:g}"

# ---------------------------------------------------------------- MMR on B-site composition
def counts(f):
    n = {e: int(c or 1) for e, c in re.findall(r"([A-Z][a-z]?)(\d*)", f)}
    return np.array([n.get(e, 0) for e in B_EL], float)


n_rank = len(pool)
pool = pool[pool.hpp >= MIN_HPC].reset_index(drop=True)       # eligible for picking / families
print(f"{len(pool)} of {n_rank} ranked compounds have HPC >= {MIN_HPC}")
Bc = np.array([counts(f) for f in pool.formula])
assert (Bc.sum(1) == 16).all()
P = (Bc >= KMIN).astype(float)                                  # B-site element sets
inter = P @ P.T
S = inter / (P.sum(1)[:, None] + P.sum(1)[None, :] - inter)      # Jaccard similarity of B-site sets
w = CRITERIA[COMBINED]
score = sum(wt * pool[f"{c}_rank"] for c, wt in w.items()).values
rel = (score - score.min()) / (score.max() - score.min())   # over the eligible compounds


def mmr(lam, k):
    chosen = []
    for _ in range(k):
        val = lam * rel - (1 - lam) * (S[:, chosen].max(1) if chosen else 0)
        val[chosen] = -np.inf
        if chosen:
            val[S[:, chosen].max(1) >= 1 - 1e-9] = -np.inf      # same B-site set as an earlier pick
        if not np.isfinite(val).any():
            break
        chosen.append(int(np.argmax(val)))
    return chosen


def b_group(row, kmin=KMIN):
    return "–".join(e for e, n in zip(B_EL, row) if n >= kmin) or "mixed"


pool["bgroup"] = [b_group(r) for r in Bc]
for lam in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):          # sensitivity
    ch = mmr(lam, K)
    print(f"lambda {lam}: picks' combined ranks {sorted(pool[COMBINED].iloc[ch].tolist())}; "
          f"groups: {', '.join(pool.bgroup.iloc[ch])}")

chosen = mmr(LAMBDA, K)
pool["family"] = np.argmax(S[:, chosen], axis=1) + 1             # ties -> earlier pick
pool["sim_to_rep"] = S[np.arange(len(pool)), np.array(chosen)[pool.family - 1]]
memb = pool.explode("found_in").groupby(["family", "found_in"]).size().unstack(fill_value=0).reindex(
    index=range(1, K + 1), columns=SOURCES, fill_value=0)
fam = pool.iloc[chosen].copy()
fam.index = range(1, K + 1)
fam["max_sim_to_earlier"] = [0.0] + [float(S[c, chosen[:i]].max()) for i, c in enumerate(chosen[1:], 1)]
fam["n_members"] = pool.groupby("family").size().reindex(fam.index, fill_value=0)
fam["sim_members_median"] = pool.groupby("family").sim_to_rep.median()
fam = fam.join(memb)
fam["n_runs"] = (fam[SOURCES[1:]] > 0).sum(axis=1)


def site_formula(f):
    n = {e: int(c or 1) for e, c in re.findall(r"([A-Z][a-z]?)(\d*)", f)}
    site = lambda els: "".join(f"{e}{n[e] if n[e] > 1 else ''}" for e in sorted(els) if e in n)
    return f"({site(A_EL)})({site(B_EL)})O48"


def site_formula_tex(f):
    """site_formula with subscripts (matplotlib mathtext), e.g. (Ca$_{2}$K$_{3}$)(Sn$_{6}$Ti$_{10}$)O$_{48}$."""
    return re.sub(r"(?<=[A-Za-z])(\d+)", r"$_{\1}$", site_formula(f))


fam["label"] = fam.bgroup + ":  " + fam.formula.map(site_formula)
cols = ["label", "formula", "bgroup", "hpp", "hpp_raw", "price", "crm", "toxicity"] + list(CRITERIA) + \
       ["max_sim_to_earlier", "n_members", "sim_members_median", "n_runs"] + SOURCES
fam[cols].to_csv(OUT / f"{STEM}.csv", index_label="pick")
pool[["family", "formula", "bgroup", "sim_to_rep", "hpp", COMBINED]].assign(
    found_in=pool.found_in.str.join(" ")).to_csv(OUT / f"{STEM}_members.csv", index=False)
pd.set_option("display.width", 250)
print(f"\n{STEM}: {K} MMR picks (lambda = {LAMBDA}), {len(pool)} compounds (HPC >= {MIN_HPC}) assigned")
print(fam[["label", "hpp"] + list(CRITERIA) + ["max_sim_to_earlier", "n_members", "sim_members_median"] + SOURCES]
      .round(2).to_string())
print("families found by >1 run:", int((fam.n_runs > 1).sum()))

# ---------------------------------------------------------------- figure (rankings only; per-run family counts are in the csv)
plt.rcParams.update({"font.size": 7})
nr = len(fam)
fig, ax = plt.subplots(figsize=(4.6, 1.5 + 0.24 * nr))
M = fam[list(CRITERIA)].values
im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=n_rank)
for i in range(nr):
    for j in range(M.shape[1]):
        ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=6,
                fontweight="bold" if j == M.shape[1] - 1 else "normal")
ax.set_xticks(range(len(CRITERIA)))
ax.set_xticklabels(list(CRITERIA), rotation=40, ha="right")
ax.set_yticks(range(nr))
ax.set_yticklabels([f"{i}. {g}:  {site_formula_tex(f)}" for i, g, f in zip(fam.index, fam.bgroup, fam.formula)],
                   fontsize=6)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("rank")
names = [f"Figure_6_{CASE_NAME}"] + (["Figure_6"] if EH_CASE == "base" and (K, LAMBDA) == (20, 0.5) else [])
for name in names:
    for ext in ("png", "pdf"):
        fig.savefig(MANUSCRIPT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {', '.join(names)} (png/pdf) in {MANUSCRIPT} and {STEM}.csv in {OUT}")

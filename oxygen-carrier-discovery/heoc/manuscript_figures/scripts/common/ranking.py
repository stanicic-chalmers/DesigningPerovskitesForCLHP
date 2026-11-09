import pandas as pd
import numpy as np
import re
import argparse 


DEFAULT_CRM=1.0
DEFAULT_TOXICITY=1.0
# EH (toxicity) score, base case: classification of the driving oxide only where harmonized or clearly notified
TOXICITY_SCORE_BASE = {
    # Inert
    "Fe": 1.00,
    "Mg": 1.00,
    "Al": 1.00,
    "Ti": 1.00,
    "Zr": 1.00,
    "Sn": 1.0,
    "Sm": 1.00,
    "La": 1.00,
    "Y": 1.00,
    "W": 0.74,
    "Mn": 0.74,
    "Ca": 0.61,
    "Sr": 0.61,
    "K": 0.61,
    "Zn": 0.61,
    "Ce": 0.61,
    "Ba":0.61,
    "Cu": 0.39,
    "Co": 0.36,
    "Ni": 0.32,
    "V": 0.16,   # Cr(VI) compounds are highly toxic
    "Cr": 0.03
}

# EH score, precautionary case: lower scores where the hazard assessment is uncertain (Ti, Sm, Sn) and a more
# cautious score for Fe, La, Y, Mn and Ba; all other elements as in the base case (TOXICITY_SCORE_BASE)
TOXICITY_SCORE_PRECAUTIONARY = {**TOXICITY_SCORE_BASE,
    "Fe": 0.87, "La": 0.87, "Y": 0.87,
    "Ti": 0.61, "Sm": 0.61, "Sn": 0.55,
    "Mn": 0.61, "Ba": 0.52,
}

# EH score used by all rankings: the base case (manuscript). The precautionary case is the sensitivity check
# (figure scripts with --eh precautionary set ranking.TOXICITY_SCORE = ranking.TOXICITY_SCORE_PRECAUTIONARY).
TOXICITY_SCORE = TOXICITY_SCORE_BASE

# Critical raw materials score
CRM_SCORE = {
    # Low supply risk / not EU critical
    "Cu": 0.98,
    "Zn": 0.95,
    "Ca": 0.93,    
    "Fe": 0.88,
    "Ti": 0.88,
    "Ni": 0.88,
    "Mg": 0.85,
    "K": 0.83,
    "Cr": 0.83, 
    "Zr": 0.80, 
    "Sn": 0.78, #strategic
    "Al": 0.7, # strategic
    "Mn": 0.7, # critical but less supply-critical than e.g. Co, W and REEs
    "W": 0.7, #strategic
    "Ba": 0.68, #strategic
    "V": 0.43, #strategic
    "Sr": 0.35,
    # EU Critical Raw Materials (including Rare earth elements) https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52020DC0474
    "Co": 0.30,
    "La": 0.13,
    "Sm": 0.13,
    "Y": 0.13,
    "Ce": 0.00,
}

# ============================================================
# FUNCTIONS
# ============================================================

def parse_formula(formula):
    """
    Returns list of (element, count)
    Example:
        La8Ca8Fe8Cr8O48 ->
        [('La',8),('Ca',8),('Fe',8),('Cr',8),('O',48)]
    """
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)

    parsed = []

    for element, number in matches:
        count = int(number) if number else 1
        parsed.append((element, count))

    return parsed

def chemistry_scores(formula):

    parsed = parse_formula(formula)

    crm = 0.0
    tox = 0.0
    total = 0

    for element, count in parsed:

        crm += CRM_SCORE.get(element, DEFAULT_CRM) * count
        tox += TOXICITY_SCORE.get(element, DEFAULT_TOXICITY) * count
        total += count

    if total == 0:
        return DEFAULT_CRM, DEFAULT_TOXICITY

    return crm / total, tox / total

def rank_materials(df, weights):
    """
    Compute a weighted ranking from any combination of criteria.
    """

    df = df.copy()
    score = 0.0

    column_map = {
        "hpp": "hpp_rank",
        "price": "price_rank",
        "crm": "crm_rank",
        "toxicity": "toxicity_rank",
    }

    for criterion, weight in weights.items():
        score += weight * df[column_map[criterion]]

    df["score"] = score

    return df.sort_values("score", ascending=False)


# =============================================
# MAIN 
# =============================================
def main():
    parser = argparse.ArgumentParser(description="Rank oxygen carrier materials.")
    parser.add_argument("--hpp", type=float, default=0.30,
                        help="Weight for HPP")
    parser.add_argument("--price", type=float, default=0.35,
                        help="Weight for price")
    parser.add_argument("--crm", type=float, default=0.15,
                    help="Weight for CRM score")
    parser.add_argument("--toxicity", type=float, default=0.15,
                    help="Weight for toxicity score")
    parser.add_argument("leaderboard",
                        help="tab-separated table with columns formula, hpp, price (e.g. complete_unformatted_leaderboard.csv)")

    args = parser.parse_args()

    # ============================================================
    # USER SETTINGS
    # ============================================================

    # Relative importance of each criterion

    
    WEIGHTS = {
        "hpp": args.hpp,
        "price": args.price,
        "crm": args.crm,
        "toxicity": args.toxicity,
    }

    total = sum(WEIGHTS.values())

    if total == 0:
        raise ValueError("At least one weight must be greater than zero.")

    WEIGHTS = {k: v / total for k, v in WEIGHTS.items()}

    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_csv(args.leaderboard, sep="\t")

    # ============================================================
    # Percentile rankings
    # ============================================================

    df["hpp_rank"] = df["hpp"].rank(pct=True)
    # lower price is better
    df["price_rank"] = 1 - df["price"].rank(pct=True)

    # ============================================================
    # Chemistry score
    # ============================================================

    df[["crm", "toxicity"]] = (
        df["formula"]
        .apply(chemistry_scores)
        .apply(pd.Series)
    )

    
    df["crm_rank"] = df["crm"].rank(pct=True)
    df["toxicity_rank"] = df["toxicity"].rank(pct=True)


    # ============================================================
    # Overall score
    # ============================================================

    df = rank_materials(df, WEIGHTS)

    # Save results
    filename = (
        f"ranked_materials"
        f"_hpp_{args.hpp}"
        f"_price_{args.price}"
        f"_crm_{args.crm}"
        f"_tox_{args.toxicity}.csv"
    )

    df.to_csv(filename, index=False)

    print(f"Saved ranking to {filename}")
    # Display best candidates
    print(df[
        [
            "formula",
            "hpp",
            "price",
            "crm",
            "toxicity",
            "score"
        ]
    ].head(30))


if __name__ == "__main__":
    main()
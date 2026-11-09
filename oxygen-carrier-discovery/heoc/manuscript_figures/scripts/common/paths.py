"""Locations shared by the figure scripts. Everything is inside this repository; change paths here only.

Inputs
  heoc/database_r1-r7/r<N>_training.db.gz   training database of each AL run r1 ... r7 (cycles 0-50), unpacked on first
                                        use to manuscript_figures/data/db/ (run_db)
  manuscript_figures/data/              exported tables (compounds.csv, al_picks.csv) and run settings (run_config/)
  manuscript_figures/output/UMAP/       saved UMAP coordinates (the maps are redrawn from them, not refitted)
Outputs
  manuscript_figures/output/            (FIGURES_OUT redirects them, e.g. to compare a rerun with the published files)
  intermediate tables                   temporary folder TABLES (TABLES_DIR), not part of the outputs

The seven runs are called r1 ... r7: r1-r5 = random seeds 1-5, r6 = seed 42, r7 = seed 100.
<TABLES> in the script docstrings = TABLES below.
Scripts in scripts/ import this with
    sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
    from paths import FIGURES, run_db  # noqa: E402
"""
import gzip
import os
import shutil
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]       # heoc/manuscript_figures
HEOC_DIR = BUNDLE.parent                           # oxygen-carrier-discovery/heoc
REPO = BUNDLE.parents[2]                           # repository root
SCRIPTS = BUNDLE / "scripts"
DATA = BUNDLE / "data"
FIGURES = Path(os.environ.get("FIGURES_OUT", BUNDLE / "output"))
MANUSCRIPT = FIGURES / "manuscript"                # Figure_3 ... Figure_7
# intermediate tables passed between the figure scripts (e.g. Figure 6 picks -> Figure 7): a temporary folder, not
# published; make_figures.sh sets TABLES_DIR to a fresh folder and removes it at the end
TABLES = Path(os.environ.get("TABLES_DIR", Path(tempfile.gettempdir()) / "clhp_manuscript_tables"))
UMAP = FIGURES / "UMAP"                            # Projection/ (Figures 4, S2), c50/<run>/ (element distributions)
RANKING_PY = SCRIPTS / "common" / "ranking.py"     # multi-objective ranking (price, Res, EH scores)
METAL_REPO = REPO / "metal"                        # put on sys.path for `import metal`
AL_SRC = REPO / "active-learning" / "src"          # plot_element_distributions.py
RUN_DATA = HEOC_DIR / "database_r1-r7"             # r<N>_training.db.gz
DB_DIR = DATA / "db"                               # unpacked databases (not in git)

RUNS_ALL = [f"r{i}" for i in range(1, 8)]
SEEDS = dict(zip(RUNS_ALL, (1, 2, 3, 4, 5, 42, 100)))


def label(run):
    """Run name r1 ... r7 (also accepts the seed names s1 ... s100; "start" stays "start")."""
    run = str(run).rsplit("_", 1)[-1]
    seed_to_run = {f"s{s}": r for r, s in SEEDS.items()}
    return seed_to_run.get(run, run)


def run_list(default=RUNS_ALL):
    """Runs a script uses: HEOC_RUNS="r1,...,r7" if set, else the script's default."""
    env = os.environ.get("HEOC_RUNS")
    return [label(r) for r in env.split(",")] if env else list(default)


def run_db(run):
    """Path of the training database of a run (r1 ... r7), unpacked from heoc/database_r1-r7/ on first use."""
    run = label(run)
    src = RUN_DATA / f"{run}_training.db.gz"
    dst = DB_DIR / f"{run}_training.db"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with gzip.open(src, "rb") as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g)
        os.replace(tmp, dst)                       # parallel readers never see a half-written file
    return dst


def connect(run):
    """Read-only SQLite connection to a run's training database."""
    import sqlite3
    return sqlite3.connect(f"file:{run_db(run)}?mode=ro", uri=True)

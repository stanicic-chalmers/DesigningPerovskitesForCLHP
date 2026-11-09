# Manuscript figures

Scripts, data and finished figures of the manuscript and the Supporting Information. Everything needed to redraw
them is in this repository; no other folder is read.

```
manuscript_figures/
├── make_figures.sh        redraws all figures below, in order
├── scripts/
│   ├── common/            paths.py (all locations, run names, database access), ranking.py (price / Res / EH
│   │                      scores and the combined ranking), cycle_fix.py (per-run cycle corrections, MAX_CYCLE = 50)
│   ├── figure3_*.py ...   one script per figure (table below)
│   └── si_*.py            Supporting Information
├── data/                  compounds.csv, al_picks.csv, run_config/<run>.json; db/ = unpacked databases (not in git)
└── output/
    ├── manuscript/        Figure_3 ... Figure_7, Figure_S1 (PNG + PDF)
    ├── UMAP/Projection/   Figure 4 map and Figure S2 (+ saved coordinates and fitted UMAP model)
    ├── UMAP/c50/<run>/    element distributions in each run's own latent space (+ saved coordinates)
```

## Data

* `../database_r1-r7/r<N>_training.db.gz`: the training database of each AL run (ASE SQLite database, cycles 0-50; cycle 0
  = the shared starting set). Unpacked on first use to `data/db/` (`paths.run_db`). Cycle numbers in the databases are
  the raw ones; `cycle_fix.py` corrects the skipped cycle numbers of r1, r4, r5, r6 and r7 and cuts every run at 50.
* `data/compounds.csv`: every computed compound (run, cycle, formula, HPC, price, E above hull, tolerance factors,
  Gibbs energies with 0-8 oxygen vacancies at air / steam / fuel reactor conditions); `data/al_picks.csv`: the
  compositions proposed in each AL cycle, with the surrogate prediction; `data/run_config/<run>.json`: AL settings.
* UMAP coordinates (`output/UMAP/**/umap_*_frozen_coords.csv`): fitting the maps needs the WAE checkpoints of the runs,
  which are not published, so the UMAP figures are redrawn from these coordinates (see `figure4_umap_projection.py`).

## Run names

The seven AL runs differ only in the random seed. Everywhere they are called **r1 ... r7**:

| Run | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|
| Seed | 1 | 2 | 3 | 4 | 5 | 42 | 100 |

## Running

```bash
bash make_figures.sh                                  # python on PATH
CONTAINER=/path/to/image.sif bash make_figures.sh     # or an apptainer image (active-learning/container/)
```

`metal/` and `active-learning/src/` are imported from this repository, so the figures use the same code as the AL
runs. Intermediate tables and each script's printout go to a temporary folder that is removed at the end; if a
script fails, the folder is kept and its location printed.

## Figures

| Figure | Script | Output |
|---|---|---|
| 3 | `figure3_mean_hpc_per_cycle.py` | `manuscript/Figure_3` (+ `Figure_3_per_cycle.csv`) |
| 4 | `figure4_umap_projection.py` | `manuscript/Figure_4` = `UMAP/Projection/umap_r6_c50_7runs_frozen` |
| 5 | `figure5_element_enrichment.py` | `manuscript/Figure_5`  |
| 6 | `figure6_multiobjective_mmr.py --eh base` / `--eh precautionary` | `manuscript/Figure_6` = `Figure_6_basecase`; `Figure_6_precautionary`  |
| 7 | `figure7_weight_sensitivity_tables.py`, then `figure7_candidates.py` | `manuscript/Figure_7` (properties of the Figure 6 picks) |
| S1 | `figureS1_sr_ratio_tables.py`, then `figureS1_element_enrichment_vs_sr_ratio.py` | `manuscript/Figure_S1` (element enrichment vs steam-reactor H₂O/H₂) |
| S2 | `figureS2_umap_site_chemistry.py` | `UMAP/Projection/umap_r6_c50_7runs_site_chemistry` |
| SI | `figure4_umap_projection.py <coords>` | `UMAP/Projection/umap_r6_c50_frozen` (r1-r5 on the r6 map) |
| SI | `si_element_distribution.py` | `UMAP/c50/<run>/` element distributions in each run's latent space |

EH (environmental and health impact) uses the base-case scores by default (`ranking.TOXICITY_SCORE`);
`--eh precautionary` gives the precautionary sensitivity (`Figure_6_precautionary`).

#!/bin/bash
# Redraw all published figures from the data in this repository (runs r1-r7: r1-r5 = seeds 1-5, r6 = seed 42,
# r7 = seed 100). Inputs: heoc/database_r1-r7/r<N>_training.db.gz (unpacked to data/db/ on first use), data/, and the saved
# UMAP coordinates in output/UMAP/. Outputs: output/manuscript/ (Figures 3-7), output/UMAP/ (Figure 4 map, Figure S2,
# element distributions). Intermediate tables (e.g. the Figure 6 picks read by Figure 7) and each script's printout
# go to a temporary folder, removed at the end unless a script failed (its log is then kept and named).
#
# Python: the one on PATH, or set CONTAINER to an apptainer image (e.g. built from active-learning/container/).
# FIGURES_OUT=/some/dir bash make_figures.sh writes everything there instead (e.g. to compare with the published files;
# copy output/UMAP/ there first, the UMAP figures are drawn from its coordinates).
# Usage: bash make_figures.sh
HERE=$(dirname "$(readlink -f "$0")")            # heoc/manuscript_figures
SCRIPTS=$HERE/scripts
OUT=${FIGURES_OUT:-$HERE/output}
TMP=$(mktemp -d)
export TABLES_DIR=$TMP/tables
LOGS=$TMP/logs
if [ -n "$CONTAINER" ]; then PY=(apptainer exec "$CONTAINER" python -u); else PY=(python -u); fi
export MPLBACKEND=Agg
mkdir -p "$LOGS"
FAILED=()

run() {  # run <script> [args...]; printout -> $LOGS/<script>[_<tag>].txt (tag from LOGTAG)
    local script=$1; shift
    local log="$LOGS/${script%.py}${LOGTAG:+_$LOGTAG}.txt"
    echo "[$(date +%H:%M:%S)] $script $*"
    if ! (cd "$HERE" && "${PY[@]}" "$SCRIPTS/$script" "$@" > "$log" 2>&1); then
        FAILED+=("$script $*"); echo "    FAILED, see $log"
    fi
}

# ---------------------------------------------------------------- manuscript
run figure3_mean_hpc_per_cycle.py                                   # Figure 3 (from data/compounds.csv)
run figure4_umap_projection.py                                      # Figure 4 (saved UMAP coordinates, r6 map)
run figure5_element_enrichment.py                                   # Figure 5
LOGTAG=base run figure6_multiobjective_mmr.py --eh base             # Figure 6 (= Figure_6_basecase)
LOGTAG=precautionary run figure6_multiobjective_mmr.py --eh precautionary   # Figure_6_precautionary (EH sensitivity)
run figure7_weight_sensitivity_tables.py --eh base                  # tables for Figure 7
run figure7_candidates.py --eh base                                 # Figure 7 (the Figure 6 picks)

# ---------------------------------------------------------------- Supporting Information
run figureS1_sr_ratio_tables.py                                    # tables: HPC vs steam-reactor H2O/H2
run figureS1_element_enrichment_vs_sr_ratio.py                      # Figure S1
run figureS2_umap_site_chemistry.py                                 # Figure S2 (HPC, dominant A/B element; Figure 4 map)
LOGTAG=r6 run figure4_umap_projection.py "$OUT/UMAP/Projection/umap_r6_c50_frozen_coords.csv"   # r6 compounds on the r6 map
run si_element_distribution.py                                      # output/UMAP/c50/<run>/ (each run's own map)

echo "Done at $(date). Failed: ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  $f"; done
if [ ${#FAILED[@]} -eq 0 ]; then rm -rf "$TMP"; else echo "Logs kept in $LOGS"; fi

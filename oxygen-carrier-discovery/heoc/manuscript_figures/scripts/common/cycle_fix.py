"""Per-run cycle corrections shared by the figure scripts.

All runs are cut at MAX_CYCLE, so that every run is compared over the same cycles.
Skipped cycle numbers: when a job ended right after creating the next cycle folder, the restart found no progress in
it and started a new, higher-numbered cycle (al_util.py: "No progress found for cycle N. Starting new cycle N+1"), so
that cycle number stays empty and every later cycle of that run is numbered one too high. No work was lost. Affected:
r1 (cycle 47 empty), r4 (cycle 30), r5 (cycle 38), r6 and r7 (cycle 24); their cycles after the empty one are shifted
down by one here. The cycle numbers stored in the databases (heoc/database_r1-r7) are the raw ones.
"""
MAX_CYCLE = 50
SHIFT = {  # run: (first raw cycle affected, offset) = (empty cycle + 1, -1)
    "r1": (48, -1), "r4": (31, -1), "r5": (39, -1), "r6": (25, -1), "r7": (25, -1)}


def fix_cycle(run, cycle):
    """Corrected cycle number, or None if the row is to be left out (beyond MAX_CYCLE)."""
    cycle = int(cycle)
    first, offset = SHIFT.get(run, (None, 0))
    if first is not None and cycle >= first:
        cycle += offset
    return None if cycle > MAX_CYCLE else cycle

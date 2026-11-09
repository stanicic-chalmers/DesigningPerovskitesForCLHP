"""Shared utilities for the heoc pyscripts pipeline.

Centralises boilerplate that was duplicated across many scripts:
path resolution, metal import setup, cycle detection, crystal-space
loading, SLURM process count, and working-directory management.
"""

import os
import sys
from contextlib import contextmanager
from glob import glob

# ---------------------------------------------------------------------------
# Path constants (relative to *this* file, i.e. heoc/pyscripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)          # heoc/


# ---------------------------------------------------------------------------
# Metal import setup
# ---------------------------------------------------------------------------
def setup_metal_path():
    """Add the ``metal/`` package to ``sys.path`` so it can be imported.

    The metal directory is located three levels above the pyscripts folder
    (i.e. ``oxygen-carrier-discovery/../../metal``).
    """
    metal_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'metal'))
    if metal_path not in sys.path:
        sys.path.append(metal_path)


# ---------------------------------------------------------------------------
# Working-directory context manager
# ---------------------------------------------------------------------------
@contextmanager
def working_directory(path):
    """Temporarily change the working directory, restoring it on exit."""
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------
def get_latest_cycle(directory, base_dir=None):
    """Return the latest cycle number from ``cycle*`` folders.

    Parameters
    ----------
    directory : str
        Data directory name (e.g. ``'clhp_r1'``).
    base_dir : str, optional
        Base directory containing *directory*.  Defaults to :data:`PARENT_DIR`.

    Returns
    -------
    int
    """
    if base_dir is None:
        base_dir = PARENT_DIR
    folders = glob(f'{base_dir}/{directory}/cycle*')
    return sorted(int(os.path.basename(f).replace('cycle', ''))
                  for f in folders)[-1]


# ---------------------------------------------------------------------------
# Crystal-space loading
# ---------------------------------------------------------------------------
def load_crystal_space(directory, sqs=False, base_dir=None):
    """Load a :class:`CrystalSpace` from a ``.spa`` file.

    Parameters
    ----------
    directory : str
        Data directory name (e.g. ``'clhp_r1'``).
    sqs : bool
        If *True*, load ``crystal_space_sqs.spa``; otherwise
        ``crystal_space.spa``.
    base_dir : str, optional
        Base directory containing *directory*.  Defaults to :data:`PARENT_DIR`.

    Returns
    -------
    CrystalSpace
    """
    from metal.applications.crystals.crystal_space import CrystalSpace

    if base_dir is None:
        base_dir = PARENT_DIR
    spa_name = 'crystal_space_sqs.spa' if sqs else 'crystal_space.spa'
    spa_path = os.path.join(base_dir, directory, spa_name)
    return CrystalSpace.read(spa_path)


# ---------------------------------------------------------------------------
# SLURM / process count
# ---------------------------------------------------------------------------
def get_process_count(args_processes=None):
    """Resolve the number of worker processes.

    Priority: *args_processes* > ``SLURM_NTASKS`` env var > 1.

    Parameters
    ----------
    args_processes : int or None
        Value from argparse (``--processes``).

    Returns
    -------
    int
    """
    if args_processes is not None:
        return args_processes
    return int(os.getenv('SLURM_NTASKS', 1))


# ---------------------------------------------------------------------------
# Proxy-element detection
# ---------------------------------------------------------------------------
def uses_proxy_elements(crystal_space):
    """Check whether *crystal_space* contains proxy elements.

    Returns *True* if any species in the first structure's chemical-symbol
    lists is a key in :data:`config.PROXY_TO_REAL`.
    """
    from config import PROXY_TO_REAL
    all_species = {s for syms in crystal_space.chemical_symbols_list[0]
                   for s in syms}
    return bool(all_species & set(PROXY_TO_REAL))


# ---------------------------------------------------------------------------
# Sublattice info convenience
# ---------------------------------------------------------------------------
def get_sublattice_info(crystal_space):
    """Return sublattice map and fractional sizes for the first structure.

    Returns
    -------
    sublattice_map : dict
        ``{sl_symbol: [chemical_symbols]}``
    sublattice_fractions : dict
        ``{sl_symbol: fraction}`` where fraction = n_sites / n_atoms_total.
    """
    from metal.applications.crystals.tools import (get_sublattice_map,
                                                   get_atoms_per_sublattice)
    prim = crystal_space.structure_list[0]
    cs = crystal_space.cluster_space_list[0]
    sl_map = get_sublattice_map(prim, cs)
    atoms_per_sl = get_atoms_per_sublattice(prim, cs)
    n_prim = len(prim)
    sl_fracs = {sl: n / n_prim for sl, n in atoms_per_sl.items()}
    return sl_map, sl_fracs

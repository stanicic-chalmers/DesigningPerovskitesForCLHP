#!/usr/bin/env python3
"""Set up a new active-learning run from the published start set.

Creates heoc/<name>/ with
    training.db         initial database (cycle 0; unpacked from heoc/clhp/training.db.gz)
    crystal_space.spa   composition space (copied from heoc/clhp/)
    cycle0/             empty; the AL loop writes cycle1, cycle2, ... next to it
The start set is the one used for the runs in the paper: 3903 random compositions, of which 2103 pass the
charge-neutrality and tolerance-factor filters and carry G(v) and HPC labels.

The run name must contain 'clhp': the stability descriptors of every new composition are computed with the
element set selected by that name (see build_training_database.get_stability_factors).

Usage (from heoc/pyscripts/):
    python setup_run.py --name clhp_r1
then start the loop from active-learning/src/:
    python -u active_learning_HEO.py --method clhp_r1 --seed 1 --log-path ../../oxygen-carrier-discovery/heoc/clhp_r1/run
"""
import argparse
import gzip
import hashlib
import os
import shutil
import sys

HEOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = os.path.join(HEOC, 'clhp')
SHA256_TRAINING_DB = 'fbba63d9f9248424bf6d6d58cbf5366c17b0847e4796f5f54a9f9553dd072d5d'  # uncompressed


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def setup_run(name, overwrite=False):
    if 'clhp' not in name:
        sys.exit(f"Run name must contain 'clhp' (e.g. clhp_r1), got '{name}'")
    if os.path.abspath(os.path.join(HEOC, name)) == os.path.abspath(START):
        sys.exit("'clhp' holds the published start set; choose another name, e.g. clhp_r1")
    run_dir = os.path.join(HEOC, name)
    db = os.path.join(run_dir, 'training.db')
    if os.path.exists(db) and not overwrite:
        sys.exit(f'{db} exists already; use --overwrite to replace it (this deletes the results of that run)')

    os.makedirs(os.path.join(run_dir, 'cycle0'), exist_ok=True)
    shutil.copy2(os.path.join(START, 'crystal_space.spa'), run_dir)
    with gzip.open(os.path.join(START, 'training.db.gz'), 'rb') as src, open(db, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    if sha256(db) != SHA256_TRAINING_DB:
        sys.exit(f'Checksum of {db} does not match the published start set')
    print(f'Run folder ready: {run_dir}')
    print(f'  training.db ({os.path.getsize(db) / 1e6:.0f} MB, checksum ok), crystal_space.spa, cycle0/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--name', required=True, help="run name, must contain 'clhp' (e.g. clhp_r1)")
    parser.add_argument('--overwrite', action='store_true', help='replace an existing training.db')
    args = parser.parse_args()
    setup_run(args.name, args.overwrite)

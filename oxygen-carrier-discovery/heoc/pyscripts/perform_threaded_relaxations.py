#!/usr/bin/env python3

import argparse
from glob import glob
import math
import os
import sys
import warnings
sys.path.append(os.path.join(os.getcwd().split('oxygen-carrier-discovery')[0],
                             'metal'))
from metal.applications.crystals.thermodynamics import\
    update_relax_structures_dbs_threaded


def perform_threaded_relaxations(directory='clhp', selection=None,
                                  pattern='sqs_*.db', cycle=None,
                                  n_threads=2, gpu_id=0, logger=None,
                                  divide_by=1, index=0):
    '''Relax all structures in SQS databases using threading to maximize GPU utilization

    This function uses threading to run multiple relaxations concurrently on a single GPU,
    which keeps the GPU busy (some clusters stop jobs with low GPU utilisation).

    Parameters
    ----------
    directory
        data directory
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    pattern
        pattern used for selecting files
    cycle
        number of cycle, the default is the latest; if specified then any
        existing data in the corresponding folder is overwritten
    n_threads
        number of concurrent relaxations (2-4 recommended for 40GB GPU)
        Higher values = more GPU utilization but more memory usage
    gpu_id
        GPU id to use
    divide_by
        divide the total files into this many chunks (default 1 = all files)
    index
        0-based index of which chunk to process
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if cycle is not None:
        overwrite = True
    else:
        cycle = sorted([
            int(os.path.basename(folder).replace('cycle', ''))
            for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
        overwrite = False

    # Get all filenames
    all_filenames = sorted(glob(f'{parent_dir}/{directory}/cycle{cycle}/{pattern}'))

    # Select this chunk (same ceil-based chunking as perform_phonon_calculations)
    divide_by = max(divide_by, 1)
    chunk_size = math.ceil(len(all_filenames) / divide_by)
    filenames = all_filenames[index * chunk_size:(index + 1) * chunk_size]

    print(f"Found {len(all_filenames)} database files in cycle{cycle}, "
          f"processing chunk {index+1} of {divide_by}: {len(filenames)} files")
    print(f"Using {n_threads} concurrent threads per database to maximize GPU utilization")

    # Process each database file with threaded relaxations
    for i, filename in enumerate(filenames, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(filenames)}: {os.path.basename(filename)}")
        print(f"{'='*60}")

        update_relax_structures_dbs_threaded(
            filename=filename,
            selection=selection,
            overwrite=overwrite,
            relax_kwargs={'verbose': False},
            potential='CHGNet',
            n_threads=n_threads,
            gpu_id=gpu_id,
            relax_cell=True
        )

        print(f"Completed: {os.path.basename(filename)}\n")

    if logger is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"All {len(filenames)} databases processed!")
        logger.info(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"All {len(filenames)} databases processed!")
        print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Relax structures using threading to maximize GPU utilization'
    )
    parser.add_argument('--directory',
                        type=str,
                        default='clhp',
                        help='data directory')
    parser.add_argument('--cycle',
                        default=None,
                        type=int,
                        help='cycle number; default is latest')
    parser.add_argument('--selection',
                        default=None,
                        nargs='+',
                        help='integers or strings used to select rows')
    parser.add_argument('--pattern',
                        default='sqs_*.db',
                        help='pattern used for selecting files')
    parser.add_argument('--n_threads',
                        type=int,
                        default=2,
                        help='number of concurrent relaxations (2-4 recommended for 40GB GPU)')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU id to use')
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # ensure that all warnings are printed
    warnings.simplefilter("always")

    # parse selection
    if args.selection is None:
        selection = args.selection
    else:
        if len(args.selection) > 1:
            try:
                selection = [int(s) for s in args.selection]
            except Exception:
                selection = args.selection
        else:
            try:
                selection = int(args.selection[0])
            except Exception:
                selection = args.selection[0]

    # Perform threaded relaxations
    perform_threaded_relaxations(
        directory=args.directory,
        selection=selection,
        pattern=args.pattern,
        cycle=args.cycle,
        n_threads=args.n_threads,
        gpu_id=args.gpu_id
    )

    os.chdir(cwd)

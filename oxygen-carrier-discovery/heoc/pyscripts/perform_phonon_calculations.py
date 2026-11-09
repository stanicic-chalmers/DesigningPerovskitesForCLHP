#!/usr/bin/env python3

import argparse
from glob import glob
import os
import sys
import warnings
import math # Import math for ceil function
from pyscripts_utils import (setup_metal_path, working_directory, PARENT_DIR,
                             get_latest_cycle, get_process_count)
setup_metal_path()
from metal.applications.crystals.thermodynamics import\
    update_phonopy_phonons_dbs


def perform_phonon_calculations(directory='clhp', selection=None,
                                pattern='sqs_*.db', cycle=None,
                                asynchronus=True, connect_each_row=True,
                                use_lock_file=False, processes=1,
                                batch_size=None,
                                logging=None,
                                use_gpus=False, semi_parallel=False,
                                divide_by=1, index=0,
                                n_relax_threads=0):
    '''Update all SQS databases found in the latest cycle folder after
    relaxing the corresponding structures and performing phonopy calculations

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
        number of cycle, the dafault is the latest; if specified then any
        existing data in the corresponding folder is overwritten
    asynchronous
        run the processes asynchronously; ignored if sequential
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    processes
        number of processes to start; if the number is less than two these are
        run sequentially
    use_gpus
        if GPUs should be used; raises exception if unavailable
    semi_parallel
        run the calculations in a semi-parallel fashion, ignored if GPUs are
        not used
    divide_by
        integer to divide the total number of files by (e.g., 20 for 1/20th)
    index
        0-based index of the chunk of files to process (e.g., 0 for the first
        chunk, 1 for the second, etc.)
    n_relax_threads
        if > 0, run threaded GPU relaxations on this chunk before phonon
        calculations (recommended 2-4 for 40GB GPU)
    '''

    parent_dir = PARENT_DIR

    if cycle is not None:
        overwrite = True
    else:
        cycle = get_latest_cycle(directory)
        overwrite = False

    # Get all filenames
    all_filenames = sorted(glob(f'{parent_dir}/{directory}/cycle{cycle}/{pattern}'))

    # --- Logic for dividing and selecting a chunk ---
    
    # Ensure divide_by is at least 1 and index is valid
    if divide_by < 1:
        warnings.warn("divide_by must be >= 1. Using 1 (all files).")
        divide_by = 1
        
    if index < 0 or index >= divide_by:
        warnings.warn(f"index ({index}) is outside the valid range [0, {divide_by-1}] for divide_by={divide_by}. Using index 0.")
        index = 0

    total_files = len(all_filenames)
    # Ceil-based chunking (consistent with perform_threaded_relaxations)
    chunk_size = math.ceil(total_files / max(divide_by, 1))
    filenames = all_filenames[index * chunk_size : (index + 1) * chunk_size]

    print(f"Total files found: {total_files}")
    print(f"Processing chunk {index+1} of {divide_by}: {len(filenames)} files")
    if not filenames:  # fewer files than chunks: this chunk has nothing to do
        return

    # Run threaded relaxations on this chunk before phonon calculations
    if n_relax_threads > 0 and use_gpus:
        from perform_threaded_relaxations import perform_threaded_relaxations as _threaded_relax
        print(f"Running threaded relaxations with {n_relax_threads} threads before phonon calculations...")
        _threaded_relax(
            directory=directory, selection=selection, pattern=pattern,
            cycle=cycle, n_threads=n_relax_threads,
            divide_by=divide_by, index=index)
        print("Threaded relaxations complete.")

    update_phonopy_phonons_dbs(
            filenames=filenames, 
            overwrite=False,
            selection=selection,
            asynchronus=asynchronus,
            connect_each_row=connect_each_row,
            use_lock_file=use_lock_file,
            processes=processes,
            use_gpus=use_gpus,
            batch_size=batch_size,
            logging=logging,
            semi_parallel=semi_parallel,
            relax_kwargs={'verbose': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='clhp',
                        help='data directory')
    parser.add_argument('--cycle',
                        default=None,
                        type=int,
                        help='cycle, default is latest; overwrites existing'
                        ' data if not None')
    parser.add_argument('--selection',
                        default=None,
                        nargs='+',
                        help='integers or strings used to select rows')
    parser.add_argument('--pattern',
                        default='sqs_*.db',
                        help='pattern used for selecting files')
    parser.add_argument('--divide_by',
                        type=int,
                        default=1,
                        help='Divide the list of database files into this many chunks (N) to process 1/N-th of the data.')
    parser.add_argument('--index',
                        type=int,
                        default=0,
                        help='0-based index (0 to N-1) of the chunk of files to process.')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    parser.add_argument('--use_gpus',
                        action='store_true',
                        help='use GPUs')
    parser.add_argument('--semi_parallel',
                        action='store_true',
                        help='run semi-parallel calculations')
    parser.add_argument('--use_preloader',
                        action='store_true',
                        help='run GPU memory preloader thread to maintain GPU utilization')
    parser.add_argument('--n_relax_threads',
                        type=int,
                        default=0,
                        help='if > 0, run threaded GPU relaxations before phonon calcs (2-4 recommended)')
    parser.add_argument('--n_batch',
                        type=int,
                        default=0)
    args = parser.parse_args()

    # Start GPU memory preloader if requested
    if args.use_preloader and args.use_gpus:
        import threading
        al_src = os.path.join(PARENT_DIR, '..', '..', 'active-learning', 'src')
        sys.path.insert(0, os.path.abspath(al_src))
        from al_util import _gpu_memory_preloader
        _preloader_stop = threading.Event()
        _preloader = threading.Thread(
            target=_gpu_memory_preloader, args=(0, _preloader_stop), daemon=True)
        _preloader.start()

    # ensure that all warnings are printed
    warnings.simplefilter("always")

    processes = get_process_count(args.processes)

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

    with working_directory(PARENT_DIR):
        perform_phonon_calculations(directory=args.directory,
                                    selection=selection,
                                    pattern=args.pattern,
                                    asynchronus=True,
                                    connect_each_row=True,
                                    use_lock_file=(processes == 1),
                                    cycle=args.cycle,
                                    processes=processes,
                                    use_gpus=args.use_gpus,
                                    semi_parallel=args.semi_parallel,
                                    divide_by=args.divide_by,
                                    batch_size=args.n_batch,
                                    index=args.index,
                                    n_relax_threads=args.n_relax_threads)

    # Stop preloader thread before exit to avoid CUDA teardown crash
    if args.use_preloader and args.use_gpus:
        _preloader_stop.set()
        _preloader.join(timeout=5)
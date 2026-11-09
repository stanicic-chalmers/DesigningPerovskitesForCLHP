"""SLURM fan-out for phonon calculations.

Submits parallel sbatch jobs for phonon calculations across multiple GPUs,
runs a chunk locally on the main GPU, and handles fallback for stuck jobs.
"""

import os
import re
import subprocess
import time

from perform_phonon_calculations import perform_phonon_calculations


def submit_phonon_slurm_jobs(config, cycle, logger):
    """Run phonon calculations on the main GPU and fan out to additional SLURM jobs.

    The work is split into (phonon_slurm_jobs + 1) chunks. The main process
    runs chunk 0 on its own GPU while the sbatch jobs handle chunks 1..N.
    After the main chunk finishes, it polls squeue until all sbatch jobs complete.

    Parameters
    ----------
    config : Config
        Configuration object with SLURM settings.
    cycle : int
        Current active learning cycle number.
    logger : logging.Logger
        Logger instance.
    """
    n_sbatch_jobs = config.phonon_slurm_jobs
    total_chunks = n_sbatch_jobs + 1  # main process + sbatch jobs
    pyscripts_dir = os.path.join(config.project_root,
                                 'oxygen-carrier-discovery', 'heoc', 'pyscripts')
    phonon_script = os.path.join(pyscripts_dir, 'perform_phonon_calculations.py')
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Submit sbatch jobs for chunks 1..N
    job_ids = []
    for i in range(1, total_chunks):
        preloader_flag = ' --use_preloader' if config.slurm_use_preloader else ''
        relax_flag = f' --n_relax_threads {config.relax_threads}' if config.relax_threads else ''
        batch_flag = f' --n_batch {config.slurm_phonon_batch}' if config.slurm_phonon_batch else ''
        python_cmd = (
            f'python -u {phonon_script}'
            f' --directory {config.method}'
            f' --cycle {cycle}'
            f' --divide_by {total_chunks}'
            f' --index {i}'
            f' --use_gpus'
            f'{preloader_flag}'
            f'{relax_flag}'
            f'{batch_flag}'
        )

        if config.slurm_container:
            run_cmd = (
                f'apptainer exec --nv {config.slurm_container}'
                f' {python_cmd}'
            )
        else:
            run_cmd = f'{python_cmd}'

        job_name = f'phonon_{config.method}_c{cycle}_chunk{i}'
        log_file = os.path.join(config.log_path, f'phonon_c{cycle}_chunk{i}.log')

        sbatch_script = f"""#!/bin/bash
#SBATCH -A {config.slurm_account}
#SBATCH --gpus=1
#SBATCH -t {config.slurm_time}
#SBATCH -J {job_name}
#SBATCH -o {log_file}

cd {src_dir}

echo "Phonon job chunk {i}/{total_chunks} started: $(date)"
{run_cmd}
echo "Phonon job chunk {i}/{total_chunks} ended: $(date)"
"""

        proc = subprocess.run(
            ['sbatch'], input=sbatch_script, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'sbatch failed for chunk {i}: {proc.stderr}')

        match = re.search(r'Submitted batch job (\d+)', proc.stdout)
        if not match:
            raise RuntimeError(f'Could not parse job ID from: {proc.stdout}')
        job_id = match.group(1)
        job_ids.append(job_id)
        logger.info(f'Submitted phonon chunk {i}/{total_chunks} as SLURM job {job_id}')

    # Run chunk 0 locally on the main GPU (threaded relaxations run inside if configured)
    logger.info(f'Running phonon chunk 0/{total_chunks} locally on main GPU...')
    perform_phonon_calculations(
        processes=1, directory=f'{config.method}', use_gpus=True,
        logging=logger, batch_size=config.phonon_batch_size,
        divide_by=total_chunks, index=0,
        n_relax_threads=config.relax_threads or 0)
    logger.info(f'Local phonon chunk 0/{total_chunks} complete.')

    # Wait for sbatch jobs, with fallback for stuck PENDING jobs
    _wait_for_slurm_jobs(config, cycle, job_ids, total_chunks, logger)


def _wait_for_slurm_jobs(config, cycle, job_ids, total_chunks, logger):
    """Poll sbatch jobs, cancel stuck PENDING ones and run them locally."""
    job_chunk_map = {jid: chunk_i for jid, chunk_i in zip(job_ids, range(1, total_chunks))}
    pending_since = {}  # job_id -> timestamp when first seen PENDING after main chunk done
    fallback_chunks = []  # chunks cancelled and run locally

    n_sbatch_jobs = len(job_ids)
    logger.info(f'Waiting for {n_sbatch_jobs} remote phonon SLURM jobs '
                f'(queue timeout: {config.slurm_queue_timeout}s)...')

    while job_chunk_map:
        time.sleep(config.slurm_poll_interval)

        remaining_ids = list(job_chunk_map.keys())
        proc = subprocess.run(
            ['squeue', '--jobs=' + ','.join(remaining_ids),
             '--noheader', '--format=%i %T'],
            capture_output=True, text=True)
        active_lines = [l for l in proc.stdout.strip().split('\n') if l.strip()]

        if not active_lines:
            break

        active_states = {l.split()[0]: l.split()[1] for l in active_lines}

        # Check for PENDING jobs that have exceeded the queue timeout
        now = time.time()
        to_cancel = []
        for jid in list(active_states.keys()):
            if active_states[jid] == 'PENDING':
                if jid not in pending_since:
                    pending_since[jid] = now
                elif now - pending_since[jid] >= config.slurm_queue_timeout:
                    to_cancel.append(jid)
            else:
                pending_since.pop(jid, None)

        # Cancel timed-out pending jobs and collect their chunks for local execution
        for jid in to_cancel:
            chunk_i = job_chunk_map.pop(jid)
            pending_since.pop(jid, None)
            subprocess.run(['scancel', jid], capture_output=True)
            logger.info(f'Cancelled PENDING job {jid} (chunk {chunk_i}) after '
                        f'{config.slurm_queue_timeout}s — will run locally.')
            fallback_chunks.append(chunk_i)

        # Remove finished jobs (no longer in squeue) from tracking
        finished = [jid for jid in remaining_ids if jid not in active_states]
        for jid in finished:
            job_chunk_map.pop(jid, None)

        if job_chunk_map:
            running = sum(1 for st in active_states.values() if st == 'RUNNING')
            pending = sum(1 for st in active_states.values() if st == 'PENDING')
            status_parts = []
            if running:
                status_parts.append(f'{running} running')
            if pending:
                status_parts.append(f'{pending} pending')
            logger.info(f'  {", ".join(status_parts)} — '
                        + ', '.join(f'{jid}({active_states.get(jid, "done")})'
                                    for jid in job_chunk_map))

    # Run fallback chunks locally on the main GPU
    for chunk_i in fallback_chunks:
        logger.info(f'Running fallback phonon chunk {chunk_i}/{total_chunks} locally...')
        perform_phonon_calculations(
            processes=1, directory=f'{config.method}', use_gpus=True,
            logging=logger, batch_size=config.phonon_batch_size,
            divide_by=total_chunks, index=chunk_i,
            n_relax_threads=config.relax_threads or 0)
        logger.info(f'Fallback chunk {chunk_i}/{total_chunks} complete.')

    # Check completed sbatch jobs for failures via sacct (skip cancelled fallback jobs)
    cancelled_ids = set()
    for jid, chunk_i in zip(job_ids, range(1, total_chunks)):
        if chunk_i in fallback_chunks:
            cancelled_ids.add(jid)
    sacct_ids = [jid for jid in job_ids if jid not in cancelled_ids]
    if sacct_ids:
        try:
            proc = subprocess.run(
                ['sacct', '-j', ','.join(sacct_ids), '--format=JobID,State,ExitCode',
                 '--noheader', '--parsable2'],
                capture_output=True, text=True)
            failed = []
            for line in proc.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    job_id_field, state = parts[0], parts[1]
                    if '.' not in job_id_field and state not in ('COMPLETED', 'CANCELLED', ''):
                        failed.append(f'{job_id_field}: {state}')
            if failed:
                raise RuntimeError(
                    f'Phonon SLURM jobs failed:\n' + '\n'.join(failed) +
                    f'\nCheck logs in {config.log_path}/phonon_c{cycle}_chunk*.log')
        except FileNotFoundError:
            logger.warning('sacct not available — skipping post-job failure check. '
                           f'Verify logs manually: {config.log_path}/phonon_c{cycle}_chunk*.log')

    logger.info(f'All {total_chunks} phonon chunks completed successfully'
                + (f' ({len(fallback_chunks)} ran locally as fallback).'
                   if fallback_chunks else '.'))

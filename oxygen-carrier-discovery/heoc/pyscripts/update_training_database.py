#!/usr/bin/env python3

import argparse
from ase.data import atomic_masses, atomic_numbers
from glob import glob
import multiprocessing
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from scipy.interpolate import CubicSpline
import sys
import traceback
import warnings
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.tools import (
    _add_structure_db_mp,
    apply_function_db,
    get_rows_db,
    get_row_ids,
    get_average_abundance,
    get_price,
    get_mu_oxygen,
    get_mu_oxygen_fr,
    get_mu_oxygen_sr)
from build_training_database import (get_stability_factors,
                                     get_average_metal_abundance,
                                     estimate_configurational_entropies)


def update_training_database(directory='clhp', pattern='sqs_*.db', cycle=None,
                             asynchronus=True, use_lock_file=True,
                             processes=1, dimensions=(1, 2, 3), quadratic_fit=True):
    '''Update the database with training structures with the results from
    phonon calculations based on the SQS structures generated from the
    original training set or samples produced by the network in later cycles

    Parameters
    ----------
    directory
        data directory
    asynchronous
        run the processes asynchronously; ignored if sequential
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    processes
        number of processes to start; if the number is less than two these are
        run sequentially
    '''
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Define list of keyword arguments
    if cycle is not None:
        overwrite = True
    else:
        cycle = sorted([
            int(os.path.basename(folder).replace('cycle', ''))
            for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
        overwrite = True 
    fun_kwargs = {'cycle': cycle,
                  'directory': directory,
                  'use_lock_file': use_lock_file,
                  'overwrite': overwrite,
                  'quadratic_fit': quadratic_fit,
                  'dimensions': tuple(dimensions)}
    filenames = glob(f'{parent_dir}/{directory}/cycle{cycle}/{pattern}')
    kwargs_list = [dict(filename=filename, **fun_kwargs) for filename in
                   filenames]
    
    processes = min(len(kwargs_list), processes)
    if processes == 1:
        for kwargs in kwargs_list:
            _update_training_structure_sqs_db_mp(kwargs)
    else:
        Pool = multiprocessing.Pool
        with Pool(processes) as pool:
            if asynchronus:
                results = pool.map_async(
                    _update_training_structure_sqs_db_mp,
                    kwargs_list
                )
                results.get()
            else:
                pool.map(
                    _update_training_structure_sqs_db_mp,
                    kwargs_list
                )


def _update_training_structure_sqs_db_mp(kwargs) -> None:
    """Parallelisable version of _update_training_structures_sqs_db"""
    try:
        _update_training_structures_sqs_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to add/update training structure for the database'
            f' {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))


def _update_training_structures_sqs_db(filename, directory,
                                       dimensions=(1, 2, 3),
                                       quadratic_fit=True,
                                       cycle=None, overwrite=True, use_lock_file=True):
    '''Get a list of structures after having calculated the vacancy formation
    energies using the data obtained from phonon calculations based on the
    SQS structures found in the databases, and use them to determine the equilibrium
    vacancy contents in each reactor, the HPC (FR - SR) and the OTC (FR - AR)

    Parameters
    ----------
    filename
        path to database with SQS structures
    crystal_space
        CrystalSpace object
    directory
        data directory
    conditions
        dictionary of reactor conditions
    dimensions
        SQS supercell dimensions
    cycle
        training cycle (used for overwrite logic)
    overwrite
        overwrite exisiting entries if the formulas match
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    '''
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    # check if row exists or has been updated
    train_file = f'{parent_dir}/{directory}/training.db'
    if cycle == 0 or not overwrite:
        formula = os.path.basename(
            filename).replace('.db', '').replace('sqs_', '')
        rows = get_rows_db(filename=train_file, selection=f'formula={formula}')
        if rows is None or len(rows) < 1:
            if cycle == 0:
                print(f'There is no row in {train_file} for {formula}')
                return
        else:
            if not overwrite and rows[0].get('energy', False) is not False:
                print(f'The row in {train_file} for {formula} has already been'
                      f' updated')
                return

    # Read crystal space
    try:
        crystal_space = CrystalSpace.read(f'{parent_dir}/{directory}/crystal_space.spa')
    except Exception:
        try:
            spa_file_path = os.path.join(parent_dir, f'{directory}/crystal_space.spa')
            crystal_space = CrystalSpace.read(spa_file_path)
        except Exception:
            print("Path to crystal_space.spa incorrect")
            raise FileNotFoundError(f"Could not find crystal_space.spa in {directory}")
            

    # define conditions in reactors (WATER SPLITTING)
    conditions = {
        'AR': {'T': 1000 + 273.15, 'pO2': 0.05},
        'FR': {'T': 900 + 273.15, 'ratio': 1/9}, # CO2/CO = 1/9, FeO/Fe transition (pO2 ~ 1.7e-17 at 900 C). Previously pO2 = 1e-16.
        'SR':{'T': 800 + 273.15, 'ratio': 4}, # ratio = 4, 
    }

    # get rows from database
    rows = get_rows_db(filename=filename, selection='energy')

    # get maximum oxygen content and supercell size
    max_oxygen, max_natoms = max([(row.count_atoms()['O'], row.natoms) for
                                  row in rows], key=itemgetter(0))
    ncells = max_natoms // len(crystal_space.structure_list[0])

    # collect formation energies
    data_dicts = []
    for irow, row in enumerate(rows):

        # get free energy from phonon calculations, remove the 0 K energy
        # and add the formation energy before creating a spline
        try:
            form_energy = row.get('form_energy', None)
            if form_energy is None:
                correction = 0.0
            else:
                correction = form_energy - row.energy
            free_energy_spline = CubicSpline(
                np.asarray(row.data['phonons']['temperature']),
                np.asarray(row.data['phonons']['free_energy']
                           ) * row.natoms + correction)
        except Exception as e:
            print(f'Unable to retrieve free energies for row with tag'
                  f' {row.tag} from the database {filename} because of\n'
                  + str(e))
            data_dicts = None
            break

        # determine number of vacancies
        vacancies = max_oxygen - row.count_atoms()['O']

        # estimate configurational entropy, including oxidation states
        pt = get_stability_factors(row, directory=directory, instance=True)
        entropy = pt.entropy(anion_vacancies=True)

        for reactor, condition in conditions.items():

            # calculate total energy
            temperature = condition['T']
            free_energy = free_energy_spline(temperature)

            # calculate oxygen chemical potential
            if reactor == "SR":
                muO = (get_mu_oxygen_sr(temperature,                              
                              condition['ratio'],
                              add_u_dft=form_energy is None) * vacancies
                       if vacancies > 0 else 0.0)
            elif reactor == "FR":
                muO = (get_mu_oxygen_fr(temperature,
                              condition['ratio'],
                              add_u_dft=form_energy is None) * vacancies
                       if vacancies > 0 else 0.0)
            else:
                muO = (get_mu_oxygen(temperature, condition['pO2'],
                                 add_u_dft=form_energy is None) * vacancies
                   if vacancies > 0 else 0.0)

            # add data
            data_dicts.append({
                'formation_energy': free_energy - entropy * temperature
                + muO,
                'vacancies': vacancies,
                'delta': vacancies / ncells,
                'reactor': reactor,
                'irow': irow
            })

    if data_dicts is None:
        print(f'Unable to obtain data from the database {filename}')
        return

    # create dataframe with lowest energy at each temperature
    df = pd.DataFrame(data_dicts).groupby(by=['reactor', 'vacancies'],
                                          as_index=False).min()    
    
    stoich_row = rows[df.loc[(df.vacancies == 0)].irow.iloc[0]]
    stoich_mass = stoich_row.toatoms().get_masses().sum()

    # calculate formation energies relative to the stoichiometric phase
    # Ensure formation_energy is scalar for all rows
    for reactor in df.reactor.unique():
        df.loc[df.reactor == reactor, 'formation_energy'] -= df.loc[
            (df.reactor == reactor) & (df.vacancies == 0)
        ].formation_energy.iloc[0]

        df.loc[df.reactor == reactor, 'formation_energy'] = df.loc[
            df.reactor == reactor, 'formation_energy'
        ].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    # find vacancy concentration in each reactor
    vacancies = {}
    for reactor in df.reactor.unique():
        df_rea = df.loc[df.reactor == reactor]
        df_min = df_rea.loc[df_rea.formation_energy ==
                            df_rea.formation_energy.min()]
        vacancies[reactor] = df_min.vacancies.iloc[0]    

    # calculate oxygen release capacity (gram O2 per gram OC)
    vacancy_diff = vacancies['FR'] - vacancies['AR']
    otc = (atomic_masses[atomic_numbers['O']] * vacancy_diff / stoich_mass)
    # calculate hydrogen production capacity (mmol H2 per gram OC)
    vacancy_diff_2 = vacancies['FR']-vacancies['SR']
    hpp = (1000 * vacancy_diff_2 / stoich_mass)

    # Extract Gibbs energies for ML target (flattened landscape)
    # Build gibbs_keys dict
    gibbs_keys = {}
    for index, df_row in df.iterrows():
        formation_energy = df_row['formation_energy']
        if isinstance(formation_energy, (list, np.ndarray)):
            formation_energy = formation_energy[0]
        key = f"gibbs_{df_row['reactor']}_{int(df_row['vacancies'])}"
        gibbs_keys[key] = formation_energy
            
    vacancies_array = np.array(sorted(set(df.vacancies)))

    # Optionally compute OTC or HPP using quadratic fitting
    if quadratic_fit:
        # Sort Gibbs energies by vacancy number to ensure correct order for fitting
        gibbs_AR = np.array([gibbs_keys[f'gibbs_AR_{v}'] for v in vacancies_array])
        gibbs_FR = np.array([gibbs_keys[f'gibbs_FR_{v}'] for v in vacancies_array])
        gibbs_SR = np.array([gibbs_keys[f'gibbs_SR_{v}'] for v in vacancies_array])
        # Fit quadratic polynomial: G(v) = a*v^2 + b*v + c
        # np.polyfit returns coefficients in descending order [a, b, c]
        coeffs_AR = np.polyfit(vacancies_array, gibbs_AR, 2)
        coeffs_FR = np.polyfit(vacancies_array, gibbs_FR, 2)
        coeffs_SR = np.polyfit(vacancies_array, gibbs_SR, 2)

        # Analytical solution: minimum of quadratic ax^2 + bx + c is at x = -b/(2a)
        def find_quadratic_minimum(coeffs, gibbs_values, vacancies, min_val, max_val):
            """Find minimum of quadratic, clamped to [min_val, max_val]."""
            a = coeffs[0]  # Quadratic coefficient
            b = coeffs[1]  # Linear coefficient

            if abs(a) < 1e-10:  # Nearly linear
                return min_val if b > 0 else max_val

            if a < 0:  # Parabola opens downward - minimum at boundary
                # Evaluate at both boundaries to find which is lower
                val_min = np.polyval(coeffs, min_val)
                val_max = np.polyval(coeffs, max_val)
                return min_val if val_min < val_max else max_val
            else:  # Parabola opens upward - minimum at critical point
                v_min = -b / (2 * a)
                return np.clip(v_min, min_val, max_vac)

        min_vac, max_vac = vacancies_array.min(), vacancies_array.max()
        v_AR_opt = find_quadratic_minimum(coeffs_AR, gibbs_AR, vacancies_array, min_vac, max_vac)
        v_FR_opt = find_quadratic_minimum(coeffs_FR, gibbs_FR, vacancies_array, min_vac, max_vac)
        v_SR_opt = find_quadratic_minimum(coeffs_SR, gibbs_SR, vacancies_array, min_vac, max_vac)

        vacancy_diff_qd = v_FR_opt - v_AR_opt
        vacancy_diff_qd_2=v_FR_opt - v_SR_opt

        otc_qd = (atomic_masses[atomic_numbers['O']] * vacancy_diff_qd / stoich_mass)
        hpc_qd = (1000* vacancy_diff_qd_2 / stoich_mass)
    else:
        otc_qd = None
        hpc_qd=None
    structure_index = 0
    concentrations = {s: c / stoich_row.natoms for s, c in
                      stoich_row.count_atoms().items()}
    cluster_space = crystal_space.cluster_space_list[structure_index]
    prototype = crystal_space.map(concentrations, structure_index,
                                  dimensions=dimensions)
    configurational_entropies = estimate_configurational_entropies(
        cluster_space=cluster_space, atoms_row=prototype)

    # add stoichiometric structure to list
    structure = stoich_row.toatoms()
    key_value_pairs = {
        'hpp': hpp,
        'otc': otc,
        'vacancy_diff': vacancy_diff,
        'vacancy_diff_2': vacancy_diff_2,
        'delta_diff': vacancy_diff / ncells,
        'delta_diff_2': vacancy_diff_2/ncells,
        **configurational_entropies,
        **gibbs_keys,
        **row.key_value_pairs}

    # Optionally add quadratic fit OTC if computed
    if quadratic_fit and hpc_qd is not None:
        key_value_pairs['otc_quadratic_fit'] = otc_qd
        key_value_pairs['otc_vacancy_diff_quadratic_fit'] = vacancy_diff_qd
        key_value_pairs['hpp_quadratic_fit'] = hpc_qd
        key_value_pairs['vacancy_diff_quadratic_fit'] = vacancy_diff_qd_2
        key_value_pairs['mass_'] = stoich_mass
    data = {'delta_energies': data_dicts}

    # add/update structures
    property_calculators = {
        'stability': lambda x: get_stability_factors(
            atoms_row=x, directory=directory),
        'abundance': get_average_abundance,
        'metal_abundance': get_average_metal_abundance,
        'price': get_price
    }
    formula = structure.get_chemical_formula()
    fun_kwarg = dict(structure=structure)
    if cycle == 0:
        fun_kwarg.update(dict(
            row_id=get_row_ids(filename=train_file,
                               selection=f'formula={formula}')[0],
            overwrite=overwrite,
            use_lock_file=use_lock_file,
            data=data,
            **key_value_pairs))
    else:
        fun_kwarg.update(dict(
            keyword_filters=crystal_space.filters_list[0],
            property_calculators=property_calculators,
            overwrite=overwrite,
            use_lock_file=use_lock_file,
            cycle=cycle,
            data=data,
            **key_value_pairs))
    apply_function_db(filename=train_file,
                      function=_add_structure_db_mp,
                      fun_kwargs=[fun_kwarg],
                      append=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='clhp',
                        choices=['wang', 'clhp'],
                        help='data directory')
    parser.add_argument('--cycle',
                        default=None,
                        type=int,
                        help='cycle, default is latest; overwrites existing'
                        ' data if not None')
    parser.add_argument('--pattern',
                        default='sqs_*.db',
                        help='pattern used for selecting files')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    parser.add_argument('--dimensions', type=int, nargs=3, default=[1, 2, 3],
                        help='SQS supercell dimensions (default: 1 2 3)')
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # ensure that all warnings are printed
    warnings.simplefilter("always")

    # determine number of processes
    processes = args.processes
    if processes is None:
        processes = int(os.getenv('SLURM_NTASKS', 1))

    # Update the database with training structures with the results from
    # phonon calculations based on the SQS structures generated from the
    # original training set or samples produced by the network in later cycles
    update_training_database(directory=args.directory,
                             pattern=args.pattern,
                             asynchronus=True,
                             cycle=args.cycle,
                             processes=processes,
                             dimensions=args.dimensions)

    os.chdir(cwd)

#!/usr/bin/env python3

from ase.db import connect
import argparse
from glob import glob
from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd().split('oxygen-carrier-discovery')[0],
                             'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.cluster_expansion import\
    build_sqs_db
from metal.applications.crystals.tools import (get_rows_db, get_sublattice_map,
                                               _atoms_row_to_structure)


def get_cubic_tag(atoms_row):
    '''Generate tag by adding ``_cubic`` to the chemical formula

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    return f'{structure.get_chemical_formula()}_cubic'


def get_orthorhombic_tag(atoms_row):
    '''Generate tag by adding ``_orthorhombic`` to the chemical formula

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    return f'{structure.get_chemical_formula()}_orthorhombic'


def get_brownmillerite_tag(atoms_row):
    '''Generate tag by adding ``_brownmillerite`` to the chemical formula

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    return f'{structure.get_chemical_formula()}_brownmillerite'


def mp_fun(kwarg_list):
    '''Create a database with SQS structures based on 80 atom cubic and
    orthorhombic perovskite supercells, from which up to ten oxygen atoms have
    been removed, as well as a corresponding brownmillerite structure with the
    same metallic elements.

    Parameters
    ----------
    kwarg_list
        list of dictionaries with keyword arguments for build_sqs_db
    '''
    print(f"mp_fun started with {len(kwarg_list)} entries")
    for kwargs in kwarg_list:
        build_sqs_db(**kwargs)


def build_sqs_structures(directory='clhp', selection=None, processes=1,
                         asynchronus=True):

    '''Create databases with SQS structures based on 80 atom cubic and
    orthorhombic perovskite supercells, from which up to ten oxygen atoms have
    been removed, as well as a corresponding brownmillerite structure with the
    same metallic elements. If no cycle folders are found the configurations
    are taken from the training database and othwerise from the sample text
    file for the latest cycle.

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
    processes : int
        number of processes to start; default is to run sequentially
        (1 process)
    asynchronus : bool
        run the processes asynchronously, ignored if single process
    '''
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    # check cycle
    if len(glob(f'{parent_dir}/{directory}/cycle*')) > 1:
        cycle = sorted([int(os.path.basename(folder).replace('cycle', ''))
                        for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
        input_file = f'{parent_dir}/{directory}/cycle{cycle}/sample.db'
    else:
        input_file = f'{parent_dir}/{directory}/training.db'
        cycle = 0
    # Try to read crystal space, otherwise change path
    # Based on relative path of this script
    try:
        crystal_space = CrystalSpace.read(f'{parent_dir}/{directory}/crystal_space.spa')
    except:
        try:
            # Add crystal space based on relative location of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            spa_file_path = os.path.join(parent_dir, f'{directory}/crystal_space.spa')
            
            crystal_space = CrystalSpace.read(spa_file_path)
        except:
            print("Path to crystal_space.spa incorrect")   

    print('Input file:', input_file)
        
    primitives = crystal_space.structure_list
    dimensions = [(4, 2, 2), (2, 1, 2), (2, 1, 2)]
    prim_sl_sym_maps = {}
    for chem_sym in set(primitives[0].get_chemical_symbols()):
        prim_sl_sym_maps[chem_sym] = []
        for i in range(crystal_space.nstructures):
            for sl_sym, chem_syms in get_sublattice_map(
                    structure=primitives[i],
                    cluster_space=crystal_space.cluster_space_list[i]).items():
                if chem_sym in chem_syms:
                    prim_sl_sym_maps[chem_sym].append(sl_sym)
                    break

    # generate kwarg list
    tag_functions = [get_cubic_tag, get_orthorhombic_tag,
                     get_brownmillerite_tag]
    kwarg_lists = []
    for row in get_rows_db(filename=input_file, selection=selection):
        output_fname = f'sqs_{row.formula}.db'
        output_file = f'{parent_dir}/{directory}/cycle{cycle}/{output_fname}'
        existing_files = glob(f'{parent_dir}/{directory}/cycle*/{output_fname}')
        if len(existing_files) > 0:
            if output_file not in existing_files:
                print(f'The database {output_fname} already exists')
                continue
            db = connect(output_file)
            if db.count() >= 19:
                continue
            file_tags = [r.tag.split('_')[-1] for r in db.select()]
        else:
            file_tags = None
        concentrations = {symbol: count / row.natoms for symbol, count in
                          row.count_atoms().items()}
        sublattice_concentrations =\
            crystal_space.get_sublattice_concentrations(
                concentrations=concentrations,
                structure_index=0,
                dimensions=dimensions[0])
        kwarg_list = []
        for i in range(crystal_space.nstructures):
            primitive = primitives[i]
            sublattice_concentrations_list = []
            if i < 2:
                property_calculators = {'tag': get_cubic_tag if i == 0 else
                                        get_orthorhombic_tag}
                vacancies = ['X']
                import math
                cX_step = 1 / (len(primitive) * int(math.prod(dimensions[i]))
                               // 5)
                for cX in np.arange(0, 0.51, cX_step):
                    sublattice_concentrations_list.append({
                        **sublattice_concentrations,
                        prim_sl_sym_maps['O'][i]: {'X': cX / 3,
                                                   'O': 1 - cX / 3}})
            else:
                property_calculators = {'tag': get_brownmillerite_tag}
                vacancies = None
                sublattice_concentrations_list.append({
                    sl_syms[2]: sublattice_concentrations[sl_syms[0]] if
                    chem_sym != 'O' else {'X': 0.0, 'O': 1.0} for
                    chem_sym, sl_syms in prim_sl_sym_maps.items()})
            if file_tags is not None:
                tag = tag_functions[i](primitive).split('_')[-1]
                if file_tags.count(tag) >= len(sublattice_concentrations_list):
                    continue
            kwarg_list.append(dict(
                primitive=primitive,
                filename=output_file,
                cutoffs=crystal_space.cutoffs_list[i],
                chemical_symbols=crystal_space.chemical_symbols_list[i],
                sublattice_concentrations_list=sublattice_concentrations_list,
                dimensions=dimensions[i],
                vacancies=vacancies,
                append=True,
                property_calculators=property_calculators))
        kwarg_lists.append(kwarg_list)

    if processes == 1:
        for kwarg_list in kwarg_lists:
            mp_fun(kwarg_list)
    else:
        with Pool(processes) as pool:
            if asynchronus:
                results = pool.map_async(mp_fun, kwarg_lists)
                results.get()
            else:
                pool.map_async(mp_fun, kwarg_lists)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='clhp',
                        choices=['wang', 'clhp'],
                        help='data directory')
    parser.add_argument('--min_id',
                        type=int,
                        default=None,
                        help='minimum row id to consider')
    parser.add_argument('--max_id',
                        type=int,
                        default=None,
                        help='maximum row id to consider')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    args = parser.parse_args()

    # parse selection
    selection = ['filtered=False']
    if args.min_id is not None:
        selection.append(f'id>={args.min_id}')
    if args.max_id is not None:
        selection.append(f'id<={args.max_id}')
    selection = ','.join(selection)

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # determine number of processes
    processes = args.processes
    if processes is None:
        processes = int(os.getenv('SLURM_NTASKS', 1))

    # Create databases with SQS structures based on 80 atom cubic and
    # orthorhombic perovskite supercells, from which up to ten oxygen atoms
    # have been removed, as well as a corresponding brownmillerite structure
    # with the same metallic elements. If no cycle folders are found the
    # configurations are taken from the training database and othwerise from
    # the sample text file for the latest cycle.
    build_sqs_structures(directory=args.directory, selection=selection,
                         processes=processes)

    os.chdir(cwd)

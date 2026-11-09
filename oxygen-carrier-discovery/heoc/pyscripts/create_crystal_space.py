#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.tools import get_structures_db


def create_crystal_space(directory='clhp'):
    '''Create a crystal space based on  prototype structures represented by the
    cubic and orthorhombic form of the perovskite CaMnO3 and the brownmillerite
    Ca2Mn2O5

    Parameters
    ----------
    directory
        data directory, must contain either 'wang' or 'clhp'
    '''

    if 'wang' in directory:
        replacements = {'Ca': ['Ca', 'K', 'Y', 'Ba', 'La', 'Sm', 'Sr'],
                        'Mn': ['Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti', 'Fe'],
                        'O': ['O', 'X']}
    elif 'clhp' in directory:
        replacements = {'Ca': [ 'Ca', 'Ce', 'K', 'Mg', 'Y', 'Ba', 'La', 'Sm', 'Sr'],
                        'Mn': ['Al', 'Cr', 'Co', 'Cu', 'Fe', 'Mn',  'Ni', 'Sn', 'Ti', 'V', 'W', 'Zr', 'Zn'],
                        'O' : ['O', 'X']}
    else:
        raise ValueError(f'The directory must contain either "wang" or "clhp", not'
                         f' {directory}')

    # read prototype structures
    structure_list = list(get_structures_db(f'{directory}/prototypes.db'))

    # define chemical symbols
    chemical_symbols_list = [[replacements[s] for s in structure.symbols] for
                             structure in structure_list]

    # define cutoffs
    cutoffs_list = [[6.0, 4.5], [6.0, 4.5], [6.0, 4.5]]

    # define filters
    filters_list = [[
        {'key': 'bartel_tolerance', 'symbol': '>', 'value': 4.18},
        {'key': 'delta_charge_neutrality', 'symbol': '=', 'value': False}
        ]] * 3

    # create and write a crystal space
    crystal_space = CrystalSpace(db_path='.',
                                 structure_list=structure_list,
                                 chemical_symbols_list=chemical_symbols_list,
                                 cutoffs_list=cutoffs_list,
                                 filters_list=filters_list)
    crystal_space.write(f'{directory}/crystal_space.spa')


if __name__ == '__main__':

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='clhp',
                        choices=['wang', 'clhp'],
                        help='data directory')

    args = parser.parse_args()

    # Create a crystal space based on  prototype structures represented by the
    # cubic and orthorhombic form of the perovskite CaMnO3 and the
    # brownmillerite Ca2Mn2O5
    create_crystal_space(directory=args.directory)

    os.chdir(cwd)

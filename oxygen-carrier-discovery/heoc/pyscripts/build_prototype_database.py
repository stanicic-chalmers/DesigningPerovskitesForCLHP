#!/usr/bin/env python3

import argparse
import os
from pymatgen.core import SETTINGS
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.tools import build_prototype_db


def build_prototype_database(directory='clhp'):
    '''Create a database with prototype structures represented by the cubic and
    orthorhombic form of the perovskite CaMnO3 and the brownmillerite Ca2Mn2O5

    Parameters
    ----------
    directory
        data directory
    '''

    # read api key, since pymatgen typically fails to do so
    api_key = SETTINGS['PMG_MAPI_KEY']

    # create a database with cubic and orthorhombic CaMnO3 as well as Ca2Mn2O5
    build_prototype_db(
        filename=f'{directory}/prototypes.db',
        formulas=['CaMnO3', 'CaMnO3', 'Ca2Mn2O5'],
        crystal_systems=['Cubic', 'Orthorhombic', 'Orthorhombic'],
        properties=['energy', 'energy_per_atom'],
        make_ideal=True,
        api_key=api_key)


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

    # Create a database with prototype structures represented by the cubic and
    # orthorhombic form of the perovskite CaMnO3 and the brownmillerite
    # Ca2Mn2O5
    build_prototype_database(directory=args.directory)

    os.chdir(cwd)

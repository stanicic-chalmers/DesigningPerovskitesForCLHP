#!/usr/bin/env python3

import argparse
from ase.data import chemical_symbols
from glob import glob
import os
import pandas as pd
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.thermodynamics import\
    estimate_configurational_entropy
from metal.applications.crystals.tools import (get_average_abundance,
                                               get_price)
from build_training_database import get_stability_factors


def build_sample_database(directory='clhp'):
    '''Create a database with training structures by randomly populating 80
    atom perovskite supercells with various elements based on the
    concentrations in the sample text file found in the latest cycle folder

    Parameters
    ----------
    directory
        data directory
    '''

    # Try to read crystal space, otherwise change path
    # Based on relative path of this script
    try:
        crystal_space = CrystalSpace.read(f'{directory}/crystal_space.spa')
    except:
        try:
            # Add crystal space based on relative location of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            spa_file_path = os.path.join(parent_dir, f'{directory}/crystal_space.spa')
            
            crystal_space = CrystalSpace.read(spa_file_path)
        except:
            print("Path to crystal_space.spa incorrect")    

    # read concentrations
    cycle = sorted([int(os.path.basename(folder).replace('cycle', ''))
                    for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
    df = pd.read_csv(f'{parent_dir}/{directory}/cycle{cycle}/sample.csv', sep='\t')
    df = df[[key for key in df.keys() if key in chemical_symbols]]
    concentrations_list = list(df.to_dict(orient='records'))

    # create database with random configurations
    property_calculators = {
        'stability': lambda x: get_stability_factors(
            atoms_row=x, directory=directory),
        'configurational_entropy': lambda x: estimate_configurational_entropy(
            cluster_space=crystal_space.cluster_space_list[0], structure=x),
        'abundance': get_average_abundance,
        'price': get_price,
        'cycle': lambda x: cycle
    }
    crystal_space.build_db_random_configurations(
            structure_index=0,
            concentrations_list=concentrations_list,
            dimensions=(4, 2, 2),
            filename=f'{parent_dir}/{directory}/cycle{cycle}/sample.db',
            apply_filter=True,
            property_calculators=property_calculators)


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

    # Create a database with training structures by randomly populating 80 atom
    # perovskite supercells with various elements based on the concentrations
    # in the sample text file found in the latest cycle folder
    build_sample_database(directory=args.directory)

    os.chdir(cwd)

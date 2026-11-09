#!/usr/bin/env python3

from ase.formula import non_metals
from ase.units import kB
import argparse
import os
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.stability_factors import PerovskiteTolerance
from metal.applications.crystals.tools import (get_average_abundance,
                                               _atoms_row_to_structure,
                                               get_price)
from metal.applications.crystals.thermodynamics import\
    estimate_configurational_entropy


def get_average_metal_abundance(atoms_row):
    '''Estimate the average abundance of the metals a structure

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # delete non-metals
    del structure[[s.index for s in structure if s.symbol in non_metals]]

    return get_average_abundance(structure)


def estimate_configurational_entropies(cluster_space, atoms_row,
                                       stirling=True):
    """Calculate an estimate of the configurational entropy based on the
    provided structure and cluster space

    Parameters
    ----------
    cluster_space
        cluster space
    atoms_row
        ASE Atoms or AtomsRow object
    stirling
        use the Stirling approximation; otherwise the result only
        approaches the macroscopic value for an infitely large supercell
    """

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # calculate configurational entropy
    configurational_entropy = estimate_configurational_entropy(
                cluster_space=cluster_space, structure=structure)

    # Sconf_R: configurational entropy per formula unit in units of R (5 atoms per ABO3 formula unit)
    return {'configurational_entropy': configurational_entropy,
            'Sconf_R': configurational_entropy / kB / len(structure) * 5}


def get_stability_factors(atoms_row, directory='clhp', instance=False):
    """Calculate various stability factors, including the Bartel and
    Goldschmidt tolerance as well as the charge neutrality and configurational
    entropy

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    directory
        data directory
    instance
        return instance of PerovskiteTolerance instead of dictionrary
    """
    # a run directory may override the A/B element lists, e.g. to put Mg on the B site
    site_file = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), directory, 'site_elements.json')
    if os.path.exists(site_file):
        import json
        with open(site_file) as f:
            sites = json.load(f)
        pt = PerovskiteTolerance(atoms_row=atoms_row,
                                 possible_a_elements=sites['a'],
                                 possible_b_elements=sites['b'])
    elif 'wang' in directory:
        pt = PerovskiteTolerance(atoms_row)
    elif  'clhp' in directory:
        pt = PerovskiteTolerance(
                atoms_row=atoms_row,
                possible_a_elements=[ 'Ca', 'Ce', 'K', 'Mg', 'Y', 'Ba', 'La', 'Sm', 'Sr'],
                possible_b_elements=['Al', 'Cr', 'Co', 'Cu', 'Fe', 'Mn', 'Mo', 'Sn','Ni', 
                             'Ti', 'V', 'W', 'Zr', 'Zn'])
        
                ## Possibly disallow some oxidation states here?
                ## Would speed things up a bit
    else:
        raise ValueError(f'The directory must contain either "wang" or "clhp", not'
                         f' {directory}')

    if instance:
        return pt
    return {
        'goldschmidt_tolerance': pt.goldschmidt_tolerance,
        'bartel_tolerance': pt.bartel_tolerance,
        'bartel_probability': pt.bartel_probability(),
        'total_charge': pt.total_charge,
        'delta_charge_neutrality': pt.delta_charge_neutrality(),
        'oxidation_state_entropy': pt.entropy()}


def build_training_database(directory='clhp'):
    '''Create a database with training structures in the form of ABO3
    perovskites with 80 atoms per supercells and up to six distinct metallic
    elements

    Parameters
    ----------
    directory
        data directory
    '''

    if directory == 'wang':
        max_total = 7
    elif directory =='clhp':
        max_total = 7
    else:
        raise ValueError(f'The directory must be either "wang" or "clhp", not'
                         f' {directory}')

    # read crystal space
    crystal_space = CrystalSpace.read(f'{directory}/crystal_space.spa')

    # create database with all possible replacements
    property_calculators = {
        'stability': lambda x: get_stability_factors(
            atoms_row=x, directory=directory),
        'abundance': get_average_abundance,
        'metal_abundance': get_average_metal_abundance,
        'price': get_price,
        'entropies': lambda x: estimate_configurational_entropies(
            cluster_space=crystal_space.cluster_space_list[0], atoms_row=x),
        'cycle': lambda x: 0
    }
    crystal_space.build_db_random_concentrations(
            structure_index=0,
            fixed_symbols=['O'],
            max_total=max_total,
            n_structures=2000,
            dimensions=(4, 2, 2),
            filename=f'{directory}/training.db',
            apply_filter=True,
            property_calculators=property_calculators,
            append=True)


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

    # Create a database with training structures in the form of ABO3
    # perovskites with 80 atoms per supercells and up to six distinct metallic
    # elements
    build_training_database(directory=args.directory)

    os.chdir(cwd)

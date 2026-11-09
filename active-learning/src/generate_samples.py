import sys
import os
import torch
import numpy as np
import random

# Generate random samples based on Joakims script for creating HEO database.

metal_path = os.path.join(os.path.dirname(os.getcwd()),'libs','metal')
sys.path.append(metal_path)

from metal.applications.crystals.crystal_space import CrystalSpace
from itertools import product
from random import choice, choices
from ase.db.row import AtomsRow
from icet import ClusterSpace

# This was to slow to use in a training loop
def generate_fake_samples_batch(batch_size, device, seed):
    #torch.manual_seed(seed)
    #random.seed(seed)
    #np.random.seed(seed)

    elements = ['Ca','Ba','K','La','Sm','Sr','Y','Co','Cu','Fe','Mn','Mg','Ni','Ti','O']
    crys_space = CrystalSpace.read('crystal_space.spa')
    structure_index=0
    fixed_symbols=['O']
    max_elements=7
    dimensions=(4, 2, 2)


    primitive=crys_space.structure_list[structure_index]
    chemical_symbols=crys_space.chemical_symbols_list[structure_index]

    # create cluster space
    cluster_space = ClusterSpace(primitive, cutoffs=[1.0],
                                    chemical_symbols=chemical_symbols)
    supercell = primitive.repeat(dimensions)
    sublattices = cluster_space.get_sublattices(supercell)

    # determine possible combinations
    if max_elements is not None:
        combos = np.array(list(product(*[
            list(range(1, len(sl.chemical_symbols))) for sl in sublattices])))
        combos = combos[np.where(np.sum(combos, axis=1) <= max_elements)]

    # generate structures, generate the same same amount as batch size
    fun_kwargs = []
    for _ in range(batch_size):
        structure = supercell.copy()
        if max_elements is not None:
            combo = choice(combos)
        for s, sl in enumerate(sublattices):
            if max_elements is not None:
                possible_symbols = choices(sl.chemical_symbols, k=combo[s])
            else:
                possible_symbols = sl.chemical_symbols
            for index in sl.indices:
                if (fixed_symbols is None or structure.symbols[index] not in
                        fixed_symbols):
                    structure.symbols[index] = choice(possible_symbols)
        tag = structure.get_chemical_formula()
        fun_kwargs.append(dict(structure=structure))

    # Get atom counts for all generated structures
    all_counts = []
    for i, row in enumerate(fun_kwargs):
        all_counts.append(AtomsRow(fun_kwargs[i]['structure']).count_atoms())
    # Go from atom count to concentration of elements
    all_concs = [{k: v / sum(counts.values()) for k, v in counts.items()} for counts in all_counts]

    # Create a torch tensor which can be used from the elemental concetrations
    db2vec = {}
    for i, element in enumerate(elements):
        db2vec[element] = i
    #print(db2vec)
        # Create tensor from list of elemental concentrations
    fake_sample_tensor = None
    for i, conc_dict in enumerate(all_concs):
        new_row = torch.zeros(len(elements))
        for key in conc_dict: 
            new_row[db2vec[key]] = conc_dict[key]
        
        # Add row or create row if first
        if fake_sample_tensor != None:
            fake_sample_tensor = torch.cat((fake_sample_tensor, new_row.unsqueeze(0)),0)
        else:
            fake_sample_tensor = new_row.unsqueeze(0)

    return fake_sample_tensor.to(device)

# Return np array of all samples, needed same amount as data used
def generate_fake_samples_np(elements, n_samples, seed):
   # torch.manual_seed(seed)
   # random.seed(seed)
   # np.random.seed(seed)


    crys_space = CrystalSpace.read('crystal_space.spa')
    structure_index=0
    fixed_symbols=['O']
    max_elements=7
    dimensions=(4, 2, 2)


    primitive=crys_space.structure_list[structure_index]
    chemical_symbols=crys_space.chemical_symbols_list[structure_index]

    # create cluster space
    cluster_space = ClusterSpace(primitive, cutoffs=[1.0],
                                    chemical_symbols=chemical_symbols)
    supercell = primitive.repeat(dimensions)
    sublattices = cluster_space.get_sublattices(supercell)

    # determine possible combinations
    if max_elements is not None:
        combos = np.array(list(product(*[
            list(range(1, len(sl.chemical_symbols))) for sl in sublattices])))
        combos = combos[np.where(np.sum(combos, axis=1) <= max_elements)]

    # generate structures, generate the same same amount as batch size
    fun_kwargs = []
    for _ in range(n_samples):
        structure = supercell.copy()
        if max_elements is not None:
            combo = choice(combos)
        for s, sl in enumerate(sublattices):
            if max_elements is not None:
                possible_symbols = choices(sl.chemical_symbols, k=combo[s])
            else:
                possible_symbols = sl.chemical_symbols
            for index in sl.indices:
                if (fixed_symbols is None or structure.symbols[index] not in
                        fixed_symbols):
                    structure.symbols[index] = choice(possible_symbols)
        tag = structure.get_chemical_formula()
        fun_kwargs.append(dict(structure=structure))

    # Get atom counts for all generated structures
    all_counts = []
    for i, row in enumerate(fun_kwargs):
        all_counts.append(AtomsRow(fun_kwargs[i]['structure']).count_atoms())
    # Go from atom count to concentration of elements
    all_concs = [{k: v / sum(counts.values()) for k, v in counts.items()} for counts in all_counts]

    # Create a torch tensor which can be used from the elemental concetrations
    db2vec = {}
    for i, element in enumerate(elements):
        db2vec[element] = i
    #print(db2vec)
        # Create tensor from list of elemental concentrations
    fake_sample_np = None
    for i, conc_dict in enumerate(all_concs):
        new_row = np.zeros(len(elements))
        for key in conc_dict: 
            new_row[db2vec[key]] = conc_dict[key]
        
        # Add row or create row if first
        if fake_sample_np is None:
            fake_sample_np = new_row
        else:
            fake_sample_np = np.vstack((fake_sample_np, new_row))
            

    return fake_sample_np

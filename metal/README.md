# metal (subset used by the active-learning runs)

Python package for the atomistic part of the active-learning loop: the composition space and SQS structures,
structure relaxation with the CHGNet interatomic potential, vibrational free energies, oxygen chemical
potentials of the reactor gases, and the perovskite stability descriptors (charge neutrality, tolerance factors).

Only the modules and data files needed by `active-learning/` and the property-evaluation scripts are included.
The folder is put on the Python path by the active-learning code, so the package is imported as
`metal.applications.crystals...` from `<root>/metal`.

## Contents

| Module (`metal/applications/crystals/`) | Content |
|---|---|
| `crystal_space.py`, `../space.py` | definition of the composition space (sublattices, allowed elements) |
| `cluster_expansion.py` | special quasi-random structures (SQS) via ICET |
| `thermodynamics.py` | structure relaxations and phonon calculations (ASE / phonopy, incl. QHA), database updates |
| `base_optimizer.py`, `chgnet_optimizer.py`, `torch_tools.py` | structure relaxation with CHGNet |
| `stability_factors.py` | charge neutrality, Goldschmidt and Bartel tolerance factors, phase stability (Materials Project) |
| `tools.py` | oxygen chemical potential of the reactor gases (`get_mu_oxygen_fr`, `get_mu_oxygen_sr`), database helpers |

MACE and NEP potentials are optional in `thermodynamics.py` and are not included; the runs used CHGNet.

| Data (`metal/applications/crystals/data/`) | Content |
|---|---|
| `O2_ref.txt`, `O2.json`, `H2O_ref.txt`, `H2_ref.txt`, `CO_ref.txt`, `CO2_ref.txt` | gas-phase thermochemical reference data (NIST-JANAF) |
| `Shannon_radii_dict.json` | Shannon effective ionic radii |
| `electronegativities.csv` | element electronegativities |
| `TableS1.csv` | experimental perovskite classification data used to calibrate the Bartel tolerance factor (Bartel et al., *Sci. Adv.* 2019) |
| `save_clf.p` | the calibrated Bartel classifier fitted on `TableS1.csv` (scikit-learn pickle; rebuilt from `TableS1.csv` if deleted) |
| `abundances.csv`, `symbolprice.csv` | element abundances and raw-material prices |

# oxygen-carrier-discovery (property-evaluation scripts)

Scripts that build the composition space and the databases, and evaluate the redox properties of the
perovskites. They are called by `active-learning/src/active_learning_HEO.py` in every cycle and are used once
beforehand to set up the search space and the initial database (cycle 0).

Each run lives in its own folder `heoc/<run>/`. The element set of this work is selected by the name `clhp`
(A site: Ba, Ca, Ce, K, La, Mg, Sm, Sr, Y; B site: Al, Co, Cr, Cu, Fe, Mn, Ni, Sn, Ti, V, W, Zn, Zr).

## Scripts (`heoc/pyscripts/`)

| Script | Step |
|---|---|
| `build_prototype_database.py` | prototype structures (cubic and orthorhombic perovskite, brownmillerite) → `prototypes.db` |
| `create_crystal_space.py` | composition space on the prototype lattices → `crystal_space.spa` |
| `build_training_database.py` | random compositions of the initial database, with stability descriptors → `training.db` |
| `build_sample_database.py` | database of the compositions selected by the AL cycle → `cycle<N>/sample.db` |
| `build_sqs_databases.py` | special quasi-random structures with 0 … 8 O vacancies per composition → `cycle<N>/sqs_*.db` |
| `perform_threaded_relaxations.py` | CHGNet relaxations of the SQS |
| `perform_phonon_calculations.py` | harmonic phonon calculations (relaxation + phonons, optionally on several GPUs) |
| `update_training_database.py` | Gibbs free energy G(v) in each reactor, equilibrium vacancy contents (quadratic fit over all vacancy contents), HPC; appends to `training.db` |
| `pyscripts_utils.py` | shared helpers (paths, loading the crystal space) |

The scripts expect `metal/` at the repository root (`<root>/metal`), as in this repository.

## Start set

`heoc/clhp/` holds the start set (cycle 0) of the runs reported in the paper:

| File | Content |
|---|---|
| `training.db.gz` | initial database (ASE SQLite, gzip; 15 MB, 54 MB unpacked): 3903 random compositions, of which 2103 pass the charge-neutrality and tolerance-factor filters and carry G(v) for v = 0 … 8 in the fuel, steam and air reactor and the HPC |
| `crystal_space.spa` | composition space (A and B elements on the cubic, orthorhombic and brownmillerite prototypes) |
| `SHA256SUMS` | checksums (`sha256sum -c SHA256SUMS`) |

The relaxed SQS structures and phonon data behind these labels (`cycle0/`, 5.7 GB) are not included.

## Setting up a run

```bash
cd oxygen-carrier-discovery/heoc/pyscripts
python setup_run.py --name clhp_r1        # -> heoc/clhp_r1/ with training.db, crystal_space.spa, cycle0/
cd ../../../active-learning/src
python -u active_learning_HEO.py --method clhp_r1 --seed 1 --log-path ../../oxygen-carrier-discovery/heoc/clhp_r1/run
```

The run name must contain `clhp`: in every cycle the stability descriptors of the new compositions are computed
with the element set selected by that name. Run folders other than `heoc/clhp/` are not tracked by git.

### Building a new start set (optional)

The scripts can also build a new search space and start set (`--directory wang|clhp`; `wang`: the element set
of the earlier study of Wang et al. on SrFeO₃-based perovskites):

```bash
cd oxygen-carrier-discovery/heoc/pyscripts
python build_prototype_database.py --directory clhp   # needs a Materials Project API key in the pymatgen settings
python create_crystal_space.py   --directory clhp
python build_training_database.py --directory clhp
```

The new compositions then have to be evaluated with the SQS, relaxation, phonon and update scripts (cycle 0)
before an AL run can start; for about 2000 compositions this takes considerable GPU time. Note that this
overwrites the published start set in `heoc/clhp/`, so work on a copy of the repository.

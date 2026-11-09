# Active learning for perovskite oxygen carriers

Active-learning (AL) loop used to search the composition space of high-entropy ABO3 perovskites for a high
hydrogen production capacity (HPC) in three-reactor chemical looping. Each cycle:

1. **Database**: compositions evaluated so far, with their Gibbs free energy as a function of the
   oxygen-vacancy content, G(v), under fuel- and steam-reactor conditions, and their HPC.
2. **Models**: a Wasserstein autoencoder (WAE) learns a latent space of the compositions; a deep ensemble of
   MLPs predicts G(v).
3. **Candidate generation**: latent vectors are sampled from a Gaussian fitted to the database latents and
   decoded into compositions (rounded to integer cation counts).
4. **Acquisition**: the HPC of each candidate is derived from the predicted G(v); candidates are selected by
   Thompson sampling over the ensemble members.
5. **Evaluation**: the selected compositions are evaluated with a machine-learning interatomic potential and
   phonon calculations on special quasi-random structures, added to the database, and the models are retrained.

## Files

| File | Content |
|---|---|
| `src/active_learning_HEO.py` | main script: runs the AL cycles |
| `src/config.py` | command-line options and the `Config` class (example command in the module docstring) |
| `src/al_functions.py` | candidate generation, rounding, acquisition, HPC from G(v) |
| `src/ANN_functions_new.py` | WAE model and training |
| `src/ensemble_mlp.py` | deep-ensemble surrogate (randomized-MAP sampling) |
| `src/data_manager.py` | loading, normalisation and train/validation split of the database |
| `src/generate_samples.py` | random compositions |
| `src/al_util.py`, `src/al_viz.py` | logging, checkpoints and latent-space plots |
| `src/slurm_phonon.py` | optional fan-out of the phonon calculations as SLURM jobs |
| `src/plot_element_distributions.py` | loads WAE checkpoints and plots element distributions of a run |
| `container/perovskite_al.def` | Apptainer recipe of the run environment (pinned versions) |
| `examples/config_paper_r1.json` | full configuration of run r1 of the paper (seed 1) (paths replaced by placeholders) |

## Requirements

- Python environment: the runs in the paper used an Apptainer container on CUDA GPUs, built from
  [`container/perovskite_al.def`](container/perovskite_al.def) (NVIDIA PyTorch 25.10 base image, Python 3.12,
  package versions pinned to those of the image used for the runs). Build it with
  `apptainer build perovskite_al.sif container/perovskite_al.def` and run the scripts with
  `apptainer exec --nv perovskite_al.sif python -u ...`. Without a container: install a CUDA build of
  PyTorch (>= 2.9), then `pip install -r requirements.txt` (same versions).
- Sibling folders in the same parent directory (imported via relative paths):

```
<parent>/
├── active-learning/                     this folder
├── metal/                               atomistic thermodynamics and phase stability
└── oxygen-carrier-discovery/heoc/pyscripts/   building the SQS databases and updating the training database
```

- A Materials Project API key for the phase-stability step, stored in the pymatgen settings
  (`PMG_MAPI_KEY` in `~/.config/.pmgrc.yaml`). Do not put the key in scripts or on the command line.

## Running

```bash
cd active-learning/src
python -u active_learning_HEO.py --method clhp_r1 --seed 1 --log-path ../../oxygen-carrier-discovery/heoc/clhp_r1/run
```

The default options are the settings of the paper runs: Gibbs-landscape surrogate (deep ensemble of 16
members), quadratic fits of G(v), rounded compositions, 15 % random validation split, 25 000 latent samples,
Thompson sampling, 50 cycles. Independent runs differ only in `--seed` and start from the same initial
database. Each on/off option can be switched off with its `--no-...` form, e.g. `--no-predict-landscape`.
The run writes its cycle folders and `training.db` to `oxygen-carrier-discovery/heoc/<method>/`, and logs and
model checkpoints to `--log-path` (inside that folder; give the same `--log-path` again to resume a stopped
run, otherwise a new time-stamped folder is used). The full command used on the cluster (container, SLURM fan-out of
the phonon calculations) is in the docstring of `src/config.py`.

**Vacancy range.** The training labels contain G(v) for v = 0 ... 8 O vacancies per 80-atom cell
(δ = 0 ... 0.5). By default (`--n-vacancies-max 8`) the surrogate is trained on and predicts all nine points.

**Naming.** `hpp` in the code and databases is the hydrogen production capacity (HPC) of the paper.

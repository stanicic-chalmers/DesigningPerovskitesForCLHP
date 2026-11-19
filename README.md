# DesigningPerovskitesForCLHP

Code for the active-learning design of high-entropy perovskite oxygen carriers for chemical-looping hydrogen production (CLHP).

> Citation details will be added following publication.

## Repository layout

```
DesigningPerovskitesForCLHP/
├── active-learning/   active-learning loop: generative model (WAE), deep-ensemble surrogate, Thompson sampling
├── metal/             atomistic part: composition space, SQS, CHGNet relaxations, phonons, stability descriptors
└── oxygen-carrier-discovery/heoc/
    ├── pyscripts/     building the databases and evaluating the HPC (called in every AL cycle); setup_run.py
    └── clhp/          start set of the paper runs (training.db.gz, crystal_space.spa)
```

The `metal/` package is a modified subset of **METAL** (Material Exploration Through Active Learning;
Brorsson *et al.*, [arXiv:2601.03933](https://arxiv.org/abs/2601.03933)). If you use it, please cite that work as
well; see [`metal/README.md`](metal/README.md) for what was changed for this study.

The three parts import each other through relative paths, so keep this layout. Start with
[`oxygen-carrier-discovery/README.md`](oxygen-carrier-discovery/README.md) (setting up a run) and
[`active-learning/README.md`](active-learning/README.md) (running the AL loop).


"""Configuration module for HEO Active Learning Pipeline.

This module contains all configuration parameters, command-line argument parsing,
and the Config class that consolidates all settings.

Example usage (settings of the runs in the paper; one run per seed, here seed 42):
    The container provides the Python environment (see environment.yml). The --bind options and the
    --phonon-slurm-* / --slurm-* flags are only needed when phonon calculations are fanned out as SLURM jobs.

apptainer exec --nv \
    --bind /usr/bin/sbatch --bind /usr/bin/scancel --bind /usr/bin/squeue --bind /usr/bin/sacct \
    --bind /usr/bin/scontrol --bind /usr/lib64/slurm --bind /etc/slurm \
    --bind /usr/lib64/libmunge.so.2 --bind /run/munge --bind /etc/passwd --bind /etc/group \
    <path/to/container.sif> \
    python -u active_learning_HEO.py \
        --method <run_name> \
        -b 2 \
        -t 10 \
        --use-preloader \
        --n-samples-total 25000 \
        --model ensemble \
        --use-validation \
        --val-split 0.15 \
        --val-split-mode random \
        --round-compositions \
        --predict-landscape \
        --quadratic-fit-hpp \
        --quadratic-fit-landscape-hpp \
        --ensemble-size 16 \
        --phonon-slurm-jobs 4 \
        --slurm-time 08:00:00 \
        --slurm-use-preloader \
        --slurm-container <path/to/container.sif> \
        --slurm-account <slurm_account> \
        --slurm-phonon-batch 3 \
        --seed 42 \
        --max-cycle 50 \
        --log-path <run_directory>/run

The Materials Project API key is read from the pymatgen settings (PMG_MAPI_KEY in ~/.config/.pmgrc.yaml);
never put it on the command line or in a script.
"""

import argparse
import os
import torch
from glob import glob
import shutil




class Config:
    """Consolidated configuration for HEO Active Learning Pipeline.
    
    This class consolidates all constants, parameters, the perovskite structure configuration,
    and parsed arguments into a single source of truth.
    """
    
    def __init__(self, args):
        """Initialize configuration from parsed arguments.
        
        Args:
            args: Parsed command-line arguments from argparse
        """
        # ==================== ARGUMENTS ====================
        # Store all parsed arguments
        self.args = args
        self.log_path = args.log_path
        self.checkpoint_dir = args.checkpoint_dir
        self.model = args.model
        self.ensemble_size = args.ensemble_size
        self.bootstrap_fraction = args.bootstrap_fraction
        self.method = args.method
        self.sampling = args.sampling
        self.max_element_count = args.max_element_count
        self.b = args.b
        self.t = args.t
        self.n_samples_total = args.n_samples_total
        self.use_preloader = args.use_preloader
        self.use_validation = args.use_validation
        self.val_split = args.val_split
        self.overwrite = args.overwrite
        self.max_cycle = args.max_cycle
        self.val_split_mode = args.val_split_mode
        self.ensemble_calibrate = args.ensemble_calibrate
        self.round_compositions = args.round_compositions
        self.predict_landscape = args.predict_landscape
        self.quadratic_fit_hpp = args.quadratic_fit_hpp
        self.quadratic_fit_landscape_hpp = args.quadratic_fit_landscape_hpp

        self.use_gp = self.model == 'gp'
        self.use_mc = self.model == 'mc'
        self.use_ensemble = self.model == 'ensemble'
        
        # ==================== CONSTANTS ====================
        self.default_cycles = 25
        self.default_samples_per_cycle = 10
        self.default_seed = args.seed
        self.initial_wae_epochs = 1500
        self.subsequent_wae_epochs = 500
        self.default_gp_training_iters = 750
        
        # ==================== PEROVSKITE STRUCTURE ====================
        self.atoms_in_structure = 80  # Total atoms in perovskite supercell
        self.a_scaling = 1.0/5.0  # A sublattice occupies 1/5 of total atoms
        self.b_scaling = 1.0/5.0  # B sublattice occupies 1/5 of total atoms
        self.x_scaling = 3.0/5.0  # X (oxygen) sublattice occupies 3/5 of total atoms
        # Largest number of O vacancies per 80-atom supercell; the Gibbs landscape is sampled at
        # v = 0, 1, ..., n_vacancies_max (n_vacancy_levels points; 8 -> 9 levels, delta = 0 ... 0.5)
        self.n_vacancies_max = args.n_vacancies_max
        self.sqs_dimensions = (1, 2, 3)  # SQS supercell dimensions
        
        # ==================== ELEMENTS ====================
        # Define perovskite structure composition elements
        self.A_elements = ['Ca', 'Ce', 'K', 'Mg', 'Y', 'Ba', 'La', 'Sm', 'Sr']
        self.B_elements = ['Al', 'Cr', 'Co', 'Cu', 'Fe', 'Mn','Ni', 'Sn', 'Ti', 'V', 'W', 'Zr', 'Zn']
        self.X_elements = ['O']
        self.elements = self.A_elements + self.B_elements + self.X_elements  
        
        # ==================== SCALING DICTIONARY ====================
        # Create scaling dictionary for sublattice concentrations
        # Sublattice fractions of the 80-atom ABO3 perovskite cell: A 1/5, B 1/5, O 3/5
        self.scaling_dictionary = {
            **{elem: self.a_scaling for elem in self.A_elements},  # 1/5
            **{elem: self.b_scaling for elem in self.B_elements},  # 1/5
            **{elem: self.x_scaling for elem in self.X_elements},  # 3/5
        }
        
        # ==================== ENSEMBLE PARAMETERS ====================
        # Architecture parameters, smaller networks seem better, less overfitting
        self.ensemble_hidden_dims = [64, 64]  # Hidden layer dimensions
        self.ensemble_dropout_rate = 0.25  # Dropout rate for ensemble members

        # Training parameters
        self.ensemble_epochs = 500  # Maximum training epochs
        self.ensemble_lr = 1e-4  # Learning rate
        self.ensemble_weight_decay = 0  # Weight decay for regularization, WARNING: May cause ensemble collapse, careful
        self.ensemble_batch_size = 64  # Batch size for training        
        self.early_stopping_patience = 150
        
        # Learning rate scheduler parameters
        self.ensemble_lr_scheduler_patience = 50  # Patience for LR scheduler
        self.ensemble_lr_scheduler_gamma = 0.5  # LR decay factor
        self.ensemble_use_scheduler = True  # Whether to use LR scheduler
        
        self.ensemble_hpp_quadratic_fit_samples = 1 # Number of samples per member for hpp distribution
        self.ensemble_hpp_quadratic_fit_member_mean = True # Whether to use member mean for hpp distribution, req. 1 sample
        
        # Ensemble Sampling parameters (Lu & Van Roy, 2017)
        # Loss_m = NLL(y + Z̃_m, f_θ) + (1/λ) ||θ - θ̃_m||²
        # where λ is the prior variance, likewies with pearce 2020 anchored ensembles
        # where θ̃_m ~ N(0, λI) and Z̃_m ~ N(0, noise_std²)
        self.ensemble_prior_variance = 1   # λ: Prior variance for Gaussian weight init
        self.ensemble_prior_reg = True    # 0 = disabled, 1 = standard ensemble sampling
        
        if self.ensemble_weight_decay:
            assert not self.ensemble_prior_reg, "When using weight decay, prior_reg must be False, or they interfere. Generally best to avoid weight decay."

        # Target perturbation noise std (in normalized units, after z-score)
        # Raw Gibbs noise increases linearly (close enough): 0.2 to 0.8 eV across vacancy range
        # Normalized noise = raw_noise / gibbs_std
        self.ensemble_noise_std = None  # Set by build_ensemble_noise_std() if enabled
        self._gibbs_norm_std_default = 7.331  # Fallback if not available from data
        self.gibbs_noise_std = args.gibbs_noise_std  # fixed G(v) std for the noise scale; None = from the data
        self._noise_raw_min = 0.2     # Raw noise std at vacancy=0 
        self._noise_raw_max = 0.8     # Raw noise std at the largest vacancy count (n_vacancies_max)

        # ==================== WAE PARAMETERS ====================
        # Network architecture parameters
        self.latent_dim = 8
        self.use_regression = True
        self.decoder_scale = 4
        self.regression_scale = 2

        # Dropout configuration
        self.full_dropout = True  # Use full dropout for MC model
        self.encoder_dropout = 0.1
        self.decoder_dropout = 0.1
        self.regression_dropout = 0.1

        # Training parameters
        self.wae_batch_size = 32
        self.wae_lr = 1e-3
        self.subsequent_wae_lr = 2e-4
        self.wae_weight_decay = 1e-5
        self.wae_sigma = 1.0
        self.wae_mmd_lambda = 1e-2
        self.wae_early_stopping_patience = 800
        self.wae_lr_patience = 100        
        self.subsequent_wae_lr_patience = 50
        self.wae_max_grad_norm = 1.0

        # Normalization method (set during data loading)
        # 'zscore' = z-score (mean=0, std=1) for ensemble/NLL
        # 'minmax' = min-max [0,1] for GP/MC
        self.normalization = None  # Will be set by load_training_data()

        # ==================== SLURM PHONON FAN-OUT ====================
        self.phonon_slurm_jobs = args.phonon_slurm_jobs
        self.slurm_account = args.slurm_account
        self.slurm_container = args.slurm_container
        self.slurm_time = args.slurm_time or '0-12:00:00'
        self.slurm_use_preloader = args.slurm_use_preloader
        self.slurm_poll_interval = 30  # seconds between squeue checks
        self.slurm_queue_timeout = 600  # seconds before cancelling 
        self.slurm_phonon_batch = args.slurm_phonon_batch

        # ==================== DERIVED PROPERTIES ====================
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.phonon_batch_size = self.b if self.use_gpu else None
        self.relax_threads = self.t if self.use_gpu else None
        self.tol = 1 / self.atoms_in_structure

        # Inverse transform function (set after data loading)
        # For landscape mode, this will be an identity function
        self.inverse_transform = None
                
        # ==================== PATHS ====================
        # Determine project root and paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.heoc_path = os.path.join(self.project_root, 'oxygen-carrier-discovery', 'heoc')
        self.folder_name = f'{self.heoc_path}/{self.method}/cycle'
        self.db_name = f'{self.heoc_path}/{self.method}/training.db'
        
    def validate(self):
        """Validate argument compatibility."""
        if self.use_ensemble and self.use_mc:
            raise ValueError("Cannot use both --use-ensemble and --use-mc. Choose one uncertainty estimation method.")
    
    def ensure_initial_cycle(self):
        """Ensure cycle0 folder exists."""
        os.makedirs(f'{self.folder_name}0', exist_ok=True)
    
    def setup_log_path(self, timestamp):
        """Set up log path and handle cycle cleanup if needed.
        
        Args:
            timestamp: Timestamp string for new log directories (format: YYYYMMDD_HHMMSS)
            
        Returns:
            str: The log path to use
            
        Raises:
            ValueError: If existing cycles found but no log path provided
        """
        
        # Check if cycles already exist (beyond cycle0)
        existing_cycles = glob(f'{self.folder_name}*')
        non_initial_cycles = [c for c in existing_cycles if not c.endswith('cycle0')]
        
        # Handle overwrite flag - remove existing cycles except cycle0
        if self.overwrite and non_initial_cycles:
            print(f"--overwrite flag detected. Removing {len(non_initial_cycles)} existing cycle(s)...")
            for cycle_dir in non_initial_cycles:
                print(f"  Removing {os.path.basename(cycle_dir)}")
                shutil.rmtree(cycle_dir)
            non_initial_cycles = []
        
        # Determine log path
        if self.log_path:
            log_path = self.log_path
        else:
            # No log path provided - check if resuming is needed
            if non_initial_cycles:
                raise ValueError(
                    f"ERROR: Found {len(non_initial_cycles)} existing cycle(s) but no --log-path was provided.\n"
                    f"Existing cycles: {[os.path.basename(c) for c in sorted(non_initial_cycles)]}\n\n"
                    f"To resume training, you must specify --log-path to the previous run's log directory.\n"
                    f"Example: --log-path {self.heoc_path}/{self.method}/20251119_172345\n\n"
                    f"To start fresh, delete existing cycles first or use --overwrite flag."
                )
            log_path = f'{self.heoc_path}/{self.method}/{timestamp}'
        
        # Ensure log_path is within the method directory
        assert f'/{self.method}/' in log_path or log_path.endswith(f'/{self.method}'), \
            f"Log path must be within the method directory. Expected '/{self.method}/' in path '{log_path}'"
            
        self.log_path = log_path
    
    def get_checkpoint_dir(self):
        """Get checkpoint directory path.
        
        Args:
            log_path: The log path being used
            
        Returns:
            str: Path to checkpoint directory
        """
        import os
        if self.checkpoint_dir:
            return self.checkpoint_dir
        else:
            return os.path.join(self.log_path, 'checkpoints')

    def log_initial_info(self, logger, log_path):
        """Log initial configuration information.
        
        Args:
            logger: Logger instance
            log_path: Path to log directory
        """
        logger.info(f'Using device: {self.device}')
        logger.info(f'folder: {self.folder_name}')
        logger.info(f'db_name: {self.db_name}')
        logger.info(f'logs saved to: {log_path}')
        
        logger.info('Input arguments: ')
        for k,v in vars(self.args).items():
            logger.info(f'{k}:\t{v}')
            
        self.set_method_list()
        if self.predict_landscape:
            logger.info(f'GP/Ensemble will predict HPP + Gibbs energy landscape ({len(self.method_list)} dimensions total: 1 HPP + {self.n_vacancy_levels*2} Gibbs energies, v = 0..{self.n_vacancies_max})')
        else:
            logger.info('GP/Ensemble will predict scalar HPP')
        
        # WAE always uses only the first property (HPP)
        logger.info('WAE will predict scalar HPP (first property in method_list)')
        
        logger.info('Initialise WAE')
        logger.info(f'Uncertainty estimation method: {self.model}')
        if self.use_ensemble:
            if self.bootstrap_fraction is None:
                logger.info(f'Deep Ensemble: {self.ensemble_size} members, NO bootstrapping (full dataset per member)')
            else:
                logger.info(f'Deep Ensemble: {self.ensemble_size} members, bootstrap fraction {self.bootstrap_fraction}')
        elif self.use_gp:
            logger.info('Using Gaussian Process for uncertainty estimation')
        else:
            logger.info(f'Using MC Dropout for uncertainty estimation (full dropout enabled)')
    
    @property
    def n_vacancy_levels(self):
        """Number of sampled vacancy contents, v = 0, 1, ..., n_vacancies_max."""
        return self.n_vacancies_max + 1

    def set_method_list(self):
        """Get list of properties to predict based on configuration.

        Returns:
            list: Property names to predict
        """
        if self.predict_landscape:
            # Predict Gibbs energies for SR and FR across vacancy range
            # IMPORTANT: HPP must be first for WAE training
            method_list = ['hpp']  # HPP first for WAE
            for r in ['SR', 'FR']:
                for v in range(self.n_vacancy_levels):
                    method_list.append(f'gibbs_{r}_{v}')
            self.method_list = method_list
        else:
            self.method_list = ['hpp']

    def build_ensemble_noise_std(self, gibbs_std=None):
        """Build per-dimension noise std vector for ensemble sampling.

        The ensemble predicts only the Gibbs energies G(v) in the steam (SR) and fuel (FR)
        reactor; the HPC is derived from them.
        Gibbs noise increases linearly with vacancy index:
            noise_raw(v) = noise_min + (noise_max - noise_min) * v / (n_vac - 1)
            noise_normalized = noise_raw / gibbs_std

        Args:
            gibbs_std: std of the grouped z-score normalisation of all G(v) targets (from the
                data). If None, the fallback _gibbs_norm_std_default is used.

        Returns:
            list: [gibbs_SR_0, ..., gibbs_SR_N, gibbs_FR_0, ..., gibbs_FR_N]
        """
        import numpy as np

        n_vac = self.n_vacancy_levels
        noise_min = self._noise_raw_min
        noise_max = self._noise_raw_max

        if gibbs_std is None:
            gibbs_std = self._gibbs_norm_std_default

        # Linear interpolation of raw noise across vacancy range
        vacancy_indices = np.arange(n_vac)
        raw_noise = noise_min + (noise_max - noise_min) * vacancy_indices / (n_vac - 1)
        normalized_noise = raw_noise / gibbs_std

        # Build vector: [SR_0..SR_N, FR_0..FR_N]
        noise_std = normalized_noise.tolist()  # SR
        noise_std.extend(normalized_noise.tolist())  # FR (same pattern)

        self.ensemble_noise_std = noise_std
        return noise_std

    def get_wae_params(self, cycle=1):
        """Get WAE training parameters for a given cycle.

        Args:
            cycle: Current AL cycle number. Cycle 1 uses initial params,
                   subsequent cycles use reduced lr/patience for fine-tuning.

        Returns:
            dict: WAE parameters
        """
        is_initial = (cycle == 1)

        return {
            # Training params (cycle-dependent)
            'num_epoch': self.initial_wae_epochs if is_initial else self.subsequent_wae_epochs,
            'batch_size': self.wae_batch_size,
            'lr': self.wae_lr if is_initial else self.subsequent_wae_lr,
            'lr_patience': self.wae_lr_patience if is_initial else self.subsequent_wae_lr_patience,
            'weight_decay': self.wae_weight_decay,
            'sigma': self.wae_sigma,
            'MMD_lambda': self.wae_mmd_lambda,
            'early_stopping_patience': self.wae_early_stopping_patience,
            'max_grad_norm': self.wae_max_grad_norm,

            # Network architecture parameters
            'latent_dim': self.latent_dim,
            'use_regression': self.use_regression,
            'lattice_A_size': len(self.A_elements),
            'lattice_B_size': len(self.B_elements),
            'n_properties': 1,  # WAE always predicts 1 property (HPP)
            'decoder_scale': self.decoder_scale,
            'regression_scale': self.regression_scale,

            # Dropout configuration
            'use_full_dropout': self.full_dropout,
            'encoder_dropout': self.encoder_dropout,
            'decoder_dropout': self.decoder_dropout,
            'regression_dropout': self.regression_dropout,
        }


    def save_to_file(self, log_path):
        """Save configuration to JSON file for reproducibility.

        Args:
            log_path: Directory to save config file

        Returns:
            str: Path to saved config file
        """
        import json
        import os

        config_dict = {}

        # Iterate through all attributes
        for key, value in vars(self).items():
            # Skip private attributes, args object, and functions
            if key.startswith('_') or key == 'args' or callable(value):
                continue

            # Convert non-serializable types
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif key == 'inverse_transform':
                # Skip inverse_transform (it's a function)
                config_dict[key] = 'identity' if self.predict_landscape else 'normalized'
            elif isinstance(value, (str, int, float, bool, type(None))):
                config_dict[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples (should be serializable)
                config_dict[key] = list(value)
            elif isinstance(value, dict):
                # Convert dicts (should be serializable)
                config_dict[key] = value
            else:
                # For other types, convert to string
                config_dict[key] = str(value)

        # Also add parsed arguments for reference
        config_dict['_command_line_args'] = vars(self.args) if hasattr(self, 'args') else {}

        # Save to file
        config_file = os.path.join(log_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        return config_file

    def load_from_file(self, config_file, logger=None):
        """Load configuration from JSON file and override current settings.

        Args:
            config_file: Path to config.json file
            logger: Optional logger for info messages

        Returns:
            self: Returns self for method chaining
        """
        import json
        import os

        if not os.path.exists(config_file):
            raise FileNotFoundError(f'Config file not found: {config_file}')

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # Remove special keys that shouldn't be loaded
        config_dict.pop('_command_line_args', None)
        config_dict.pop('inverse_transform', None)  # Will be set during data loading

        # Convert device string back to torch.device
        if 'device' in config_dict:
            device_str = config_dict['device']
            if 'cuda' in device_str:
                config_dict['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                config_dict['device'] = torch.device('cpu')

        # Update attributes
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if logger:
                    logger.debug(f'Loaded config.{key} = {value}')

        if logger:
            logger.info(f'Configuration loaded from: {config_file}')

        return self


def parse_args():
    """Parse command line arguments.

    If --config-file is provided, loads saved command-line args from that file
    and uses them as defaults. Any additional CLI args will override the saved values.
    """
    import sys
    import json

    # Check for --config-file early to load saved args as defaults
    config_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config-file' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break
        elif arg.startswith('--config-file='):
            config_file = arg.split('=', 1)[1]
            break

    # Load saved arguments if config file provided
    saved_args = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        saved_args = config_dict.get('_command_line_args', {})
        # Remove config_file from saved args to avoid recursion
        saved_args.pop('config_file', None)
        print(f"Loaded saved arguments from: {config_file}")
        
    parser = argparse.ArgumentParser(description='HEO Active Learning Pipeline')
    parser.add_argument('--config-file', type=str,
                       help='Load saved arguments from config.json file (CLI args override saved values)')
    parser.add_argument('--log-path', type=str,
                       help='Custom path for logs/saves. If not provided, uses default: heoc_path/method/timestamp')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory to save checkpoints. If not provided, uses log_path')
    parser.add_argument('--model', type=str, default='ensemble', choices=['gp', 'mc', 'ensemble'],
                       help='Surrogate model: deep ensemble (default, used in the paper), Gaussian process or MC dropout')
    parser.add_argument('--ensemble-size', type=int, default=16,
                       help='Number of ensemble members (default: 16)')
    parser.add_argument('--quadratic-fit-hpp', action=argparse.BooleanOptionalAction, default=True,
                       help='Use the HPC from the quadratic fit of G(v) as training label (default: on; --no-quadratic-fit-hpp to disable)')
    parser.add_argument('--quadratic-fit-landscape-hpp', action=argparse.BooleanOptionalAction, default=True,
                       help='Derive the predicted HPC from a quadratic fit of the predicted G(v) (default: on; --no-quadratic-fit-landscape-hpp to disable)')
    

    parser.add_argument('--bootstrap-fraction', type=lambda x: None if x.lower() == 'none' else float(x), default=None,
                       help='Bootstrap sampling fraction for ensemble training (default: 0.8, use "None" to train all members on full dataset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for WAE, ensemble, candidate sampling and val split (default: 42)')
    parser.add_argument('--method', type=str, default='clhp',
                       help='Method/directory name for organizing outputs (default: clhp)')
    parser.add_argument('--sampling', type=str, default='thompson',
                       choices=['thompson', 'explore', 'greedy'],
                       help='Sampling strategy for active learning (default: thompson)')
    parser.add_argument('--max-element-count', type=int, default=None,
                       help='Maximum number of unique cations allowed in decoded samples.')
    parser.add_argument('-b', type=int, default=2,
                       help='Phonon batch size (default: 2, for an A100 GPU)')
    parser.add_argument('-t', type=int, default=10,
                       help='Relaxation threads (default: 10, for an A100 GPU)')
    parser.add_argument('--n-samples-total', type=int, default=25000,
                       help='Number of samples total, before selective sampling.')
    parser.add_argument('--use-preloader', action='store_true',
                       help='Enable GPU memory preloader thread (default: False)')
    parser.add_argument('--use-validation', action=argparse.BooleanOptionalAction, default=True,
                       help='Hold out a random validation split for early stopping and LR scheduling (default: on; --no-use-validation to disable)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    

    parser.add_argument('--overwrite', action='store_true',
                       help='Remove all existing cycles (except cycle0) before starting (default: False)')
    parser.add_argument('--n-vacancies-max', type=int, default=8,
                       help='Largest number of O vacancies per 80-atom supercell; the surrogate predicts G(v) for '
                            'v = 0..N (default 8, i.e. delta = 0..0.5, as in the training labels). '
                            'The runs reported in the paper used 7 (v = 0..7).')
    parser.add_argument('--gibbs-noise-std', type=float, default=None,
                       help='Fixed std (eV) used to convert the target-noise std (0.2 to 0.8 eV) to normalised units. '
                            'Default: the std of the grouped z-score normalisation of the G(v) data. '
                            'The runs reported in the paper used a fixed value of 7.331.')
    parser.add_argument('--max-cycle', type=int, default=50,
                       help='Stop after this cycle has been completed (default: 50)')
    

    parser.add_argument('--val-split-mode', type=str, default='random',
                       choices=['random', 'sequential'],
                       help='Validation split mode: "random" (default) or "sequential" (first N%% samples)')
    parser.add_argument('--ensemble-calibrate', action='store_true',
                       help='Apply post-hoc temperature calibration to ensemble uncertainty (default: False)')
    parser.add_argument('--round-compositions', action=argparse.BooleanOptionalAction, default=True,
                       help='Round decoded compositions to integer cation counts on the 80-atom cell (default: on; --no-round-compositions to disable)')
    parser.add_argument('--predict-landscape', action=argparse.BooleanOptionalAction, default=True,
                       help='Predict the Gibbs energy landscape G(v) in the fuel and steam reactor and derive the HPC from it, instead of predicting the HPC directly (default: on; --no-predict-landscape to disable)')

    # SLURM fan-out for phonon calculations
    parser.add_argument('--phonon-slurm-jobs', type=int, default=0,
                       help='Number of SLURM GPU jobs to fan out for phonon calculations. '
                            '0 = run locally (default), N > 0 = submit N sbatch jobs each processing 1/N of the SQS dbs.')
    parser.add_argument('--slurm-account', type=str, default=None,
                       help='SLURM account for phonon fan-out jobs')
    parser.add_argument('--slurm-container', type=str, default=None,
                       help='Path to Apptainer/Singularity container image for phonon jobs')
    parser.add_argument('--slurm-time', type=str, default='0-08:00:00',
                       help='Time limit for each phonon SLURM job (default: 0-08:00:00)')
    parser.add_argument('--slurm-use-preloader', action='store_true',
                       help='Run GPU memory preloader in each phonon SLURM job (for GPU util limits)')
    parser.add_argument('--slurm-phonon-batch', type=int, default=0,
                       help='Batch size for slurm phonon.')
    

        # Apply saved args as defaults (CLI args will override)
    if saved_args:
        # Convert underscores back to match argparse attribute names
        parser.set_defaults(**saved_args)
        
    return parser.parse_args()

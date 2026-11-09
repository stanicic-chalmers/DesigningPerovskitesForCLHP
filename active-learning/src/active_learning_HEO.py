# Standard library imports
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
import threading
import random
import shutil

# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

# Determine the directory based on whether the environment is a notebook or a script
try:
    # This will raise an error in notebooks since __file__ doesn't exist
    directory = Path(__file__).resolve().parent
except NameError:
    # In Jupyter notebooks, the current working directory is used
    directory = Path().resolve()

sys.path.append(str(directory.parent))

# Go up 3 levels: py -> src -> active-learning
# Folder structure:
# <Parent folder> 
#               ├── active-learning
#               |   └── src
#               |       └── active_learning_HEO.py
#               ├── oxygen-carrier-discovery
#               └── metal

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

metal_path = os.path.join(project_root, 'metal')
pyscripts_path = os.path.join(project_root, 'oxygen-carrier-discovery','heoc','pyscripts')

sys.path.append(metal_path)
sys.path.append(pyscripts_path)

# Local imports
from config import Config, parse_args
from ANN_functions_new import WAE_modular, train_WAE, same_seeds, get_latents
from al_functions import *
from ensemble_mlp import DeepEnsemble, train_deep_ensemble
from al_util import (
    save_checkpoint,
    save_progress_state, should_skip_stage,
    setup_logging, validate_cycle_integrity,
    auto_detect_and_load_checkpoint, log_model_info,
    STAGE_TRAINING_AND_SAMPLING, STAGE_SAMPLE_DB,
    STAGE_SQS, STAGE_PHONON, STAGE_PHASE_STABILITY, STAGE_TRAINING_DB,
    STAGE_COMPARE_PREDICTIONS, log_wae_loss,
    _gpu_memory_preloader
)
from al_viz import visualize_latent_space, plot_latent_evolution, analyze_latent_structure
from build_sample_database import build_sample_database
from build_sqs_databases import build_sqs_structures
from perform_phonon_calculations import perform_phonon_calculations
from slurm_phonon import submit_phonon_slurm_jobs
from update_training_database import update_training_database
from data_manager import DataManager

# External project imports
from metal.applications.crystals.stability_factors import update_phase_stability_dbs
from pymatgen.core import SETTINGS


args = parse_args()
config = Config(args)
config.validate()

def main():
    _preloader_stop = None
    _preloader = None

    if config.use_gpu and config.use_preloader:
        _preloader_stop = threading.Event()
        _preloader = threading.Thread(
            target=_gpu_memory_preloader,
            args=(0, _preloader_stop),
            daemon=True
        )
        _preloader.start()
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up initial cycle and log path using config methods
    config.ensure_initial_cycle()
    config.setup_log_path(now)

    # Set up logging early so we can use it for all output
    logger = setup_logging(config.log_path)

    # Log initial configuration
    config.log_initial_info(logger, config.log_path)

    # Save configuration to file for reproducibility
    config_file = config.save_to_file(config.log_path)
    logger.info(f'Configuration saved to: {config_file}')

    # Set random seed for reproducibility
    y_name = 'raw_y'
    same_seeds(config.default_seed)
    np.random.seed(config.default_seed)
    random.seed(config.default_seed)

    # Get properties to predict from config
    config.set_method_list()
    n_props = len(config.method_list)

    # Get WAE parameters from config
    params = config.get_wae_params()

    # Set up checkpoint directory using config method
    checkpoint_dir = config.get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)

    wae_model = WAE_modular(params).to(config.device)

    # Log model architecture and parameters
    log_model_info(wae_model, params, logger, is_ensemble=False)

    # Auto-detect if we should resume from a checkpoint and determine starting cycle
    start_iteration, cycle, progress_state = auto_detect_and_load_checkpoint(
        config.log_path, checkpoint_dir, wae_model, config.device, logger, config.folder_name
    )

    # Validate cycle integrity before starting the loop
    validate_cycle_integrity(config.folder_name, logger)

    # ==================================================================================
    # MAIN ACTIVE LEARNING LOOP
    # ==================================================================================
    # Run n_cycles iterations from where we resume (cycle 0 if starting fresh)
    end_iteration = start_iteration + config.default_cycles
    logger.info(f'Will run {config.default_cycles} iterations from {start_iteration} to {end_iteration - 1}')

    for iteration in range(start_iteration, end_iteration):
        if config.max_cycle is not None and cycle > config.max_cycle:
            logger.info(f'Cycle {config.max_cycle} (--max-cycle) reached, stopping.')
            break
        logger.info('=' * 80)
        logger.info(f'ITERATION {iteration} / CYCLE {cycle}')
        logger.info('=' * 80)

        # Reduce WAE training epochs for subsequent iterations (initial training takes longer)
        if iteration != 0:
            params['num_epoch'] = config.subsequent_wae_epochs

        new_folder_name = f'{config.heoc_path}/{config.method}/cycle{cycle}'

        # Data Management
        data_manager = DataManager(config, logger)
        data_manager.load_data(config.method_list, config.db_name, config.elements)
        
        # Decide normalization based on models used
        norm_mode = 'zscore' if config.model == 'ensemble' else 'minmax'
        data_manager.normalize(mode=norm_mode)

        # Build ensemble noise std for ensemble sampling (perturbed targets)
        if config.predict_landscape and config.use_ensemble:
            gibbs_std = (config.gibbs_noise_std if config.gibbs_noise_std is not None   # fixed value (paper runs: 7.331)
                         else data_manager.norm_stats.get('gibbs_SR_0', {}).get('std', None))  # grouped z-score: same std for all G(v)
            noise_std = config.build_ensemble_noise_std(gibbs_std=gibbs_std)
            logger.info(f'Ensemble sampling noise_std (normalized): {[f"{x:.4f}" for x in noise_std]}')
        if config.use_validation:
             data_manager.split_data(val_split=config.val_split, mode=config.val_split_mode,
                                     seed=config.default_seed)
        
        # WAE Data (Includes HPC for WAE training)
        if config.use_validation:
             dataloader, val_dataloader = data_manager.get_wae_data(batch_size=params['batch_size'], device=config.device)
        else:
             dataloader = data_manager.get_wae_data(batch_size=params['batch_size'], device=config.device)
             val_dataloader = None
        
        # Get dataset reference for later use (visualization)
        dataset = dataloader.dataset
        
        # Get dataset reference for later use (visualization)
        dataset = dataloader.dataset
             
        # Ensemble/GP Data 
        # get_ensemble_data always returns 6 values: train_x, train_y, val_x, val_y, test_x, test_y
        ens_train_x, ens_train_y_dict, ens_val_x, ens_val_y_dict, _, _ = data_manager.get_ensemble_data()
        
        # Determine effective output dimension for ensemble
        # If landscape mode, we exclude HPP, so dim is n_props - 1
        # If not, dim is n_props (usually 1 for HPP)
        if config.predict_landscape:
            ensemble_output_dim = n_props - 1
        else:
            ensemble_output_dim = n_props

        # ---------------------------Train models and generate samples---------------------
        if not should_skip_stage(STAGE_TRAINING_AND_SAMPLING, progress_state):
            # Ensure cycle folder exists before any operations
            os.makedirs(new_folder_name, exist_ok=True)

            # Initialize model_dict and ensemble_model
            model_dict = None
            ensemble_model = None
            
            # Train GP models if enabled (AFTER WAE training so latents are meaningful)
            if config.use_gp:
                logger.info('\nTraining Gaussian Process models...')
                # Prepare GP input: use latent vectors if flag is set, otherwise use compositions
                gp_train_x = ens_train_x # Use ensemble/GP training data
                model_dict = {}
                for i in range(ensemble_output_dim):
                    model_key = f'model{i}'
                    # Use keys y0, y1... from DataManager
                    model_dict[model_key] = GPRegressor(gp_train_x, ens_train_y_dict[f'y{i}'], device=config.device).to(config.device)
                    _gp_loss_history = train_GP_new(model_dict[model_key], gp_train_x, ens_train_y_dict[f'y{i}'], config.default_gp_training_iters, logger=logger)
                    model_dict[model_key].print_parameters(logger)

            if config.use_ensemble:
                # Prepare ensemble training data: use compositions (ens_train_x)
                ensemble_train_x = ens_train_x  # Compositions [N, n_elements]
                
                # Prepare ensemble targets: stack all properties
                if ensemble_output_dim > 1:
                    # Stack properties [N, ensemble_output_dim]
                    train_y_list = [ens_train_y_dict[f'y{i}'] for i in range(ensemble_output_dim)]
                    ensemble_train_y = torch.stack(train_y_list, dim=1)
                    
                    if config.use_validation:
                        val_y_list = [ens_val_y_dict[f'y{i}'] for i in range(ensemble_output_dim)]
                        ensemble_val_y = torch.stack(val_y_list, dim=1)
                        ensemble_val_x = ens_val_x
                    else:
                        ensemble_val_y = None
                        ensemble_val_x = None
                else:
                    # Single property [N]
                    ensemble_train_y = ens_train_y_dict['y0']
                    ensemble_val_x = ens_val_x if config.use_validation else None
                    ensemble_val_y = ens_val_y_dict['y0'] if config.use_validation else None

                # Get input dimension from train_x (number of elements)
                input_dim = ensemble_train_x.shape[1]
                
                # Assertions to verify dimensions
                if ensemble_train_y.ndim == 1:
                    assert ensemble_output_dim == 1, f"Expected 1 output dim but got {ensemble_output_dim}"
                else:
                    assert ensemble_train_y.shape[1] == ensemble_output_dim, \
                        f"Ensemble targets have {ensemble_train_y.shape[1]} dims but expected {ensemble_output_dim}"

                if config.predict_landscape:
                    # In landscape mode, we expect Gibbs only (16)
                    # Total methods = 17 (HPC + 16 Gibbs). 
                    # Ensemble output should be 16.
                    expected_dim = n_props - 1
                    assert ensemble_output_dim == expected_dim, \
                        f"Landscape mode: Expected {expected_dim} outputs (Gibbs), got {ensemble_output_dim}"
                    logger.info(f"Verified ensemble dimensions for landscape mode: {ensemble_output_dim} outputs (HPC excluded)")
                else:
                    # standard mode
                    logger.info(f"Verified ensemble dimensions: {ensemble_output_dim} outputs")


                # Create ensemble
                ensemble_model = DeepEnsemble(
                    n_members=config.ensemble_size,
                    input_dim=input_dim,
                    hidden_dims=config.ensemble_hidden_dims,
                    dropout_rate=config.ensemble_dropout_rate,
                    device=config.device,
                    output_dim=ensemble_output_dim, # 16 if landscape (HPC excluded), 1 if HPC
                    prior_variance=config.ensemble_prior_variance,
                    seed=config.default_seed,
                )

                # Train ensemble
                _ensemble_loss_histories = train_deep_ensemble(
                    ensemble_model,
                    ensemble_train_x,
                    ensemble_train_y,
                    val_x=ensemble_val_x,
                    val_y=ensemble_val_y,
                    bootstrap_fraction=config.bootstrap_fraction,
                    epochs=config.ensemble_epochs,
                    batch_size=config.ensemble_batch_size,
                    lr=config.ensemble_lr,
                    weight_decay=config.ensemble_weight_decay,
                    base_seed=config.default_seed,
                    config=config,
                    patience=config.early_stopping_patience,
                    prior_reg=config.ensemble_prior_reg,
                    noise_std=config.ensemble_noise_std,
                    logger=logger
                )

                # Apply post-hoc calibration if requested
                if config.ensemble_calibrate and ensemble_val_x is not None:
                    logger.info('Applying post-hoc temperature calibration to ensemble...')
                    from ensemble_mlp import calibrate_ensemble
                    temperature = calibrate_ensemble(ensemble_model, ensemble_val_x, ensemble_val_y, logger)
                    ensemble_model.temperature = temperature
                    logger.info(f'Calibration temperature: {temperature:.4f}')
                elif config.ensemble_calibrate:
                    logger.warning('--ensemble-calibrate requires --use-validation, skipping calibration')
            
            # Train WAE (after regression models so latent space learns from their predictions)
            logger.info('Train WAE')
            params = config.get_wae_params(cycle)
            optimizer = Adam(wae_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            wae_loss_history = train_WAE(wae_model, optimizer, dataloader, params, logger=logger, val_dataloader=val_dataloader)
            log_wae_loss(wae_loss_history, cycle, config.log_path)

            save_checkpoint(wae_model, iteration, checkpoint_dir, optimizer=optimizer)

            # --------------------Visualize latent space------------------------
            logger.info('Visualizing latent space...')
            # Extract property values (HPC is first column in dataset)
            property_values = dataset.x[:, 0]  # HPC values
            viz_results = visualize_latent_space(
                wae_model=wae_model,
                dataset=dataset,
                property_values=property_values,
                cycle=cycle,
                log_path=config.log_path,
                property_name=config.method_list[0],
                device=config.device,
                logger=logger
            )

            # --------------------Analyze latent structure------------------------
            analyze_latent_structure(
                latent_reps=viz_results['latent_reps'],
                property_values=viz_results['property_values'],
                logger=logger,
                cycle=cycle,
                log_path=config.log_path
            )

            # --------------------Generate samples-----------------------------
            sample_name = new_folder_name + f'/sample.csv'
            latents = get_latents(wae_model, dataset)

            if config.use_ensemble:
                # Deep Ensemble mode
                samples_ensemble(wae_model, ensemble_model, sample_name, latents, iteration,
                                config.method_list, config, config.log_path, logger=logger, cycle=cycle)
            elif config.use_gp:
                # GP mode
                samples_GP(wae_model, model_dict, sample_name, latents, iteration,
                           config.method_list, config, config.log_path, logger=logger, cycle=cycle)
            else:
                # MC Dropout mode
                samples_mcdrop(wae_model, sample_name, latents, iteration,
                               config.method_list, config, config.log_path, logger=logger, cycle=cycle)
            # Save progress after both training and sampling complete
            save_progress_state(config.log_path, cycle, STAGE_TRAINING_AND_SAMPLING)
        else:
            logger.info('Skipping training and sampling - already completed')
    
        # ==================================================================================
        # STAGE 2: PREPARE SAMPLES FOR PROPERTY CALCULATIONS
        # ==================================================================================
        # Build sample database from selected candidates
        if not should_skip_stage(STAGE_SAMPLE_DB, progress_state):
            logger.info('Building sample database...')
            build_sample_database(directory=f'{config.method}')
            logger.info('Sample db created!')
            save_progress_state(config.log_path, cycle, STAGE_SAMPLE_DB)
        else:
            logger.info('Skipping sample database - already created')

        if not should_skip_stage(STAGE_SQS, progress_state):
            logger.info('Building SQS structures...')
            build_sqs_structures(processes=config.default_samples_per_cycle, directory=f'{config.method}')
            logger.info('SQS structures built!')
            save_progress_state(config.log_path, cycle, STAGE_SQS)
        else:
            logger.info('Skipping SQS structures - already built')

        if not should_skip_stage(STAGE_PHONON, progress_state):
            logger.info('Performing phonon calculations (this may take a while)...')
            if config.phonon_slurm_jobs > 0:
                # Fan out phonon calculations across multiple SLURM GPU jobs
                submit_phonon_slurm_jobs(config, cycle, logger)
            else:
                # Run locally on current node (threaded relaxations run inside if configured)
                perform_phonon_calculations(
                    processes=1, directory=f'{config.method}', use_gpus=config.use_gpu,
                    logging=logger, batch_size=config.phonon_batch_size,
                    n_relax_threads=config.relax_threads or 0)
            logger.info('Phonon calculations complete!')
            save_progress_state(config.log_path, cycle, STAGE_PHONON)
        else:
            logger.info('Skipping phonon calculations - already completed')

        if not should_skip_stage(STAGE_PHASE_STABILITY, progress_state):
            logger.info('Updating phase stability...')
            api_key = SETTINGS['PMG_MAPI_KEY']
            update_phase_stability_dbs(filenames=glob(f'{new_folder_name}/sqs_*.db'), api_key=api_key, processes=10)
            logger.info('Phase stability updated!')
            save_progress_state(config.log_path, cycle, STAGE_PHASE_STABILITY)
        else:
            logger.info('Skipping phase stability - already updated')

        if not should_skip_stage(STAGE_TRAINING_DB, progress_state):
            logger.info('Updating training database...')
            update_training_database(
                directory=f'{config.method}',
                quadratic_fit=config.quadratic_fit_hpp
            )
            logger.info('Training database updated!')
            save_progress_state(config.log_path, cycle, STAGE_TRAINING_DB)
        else:
            logger.info('Skipping training database update - already completed')

        if not should_skip_stage(STAGE_COMPARE_PREDICTIONS, progress_state):
            # Compare predictions with actual values for this cycle
            logger.info('Comparing predictions with actual values...')
            from al_functions import compare_predictions_with_actuals
            compare_predictions_with_actuals(
                cycle=cycle,
                log_path=config.log_path,
                sample_file=f'{new_folder_name}/sample.csv',
                db_name=config.db_name,
                elements=config.elements,
                method_list=config.method_list,
                config=config,
                logger=logger
            )
            save_progress_state(config.log_path, cycle, STAGE_COMPARE_PREDICTIONS)
        else:
            logger.info('Skipping prediction comparison - already completed')

        logger.info(f'Cycle {cycle} complete!')

        # Prepare for next iteration
        cycle += 1
        progress_state = None  # Reset progress for new cycle

    # ---------------------------Create evolution plot---------------------
    logger.info('Creating latent space evolution plot...')
    plot_latent_evolution(log_path=config.log_path, property_name=config.method_list[0], logger=logger)
    logger.info('Active learning pipeline complete!')

    if config.use_gpu and config.use_preloader and _preloader_stop is not None:
        # Cleanup background threads
        _preloader_stop.set()
        _preloader.join(timeout=2.0)


if __name__ == '__main__':
    main()

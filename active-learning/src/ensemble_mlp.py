"""
Deep Ensemble MLP for uncertainty-aware property prediction.

Created 2025-11-19 for active learning pipeline.
Implements standalone ensemble of MLPs trained on compositions.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class EnsembleMember(nn.Module):
    """Single member of the deep ensemble.

    Architecture:
        Input → Hidden1 → Dropout → Hidden2 → Dropout → {Mean head, LogVar head}

    Parameters
    ----------
    input_dim : int
        Input dimension (number of cation species; 22 for the A and B elements of this work)
    hidden_dims : list of int
        Sizes of hidden layers (default: [96, 48])
    dropout_rate : float
        Dropout rate for regularization (default: 0.1)
    """

    def __init__(self, input_dim, hidden_dims=[96, 48], dropout_rate=0.1, output_dim=1):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(hidden_dims[1], output_dim)
        self.log_var_head = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input compositions [batch_size, input_dim]

        Returns
        -------
        mean : torch.Tensor
            Predicted means [batch_size, 1]
        log_var : torch.Tensor
            Predicted log-variances [batch_size, 1]
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        return mean, log_var


class DeepEnsemble:
    """Container for multiple ensemble members.

    Parameters
    ----------
    n_members : int
        Number of ensemble members (default: 5)
    input_dim : int
        Input dimension (number of elements)
    hidden_dims : list of int
        Hidden layer sizes (default: [96, 48])
    dropout_rate : float
        Dropout rate (default: 0.1)
    device : torch.device
        Device for computation
    """

    def __init__(self, n_members=5, input_dim=15, hidden_dims=[96, 48],
                 dropout_rate=0.1, device=None, output_dim=1, prior_variance=1, seed=42):
        self.n_members = n_members
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.prior_variance = prior_variance  # λ: Variance for Gaussian prior N(0, λI)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Create ensemble members with unique random seeds for diverse weight initialization
        self.members = []
        self.anchors = []  # Random anchors for anchored ensembles (Pearce et al. 2020)

        for i in range(n_members):
            # Seed before member creation to ensure diverse weight initialization
            torch.manual_seed(seed + i * 1000)
            member = EnsembleMember(input_dim, hidden_dims, dropout_rate, output_dim=output_dim)
            # Uses PyTorch default initialization (Kaiming uniform for Linear layers)
            member = member.to(self.device)
            self.members.append(member)

            # Generate random anchor θ̃₀,m ~ N(0, λI) for this member
            # Anchors are separate from initialization - this decouples:
            # - Good init (Kaiming) for fast convergence
            # - Random anchors for diversity via regularization
            anchor = {}
            prior_std = np.sqrt(prior_variance)
            for name, param in member.named_parameters():
                if 'weight' in name:
                    # Sample anchor from N(0, λI)
                    anchor[name] = torch.randn_like(param) * prior_std
                else:
                    # Bias anchors at zero
                    anchor[name] = torch.zeros_like(param)
            self.anchors.append(anchor)

        # Temperatures for post-hoc calibration (default 1.0 = no scaling)
        self.temperature = 1.0  # Scales aleatoric variance
        

    def predict(self, x):
        """
        Make predictions with all ensemble members.

        Parameters
        ----------
        x : torch.Tensor
            Input compositions [N, input_dim]

        Returns
        -------
        dict with keys:
            'mean' : torch.Tensor [N, 1] - Ensemble mean prediction
            'aleatoric_std' : torch.Tensor [N, 1] - Aleatoric uncertainty
            'epistemic_std' : torch.Tensor [N, 1] - Epistemic uncertainty
            'total_std' : torch.Tensor [N, 1] - Total uncertainty
            'individual_means' : torch.Tensor [N, n_members] - Individual predictions
            'individual_log_vars' : torch.Tensor [N, n_members] - Individual log-variances
        """
        means_list = []
        log_vars_list = []

        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)

        with torch.no_grad():
            for member in self.members:
                member.eval()
                mean, log_var = member(x)
                means_list.append(mean)
                log_vars_list.append(log_var)

        # Stack predictions: [N, output_dim, M]
        means = torch.stack(means_list, dim=-1)
        log_vars = torch.stack(log_vars_list, dim=-1)
        
        # Clamp log_vars to match training (prevent numerical instability)
        log_vars = torch.clamp(log_vars, min=-5, max=1)

        # Compute ensemble statistics
        pred_mean = means.mean(dim=-1)  # [N, output_dim]
        
        # Apply temperature scaling to variances (for calibration)
        var_aleatoric = (self.temperature ** 2) * torch.exp(log_vars).mean(dim=-1)  # Average predicted variance
        var_epistemic = means.var(dim=-1, unbiased=False)  # Variance of predictions
        var_total = var_aleatoric + var_epistemic

        # Handle output dimensions for backward compatibility
        # If output_dim=1, we might want to squeeze to [N] or keep [N, 1]
        # The original code returned [N, 1] for mean/std, but squeezed individual_means to [N, M]
        
        individual_means = means
        individual_log_vars = log_vars
        
        if self.output_dim == 1:
            individual_means = means.squeeze(1) # [N, M]
            individual_log_vars = log_vars.squeeze(1) # [N, M]

        return {
            'mean': pred_mean,
            'aleatoric_std': torch.sqrt(var_aleatoric),
            'epistemic_std': torch.sqrt(var_epistemic),
            'total_std': torch.sqrt(var_total),
            'individual_means': individual_means,
            'individual_log_vars': individual_log_vars,
        }

    def save(self, path):
        """Save all ensemble members and anchors to a file."""
        state_dicts = [member.state_dict() for member in self.members]
        torch.save({
            'n_members': self.n_members,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'output_dim': self.output_dim,
            'prior_variance': self.prior_variance,
            'temperature': self.temperature,
            'state_dicts': state_dicts,
            'anchors': self.anchors,  # Save anchors for reproducibility
        }, path)

    def load(self, path):
        """Load all ensemble members and anchors from a file."""
        checkpoint = torch.load(path, map_location=self.device)

        # Verify architecture matches
        assert checkpoint['n_members'] == self.n_members
        assert checkpoint['input_dim'] == self.input_dim

        # Load calibration parameters if available
        if 'temperature' in checkpoint:
            self.temperature = checkpoint['temperature']
        if 'prior_variance' in checkpoint:
            self.prior_variance = checkpoint['prior_variance']
        elif 'prior_std' in checkpoint:
            # Backward compatibility: convert std to variance
            self.prior_variance = checkpoint['prior_std'] ** 2

        # Load anchors if available (for anchored ensembles)
        if 'anchors' in checkpoint:
            self.anchors = checkpoint['anchors']

        # Load state dicts
        for i, state_dict in enumerate(checkpoint['state_dicts']):
            self.members[i].load_state_dict(state_dict)
            self.members[i] = self.members[i].to(self.device)


def nll_loss(y_true, y_pred_mean, y_pred_log_var):
    """
    Negative log-likelihood loss for training ensemble members.

    Loss = 0.5 * (log(σ²) + (y - μ)² / σ²)

    Parameters
    ----------
    y_true : torch.Tensor
        True target values [batch_size] or [batch_size, output_dim]
    y_pred_mean : torch.Tensor
        Predicted means [batch_size] or [batch_size, output_dim]
    y_pred_log_var : torch.Tensor
        Predicted log-variances [batch_size] or [batch_size, output_dim]

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    # Clamp log_var to prevent numerical instability
    log_var = torch.clamp(y_pred_log_var, min=-5, max=1)
    var = torch.exp(log_var)

    # NLL loss
    nll = 0.5 * (log_var + (y_true - y_pred_mean) ** 2 / var)

    return nll.mean()

def train_deep_ensemble(ensemble, train_x, train_y, val_x=None, val_y=None,
                       bootstrap_fraction=None, epochs=300, batch_size=32,
                       lr=1e-3, weight_decay=1e-4, base_seed=42,
                       patience=50, prior_reg=True, noise_std=None,
                       config=None, logger=None):
    """
    Train the deep ensemble.

    (For Thompson) Perturbed Prior Regularization (ensemble sampling)

    Parameters
    ----------
    ensemble : DeepEnsemble
        The model to train
    train_x : torch.Tensor
        Training inputs
    train_y : torch.Tensor
        Training targets
    val_x : torch.Tensor, optional
        Validation inputs
    val_y : torch.Tensor, optional
        Validation targets
    bootstrap_fraction : float, optional
        If set, train_loader is ignored and new loaders are created.
    epochs : int
        Max epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
 prior_reg : float
        Scaling factor for perturbed prior regularization.
        Loss = NLL +  prior_reg / λ) * ||θ - θ̃₀||²
        where λ = ensemble.prior_variance
        - prior_reg=False: No regularization (default behavior)
        - prior_reg=True: Standard ensemble sampling formulation
    noise_std : float, array-like, or None
        Std for target perturbation (ensemble sampling, Lu & Van Roy 2017).
        Each member m sees: y_perturbed = y + Z̃_m where Z̃_m ~ N(0, noise_std²)
        - None or 0: No target perturbation
        - scalar: Same noise std for all output dimensions
        - array [output_dim]: Per-dimension noise std
    Returns
    -------
    list of dict
        List of loss history dictionaries for each member.
    """
    n_samples = len(train_x)
    use_bootstrap = bootstrap_fraction is not None
    bootstrap_size = int(bootstrap_fraction * n_samples) if use_bootstrap else n_samples
    
    if logger:
        logger.info(f'Training {ensemble.n_members} ensemble members (Interleaved)')
        if use_bootstrap:
            logger.info(f'Dataset size: {n_samples}, Bootstrap size: {bootstrap_size} (fraction={bootstrap_fraction})')
        else:
            logger.info(f'Dataset size: {n_samples}, Training on full dataset (no bootstrapping)')

        if prior_reg and noise_std is not None:
            logger.info(f'Anchored Ensemble Regularization (Pearce et al. 2020, Eq. 8):')
            logger.info(f'  Anchors: θ_anc ~ N(0, σ²_prior·I) with σ²_prior={ensemble.prior_variance}')
            logger.info(f'  Reg term: (1/N)(1/2σ²_prior)||(θ - θ_anc)||²')
            logger.info(f'  (Coefficient computed dynamically using noise_std)')

    # Setup DataLoaders and Optimizers for each member
    optimizers = []

    # Use pre-computed anchors from ensemble (random samples from prior)
    # Anchors are separate from initialization:
    # - Weights initialized with Kaiming (good starting point, fast convergence)
    # - Anchors sampled from N(0, λI) (diversity through regularization)
    anchors = ensemble.anchors
    
    # Ensure data is on device
    train_x = train_x.to(ensemble.device)
    train_y = train_y.to(ensemble.device)
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(1)
        
    if val_x is not None:
        val_x = val_x.to(ensemble.device)
        val_y = val_y.to(ensemble.device)
        if val_y.dim() == 1:
            val_y = val_y.unsqueeze(1)

    # Generate perturbed targets for each member (ensemble sampling)
    # Each member sees: y_m = y + Z̃_m where Z̃_m ~ N(0, noise_std²)
    use_noise = noise_std is not None
    perturbed_targets = []  # List of [N, output_dim] tensors, one per member

    if use_noise:
        # Convert noise_std to tensor
        if isinstance(noise_std, (int, float)):
            noise_std_tensor = torch.tensor([noise_std], device=ensemble.device).float()
        else:
            noise_std_tensor = torch.tensor(noise_std, device=ensemble.device).float()

        output_dim = train_y.shape[1]

        # Broadcast scalar to all dimensions if needed
        if noise_std_tensor.numel() == 1:
            noise_std_tensor = noise_std_tensor.expand(output_dim)

        if logger:
            logger.info(f'Target perturbation enabled (ensemble sampling):')
            logger.info(f'  noise_std per dim: {noise_std_tensor.cpu().numpy()}')

        # Generate FIXED noise perturbations for each member (sampled once, used throughout training)
        for m in range(ensemble.n_members):
            torch.manual_seed(base_seed + 2000 * m)  # Different seed per member
            # Z̃_m ~ N(0, noise_std²) for each sample
            noise = torch.randn(n_samples, output_dim, device=ensemble.device) * noise_std_tensor
            perturbed_targets.append(train_y + noise)

        # Reset seed for reproducibility
        torch.manual_seed(base_seed)
    else:
        # No perturbation - all members see the same targets
        for _ in range(ensemble.n_members):
            perturbed_targets.append(train_y)

    for i, member in enumerate(ensemble.members):
        # Seed numpy only for bootstrap index selection (torch already seeded in __init__)
        if use_bootstrap:
            np.random.seed(base_seed + 1000 * i)
        
        # Optimizer with per-member hyperparameters
        if prior_reg and weight_decay > 0:
            weight_decay = 0.0
            if logger: logger.warning('Disabling weight decay due to anchored ensemble regularization.')
            else: print('Disabling weight decay due to anchored ensemble regularization.')
        opt = torch.optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay)
        optimizers.append(opt)        
        member.train()        
    
    # Create static loaders ONCE (Standard Bagging)
    # This ensures each member trains on a fixed subset, preserving diversity
    train_loaders = []

    # Independent training - Create separate loaders for each member
    # Each member uses its own perturbed targets and independent shuffling
    for i in range(ensemble.n_members):
        member_y = perturbed_targets[i]  # y + Z̃_m for this member

        # Create independent generator for shuffling (different seed per member)
        generator = torch.Generator()
        generator.manual_seed(base_seed + 3000 * i)

        if use_bootstrap:
            # Bootstrap sampling (with replacement) - FIXED seed per member
            # Sample indices once so member sees consistent subset (Bagging)
            rng = np.random.RandomState(base_seed + 1000 * i)
            indices = rng.choice(n_samples, size=bootstrap_size, replace=True)

            sub_x = train_x[indices]
            sub_y = member_y[indices]
            ds = TensorDataset(sub_x, sub_y)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)
        else:
            # Full dataset with independent shuffling per member
            ds = TensorDataset(train_x, member_y)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

        train_loaders.append(loader)


    # Schedulers
    schedulers = []
    use_scheduler = True
    if config is not None:
        use_scheduler = getattr(config, 'ensemble_use_scheduler', False)
        
    if use_scheduler:
        scheduler_patience = getattr(config, 'ensemble_lr_scheduler_patience', 100)
        scheduler_gamma = getattr(config, 'ensemble_lr_scheduler_gamma', 0.5)
        
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=scheduler_gamma, patience=scheduler_patience
        ) for opt in optimizers]
        
        if logger:
            logger.info(f"LR Schedulers enabled: patience={scheduler_patience}, gamma={scheduler_gamma}")
    elif logger:
        logger.info("LR Schedulers disabled (constant learning rate)")

    # History tracking
    train_loss_histories = [[] for _ in range(ensemble.n_members)]
    val_loss_history = [] # Ensemble validation loss
    prior_loss_history = [] # Prior regularization loss
    epistemic_variance_history = [] # Epistemic uncertainty over training
    aleatoric_variance_history = [] # Aleatoric uncertainty over training

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_member_states = [None] * ensemble.n_members

    for epoch in range(epochs):
        epoch_losses = [[] for _ in range(ensemble.n_members)]
        epoch_prior_losses = []

        # Zip the static loaders
        current_loader = zip(*train_loaders)
        
        for batch_data in current_loader:
            # Prepare batches for each member
            # Independent training: batch_data is a tuple of (x,y) for each member
            member_batches = batch_data
            
            # Iterate through members and their corresponding batches
            if epoch == 0 and len(epoch_losses[0]) == 0 and ensemble.n_members > 1:
                b0_x, _ = member_batches[0]
                b1_x, _ = member_batches[1]
                if torch.equal(b0_x, b1_x):
                    msg = "Warning supplied batches for member 0 and 1 are identical!"
                    if logger: logger.warning(msg)
                    else: print(msg)
                else: 
                     if logger: logger.info("Verified: Members receiving different batches.")

            # Joint Forward Pass & Prediction Loss
            member_outputs = [] # List of (mean, log_var)
            nll_losses = []
            prior_losses = []

            for i, member in enumerate(ensemble.members):
                batch_x, batch_y = member_batches[i]

                # (requires_grad is already handled above for joint training)
                mean, log_var = member(batch_x)

                member_outputs.append((mean, log_var))
                nll_losses.append(nll_loss(batch_y, mean, log_var))

                # Anchored ensemble regularization (Pearce et al. 2020, Eq. 8):
                # Loss_j = (1/N)||y - ŷ||² + (1/N)||(1/2σ_prior)·(θ - θ_anc)||²
                # With mini-batch SGD: nll_loss is mean over batch, prior term uses Γ directly
                # (the 1/N scaling happens implicitly over the epoch)
                if use_noise and prior_reg > 0:
                    prior_loss = 0.0
                    for name, param in member.named_parameters():
                        anchor_param = anchors[i][name].to(param.device)
                        prior_loss += torch.sum((param - anchor_param) ** 2)
                    # Γ = 1 / (2 σ²_prior)
                    noise_var = (noise_std_tensor ** 2).mean().item()
                    gamma = 1 / ( 2 * ensemble.prior_variance)
                    prior_losses.append(gamma * prior_loss / n_samples)
                else:
                    prior_losses.append(0.0)

            # Backward Pass
            # Zero all gradients first
            for opt in optimizers:
                opt.zero_grad()

            # Total loss = NLL + anchor regularization
            total_prior = sum(prior_losses)
            total_nll = sum(nll_losses)

            total_loss = total_nll + total_prior

            # Single backward pass - gradients flow to all member parameters
            total_loss.backward()
            
            # Gradient clipping to prevent spikes (especially with adversarial + diversity)
            for member in ensemble.members:
                torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=1.0)
            
            # Step all optimizers
            for opt in optimizers:
                opt.step()

            for i in range(ensemble.n_members):
                epoch_losses[i].append(nll_losses[i].item())

            # Track prior regularization loss
            if prior_reg > 0:
                epoch_prior_losses.append(total_prior.item())

        # End of epoch - record train losses
        avg_train_losses = [np.mean(l) if l else 0.0 for l in epoch_losses]
        for i in range(ensemble.n_members):
            train_loss_histories[i].append(avg_train_losses[i])

        # Record prior loss
        if prior_reg > 0 and len(epoch_prior_losses) > 0:
            prior_loss_history.append(np.mean(epoch_prior_losses))

        # Validation & Early Stopping
        if val_x is not None:
            # Evaluate ensemble: Mean of individual member losses
            with torch.no_grad():
                member_val_losses = []
                for member in ensemble.members:
                    member.eval()
                    m, lv = member(val_x)
                    # Compute NLL for this member
                    # nll_loss returns scalar mean over batch
                    l = nll_loss(val_y, m, lv)
                    member_val_losses.append(l.item())
                    member.train() # Switch back to train
                
                # Mean ensemble loss (average of individual losses)
                val_loss_item = np.mean(member_val_losses)
                val_loss_history.append(val_loss_item)
                
                # Step schedulers with individual member losses (not shared average)
                for i, scheduler in enumerate(schedulers):
                    scheduler.step(member_val_losses[i])
                    
                # Calculate and log epistemic and aleatoric uncertainties
                # ensemble.predict returns: mean, aleatoric_std, epistemic_std, total_std
                with torch.no_grad():
                    predictions = ensemble.predict(val_x)
                    mean_epistemic_var = (predictions['epistemic_std'].mean().item())**2
                    mean_aleatoric_var = (predictions['aleatoric_std'].mean().item())**2
                    epistemic_variance_history.append(mean_epistemic_var)
                    aleatoric_variance_history.append(mean_aleatoric_var)
            
            # Check early stopping
            if patience is not None:
                if val_loss_item < best_val_loss:
                    best_val_loss = val_loss_item
                    epochs_without_improvement = 0
                    # Save all states

                    for i, member in enumerate(ensemble.members):
                        best_member_states[i] = {k: v.cpu().clone() for k, v in member.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if logger:
                            logger.info(f'Early stopping triggered at epoch {epoch}. Best Ensemble Val Loss: {best_val_loss:.6f}')
                        else:
                            print(f'Early stopping triggered at epoch {epoch}. Best Ensemble Val Loss: {best_val_loss:.6f}')
                        
                        # Restore all members
                        for i, member in enumerate(ensemble.members):
                            if best_member_states[i] is not None:
                                member.load_state_dict(best_member_states[i])
                        break
        
        # Logging
        if epoch % 50 == 0 or epoch == epochs - 1:
            epoch_means = [np.mean(l) for l in epoch_losses]
            mean_train_loss = np.mean(epoch_means)
            current_lr = optimizers[0].param_groups[0]['lr']
            msg = f'Epoch {epoch}/{epochs-1}, LR: {current_lr:.2e}, Mean Train Loss: {mean_train_loss:.6f}'

            if prior_reg > 0 and len(epoch_prior_losses) > 0:
                msg += f', Prior Loss: {np.mean(epoch_prior_losses):.6f}'

            if val_x is not None:
                msg += f', Val Loss: {val_loss_item:.6f}'
                msg += f', Epistemic Var: {mean_epistemic_var:.6f}, Aleatoric Var: {mean_aleatoric_var:.6f}'
                epistemic_aleatoric_ratio = mean_epistemic_var / (mean_aleatoric_var + 1e-10)
                msg += f', Epi/Ale Ratio: {epistemic_aleatoric_ratio:.3f}'

            if logger:
                logger.info(msg)
            else:
                print(msg)
            
            print(f'Batch Loss: {total_loss.item():.6f} (NLL: {total_nll.item():.6f}, Prior: {total_prior.item():.6f})')


    # Restore best models if finished without early stopping
    if val_x is not None and patience is not None and best_member_states[0] is not None and epochs_without_improvement < patience:
         for i, member in enumerate(ensemble.members):
            member.load_state_dict(best_member_states[i])
         if logger:
             logger.info(f'Restored ensemble with best validation loss: {best_val_loss:.6f}')
         else:
             print(f'Restored ensemble with best validation loss: {best_val_loss:.6f}')

    results = []
    for i in range(ensemble.n_members):
        res = {'train_loss': train_loss_histories[i]}
        if val_x is not None:
            res['val_loss'] = val_loss_history
        results.append(res)

    # Add ensemble-level statistics to first member's results
    if len(results) > 0:
        results[0]['prior_loss_history'] = prior_loss_history
        results[0]['epistemic_variance_history'] = epistemic_variance_history
        results[0]['aleatoric_variance_history'] = aleatoric_variance_history
    

    return results


def calibrate_ensemble(ensemble, val_x, val_y, logger=None):
    """
    Currently not really needed since it doesnt affect ensemble sampling.
    
    Perform post-hoc temperature calibration on ensemble using validation data.
    
    Finds optimal temperature T that minimizes validation NLL by scaling the
    predicted variances: var_calibrated = T² * var_predicted
    
    
    Parameters
    ----------
    ensemble : DeepEnsemble
        Trained ensemble to calibrate
    val_x : torch.Tensor
        Validation inputs [N_val, input_dim]
    val_y : torch.Tensor
        Validation targets [N_val]
    logger : logging.Logger, optional
        Logger for progress messages
    
    Returns
    -------
    float
        Optimal temperature value (will be stored in ensemble.temperature)
    """
    import scipy.optimize
    
    # Ensure tensors are on correct device
    val_x = val_x.to(ensemble.device)
    val_y = val_y.to(ensemble.device)
    if val_y.dim() == 1:
        val_y = val_y.unsqueeze(1)
    
    # Save original temperature
    orig_temp = ensemble.temperature
    
    def nll_with_temperature(T):
        """Compute validation NLL with temperature scaling"""
        ensemble.temperature = T
        
        with torch.no_grad():
            pred = ensemble.predict(val_x)
            mean = pred['mean']
            var = pred['aleatoric_std'] ** 2  # Already scaled by T² in predict()
            
            # NLL: 0.5 * (log(2π*var) + (y - μ)² / var)
            nll = 0.5 * (torch.log(2 * np.pi * var) + (val_y - mean) ** 2 / var)
            return nll.mean().item()
    
    # Optimize temperature (search range 0.1 to 10.0)
    result = scipy.optimize.minimize_scalar(
        nll_with_temperature,
        bounds=(0.1, 10.0),
        method='bounded'
    )
    
    optimal_temp = result.x
    optimal_nll = result.fun
    
    # Restore to optimal temperature
    ensemble.temperature = optimal_temp
    
    if logger:
        logger.info(f'Temperature calibration: T={optimal_temp:.4f}, Val NLL={optimal_nll:.6f}')
        if optimal_temp < 0.5:
            logger.warning(f'Temperature < 0.5 indicates significant overconfidence')
        elif optimal_temp > 2.0:
            logger.warning(f'Temperature > 2.0 indicates significant underconfidence')
    
    return optimal_temp

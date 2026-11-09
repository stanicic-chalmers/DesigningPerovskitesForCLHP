import numpy as np
import torch
from torch.utils.data import DataLoader
from ANN_functions_new import FeatureDataset
from al_functions import create_df
import logging

class DataManager:
    """
    Centralized data manager.
    Handles loading, preprocessing, normalization, and splitting of data.
    """
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Data state
        self.raw_df = None
        self.raw_x = None      # Numpy array [N, n_props + n_elements] (scaled but not normalized)
        self.props_x = None    # Numpy array [N, n_props] (original property values)
        self.comps_x = None    # Numpy array [N, n_elements] (compositions)
        
        # Normalization state
        self.norm_stats = {}   # Dictionary storing mean/std or min/max for each property
        self.normalized_x = None
        
        # Splits
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        
        # Feature names
        self.method_list = []
        self.elements = []

    def load_data(self, method_list, db_name, elements):
        """
        Load data from ASE database and apply sublattice scaling.
        """
        self.method_list = method_list
        self.elements = elements
        
        self.logger.info(f"Loading data from {db_name}...")
        
        # Use quadratic fit HPP if config flag is enabled
        use_quadratic_fit_hpp = getattr(self.config, 'quadratic_fit_hpp', False)
        if use_quadratic_fit_hpp:
            self.logger.info('Using quadratic fit HPP from database')
            
        # 1. Load DataFrame using existing helper
        self.raw_df = create_df(method_list, db_name, elements, use_quadratic_fit_hpp=use_quadratic_fit_hpp)
        self.logger.info(f"Loaded {len(self.raw_df)} samples.")
        
        # 2. Convert to numpy (exclude oxygen column which is last)
        all_data = self.raw_df.to_numpy()
        data_matrix = all_data[:, 0:-1].copy()  # Oxygen excluded
        
        # 3. Handle MMD edge case (odd number of samples)
        if data_matrix.shape[0] % 2 != 0:
            self.logger.warning(f"Odd number of samples ({data_matrix.shape[0]}). Removing one.")
            random_idx = np.random.randint(0, data_matrix.shape[0])
            data_matrix = np.delete(data_matrix, random_idx, axis=0)
            
        # 4. Apply Sublattice Scaling
        # data_matrix columns: [prop1, prop2, ..., el1, el2, ...]
        n_props = len(method_list)
        
        # Store original property values (before any normalization)
        self.props_x = data_matrix[:, :n_props].copy()
        
        # Scale compositions
        for element, scale in self.config.scaling_dictionary.items():
            if element in self.raw_df.columns and element != 'O':
                col_idx = self.raw_df.columns.get_loc(element)
                # Note: raw_df includes O, but data_matrix excluded it. 
                data_matrix[:, col_idx] = data_matrix[:, col_idx] / scale
                 
        self.comps_x = data_matrix[:, n_props:].copy()
        self.raw_x = data_matrix
        
        return self

    def normalize(self, mode='zscore'):
        """
        Normalize the property columns.
        mode: 'zscore' or 'minmax'
        """
        if self.raw_x is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        self.logger.info(f"Normalizing data using {mode}...")
        self.normalized_x = self.raw_x.copy()
        n_props = len(self.method_list)
        
        # Reset stats
        self.norm_stats = {}
        
        # Special handling for Landscape Mode (Ensemble only)
        # If we are in ensemble mode and predicting landscape, we treat HPP and Gibbs differently
        if mode == 'zscore' and self.config.predict_landscape:
            self._normalize_landscape_mode()
        else:
            # Standard independent normalization
            for i, prop in enumerate(self.method_list):
                values = self.raw_x[:, i]
                
                if mode == 'zscore':
                    mean = values.mean()
                    std = values.std()
                    # Avoid div by zero
                    if std <= 1e-8: std = 1.0
                        
                    self.normalized_x[:, i] = (values - mean) / std
                    
                    self.norm_stats[prop] = {
                        'type': 'zscore',
                        'mean': mean,
                        'std': std
                    }
                elif mode == 'minmax':
                    min_val = values.min()
                    max_val = values.max()
                    denom = max_val - min_val
                    if denom <= 1e-8: denom = 1.0
                        
                    self.normalized_x[:, i] = (values - min_val) / denom
                    
                    self.norm_stats[prop] = {
                        'type': 'minmax',
                        'min': min_val,
                        'max': max_val
                    }
                    
        # Update config with inverse transform immediately
        self.config.inverse_transform = self.get_inverse_transform_fn()
        if self.config.predict_landscape:
             self.config.inverse_transform_gibbs = self.get_inverse_transform_fn(group='gibbs')
             
        return self


    def _normalize_landscape_mode(self):
        """
        Special normalization for landscape mode:
        - HPP: Z-score
        - Gibbs: Grouped Z-score (shared mean/std across all Gibbs columns)
        """
        self.logger.info("Applying Landscape Mode Normalization (Grouped Gibbs)...")
        
        # 1. HPP (index 0)
        hpp_vals = self.raw_x[:, 0]
        hpp_mean = hpp_vals.mean()
        hpp_std = hpp_vals.std()
        if hpp_std <= 1e-8: hpp_std = 1.0
        
        self.normalized_x[:, 0] = (hpp_vals - hpp_mean) / hpp_std
        self.norm_stats['hpp'] = {'type': 'zscore', 'mean': hpp_mean, 'std': hpp_std}
        
        # 2. Gibbs (indices 1 to end of props)
        gibbs_indices = slice(1, len(self.method_list))
        gibbs_vals = self.raw_x[:, gibbs_indices]
        
        global_mean = gibbs_vals.mean()
        global_std = gibbs_vals.std()
        if global_std <= 1e-8: global_std = 1.0
        
        self.normalized_x[:, gibbs_indices] = (gibbs_vals - global_mean) / global_std
        
        # Store same stats for all Gibbs props
        gibbs_props = self.method_list[1:]
        for prop in gibbs_props:
            self.norm_stats[prop] = {
                'type': 'zscore',
                'mean': global_mean,
                'std': global_std,
                'group': 'gibbs'
            }
            
        self.logger.info(f"HPP: mean={hpp_mean:.3f}, std={hpp_std:.3f}")
        self.logger.info(f"Gibbs (Grouped): mean={global_mean:.3f}, std={global_std:.3f}")

    def split_data(self, val_split=0.15, test_split=0.0, mode='random', seed=42):
        """
        Create train/val/test split indices.
        """
        n_total = self.raw_x.shape[0]
        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)
        n_train = n_total - n_val - n_test
        
        # Ensure even numbers for MMD loss in WAE (requires splitting batches in half)
        n_val = n_val - (n_val % 2)
        n_test = n_test - (n_test % 2)
        n_train = n_train - (n_train % 2)
        
        if mode == 'sequential':
            self.val_indices = np.arange(n_val)
            self.test_indices = np.arange(n_val, n_val + n_test)
            self.train_indices = np.arange(n_val + n_test, n_total)
        else:
            np.random.seed(seed)
            indices = np.random.permutation(n_total)
            self.val_indices = indices[:n_val]
            self.test_indices = indices[n_val:n_val+n_test]
            self.train_indices = indices[n_val+n_test:]
            
        self.logger.info(f"Split data: {len(self.train_indices)} Train, {len(self.val_indices)} Val, {len(self.test_indices)} Test")
        return self

    def get_wae_data(self, batch_size=32, device='cpu'):
        """
        Get DataLoaders for WAE training. 
        WAE typically uses [HPP, Compositions] or just [Compositions].
        """
        # We usually use the full normalized matrix (Props + Comps) for WAE dataset
        # But we might need to mask out Gibbs energies if WAE only expects HPP.
        
        # Check if we need to reduce properties
        # Typically WAE is trained on HPP + Composition
        n_wae_props = 1 # Just HPP
        
        if len(self.method_list) > n_wae_props:
             # Create subset matrix: [HPP, Comps...] ignoring Gibbs cols at [1:n_props]
             # raw_x structure: [p0, p1, ... pn, c0, ... cn]
             # We want: [p0, c0, ... cn]
             
             comps_start_idx = len(self.method_list)
             prop_subset = self.normalized_x[:, :n_wae_props]
             comps_subset = self.normalized_x[:, comps_start_idx:]
             wae_data = np.hstack([prop_subset, comps_subset])
        else:
             wae_data = self.normalized_x
             
        # Create Datasets
        if self.train_indices is not None:
            train_x = torch.from_numpy(wae_data[self.train_indices]).float()
            val_x = torch.from_numpy(wae_data[self.val_indices]).float()
            
            train_ds = FeatureDataset(train_x)
            val_ds = FeatureDataset(val_x)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader
        else:
            # Full dataset
            x = torch.from_numpy(wae_data).float()
            ds = FeatureDataset(x)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            return loader

    def get_ensemble_data(self, batch_size=32, device='cpu'):
        """
        Get Data for Ensemble training.
        
        Returns (always 6 values):
            train_x: [N_train, n_elements] (Compositions only)
            train_y_dict: { 'y0': [N_train], ... } (Targets)
            val_x: [N_val, n_elements] or None
            val_y_dict: { 'y0': [N_val], ... } or None
            test_x: [N_test, n_elements] or None
            test_y_dict: { 'y0': [N_test], ... } or None
        """
        # Ensemble Inputs: Compositions Only
        n_props = len(self.method_list)
        comps_np = self.normalized_x[:, n_props:]
        
        # Ensemble Targets: 
        # If Landscape mode: Gibbs only (Skip HPP at index 0)
        # If Standard mode: All properties
        
        target_indices = []
        
        if self.config.predict_landscape:
            # Skip HPP (idx 0), take remaining
            target_indices = list(range(1, n_props))
        else:
            target_indices = list(range(n_props))
            
        # Create Tensors
        inputs = torch.from_numpy(comps_np).float()
        targets = {}
        
        for i, prop_idx in enumerate(target_indices):
            targets[f'y{i}'] = torch.from_numpy(self.normalized_x[:, prop_idx]).float()
            
        # Split - always return 6 values for consistency
        inputs_train = inputs
        targets_train = targets
        inputs_val = None
        targets_val = None
        inputs_test = None
        targets_test = None
        
        if self.train_indices is not None:
            inputs_train = inputs[self.train_indices]
            targets_train = {k: v[self.train_indices] for k, v in targets.items()}
            
            if self.val_indices is not None and len(self.val_indices) > 0:
                inputs_val = inputs[self.val_indices]
                targets_val = {k: v[self.val_indices] for k, v in targets.items()}
            
            if self.test_indices is not None and len(self.test_indices) > 0:
                inputs_test = inputs[self.test_indices]
                targets_test = {k: v[self.test_indices] for k, v in targets.items()}
        
        return inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test


    def get_inverse_transform_fn(self, group=None):
        """
        Returns a function that can denormalize data.
        """
        def inverse_transform(value, prop_name, is_uncertainty=False):
            if prop_name not in self.norm_stats and group is None:
                return value
                
            # If finding specific prop fails, try group look up if provided (e.g. for generic 'gibbs')
            stats = self.norm_stats.get(prop_name)
            
            # If we requested a group transform (e.g. 'gibbs') and didn't find specific name
            if stats is None and group is not None:
                # Find any stat with that group
                for k, v in self.norm_stats.items():
                    if v.get('group') == group:
                        stats = v
                        break
            
            if stats is None:
                return value
                
            if stats['type'] == 'zscore':
                # value * std + mean
                # for uncertainty: value * std
                if is_uncertainty:
                    return value * stats['std']
                else:
                    return value * stats['std'] + stats['mean']
            elif stats['type'] == 'minmax':
                scale = stats['max'] - stats['min']
                if is_uncertainty:
                    return value * scale
                else:
                    return value * scale + stats['min']
            return value
            
        return inverse_transform

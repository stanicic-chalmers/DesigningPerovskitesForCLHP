"""
Visualization functions for active learning pipeline.
Handles latent space visualization with t-SNE and UMAP, and evolution plots.
"""

import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def visualize_latent_space(wae_model, dataset, property_values, cycle, log_path,
                          property_name, device, logger):
    """
    Visualize the latent space using t-SNE and UMAP, colored by property values.

    Parameters
    ----------
    wae_model : torch.nn.Module
        Trained WAE model
    dataset : torch.utils.data.Dataset
        Dataset containing the training data
    property_values : torch.Tensor or np.ndarray
        Property values for coloring the plots
    cycle : int
        Current cycle number
    log_path : str
        Path to save visualization plots
    property_name : str
        Name of the property being visualized
    device : torch.device
        Device for computation
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary containing 'latent_reps' and 'property_values' as numpy arrays
    """
    # Extract latent representations
    wae_model.eval()
    latent_reps = []

    with torch.no_grad():
        for data in dataset:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            # Add batch dimension if it's a single sample
            if data.dim() == 1:
                data = data.unsqueeze(0)
            latent = wae_model._encode(data)
            latent_reps.append(latent.cpu().numpy())

    latent_reps = np.concatenate(latent_reps, axis=0)

    # Convert property values to numpy
    if torch.is_tensor(property_values):
        property_values = property_values.cpu().numpy()
    else:
        property_values = np.array(property_values)

    # Ensure shapes match
    if len(property_values) != len(latent_reps):
        logger.warning(f'Property values length ({len(property_values)}) != '
                      f'latent reps length ({len(latent_reps)}). Truncating to match.')
        min_len = min(len(property_values), len(latent_reps))
        property_values = property_values[:min_len]
        latent_reps = latent_reps[:min_len]

    logger.info(f'Extracted {len(latent_reps)} latent representations with dim={latent_reps.shape[1]}')

    # Create visualization directory
    viz_dir = os.path.join(log_path, 'latent_visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # t-SNE visualization
    logger.info('Computing t-SNE embedding...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_reps) - 1))
    tsne_result = tsne.fit_transform(latent_reps)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                         c=property_values, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label=property_name)
    plt.title(f't-SNE Visualization - Cycle {cycle}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    tsne_path = os.path.join(viz_dir, f'tsne_cycle_{cycle}.png')
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    logger.info(f't-SNE plot saved to {tsne_path}')

    # UMAP visualization (if available)
    if UMAP_AVAILABLE:
        logger.info('Computing UMAP embedding...')
        reducer = umap.UMAP(n_components=2, random_state=42,
                           n_neighbors=min(15, len(latent_reps) - 1))
        umap_result = reducer.fit_transform(latent_reps)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1],
                             c=property_values, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label=property_name)
        plt.title(f'UMAP Visualization - Cycle {cycle}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        umap_path = os.path.join(viz_dir, f'umap_cycle_{cycle}.png')
        plt.savefig(umap_path, dpi=150)
        plt.close()
        logger.info(f'UMAP plot saved to {umap_path}')
    else:
        logger.warning('UMAP not available. Install with: pip install umap-learn')

    return {
        'latent_reps': latent_reps,
        'property_values': property_values
    }


def analyze_latent_structure(latent_reps, property_values, logger, cycle, log_path):
    """
    Analyze the structure of the latent space by computing distance correlations.

    Measures how well the latent space preserves property relationships by
    computing the correlation between pairwise distances in latent space
    and pairwise distances in property space.

    Parameters
    ----------
    latent_reps : np.ndarray
        Latent space representations, shape (n_samples, latent_dim)
    property_values : np.ndarray
        Property values, shape (n_samples,)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary containing distance correlation statistics
    """
    logger.info('Analyzing latent space structure...')

    # Compute pairwise distances in latent space
    latent_distances = pdist(latent_reps, metric='euclidean')

    # Compute pairwise distances in property space (1D)
    property_values_2d = property_values.reshape(-1, 1)
    property_distances = pdist(property_values_2d, metric='euclidean')

    # Compute Pearson correlation between distance matrices
    correlation, p_value = pearsonr(latent_distances, property_distances)

    # Compute statistics
    mean_latent_dist = np.mean(latent_distances)
    std_latent_dist = np.std(latent_distances)
    mean_property_dist = np.mean(property_distances)
    std_property_dist = np.std(property_distances)

    logger.info(f'Distance correlation: {correlation:.4f} (p={p_value:.4e})')
    logger.info(f'Mean latent distance: {mean_latent_dist:.4f} ± {std_latent_dist:.4f}')
    logger.info(f'Mean property distance: {mean_property_dist:.4f} ± {std_property_dist:.4f}')

    structure_analysis = {
        'distance_correlation': float(correlation),
        'p_value': float(p_value),
        'mean_latent_distance': float(mean_latent_dist),
        'std_latent_distance': float(std_latent_dist),
        'mean_property_distance': float(mean_property_dist),
        'std_property_distance': float(std_property_dist)
    }    

    structure_row = pd.DataFrame([{
        'cycle': cycle,
        'distance_correlation': structure_analysis['distance_correlation'],
        'p_value': structure_analysis['p_value'],
        'mean_latent_distance': structure_analysis['mean_latent_distance'],
        'std_latent_distance': structure_analysis['std_latent_distance'],
        'mean_property_distance': structure_analysis['mean_property_distance'],
        'std_property_distance': structure_analysis['std_property_distance'],
    }])

    structure_file = f'{log_path}/latent_structure_analysis.csv'
    if not os.path.exists(structure_file):
        structure_row.to_csv(structure_file, index=False)
    else:
        structure_row.to_csv(structure_file, mode='a', header=False, index=False)


def plot_latent_evolution(log_path, property_name, logger):
    """
    Create a plot showing the evolution of latent space structure across cycles.

    Reads the latent structure analysis CSV and creates visualizations of how
    the latent space quality evolves during active learning.

    Parameters
    ----------
    log_path : str
        Path to log directory containing latent_structure_analysis.csv
    property_name : str
        Name of the property being tracked
    logger : logging.Logger
        Logger instance
    """
    import pandas as pd

    structure_file = os.path.join(log_path, 'latent_structure_analysis.csv')

    if not os.path.exists(structure_file):
        logger.warning(f'Structure analysis file not found: {structure_file}')
        return

    # Load the data
    df = pd.read_csv(structure_file)
    logger.info(f'Loaded structure analysis for {len(df)} cycles')

    if len(df) == 0:
        logger.warning('No data in structure analysis file')
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distance correlation over cycles
    ax = axes[0, 0]
    ax.plot(df['cycle'], df['distance_correlation'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Distance Correlation')
    ax.set_title('Latent-Property Distance Correlation')
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean distances over cycles
    ax = axes[0, 1]
    ax.plot(df['cycle'], df['mean_latent_distance'], 'o-', label='Latent space',
           linewidth=2, markersize=6)
    ax.plot(df['cycle'], df['mean_property_distance'], 's-', label='Property space',
           linewidth=2, markersize=6)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Mean Pairwise Distance')
    ax.set_title('Mean Distances Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Latent distance distribution
    ax = axes[1, 0]
    ax.errorbar(df['cycle'], df['mean_latent_distance'], yerr=df['std_latent_distance'],
               fmt='o-', linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Latent Distance')
    ax.set_title('Latent Distance (mean ± std)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Property distance distribution
    ax = axes[1, 1]
    ax.errorbar(df['cycle'], df['mean_property_distance'], yerr=df['std_property_distance'],
               fmt='s-', linewidth=2, markersize=6, capsize=4, color='orange')
    ax.set_xlabel('Cycle')
    ax.set_ylabel(f'{property_name} Distance')
    ax.set_title(f'{property_name} Distance (mean ± std)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Latent Space Evolution - {property_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    evolution_path = os.path.join(log_path, 'latent_space_evolution.png')
    plt.savefig(evolution_path, dpi=150)
    plt.close()

    logger.info(f'Evolution plot saved to {evolution_path}')

    # Also create individual t-SNE/UMAP comparison if multiple cycles exist
    viz_dir = os.path.join(log_path, 'latent_visualizations')
    if os.path.exists(viz_dir):
        tsne_files = sorted(glob(os.path.join(viz_dir, 'tsne_cycle_*.png')))
        if len(tsne_files) > 1:
            logger.info(f'Found {len(tsne_files)} t-SNE visualizations across cycles')

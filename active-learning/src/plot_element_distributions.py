#!/usr/bin/env python3
"""
Element Distribution in Latent Space Visualization
Usage:
    python plot_element_distributions.py --log-path /path/to/logs [OPTIONS]

Options:
    --cycle N         Specific cycle checkpoint to load (default: latest)
    --log-scale       Use logarithmic scale for element concentration colormaps
    --cmap CMAP       Colormap to use (default: viridis; recommended: inferno, magma, cividis)
    --include-zeros   Include points with zero concentration for each element
    --marker-size N   Size of scatter plot markers (default: 3.0)
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Polygon
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import ConvexHull

# Set up path resolution (same as interpolate_latent_space.py)
try:
    directory = Path(__file__).resolve().parent
except NameError:
    directory = Path().resolve()

sys.path.append(str(directory.parent))

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

metal_path = os.path.join(project_root, 'metal')
pyscripts_path = os.path.join(project_root, 'oxygen-carrier-discovery', 'heoc', 'pyscripts')

sys.path.append(metal_path)
sys.path.append(pyscripts_path)

# Local imports
from ANN_functions_new import WAE_modular, FeatureDataset
from al_functions import create_df, normalize_and_scale
from al_util import setup_logging
from data_manager import DataManager
from config import Config

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Element Distribution in Latent Space Visualization')
    parser.add_argument('--log-path', type=str, required=True,
                       help='Path to logs directory containing checkpoints and config.json')
    parser.add_argument('--cycle', type=int, default=None,
                       help='Specific cycle to load checkpoint from. If not provided, uses latest.')
    parser.add_argument('--hpp-percentile', type=int, default=8,
                       help='Percentile threshold for high/low HPC regions (default: 8)')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for element concentration colormaps')
    parser.add_argument('--cmap', type=str, default='magma',
                       help='Colormap for element concentrations (default: magma). '
                            'Recommended for log scale: cividis, inferno, magma')
    parser.add_argument('--include-zeros', action='store_true',
                       help='Include points with zero concentration for each element (default: False)')
    parser.add_argument('--marker-size', type=float, default=10.0,
                       help='Size of scatter plot markers (default: 10.0)')

    return parser.parse_args()


def load_latest_checkpoint(log_path, cycle=None):
    """Find and return the path to the latest (or specified) checkpoint."""
    checkpoint_dir = os.path.join(log_path, 'checkpoints')

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = list(Path(checkpoint_dir).glob('checkpoint_iter_*.pt'))

    if not checkpoints:
        return None

    if cycle is not None:
        target = Path(checkpoint_dir) / f'checkpoint_iter_{cycle}.pt'
        if target.exists():
            return str(target)
        else:
            print(f"Warning: Checkpoint for cycle {cycle} not found. Using latest.")

    def extract_iter(path):
        stem = path.stem
        return int(stem.split('_')[-1])

    checkpoints_sorted = sorted(checkpoints, key=extract_iter)
    return str(checkpoints_sorted[-1])


def load_wae_from_checkpoint(checkpoint_path, device, config):
    """Load WAE model from checkpoint using config parameters.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    device : torch.device
        Device to load model on
    config : Config
        Configuration object containing WAE parameters

    Returns
    -------
    tuple
        (wae_model, iteration)
    """
    params = {
        'latent_dim': getattr(config, 'latent_dim', 8),
        'use_regression': getattr(config, 'use_regression', True),
        'lattice_A_size': len(config.A_elements),
        'lattice_B_size': len(config.B_elements),
        'n_properties': 1,
        'decoder_scale': getattr(config, 'decoder_scale', 4.0),
        'regression_scale': getattr(config, 'regression_scale', 2.0),
        'use_full_dropout': getattr(config, 'full_dropout', True),
        'encoder_dropout': getattr(config, 'encoder_dropout', 0.1),
        'decoder_dropout': getattr(config, 'decoder_dropout', 0.1),
        'regression_dropout': getattr(config, 'regression_dropout', 0.1),
    }

    wae_model = WAE_modular(params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    wae_model.load_state_dict(checkpoint['model_state_dict'], strict=False) #OBS CHECK THIS
    wae_model.eval()

    iteration = checkpoint.get('iteration', -1)

    return wae_model, iteration


def get_latent_representations(wae_model, dataset, device):
    """Extract latent representations from dataset."""
    wae_model.eval()
    latent_reps = []

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        for data in dataloader:
            data = data.to(device)
            latent = wae_model._encode(data)
            latent_reps.append(latent.cpu().numpy())

    return np.concatenate(latent_reps, axis=0)


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.labelpad': 3,
        'axes.titlepad': 6,
    })


def add_hpp_region_overlay(ax, umap_coords, property_values, threshold_percentile=25, alpha=0.00):
    """
    Add transparent overlay showing high vs low HPC regions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add overlay to
    umap_coords : np.ndarray
        UMAP coordinates [N, 2]
    property_values : np.ndarray
        Property values (HPC) [N]
    threshold_percentile : int
        Percentile threshold for high/low (default: 25 = top/bottom quartile)
    alpha : float
        Transparency of overlay (default: 0.15)
    """
    # Define high and low thresholds
    low_threshold = np.percentile(property_values, threshold_percentile)
    high_threshold = np.percentile(property_values, 100 - threshold_percentile)

    # Get indices for high and low HPC samples
    high_hpp_mask = property_values >= high_threshold
    low_hpp_mask = property_values <= low_threshold

    high_hpp_points = umap_coords[high_hpp_mask]
    low_hpp_points = umap_coords[low_hpp_mask]

    # Draw convex hulls if we have enough points
    try:
        if len(high_hpp_points) >= 3:
            hull_high = ConvexHull(high_hpp_points)
            hull_points_high = high_hpp_points[hull_high.vertices]
            # Use rgba for edge to control alpha separately (more opaque edge)
            edge_alpha = min(alpha * 8, 0.7)  # Make edge more visible
            poly_high = Polygon(hull_points_high, facecolor='green',
                               edgecolor=(0, 0.392, 0, edge_alpha),  # darkgreen with custom alpha
                               alpha=alpha, linewidth=2.0,
                               label=f'High HPC (>{high_threshold:.1f}mmol H$_2$/g)')
            ax.add_patch(poly_high)
    except Exception:  
        pass

def create_standalone_element_plot(umap_coords, compositions, elem, elem_idx,
                                   output_dir, property_values=None, hpp_percentile=25,
                                   log_scale=False, cmap='viridis', exclude_zeros=False,
                                   marker_size=3.0):
    """
    Create a standalone plot for a single element.

    Parameters
    ----------
    umap_coords : np.ndarray
        UMAP coordinates [N, 2]
    compositions : np.ndarray
        Composition array [N, n_elements] excluding oxygen
    elem : str
        Element name
    elem_idx : int
        Index of element in compositions array
    output_dir : str
        Directory to save plot
    property_values : np.ndarray, optional
        Property values (e.g., HPC) for each sample [N]
    hpp_percentile : int
        Percentile threshold for high/low HPC regions (default: 25)
    log_scale : bool
        Use log scale for colormap (default: False)
    cmap : str
        Colormap name (default: 'viridis')
    exclude_zeros : bool
        Hide points with zero concentration (default: False)
    marker_size : float
        Size of scatter markers (default: 3.0)
    """
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

    # Prepare normalization
    elem_values = compositions[:, elem_idx]

    # Filter out zeros if requested
    if exclude_zeros:
        nonzero_mask = elem_values > 0
        if not nonzero_mask.any():
            # All zeros - create empty plot with message
            ax.text(0.5, 0.5, f'No samples with {elem}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9, color='gray')
            ax.set_title(f'{elem}', fontweight='normal', fontsize=9, pad=4)
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            output_path = os.path.join(output_dir, f'{elem}_concentration.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path

        # Filter data
        umap_coords_filtered = umap_coords[nonzero_mask]
        elem_values_filtered = elem_values[nonzero_mask]
    else:
        umap_coords_filtered = umap_coords
        elem_values_filtered = elem_values

    norm = None
    vmin = 0
    vmax = elem_values_filtered.max()

    if log_scale:
        # For log scale, exclude zeros and use a small positive minimum
        nonzero_mask_scale = elem_values_filtered > 0
        if nonzero_mask_scale.any():
            vmin = max(elem_values_filtered[nonzero_mask_scale].min() / 10, 1e-4)
            vmax = elem_values_filtered.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            # Fallback to linear if all zeros
            log_scale = False

    scatter_kwargs = {
        'c': elem_values_filtered,
        'cmap': cmap,
        's': marker_size,
        'alpha': 0.6,
        'edgecolors': 'none',
        'rasterized': True
    }

    if log_scale and norm is not None:
        scatter_kwargs['norm'] = norm
    else:
        scatter_kwargs['vmin'] = 0
        scatter_kwargs['vmax'] = vmax

    scatter = ax.scatter(
        umap_coords_filtered[:, 0],
        umap_coords_filtered[:, 1],
        **scatter_kwargs
    )

    #if property_values is not None:
        # add_hpp_region_overlay(ax, umap_coords, property_values,
         #                     threshold_percentile=hpp_percentile, alpha=0.08)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.046, aspect=25)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.set_label('Concentration', fontsize=7, labelpad=2)
    cbar.outline.set_linewidth(0.5)

    # Update title to show count if zeros are excluded
    if exclude_zeros:
        n_samples = len(elem_values_filtered)
        title = f'{elem} (n={n_samples})'
    else:
        title = f'{elem}'

    ax.set_title(title, fontweight='normal', fontsize=9, pad=4)
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)

    output_path = os.path.join(output_dir, f'{elem}_concentration.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_multi_panel_heatmaps(umap_coords, compositions, A_elements, B_elements,
                                output_dir, logger, property_values=None, hpp_percentile=25,
                                log_scale=False, cmap='viridis', exclude_zeros=False,
                                marker_size=3.0):
    """
    Create Option 1: Multi-panel heatmaps showing each element's concentration.

    Also creates standalone plots for each element in a subfolder.

    Parameters
    ----------
    umap_coords : np.ndarray
        UMAP coordinates [N, 2]
    compositions : np.ndarray
        Composition array [N, n_elements] excluding oxygen
    A_elements : list
        A-site element names
    B_elements : list
        B-site element names
    output_dir : str
        Directory to save plot
    logger : logging.Logger
        Logger instance
    property_values : np.ndarray, optional
        Property values (e.g., HPC) for each sample [N]
    hpp_percentile : int
        Percentile threshold for high/low HPC regions (default: 25)
    log_scale : bool
        Use log scale for colormap (default: False)
    cmap : str
        Colormap name (default: 'viridis')
    exclude_zeros : bool
        Hide points with zero concentration for each element (default: False)
    marker_size : float
        Size of scatter markers (default: 3.0)
    """
    setup_style()

    # Sort elements alphabetically within each site
    A_elements_sorted = sorted(A_elements)
    B_elements_sorted = sorted(B_elements)
    elements = A_elements_sorted + B_elements_sorted

    # Log the panel ordering
    logger.info(f'Multi-panel ordering: HPC (top-left), then A-site: {A_elements_sorted}, then B-site: {B_elements_sorted}')

    # Create subfolder for individual element plots
    element_plots_dir = os.path.join(output_dir, 'individual_elements')
    os.makedirs(element_plots_dir, exist_ok=True)
    logger.info(f'Creating individual element plots in: {element_plots_dir}')

    # 6 x 4 grid: 22 element panels + 1 HPC panel (23 of 24 slots used)
    n_cols = 4
    n_rows = 6

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.35)

    # Add HPC panel first (top-left position)
    if property_values is not None:
        # Standalone HPC plot
        fig_hpp, ax_hpp = plt.subplots(1, 1, figsize=(3.5, 3))
        scatter_hpp = ax_hpp.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=property_values,
            cmap='plasma',
            s=marker_size,
            alpha=0.6,
            vmin=0,
            vmax=(property_values).max(),
            edgecolors='none',
            rasterized=True
        )
        #add_hpp_region_overlay(ax_hpp, umap_coords, property_values,
        #                      threshold_percentile=hpp_percentile, alpha=0.1)
        ax_hpp.legend(loc='upper left', fontsize=6, framealpha=0.9,
                     edgecolor='none', handlelength=1, handletextpad=0.5)
        cbar_hpp = plt.colorbar(scatter_hpp, ax=ax_hpp, pad=0.01, fraction=0.046, aspect=25)
        cbar_hpp.ax.tick_params(labelsize=6, width=0.5, length=2)
        cbar_hpp.set_label('HPC (mmol H$_2$/g)', fontsize=7, labelpad=2)
        cbar_hpp.outline.set_linewidth(0.5)
        ax_hpp.set_title('HPC', fontweight='normal', fontsize=9, pad=4)
        ax_hpp.set_xlabel('UMAP 1', fontsize=8)
        ax_hpp.set_ylabel('UMAP 2', fontsize=8)
        ax_hpp.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)
        hpp_output_path = os.path.join(element_plots_dir, 'HPC_distribution.png')
        plt.savefig(hpp_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_hpp)
        logger.info(f'  Created HPC plot: {os.path.basename(hpp_output_path)}')

        # Multi-panel HPC (top-left)
        ax = fig.add_subplot(gs[0, 0])
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=property_values,
            cmap='plasma',
            s=marker_size * 0.67,  # Smaller for multi-panel layout
            alpha=0.6,
            vmin=0,
            vmax=(property_values).max(),
            edgecolors='none',
            rasterized=True
        )

        #add_hpp_region_overlay(ax, umap_coords, property_values,
        #                      threshold_percentile=hpp_percentile, alpha=0.1)
        ax.legend(loc='upper left', fontsize=5, framealpha=0.9,
                 edgecolor='none', handlelength=0.8, handletextpad=0.4)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.046, aspect=20)
        cbar.ax.tick_params(labelsize=5, width=0.5, length=1.5)
        cbar.set_label('mmol H$_2$/g', fontsize=6, labelpad=1)
        cbar.outline.set_linewidth(0.5)

        ax.set_title('HPC', fontweight='normal', fontsize=8, pad=3)
        ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot element concentrations (starting from position 1 since HPC is in position 0)
    for i, elem in enumerate(elements):
        # Get the original element index in the unsorted list
        if elem in A_elements:
            elem_idx = A_elements.index(elem)
        else:
            elem_idx = len(A_elements) + B_elements.index(elem)
        # Create standalone plot
        standalone_path = create_standalone_element_plot(
            umap_coords, compositions, elem, elem_idx, element_plots_dir,
            property_values, hpp_percentile, log_scale, cmap, exclude_zeros, marker_size
        )
        logger.info(f'  Created {elem} plot: {os.path.basename(standalone_path)}')

        # Continue with multi-panel plot (skip position 0 which is HPC)
        panel_idx = i + 1  # Offset by 1 since HPC is in position 0
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Prepare normalization for multi-panel plot
        elem_values = compositions[:, elem_idx]

        # Filter out zeros if requested
        if exclude_zeros:
            nonzero_mask = elem_values > 0
            if not nonzero_mask.any():
                # All zeros - show message
                ax.text(0.5, 0.5, f'No {elem}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=6, color='gray')
                ax.set_title(elem, fontweight='normal', fontsize=8, pad=3)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            umap_coords_filtered = umap_coords[nonzero_mask]
            elem_values_filtered = elem_values[nonzero_mask]
        else:
            umap_coords_filtered = umap_coords
            elem_values_filtered = elem_values

        norm = None
        vmin = 0
        vmax = elem_values_filtered.max()

        if log_scale:
            nonzero_mask_scale = elem_values_filtered > 0
            if nonzero_mask_scale.any():
                vmin = max(elem_values_filtered[nonzero_mask_scale].min() / 10, 1e-4)
                vmax = elem_values_filtered.max()
                norm = LogNorm(vmin=vmin, vmax=vmax)

        scatter_kwargs = {
            'c': elem_values_filtered,
            'cmap': cmap,
            's': marker_size * 0.67,  # Smaller for multi-panel layout
            'alpha': 0.6,
            'edgecolors': 'none',
            'rasterized': True
        }

        if log_scale and norm is not None:
            scatter_kwargs['norm'] = norm
        else:
            scatter_kwargs['vmin'] = 0
            scatter_kwargs['vmax'] = vmax

        scatter = ax.scatter(
            umap_coords_filtered[:, 0],
            umap_coords_filtered[:, 1],
            **scatter_kwargs
        )

        #if property_values is not None:
            #add_hpp_region_overlay(ax, umap_coords, property_values,
             #                     threshold_percentile=hpp_percentile, alpha=0.08)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.046, aspect=20)
        cbar.ax.tick_params(labelsize=5, width=0.5, length=1.5)
        cbar.outline.set_linewidth(0.5)

        # Update title to show count if zeros are excluded
        if exclude_zeros:
            n_samples = len(elem_values_filtered)
            title = f'{elem} ({n_samples})'
        else:
            title = elem

        ax.set_title(title, fontweight='normal', fontsize=8, pad=3)
        ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    output_path = os.path.join(output_dir, 'element_distribution_heatmaps.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    logger.info(f'Multi-panel heatmap saved to {output_path}')


def create_dominant_element_plots(umap_coords, compositions, A_elements, B_elements,
                                  output_dir, logger, property_values=None, hpp_percentile=25,
                                  marker_size=3.0):
    """
    Create Option 2: Categorical plots showing dominant element in each sublattice.

    Parameters
    ----------
    umap_coords : np.ndarray
        UMAP coordinates [N, 2]
    compositions : np.ndarray
        Composition array [N, n_elements] excluding oxygen
    A_elements : list
        A-site element names
    B_elements : list
        B-site element names
    output_dir : str
        Directory to save plot
    logger : logging.Logger
        Logger instance
    property_values : np.ndarray, optional
        Property values (e.g., HPC) for each sample [N]
    hpp_percentile : int
        Percentile threshold for high/low HPC regions (default: 25)
    marker_size : float
        Size of scatter markers (default: 3.0)
    """
    setup_style()

    n_A = len(A_elements)
    n_B = len(B_elements)

    # Split compositions by sublattice
    A_compositions = compositions[:, :n_A]
    B_compositions = compositions[:, n_A:n_A+n_B]

    # Find dominant element in each sublattice
    dominant_A = np.argmax(A_compositions, axis=1)
    dominant_B = np.argmax(B_compositions, axis=1)

    # Create discrete colormaps with exact number of colors needed
    tab10_colors = plt.cm.tab10.colors
    tab20_colors = plt.cm.tab20.colors
    cmap_A = ListedColormap(tab10_colors[:n_A])
    cmap_B = ListedColormap(tab20_colors[:n_B])

    fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(10.5, 3.5))

    # Plot A-site dominant elements
    scatter1 = ax1.scatter(
        umap_coords[:, 0], umap_coords[:, 1], c=dominant_A, cmap=cmap_A,
        s=marker_size, alpha=0.6, edgecolors='none', vmin=-0.5, vmax=n_A - 0.5, rasterized=True
    )

    #if property_values is not None:
        #add_hpp_region_overlay(ax1, umap_coords, property_values,
         #                     threshold_percentile=hpp_percentile, alpha=0.06)

    ax1.set_title('A-site', fontsize=9, fontweight='normal', pad=4)
    ax1.set_xlabel('UMAP 1', fontsize=8)
    ax1.set_ylabel('UMAP 2', fontsize=8)
    ax1.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)

    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.01, ticks=range(n_A), aspect=25)
    cbar1.set_ticklabels(A_elements)
    cbar1.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar1.outline.set_linewidth(0.5)

    scatter2 = ax2.scatter(
        umap_coords[:, 0], umap_coords[:, 1], c=dominant_B, cmap=cmap_B,
        s=marker_size, alpha=0.6, edgecolors='none', vmin=-0.5, vmax=n_B - 0.5, rasterized=True
    )

    #if property_values is not None:
        #add_hpp_region_overlay(ax2, umap_coords, property_values,
         #                     threshold_percentile=hpp_percentile, alpha=0.06)

    ax2.set_title('B-site', fontsize=9, fontweight='normal', pad=4)
    ax2.set_xlabel('UMAP 1', fontsize=8)
    ax2.set_ylabel('UMAP 2', fontsize=8)
    ax2.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)

    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.01, ticks=range(n_B), aspect=25)
    cbar2.set_ticklabels(B_elements)
    cbar2.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar2.outline.set_linewidth(0.5)

    if property_values is not None:
        scatter3 = ax3.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=property_values, cmap='plasma',
            s=marker_size, alpha=0.6, edgecolors='none', rasterized=True
        )

        #add_hpp_region_overlay(ax3, umap_coords, property_values,
         #                     threshold_percentile=hpp_percentile, alpha=0.08)
        ax3.legend(loc='upper left', fontsize=6, framealpha=0.9,
                  edgecolor='none', handlelength=1, handletextpad=0.5)

        ax3.set_title('HPC', fontsize=9, fontweight='normal', pad=4)
        ax3.set_xlabel('UMAP 1', fontsize=8)
        ax3.set_ylabel('UMAP 2', fontsize=8)
        ax3.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)

        cbar3 = plt.colorbar(scatter3, ax=ax3, pad=0.01, aspect=25)
        cbar3.set_label('HPC (mmol H$_2$/g)', fontsize=7, labelpad=2)
        cbar3.ax.tick_params(labelsize=6, width=0.5, length=2)
        cbar3.outline.set_linewidth(0.5)
    else:
        ax3.axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'dominant_element_distribution.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    logger.info(f'Dominant element plot saved to {output_path}')

    # Print statistics
    logger.info('\nDominant A-site element statistics:')
    for i, elem in enumerate(A_elements):
        count = np.sum(dominant_A == i)
        pct = 100 * count / len(dominant_A)
        logger.info(f'  {elem}: {count} ({pct:.1f}%)')

    logger.info('\nDominant B-site element statistics:')
    for i, elem in enumerate(B_elements):
        count = np.sum(dominant_B == i)
        pct = 100 * count / len(dominant_B)
        logger.info(f'  {elem}: {count} ({pct:.1f}%)')


def main():
    args = parse_args()

    # Normalize log path FIRST to handle trailing slashes and relative paths
    log_path_normalized = os.path.normpath(os.path.abspath(args.log_path))

    # Set up logging with normalized path
    output_dir = os.path.join(log_path_normalized, 'element_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Log path (raw): {args.log_path}')
    logger.info(f'Log path (normalized): {log_path_normalized}')
    logger.info(f'Current working directory: {os.getcwd()}')

    # Load configuration from log path
    config_file = os.path.join(log_path_normalized, 'config.json')
    logger.info(f'Looking for config file: {config_file}')

    if not os.path.exists(config_file):
        logger.error(f'Config file not found: {config_file}')
        logger.error(f'Directory exists: {os.path.exists(log_path_normalized)}')
        if os.path.exists(log_path_normalized):
            logger.error(f'Directory contents: {os.listdir(log_path_normalized)[:10]}')
        logger.error('Please ensure the log path contains a config.json file from an active learning run')
        return

    logger.info(f'Loading configuration from: {config_file}')

    # Load config JSON directly (Config class requires args which we don't have)
    import json
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    # Create a simple object to hold config values
    class SimpleConfig:
        pass

    config = SimpleConfig()

    # Load relevant values from config dict
    for key, value in config_dict.items():
        if not key.startswith('_'):  # Skip special keys
            setattr(config, key, value)

    logger.info(f'Loaded configuration with {len(config_dict)} parameters')

    # Use config values
    A_elements = config.A_elements
    B_elements = config.B_elements

    logger.info(f'A-site elements: {A_elements}')
    logger.info(f'B-site elements: {B_elements}')
    logger.info(f'Model type: {config.model}')

    # Find and load checkpoint
    logger.info('Loading checkpoint...')
    checkpoint_path = load_latest_checkpoint(log_path_normalized, args.cycle)

    if checkpoint_path is None:
        logger.error(f'No checkpoint found in {args.log_path}/checkpoints')
        return

    logger.info(f'Loading checkpoint from: {checkpoint_path}')
    wae_model, iteration = load_wae_from_checkpoint(checkpoint_path, device, config)
    logger.info(f'Loaded checkpoint from iteration {iteration}')

    # Load training database - go up one level from log path
    method_dir = os.path.dirname(log_path_normalized)
    db_name = os.path.join(method_dir, 'training.db')

    logger.info(f'Method directory: {method_dir}')
    logger.info(f'Looking for training database: {db_name}')

    if not os.path.exists(db_name):
        logger.error(f'Training database not found: {db_name}')
        return

    logger.info(f'Successfully loaded training database')

    # First, load raw data to get HPC values before normalization
    method_list = config.method_list if hasattr(config, 'method_list') else ['hpp']
    use_quadratic_fit_hpp = getattr(config, 'quadratic_fit_hpp', False)

    # Initialize DataManager
    data_manager = DataManager(config, logger)
    data_manager.load_data(config.method_list, db_name, config.elements)
    data_manager.normalize()
    
    # Get WAE data (filters out Gibbs energies if present, keeping only HPC + Compositions)
    # get_wae_data returns a DataLoader, we access the underlying dataset
    loader = data_manager.get_wae_data(batch_size=128)
    dataset = loader.dataset
    
    # Extract unnormalized compositions (already scaled by sublattice factors)
    compositions_unnormalized = data_manager.comps_x
    
    # Extract OCT values (first property)
    hpp_values = data_manager.props_x[:, 0]
    n_props = len(config.method_list)
    logger.info(f'Loaded {len(dataset)} samples from database')

    # Extract normalized data from dataset
    normalized_data = dataset.x

    logger.info(f'Data prepared with shape {normalized_data.shape}')
    logger.info(f'Compositions shape: {compositions_unnormalized.shape}')
    logger.info(f'Number of properties: {n_props}')

    # Extract latent representations
    logger.info('Extracting latent representations...')
    latent_reps = get_latent_representations(wae_model, dataset, device)
    logger.info(f'Extracted latent representations with shape {latent_reps.shape}')

    # Compute UMAP embedding
    if not UMAP_AVAILABLE:
        logger.error('UMAP is required for this script. Install with: pip install umap-learn')
        return

    logger.info('Computing UMAP embedding...')
    reducer = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=20)
    umap_coords = reducer.fit_transform(latent_reps)
    logger.info('UMAP embedding computed')

    # Use unnormalized compositions for visualization (natural [0,1] range after sublattice scaling)
    logger.info(f'Composition shape: {compositions_unnormalized.shape}')

    # Log HPC region overlay settings
    low_threshold = np.percentile(hpp_values, args.hpp_percentile)
    high_threshold = np.percentile(hpp_values, 100 - args.hpp_percentile)
    logger.info(f'HPC region overlay: {args.hpp_percentile}th percentile threshold')
    logger.info(f'  Low HPC region: <{low_threshold:.2f}mmol H$_2$/g')
    logger.info(f'  High HPC region: >{high_threshold:.2f}mmol H$_2$/g')

    # Log colormap settings
    if args.log_scale:
        logger.info(f'Using log scale for element concentration colormaps')
    logger.info(f'Using colormap: {args.cmap}')
    if not args.include_zeros:
        logger.info(f'Excluding zero-concentration points from element plots')
    logger.info(f'Marker size: {args.marker_size}')

    # Create visualizations
    logger.info('\nCreating Option 1: Multi-panel heatmaps...')
    create_multi_panel_heatmaps(umap_coords, compositions_unnormalized, A_elements, B_elements,
                               output_dir, logger, property_values=hpp_values,
                               hpp_percentile=args.hpp_percentile,
                               log_scale=args.log_scale, cmap=args.cmap,
                               exclude_zeros=not args.include_zeros, marker_size=args.marker_size)

    logger.info('\nCreating Option 2: Dominant element categorical plots...')
    create_dominant_element_plots(umap_coords, compositions_unnormalized, A_elements, B_elements,
                                 output_dir, logger, property_values=hpp_values,
                                 hpp_percentile=args.hpp_percentile, marker_size=args.marker_size)

    logger.info(f'\nAnalysis complete! Results saved to {output_dir}')


if __name__ == '__main__':
    main()

"""
Report Utilities for SEU Sensitivity Project

This module provides common utilities for generating reports, including:
- Loading and processing results from analysis runs
- Standard plotting styles and themes
- Common data transformations
- Helper functions for Quarto reports

Usage in Quarto reports:
    ```{python}
    import sys
    sys.path.insert(0, '..')
    from reports.report_utils import load_results, plot_recovery, SEU_COLORS
    ```
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Union, Any

# =============================================================================
# Color Scheme and Styling
# =============================================================================

# Primary color palette for SEU Sensitivity reports
SEU_COLORS = {
    'primary': '#2c5282',      # Deep blue
    'secondary': '#4a90a4',    # Teal
    'accent': '#ed8936',       # Orange
    'success': '#48bb78',      # Green
    'warning': '#ecc94b',      # Yellow
    'error': '#f56565',        # Red
    'text': '#2d3748',         # Dark gray
    'background': '#f7fafc',   # Light gray
    'grid': '#e2e8f0',         # Grid lines
}

# Alternative palette for multiple series
SEU_PALETTE = [
    '#2c5282',  # primary blue
    '#ed8936',  # accent orange
    '#48bb78',  # green
    '#9f7aea',  # purple
    '#f56565',  # red
    '#38b2ac',  # teal
    '#ecc94b',  # yellow
    '#667eea',  # indigo
]

def set_seu_style():
    """
    Set the default matplotlib style for SEU Sensitivity reports.
    
    This creates a consistent, professional look across all report figures.
    """
    plt.rcParams.update({
        # Figure
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        
        # Fonts
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Colors
        'axes.prop_cycle': plt.cycler('color', SEU_PALETTE),
        'axes.facecolor': 'white',
        'axes.edgecolor': SEU_COLORS['text'],
        'axes.labelcolor': SEU_COLORS['text'],
        'text.color': SEU_COLORS['text'],
        'xtick.color': SEU_COLORS['text'],
        'ytick.color': SEU_COLORS['text'],
        
        # Grid
        'axes.grid': True,
        'grid.color': SEU_COLORS['grid'],
        'grid.alpha': 0.5,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': SEU_COLORS['grid'],
        
        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# Data Loading Utilities
# =============================================================================

def get_project_root() -> Path:
    """Get the root directory of the SEU sensitivity project."""
    # Navigate up from reports/ to project root
    current = Path(__file__).resolve().parent
    while current.name != 'seu-sensitivity' and current.parent != current:
        current = current.parent
    return current


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_study_design(path: Optional[Union[str, Path]] = None, 
                      design_name: str = 'study') -> Dict:
    """
    Load a study design from the results directory.
    
    Parameters:
        path: Full path to design file, or None to use default
        design_name: Name of design file (without .json) if path is None
        
    Returns:
        Dictionary containing the study design
    """
    if path is None:
        root = get_project_root()
        path = root / 'results' / 'designs' / f'{design_name}.json'
    
    return load_json(path)


def load_recovery_results(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load parameter recovery results from a results directory.
    
    Parameters:
        results_dir: Path to the recovery results directory
        
    Returns:
        Dictionary with keys:
        - 'summary': DataFrame with recovery statistics
        - 'true_params': List of true parameter dictionaries
        - 'posterior_summaries': List of posterior summary dictionaries
        - 'config': Configuration dictionary (if available)
    """
    results_dir = Path(results_dir)
    
    results = {}
    
    # Load summary if available
    summary_path = results_dir / 'recovery_summary.csv'
    if summary_path.exists():
        results['summary'] = pd.read_csv(summary_path)
    
    # Load config if available
    config_path = results_dir / 'config.json'
    if config_path.exists():
        results['config'] = load_json(config_path)
    
    # Load individual iteration results
    true_params = []
    posterior_summaries = []
    
    for iter_dir in sorted(results_dir.glob('iteration_*')):
        true_path = iter_dir / 'true_parameters.json'
        posterior_path = iter_dir / 'posterior_summary.json'
        
        if true_path.exists():
            true_params.append(load_json(true_path))
        if posterior_path.exists():
            posterior_summaries.append(load_json(posterior_path))
    
    if true_params:
        results['true_params'] = true_params
    if posterior_summaries:
        results['posterior_summaries'] = posterior_summaries
    
    return results


def load_sbc_results(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load SBC results from a results directory.
    
    Parameters:
        results_dir: Path to the SBC results directory
        
    Returns:
        Dictionary with keys:
        - 'ranks': DataFrame with rank statistics
        - 'true_params': DataFrame with true parameter values
        - 'config': Configuration dictionary (if available)
    """
    results_dir = Path(results_dir)
    
    results = {}
    
    # Load ranks
    ranks_path = results_dir / 'ranks.csv'
    if ranks_path.exists():
        results['ranks'] = pd.read_csv(ranks_path)
    
    # Load true params
    true_path = results_dir / 'true_params.csv'
    if true_path.exists():
        results['true_params'] = pd.read_csv(true_path)
    
    # Load config
    config_path = results_dir / 'config_info.json'
    if config_path.exists():
        results['config'] = load_json(config_path)
    
    return results


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_recovery_scatter(true_values: np.ndarray, 
                         estimated_values: np.ndarray,
                         ci_lower: Optional[np.ndarray] = None,
                         ci_upper: Optional[np.ndarray] = None,
                         param_name: str = 'Parameter',
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a scatter plot comparing true vs estimated parameter values.
    
    Parameters:
        true_values: Array of true parameter values
        estimated_values: Array of estimated values (posterior means)
        ci_lower: Lower bounds of credible intervals (optional)
        ci_upper: Upper bounds of credible intervals (optional)
        param_name: Name of parameter for labels
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot identity line
    min_val = min(true_values.min(), estimated_values.min())
    max_val = max(true_values.max(), estimated_values.max())
    margin = 0.1 * (max_val - min_val)
    
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin],
            'k--', alpha=0.5, label='Identity', linewidth=1)
    
    # Plot error bars if provided
    if ci_lower is not None and ci_upper is not None:
        ax.errorbar(true_values, estimated_values,
                   yerr=[estimated_values - ci_lower, ci_upper - estimated_values],
                   fmt='o', color=SEU_COLORS['primary'], alpha=0.7,
                   capsize=3, markersize=6, label='Estimate ± 95% CI')
    else:
        ax.scatter(true_values, estimated_values, 
                  color=SEU_COLORS['primary'], alpha=0.7, s=50)
    
    ax.set_xlabel(f'True {param_name}')
    ax.set_ylabel(f'Estimated {param_name}')
    ax.set_title(f'Parameter Recovery: {param_name}')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    return ax


def plot_rank_histogram(ranks: np.ndarray,
                       n_bins: int = 20,
                       param_name: str = 'Parameter',
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a rank histogram for SBC diagnostics.
    
    A well-calibrated model should show uniform ranks.
    
    Parameters:
        ranks: Array of rank statistics
        n_bins: Number of histogram bins
        param_name: Name of parameter for labels
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot histogram
    ax.hist(ranks, bins=n_bins, density=True, alpha=0.7, 
            color=SEU_COLORS['primary'], edgecolor='white')
    
    # Expected uniform density
    expected = 1.0 / (ranks.max() - ranks.min() + 1) if ranks.max() != ranks.min() else 1.0
    ax.axhline(y=expected, color=SEU_COLORS['accent'], linestyle='--', 
               linewidth=2, label='Expected (uniform)')
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Density')
    ax.set_title(f'SBC Rank Histogram: {param_name}')
    ax.legend()
    
    return ax


def plot_ecdf(ranks: np.ndarray,
             param_name: str = 'Parameter',
             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create an ECDF plot for SBC diagnostics with tolerance bands.
    
    Parameters:
        ranks: Array of rank statistics (should be normalized to [0, 1])
        param_name: Name of parameter for labels
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Normalize ranks to [0, 1] if not already
    if ranks.max() > 1:
        ranks = ranks / (ranks.max() + 1)
    
    # Compute ECDF
    sorted_ranks = np.sort(ranks)
    ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    
    # Plot ECDF
    ax.step(sorted_ranks, ecdf, where='post', 
            color=SEU_COLORS['primary'], linewidth=2, label='Observed ECDF')
    
    # Plot diagonal (expected under uniformity)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Expected (uniform)')
    
    # Add tolerance bands (approximate 95% CI)
    n = len(ranks)
    alpha = 0.05
    z = 1.96  # For 95% CI
    
    x_grid = np.linspace(0, 1, 100)
    se = np.sqrt(x_grid * (1 - x_grid) / n)
    ax.fill_between(x_grid, x_grid - z*se, x_grid + z*se, 
                   alpha=0.2, color=SEU_COLORS['secondary'],
                   label='95% tolerance band')
    
    ax.set_xlabel('Rank (normalized)')
    ax.set_ylabel('ECDF')
    ax.set_title(f'SBC ECDF: {param_name}')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return ax


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_recovery_stats(true_values: np.ndarray,
                          estimated_values: np.ndarray,
                          ci_lower: np.ndarray,
                          ci_upper: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for parameter recovery.
    
    Parameters:
        true_values: Array of true parameter values
        estimated_values: Array of posterior means
        ci_lower: Lower bounds of 95% credible intervals
        ci_upper: Upper bounds of 95% credible intervals
        
    Returns:
        Dictionary with bias, RMSE, coverage, and mean CI width
    """
    errors = estimated_values - true_values
    
    # Bias: mean error
    bias = np.mean(errors)
    
    # RMSE: root mean squared error
    rmse = np.sqrt(np.mean(errors**2))
    
    # Coverage: proportion of true values within CIs
    covered = (true_values >= ci_lower) & (true_values <= ci_upper)
    coverage = np.mean(covered)
    
    # Mean CI width
    ci_width = np.mean(ci_upper - ci_lower)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'coverage': coverage,
        'ci_width': ci_width,
        'n': len(true_values)
    }


def format_recovery_table(stats_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Format recovery statistics as a nice DataFrame.
    
    Parameters:
        stats_dict: Dictionary mapping parameter names to their stats
        
    Returns:
        Formatted DataFrame
    """
    rows = []
    for param_name, stats in stats_dict.items():
        rows.append({
            'Parameter': param_name,
            'Bias': f"{stats['bias']:.3f}",
            'RMSE': f"{stats['rmse']:.3f}",
            'Coverage': f"{stats['coverage']:.1%}",
            'CI Width': f"{stats['ci_width']:.3f}",
            'N': int(stats['n'])
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Convenience Functions for Reports
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax values for array x.
    
    Parameters:
        x: Input array
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_odds_to_prob(log_odds: float) -> float:
    """Convert log-odds to probability."""
    odds = np.exp(log_odds)
    return odds / (1 + odds)


def prob_to_log_odds(p: float) -> float:
    """Convert probability to log-odds."""
    return np.log(p / (1 - p))


# =============================================================================
# Initialize style when module is imported
# =============================================================================

# Optionally set style on import (comment out if not desired)
# set_seu_style()

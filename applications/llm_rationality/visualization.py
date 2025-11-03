"""
Visualization utilities for LLM rationality benchmarks.

This module provides functions to visualize benchmark results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

def plot_rationality_comparison(results: Dict[str, Any], 
                              title: str = "LLM Rationality Comparison",
                              figsize: tuple = (10, 6),
                              sort_by_alpha: bool = True) -> plt.Figure:
    """
    Create a bar chart comparing the rationality scores (alpha) across LLM systems.
    
    Args:
        results: Dictionary of benchmark results by LLM
        title: Plot title
        figsize: Figure size
        sort_by_alpha: Whether to sort LLMs by alpha value
        
    Returns:
        Matplotlib figure
    """
    # Extract model names and alpha values
    models = list(results.keys())
    alpha_means = [results[model]["alpha"]["mean"] for model in models]
    alpha_lower = [results[model]["alpha"]["ci_95"][0] for model in models]
    alpha_upper = [results[model]["alpha"]["ci_95"][1] for model in models]
    
    # Calculate error bar sizes
    alpha_errors = [
        [m - l for m, l in zip(alpha_means, alpha_lower)],
        [u - m for m, u in zip(alpha_means, alpha_upper)]
    ]
    
    # Sort by alpha if requested
    if sort_by_alpha:
        sorted_indices = np.argsort(alpha_means)[::-1]  # Descending
        models = [models[i] for i in sorted_indices]
        alpha_means = [alpha_means[i] for i in sorted_indices]
        alpha_errors = [
            [alpha_errors[0][i] for i in sorted_indices],
            [alpha_errors[1][i] for i in sorted_indices]
        ]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with error bars
    sns.set_style("whitegrid")
    bars = ax.barh(models, alpha_means, xerr=alpha_errors, capsize=5)
    
    # Add data labels
    for bar, value in zip(bars, alpha_means):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height()/2,
            f'{value:.2f}',
            va='center'
        )
    
    # Format plot
    ax.set_xlabel('Rationality Score (α)')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set x-axis limits to start from 0
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([0, xmax * 1.1])
    
    plt.tight_layout()
    return fig

def plot_choice_analysis(results: Dict[str, Any], 
                        feature_data: Dict[str, Any],
                        problems: List[Dict[str, Any]],
                        figsize: tuple = (14, 10)) -> plt.Figure:
    """
    Plot analysis of where different LLMs made similar or different choices.
    
    Args:
        results: Dictionary of benchmark results by LLM
        feature_data: Feature data from TextFeatureGenerator
        problems: List of decision problems
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    models = list(results.keys())
    
    # Create a matrix of choices
    choices = np.array([results[model]["choices"] for model in models])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Choice agreement heatmap
    agreement_matrix = np.zeros((len(models), len(models)))
    for i in range(len(models)):
        for j in range(len(models)):
            agreement = np.mean(choices[i] == choices[j])
            agreement_matrix[i, j] = agreement
    
    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=models,
        yticklabels=models,
        ax=axes[0],
        cmap="YlGnBu",
        vmin=0,
        vmax=1
    )
    axes[0].set_title("Choice Agreement Between LLMs")
    
    # Plot 2: Choice distribution by problem
    # Find problems with most disagreement
    problem_agreement = []
    for p in range(feature_data["M"]):
        # Get choices for this problem across models
        problem_choices = choices[:, p]
        # Count how many models made the same choice
        unique_choices, counts = np.unique(problem_choices, return_counts=True)
        # Calculate max agreement
        max_agreement = np.max(counts) / len(models)
        problem_agreement.append(max_agreement)
    
    # Get top 10 problems with lowest agreement
    top_disagreement = np.argsort(problem_agreement)[:10]
    
    # Plot choice distribution for these problems
    x_positions = np.arange(len(top_disagreement))
    bar_width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_choices = choices[i, top_disagreement]
        axes[1].bar(
            x_positions + i * bar_width - 0.4 + bar_width/2, 
            model_choices,
            width=bar_width,
            label=model
        )
    
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([f"P{i}" for i in top_disagreement])
    axes[1].set_ylabel("Choice")
    axes[1].set_title("Choices for Problems with Most Disagreement")
    axes[1].legend()
    
    plt.tight_layout()
    return fig
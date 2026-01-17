"""
Visualization Module for Prompt Framing Study

Generates plots and visualizations for study results.
"""
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Plotting disabled.")


class StudyVisualizer:
    """
    Generate visualizations for prompt framing study results.
    """
    
    # Color palette for variants (colorblind-friendly)
    VARIANT_COLORS = {
        "minimal": "#E69F00",    # Orange
        "baseline": "#56B4E9",   # Sky blue
        "enhanced": "#009E73",   # Green
        "maximal": "#CC79A7"     # Pink
    }
    
    def __init__(
        self,
        results_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing study results
            figsize: Default figure size
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.figsize = figsize
        
        if HAS_MATPLOTLIB:
            plt.rcParams['figure.figsize'] = figsize
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    def plot_alpha_comparison(
        self,
        model_fits: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Plot alpha estimates across prompt variants.
        
        Args:
            model_fits: Dict mapping variant_name -> fit results with alpha statistics
            save_path: Path to save figure
            show: Whether to display the figure
            
        Returns:
            matplotlib Figure object if available
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for plotting")
            return None
        
        # Sort variants by level (assumed in name order)
        variant_order = ["minimal", "baseline", "enhanced", "maximal"]
        variants = [v for v in variant_order if v in model_fits]
        
        # Extract alpha statistics
        means = [model_fits[v].get("alpha_mean", 0) for v in variants]
        stds = [model_fits[v].get("alpha_std", 0) for v in variants]
        q05 = [model_fits[v].get("alpha_q05", 0) for v in variants]
        q95 = [model_fits[v].get("alpha_q95", 0) for v in variants]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(variants))
        colors = [self.VARIANT_COLORS.get(v, "#999999") for v in variants]
        
        # Plot means with error bars (90% CI)
        yerr = np.array([
            [means[i] - q05[i] for i in range(len(variants))],
            [q95[i] - means[i] for i in range(len(variants))]
        ])
        
        bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yerr[1, i] + 0.02,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Formatting
        ax.set_xlabel('Prompt Variant (Rationality Emphasis)', fontsize=12)
        ax.set_ylabel('Estimated α (Sensitivity Parameter)', fontsize=12)
        ax.set_title('SEU Sensitivity by Prompt Framing', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([v.capitalize() for v in variants])
        
        # Add horizontal line at α=1 (random choice)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, 
                   label='α=1 (random choice)')
        ax.legend(loc='upper right')
        
        # Set y-axis limits
        ax.set_ylim(0, max(q95) * 1.2)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved alpha comparison plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_alpha_posteriors(
        self,
        alpha_samples: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Plot posterior distributions of alpha for each variant.
        
        Args:
            alpha_samples: Dict mapping variant_name -> array of alpha samples
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        variant_order = ["minimal", "baseline", "enhanced", "maximal"]
        variants = [v for v in variant_order if v in alpha_samples]
        
        for variant in variants:
            samples = alpha_samples[variant]
            color = self.VARIANT_COLORS.get(variant, "#999999")
            
            # Use scipy for KDE if available, otherwise histogram
            try:
                from scipy import stats
                kde = stats.gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 200)
                ax.plot(x_range, kde(x_range), label=variant.capitalize(),
                       color=color, linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
            except ImportError:
                ax.hist(samples, bins=50, density=True, alpha=0.5, 
                       label=variant.capitalize(), color=color)
        
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1,
                   label='α=1 (random)')
        
        ax.set_xlabel('α (Sensitivity Parameter)', fontsize=12)
        ax.set_ylabel('Posterior Density', fontsize=12)
        ax.set_title('Posterior Distributions of α by Prompt Variant', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_embedding_space(
        self,
        embeddings: Dict[str, np.ndarray],
        variant_name: str,
        claims: List[Dict[str, str]],
        save_path: Optional[str] = None,
        show: bool = True,
        method: str = "tsne"
    ) -> Optional[Any]:
        """
        Visualize embedding space using t-SNE or PCA.
        
        Args:
            embeddings: Dict mapping claim_id -> embedding
            variant_name: Name of variant (for title)
            claims: List of claims with 'id' and 'description'
            save_path: Path to save figure
            show: Whether to display
            method: "tsne" or "pca"
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Stack embeddings
        claim_ids = list(embeddings.keys())
        X = np.vstack([embeddings[cid] for cid in claim_ids])
        
        # Reduce to 2D
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_2d = reducer.fit_transform(X)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c='steelblue', 
                            alpha=0.7, s=100, edgecolors='white')
        
        # Add claim IDs as labels
        for i, cid in enumerate(claim_ids):
            ax.annotate(cid, (X_2d[i, 0], X_2d[i, 1]), 
                       fontsize=8, alpha=0.7)
        
        method_name = "t-SNE" if method == "tsne" else "PCA"
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12)
        ax.set_title(f'Claim Embeddings ({variant_name.capitalize()} Variant)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_choice_distribution(
        self,
        choices: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Plot distribution of choices across variants.
        
        Args:
            choices: Dict mapping variant_name -> choice results
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        variant_order = ["minimal", "baseline", "enhanced", "maximal"]
        variants = [v for v in variant_order if v in choices]
        
        for i, variant in enumerate(variants[:4]):
            ax = axes[i]
            
            # Extract choice counts
            variant_choices = choices[variant].get("choices", [])
            if isinstance(variant_choices[0], dict):
                choice_values = [c["choice_1indexed"] for c in variant_choices]
            else:
                choice_values = [c["choice_1indexed"] for c in variant_choices[0]]
            
            # Count choices
            from collections import Counter
            counts = Counter(choice_values)
            
            x = sorted(counts.keys())
            y = [counts[k] for k in x]
            
            color = self.VARIANT_COLORS.get(variant, "#999999")
            ax.bar(x, y, color=color, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Choice (1-indexed)', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{variant.capitalize()} Variant', fontsize=12)
        
        plt.suptitle('Choice Distributions by Prompt Variant', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_robustness_heatmap(
        self,
        robustness_results: Dict[str, Any],
        metric: str = "alpha_mean",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Plot heatmap of alpha estimates across PCA dimensions and variants.
        
        Args:
            robustness_results: Results from RobustnessAnalyzer
            metric: Which metric to plot
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_MATPLOTLIB:
            return None
        
        pca_analysis = robustness_results.get("pca_sensitivity", {}).get("analysis", {})
        if not pca_analysis:
            logger.warning("No PCA sensitivity data to plot")
            return None
        
        # Build data matrix
        dims = sorted([int(k.split("_")[1]) for k in pca_analysis.keys()])
        variants = robustness_results.get("pca_sensitivity", {}).get("variants", [])
        
        # For now, use explained variance as the metric
        data = np.zeros((len(dims), len(variants)))
        
        for i, dim in enumerate(dims):
            dim_data = pca_analysis.get(f"dim_{dim}", {})
            for j, variant in enumerate(variants):
                var_data = dim_data.get(variant, {})
                data[i, j] = var_data.get("explained_variance", 0) or 0
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Use matplotlib imshow for heatmap
        im = ax.imshow(data, cmap='YlGnBu', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Explained Variance', fontsize=10)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(variants)))
        ax.set_yticks(np.arange(len(dims)))
        ax.set_xticklabels([v.capitalize() for v in variants])
        ax.set_yticklabels([f"D={d}" for d in dims])
        
        # Add text annotations
        for i in range(len(dims)):
            for j in range(len(variants)):
                text_color = 'white' if data[i, j] > data.max() * 0.5 else 'black'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                       color=text_color, fontsize=10)
        
        ax.set_xlabel('Prompt Variant', fontsize=12)
        ax.set_ylabel('PCA Dimensions', fontsize=12)
        ax.set_title('Explained Variance by Dimension and Variant', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_summary_report(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive visual report.
        
        Args:
            results: Full study results
            output_dir: Directory for saving figures
            
        Returns:
            Path to report directory
        """
        if output_dir is None:
            output_dir = self.results_dir / "figures" if self.results_dir else Path("figures")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_figures = []
        
        # Alpha comparison
        if "model_fits" in results:
            fig_path = output_dir / "alpha_comparison.png"
            self.plot_alpha_comparison(results["model_fits"], 
                                       save_path=str(fig_path), show=False)
            generated_figures.append(str(fig_path))
        
        # Choice distributions
        if "raw_choices" in results or self.results_dir:
            choices_file = self.results_dir / "raw_choices.json" if self.results_dir else None
            if choices_file and choices_file.exists():
                import json
                with open(choices_file, 'r') as f:
                    choices = json.load(f)
                fig_path = output_dir / "choice_distributions.png"
                self.plot_choice_distribution(choices, save_path=str(fig_path), show=False)
                generated_figures.append(str(fig_path))
        
        logger.info(f"Generated {len(generated_figures)} figures in {output_dir}")
        return str(output_dir)
    
    @staticmethod
    def set_publication_style():
        """Set matplotlib style for publication-quality figures."""
        if HAS_MATPLOTLIB:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.figsize': (8, 6),
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'axes.spines.top': False,
                'axes.spines.right': False
            })

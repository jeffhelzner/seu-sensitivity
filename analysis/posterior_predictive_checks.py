"""
Posterior Predictive Checks for SEU Decision Models

This module provides tools for computing and visualizing posterior predictive checks (PPCs)
to assess model fit. PPCs compare observed data to data simulated from the posterior 
predictive distribution using test statistics.

For each test statistic T:
- T_obs = T(y, theta) computed on observed data
- T_rep = T(y_rep, theta) computed on replicated data from posterior predictive

The posterior p-value is P(T_rep >= T_obs | y), estimated as mean(ppc_indicator) from Stan.
Values near 0.5 indicate good calibration; values near 0 or 1 indicate potential misfit.

Test statistics implemented:
1. Log-likelihood: Overall model fit
2. Modal accuracy: Classification performance (did agent choose highest-probability option?)
3. Sum of chosen probabilities: Probability calibration

Example usage:
    from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
    
    checker = PosteriorPredictiveChecker(fit, observed_data)
    p_values = checker.compute_p_values()
    print(checker.summary())
    checker.plot_discrepancy_scatter("ll", save_path="ppc_ll.png")
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


class PosteriorPredictiveChecker:
    """
    Compute and visualize posterior predictive checks for SEU decision models.
    
    This class extracts PPC statistics from a fitted Stan model and provides
    methods for computing posterior p-values, generating summaries, and 
    creating diagnostic plots.
    
    Attributes:
        fit: CmdStanMCMC fit object
        data: Dictionary of observed data passed to Stan
        model_type: "m_0" or "m_1" (detected automatically)
        p_values: Cached dictionary of posterior p-values
    """
    
    # Interpretation thresholds
    EXTREME_THRESHOLD = 0.05  # p < 0.05 or p > 0.95 is extreme
    WARNING_THRESHOLD = 0.10  # p < 0.10 or p > 0.90 warrants attention
    
    def __init__(self, fit, observed_data: Dict[str, Any]):
        """
        Initialize the posterior predictive checker.
        
        Args:
            fit: CmdStanMCMC object from model.sample()
            observed_data: Dictionary of data passed to Stan (must include M, y, etc.)
        """
        self.fit = fit
        self.data = observed_data
        self.p_values = None
        
        # Detect model type
        if 'N' in observed_data and 'z' in observed_data:
            self.model_type = "m_1"
        else:
            self.model_type = "m_0"
    
    def compute_p_values(self) -> Dict[str, float]:
        """
        Compute posterior p-values for all PPC statistics.
        
        The p-value is the proportion of posterior draws where T_rep >= T_obs.
        
        Returns:
            Dictionary mapping statistic names to p-values
        """
        if self.p_values is not None:
            return self.p_values
        
        self.p_values = {}
        
        if self.model_type == "m_0":
            self.p_values = self._compute_m0_p_values()
        else:
            self.p_values = self._compute_m1_p_values()
        
        return self.p_values
    
    def _compute_m0_p_values(self) -> Dict[str, float]:
        """Compute p-values for m_0 model."""
        p_values = {}
        
        # Extract PPC indicators
        ppc_ll = self.fit.stan_variable("ppc_ll")
        ppc_modal = self.fit.stan_variable("ppc_modal")
        ppc_prob = self.fit.stan_variable("ppc_prob")
        
        p_values["ll"] = float(np.mean(ppc_ll))
        p_values["modal"] = float(np.mean(ppc_modal))
        p_values["prob"] = float(np.mean(ppc_prob))
        
        # Also extract the raw statistics for plotting
        self._T_obs_ll = self.fit.stan_variable("T_obs_ll")
        self._T_rep_ll = self.fit.stan_variable("T_rep_ll")
        self._T_obs_modal = self.fit.stan_variable("T_obs_modal")
        self._T_rep_modal = self.fit.stan_variable("T_rep_modal")
        self._T_obs_prob = self.fit.stan_variable("T_obs_prob")
        self._T_rep_prob = self.fit.stan_variable("T_rep_prob")
        
        return p_values
    
    def _compute_m1_p_values(self) -> Dict[str, float]:
        """Compute p-values for m_1 model (separate for uncertain and risky)."""
        p_values = {}
        
        # Uncertain choices
        ppc_ll_uncertain = self.fit.stan_variable("ppc_ll_uncertain")
        ppc_modal_uncertain = self.fit.stan_variable("ppc_modal_uncertain")
        ppc_prob_uncertain = self.fit.stan_variable("ppc_prob_uncertain")
        
        p_values["ll_uncertain"] = float(np.mean(ppc_ll_uncertain))
        p_values["modal_uncertain"] = float(np.mean(ppc_modal_uncertain))
        p_values["prob_uncertain"] = float(np.mean(ppc_prob_uncertain))
        
        # Risky choices
        ppc_ll_risky = self.fit.stan_variable("ppc_ll_risky")
        ppc_modal_risky = self.fit.stan_variable("ppc_modal_risky")
        ppc_prob_risky = self.fit.stan_variable("ppc_prob_risky")
        
        p_values["ll_risky"] = float(np.mean(ppc_ll_risky))
        p_values["modal_risky"] = float(np.mean(ppc_modal_risky))
        p_values["prob_risky"] = float(np.mean(ppc_prob_risky))
        
        # Combined
        ppc_ll_combined = self.fit.stan_variable("ppc_ll_combined")
        p_values["ll_combined"] = float(np.mean(ppc_ll_combined))
        
        # Store raw statistics for uncertain choices
        self._T_obs_ll = self.fit.stan_variable("T_obs_ll_uncertain")
        self._T_rep_ll = self.fit.stan_variable("T_rep_ll_uncertain")
        self._T_obs_modal = self.fit.stan_variable("T_obs_modal_uncertain")
        self._T_rep_modal = self.fit.stan_variable("T_rep_modal_uncertain")
        self._T_obs_prob = self.fit.stan_variable("T_obs_prob_uncertain")
        self._T_rep_prob = self.fit.stan_variable("T_rep_prob_uncertain")
        
        return p_values
    
    def interpret_p_value(self, p: float) -> Tuple[str, str]:
        """
        Interpret a posterior p-value.
        
        Args:
            p: Posterior p-value in [0, 1]
            
        Returns:
            Tuple of (status_symbol, interpretation_text)
        """
        two_sided = 2 * min(p, 1 - p)
        
        if two_sided < self.EXTREME_THRESHOLD:
            if p < 0.5:
                return ("⚠️", f"Observed data fits BETTER than expected (p={p:.3f})")
            else:
                return ("⚠️", f"Observed data fits WORSE than expected (p={p:.3f})")
        elif two_sided < self.WARNING_THRESHOLD:
            return ("⚡", f"Marginal (p={p:.3f})")
        else:
            return ("✓", f"Good fit (p={p:.3f})")
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of PPC results.
        
        Returns:
            Multi-line string with interpretation of each statistic
        """
        p_values = self.compute_p_values()
        
        lines = [
            "=" * 50,
            "Posterior Predictive Check Summary",
            f"Model: {self.model_type}",
            "=" * 50,
            "",
            "Statistic              p-value    Status",
            "-" * 50
        ]
        
        for stat_name, p in p_values.items():
            symbol, interpretation = self.interpret_p_value(p)
            lines.append(f"{stat_name:<20}   {p:.3f}      {symbol}")
        
        lines.append("-" * 50)
        lines.append("")
        
        # Overall assessment
        extreme_count = sum(
            1 for p in p_values.values() 
            if 2 * min(p, 1 - p) < self.EXTREME_THRESHOLD
        )
        
        if extreme_count == 0:
            lines.append("Overall: Model appears well-calibrated.")
        elif extreme_count == 1:
            lines.append(f"Overall: {extreme_count} statistic shows potential misfit.")
        else:
            lines.append(f"Overall: {extreme_count} statistics show potential misfit.")
        
        lines.append("")
        lines.append("Interpretation guide:")
        lines.append("  p ≈ 0.5: Well-calibrated")
        lines.append("  p < 0.05: Observed data fits better than replicated (suspicious)")
        lines.append("  p > 0.95: Observed data fits worse than replicated (misspecified)")
        
        return "\n".join(lines)
    
    def summary_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame summary of PPC results.
        
        Returns:
            DataFrame with columns: statistic, p_value, two_sided_p, status
        """
        p_values = self.compute_p_values()
        
        rows = []
        for stat_name, p in p_values.items():
            two_sided = 2 * min(p, 1 - p)
            symbol, _ = self.interpret_p_value(p)
            rows.append({
                "statistic": stat_name,
                "p_value": p,
                "two_sided_p": two_sided,
                "status": symbol
            })
        
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export PPC results as a dictionary for JSON serialization.
        
        Returns:
            Dictionary with p-values and metadata
        """
        p_values = self.compute_p_values()
        
        result = {
            "model_type": self.model_type,
            "p_values": p_values,
            "n_problems": self.data.get("M", 0),
            "interpretation": {}
        }
        
        for stat_name, p in p_values.items():
            symbol, text = self.interpret_p_value(p)
            result["interpretation"][stat_name] = {
                "status": symbol,
                "text": text,
                "two_sided_p": 2 * min(p, 1 - p)
            }
        
        # Flag any issues
        result["has_extreme_p_values"] = any(
            2 * min(p, 1 - p) < self.EXTREME_THRESHOLD 
            for p in p_values.values()
        )
        
        return result
    
    def plot_discrepancy_scatter(
        self, 
        statistic: str = "ll",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create a scatter plot of T_obs vs T_rep for a given statistic.
        
        Points above the diagonal indicate T_rep > T_obs for that draw.
        The proportion above the line equals the posterior p-value.
        
        Args:
            statistic: One of "ll", "modal", "prob"
            save_path: Path to save figure (optional)
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return None
        
        # Get the appropriate statistics
        stat_map = {
            "ll": (self._T_obs_ll, self._T_rep_ll, "Log-Likelihood"),
            "modal": (self._T_obs_modal, self._T_rep_modal, "Modal Accuracy"),
            "prob": (self._T_obs_prob, self._T_rep_prob, "Sum of Chosen Probabilities")
        }
        
        if statistic not in stat_map:
            raise ValueError(f"Unknown statistic: {statistic}. Choose from {list(stat_map.keys())}")
        
        T_obs, T_rep, title = stat_map[statistic]
        
        # Compute p-value
        p_values = self.compute_p_values()
        if self.model_type == "m_0":
            p = p_values[statistic]
        else:
            p = p_values.get(f"{statistic}_uncertain", p_values.get(statistic, 0.5))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(T_obs, T_rep, alpha=0.3, s=10, c='steelblue')
        
        # Diagonal line
        all_vals = np.concatenate([T_obs, T_rep])
        lims = [np.min(all_vals), np.max(all_vals)]
        ax.plot(lims, lims, 'k--', linewidth=1, label='T_rep = T_obs')
        
        # Labels and title
        ax.set_xlabel(f'T_obs ({title})', fontsize=12)
        ax.set_ylabel(f'T_rep ({title})', fontsize=12)
        ax.set_title(f'Posterior Predictive Check: {title}\np-value = {p:.3f}', fontsize=14)
        
        # Add text annotation
        symbol, interpretation = self.interpret_p_value(p)
        ax.text(0.05, 0.95, f"{symbol} {interpretation}", 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_all_diagnostics(
        self,
        output_dir: str,
        show: bool = False
    ) -> List[str]:
        """
        Generate and save all diagnostic plots.
        
        Args:
            output_dir: Directory to save plots
            show: Whether to display plots
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for stat in ["ll", "modal", "prob"]:
            try:
                path = os.path.join(output_dir, f"ppc_{stat}.png")
                self.plot_discrepancy_scatter(stat, save_path=path, show=show)
                saved_files.append(path)
            except Exception as e:
                print(f"Warning: Could not generate plot for {stat}: {e}")
        
        return saved_files


def run_posterior_predictive_checks(
    fit,
    observed_data: Dict[str, Any],
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run all posterior predictive checks.
    
    Args:
        fit: CmdStanMCMC fit object
        observed_data: Data dictionary passed to Stan
        output_dir: Directory for saving results (optional)
        verbose: Whether to print summary
        
    Returns:
        Dictionary with p-values and diagnostics
    """
    checker = PosteriorPredictiveChecker(fit, observed_data)
    
    if verbose:
        print(checker.summary())
    
    results = checker.to_dict()
    
    if output_dir:
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON summary
        json_path = os.path.join(output_dir, "ppc_summary.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save plots
        checker.plot_all_diagnostics(output_dir, show=False)
        
        # Save text summary
        txt_path = os.path.join(output_dir, "ppc_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(checker.summary())
    
    return results

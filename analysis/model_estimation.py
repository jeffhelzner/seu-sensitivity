"""
Model Estimation for Epistemic Agents

This module provides tools for fitting Bayesian Decision Theory models to observed choice data.
It implements a workflow for:

1. Loading a Stan model and observed data
2. Running MCMC sampling to estimate model parameters
3. Saving posterior samples and summaries
4. Generating diagnostic plots and posterior predictive checks
5. Visualizing key parameter relationships and distributions

Typical use cases:
- Fitting the model to human or agent choice data
- Assessing model fit and parameter uncertainty
- Visualizing posterior distributions and diagnostics
- Integrating with other workflows (parameter recovery, prior predictive, SBC)

Example usage:
    from analysis.model_estimation import ModelEstimation
    estimation = ModelEstimation(data_path="path/to/data.json")
    fit = estimation.run()

Configurable parameters:
- model_path: Path to Stan model file
- data_path: Path to JSON data file
- output_dir: Directory for results
- n_mcmc_samples: Number of posterior samples per chain
- n_mcmc_chains: Number of MCMC chains

Outputs:
- Posterior samples and summary statistics
- Diagnostic plots (trace, energy, pair plots)
- Posterior predictive checks
- All results saved to output_dir

See scripts/run_model_estimation.py for CLI usage.
"""

import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


class ModelEstimation:
    """
    Bayesian Model Estimation Workflow

    This class fits a Bayesian Decision Theory model to observed choice data using Stan.
    It handles:
    - Model compilation and sampling
    - Posterior analysis and visualization
    - Saving results and diagnostics

    Parameters:
        model_path (str, optional): Path to Stan model file. Defaults to models/m_0.stan.
        data_path (str, required): Path to JSON data file containing observed choices.
        output_dir (str, optional): Directory to save results. Defaults to timestamped results/estimation/run_YYYYMMDD_HHMMSS.
        n_mcmc_samples (int): Number of posterior samples per chain.
        n_mcmc_chains (int): Number of MCMC chains.

    Usage:
        estimation = ModelEstimation(data_path="path/to/data.json")
        fit = estimation.run()
    """
    def __init__(
        self,
        model_path=None,
        data_path=None,
        output_dir=None,
        n_mcmc_samples=2000,
        n_mcmc_chains=4
    ):
        """
        Initialize the model estimation workflow.

        Args:
            model_path (str, optional): Path to Stan model file.
            data_path (str, required): Path to JSON data file.
            output_dir (str, optional): Directory for results.
            n_mcmc_samples (int): Number of posterior samples per chain.
            n_mcmc_chains (int): Number of MCMC chains.
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", 
                "m_0.stan"
            )
            
        self.model_path = model_path
        self.data_path = data_path
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_chains = n_mcmc_chains
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "estimation",
                f"run_{timestamp}"
            )
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
        # Compile the model
        self.model = CmdStanModel(stan_file=self.model_path)
        
    def run(self):
        """
        Run the model estimation workflow.

        Steps:
        1. Load data and compile Stan model
        2. Run MCMC sampling
        3. Save posterior samples and summary
        4. Generate diagnostics and plots
        5. Run posterior predictive checks

        Returns:
            CmdStanMCMC: Fitted Stan model object
        """
        # Check if data path is provided
        if self.data_path is None:
            raise ValueError("Data path must be provided for model estimation")
            
        # Load the data
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        # Make a copy of the data for the output directory
        with open(os.path.join(self.output_dir, "input_data.json"), 'w') as f:
            json.dump(data, f, indent=2)
            
        # Extract data dimensions
        M = data["M"]
        K = data["K"]
        D = data["D"]
        
        # Fit the model
        fit = self.model.sample(
            data=data,
            seed=12345,
            iter_sampling=self.n_mcmc_samples,
            iter_warmup=self.n_mcmc_samples // 2,
            chains=self.n_mcmc_chains
        )
        
        # Save the fit summary
        fit.save_csvfiles(self.output_dir)
        
        # Extract posterior samples
        self.posterior_samples = fit.draws_pd()
        
        # Create posterior summary
        posterior_summary = fit.summary()
        posterior_summary.to_csv(os.path.join(self.output_dir, "posterior_summary.csv"))
        
        # Run analyses and create plots
        self._analyze_posterior(M, K, D)
        
        # Save diagnostics
        diagnostics = fit.diagnose()
        with open(os.path.join(self.output_dir, "diagnostics.txt"), 'w') as f:
            f.write(diagnostics)

        # Try to import arviz for diagnostics plots
        try:
            import arviz as az
            idata = az.from_cmdstan(fit.stan_outputs)
            az.plot_trace(idata)
            plt.savefig(os.path.join(self.output_dir, "trace_plots.png"))
            plt.close()

            az.plot_energy(idata)
            plt.savefig(os.path.join(self.output_dir, "energy_plot.png"))
            plt.close()
        except ImportError as e:
            print("Warning: arviz could not be imported. Skipping arviz-based diagnostics plots.")
            print(f"ImportError: {e}")

        # Create pair plots for main parameters
        self._create_pair_plots(K, D)

        return fit
        
    def _analyze_posterior(self, M, K, D):
        """
        Analyze the posterior distribution and generate plots.

        - Posterior distribution of alpha (sensitivity)
        - Posterior boxplots for utilities (upsilon)
        - Heatmap of median beta coefficients
        - Posterior predictive checks for choices

        Args:
            M (int): Number of decision problems
            K (int): Number of consequences
            D (int): Number of dimensions
        """
        # Create posterior analysis directory
        posterior_dir = os.path.join(self.output_dir, "posterior")
        os.makedirs(posterior_dir, exist_ok=True)
        
        # 1. Analyze alpha (sensitivity parameter)
        post_alpha = self.posterior_samples["alpha"]
        
        plt.figure(figsize=(10, 6))
        plt.hist(post_alpha, bins=30, density=True, alpha=0.7)
        plt.axvline(
            np.median(post_alpha), 
            color='red', 
            linestyle='-', 
            label=f'Median: {np.median(post_alpha):.2f}'
        )
        plt.axvline(
            np.quantile(post_alpha, 0.025), 
            color='blue', 
            linestyle='--', 
            label=f'95% CI: [{np.quantile(post_alpha, 0.025):.2f}, {np.quantile(post_alpha, 0.975):.2f}]'
        )
        plt.axvline(np.quantile(post_alpha, 0.975), color='blue', linestyle='--')
        plt.title('Posterior Distribution of Alpha (Sensitivity)')
        plt.xlabel('Alpha')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(posterior_dir, 'alpha_posterior.png'))
        plt.close()
        
        # 2. Analyze utility parameters (upsilon)
        upsilon_cols = [f'upsilon[{k+1}]' for k in range(K)]
        if all(col in self.posterior_samples.columns for col in upsilon_cols):
            upsilon_data = {}
            for k in range(K):
                key = f'upsilon[{k+1}]'
                upsilon_data[key] = self.posterior_samples[key]
                
            upsilon_df = pd.DataFrame(upsilon_data)
            
            plt.figure(figsize=(10, 6))
            plt.boxplot([upsilon_df[col] for col in upsilon_df.columns], labels=upsilon_df.columns)
            plt.title('Posterior Distribution of Utilities')
            plt.xlabel('Consequence')
            plt.ylabel('Utility')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(posterior_dir, 'upsilon_posterior.png'))
            plt.close()
        
        # 3. Analyze beta parameters (subjective probability coefficients)
        # Create a heatmap of median beta values
        beta_medians = np.zeros((K, D))
        for k in range(K):
            for d in range(D):
                col = f'beta[{k+1},{d+1}]'
                if col in self.posterior_samples.columns:
                    beta_medians[k, d] = np.median(self.posterior_samples[col])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(beta_medians, aspect='auto', cmap="coolwarm")
        plt.colorbar(label="Median Value")
        plt.xticks(ticks=np.arange(D), labels=[f"Dim {d+1}" for d in range(D)])
        plt.yticks(ticks=np.arange(K), labels=[f"Cons {k+1}" for k in range(K)])
        for i in range(K):
            for j in range(D):
                plt.text(j, i, f"{beta_medians[i, j]:.2f}", ha='center', va='center', color='black')
        plt.title('Median Beta Coefficients')
        plt.tight_layout()
        plt.savefig(os.path.join(posterior_dir, 'beta_heatmap.png'))
        plt.close()
        
        # 4. Posterior predictive check
        y_pred_cols = [col for col in self.posterior_samples.columns if col.startswith("y_pred")]
        if y_pred_cols:
            y_pred_data = self.posterior_samples[y_pred_cols]
            
            # Count frequencies of predicted outcomes for each decision problem
            pred_freqs = {}
            for i in range(M):
                col = f'y_pred[{i+1}]'
                if col in y_pred_data.columns:
                    counts = y_pred_data[col].value_counts().sort_index()
                    pred_freqs[f'Problem {i+1}'] = counts
            
            # Create bar chart for each problem's predictions
            for problem, counts in pred_freqs.items():
                plt.figure(figsize=(8, 5))
                counts.plot(kind='bar')
                plt.title(f'Posterior Predictive Distribution for {problem}')
                plt.xlabel('Choice')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(posterior_dir, f'ppc_{problem.replace(" ", "_")}.png'))
                plt.close()
    
    def _create_pair_plots(self, K, D):
        """
        Create pair plots for key parameters.

        - Pair plots for alpha and delta
        - Pair plots for beta parameters (by consequence and dimension)

        Args:
            K (int): Number of consequences
            D (int): Number of dimensions
        """
        pairs_dir = os.path.join(self.output_dir, "pair_plots")
        os.makedirs(pairs_dir, exist_ok=True)
        
        # 1. Create pair plot for alpha and delta parameters
        delta_cols = [f'delta[{k+1}]' for k in range(K-1)]
        if all(col in self.posterior_samples.columns for col in delta_cols):
            pair_data = pd.DataFrame({'alpha': self.posterior_samples["alpha"]})
            for k in range(K-1):
                col = f'delta[{k+1}]'
                pair_data[col] = self.posterior_samples[col]
            
            pd.plotting.scatter_matrix(pair_data, figsize=(10, 10))
            plt.savefig(os.path.join(pairs_dir, 'alpha_delta_pairs.png'))
            plt.close()
        
        # 2. Create pair plots for selected beta parameters
        max_pairs = 5  # Maximum number of parameters to include in a single plot
        
        if K*D <= max_pairs:
            beta_cols = [f'beta[{k+1},{d+1}]' for k in range(K) for d in range(D)]
            if all(col in self.posterior_samples.columns for col in beta_cols):
                pair_data = pd.DataFrame()
                for col in beta_cols:
                    pair_data[col] = self.posterior_samples[col]
                
                pd.plotting.scatter_matrix(pair_data, figsize=(12, 12))
                plt.savefig(os.path.join(pairs_dir, 'beta_pairs.png'))
                plt.close()
        else:
            for k in range(K):
                beta_cols = [f'beta[{k+1},{d+1}]' for d in range(D)]
                if all(col in self.posterior_samples.columns for col in beta_cols):
                    pair_data = pd.DataFrame()
                    for col in beta_cols:
                        pair_data[col] = self.posterior_samples[col]
                    
                    pd.plotting.scatter_matrix(pair_data, figsize=(10, 10))
                    plt.savefig(os.path.join(pairs_dir, f'beta_consequence{k+1}_pairs.png'))
                    plt.close()
            
            for d in range(D):
                beta_cols = [f'beta[{k+1},{d+1}]' for k in range(K)]
                if all(col in self.posterior_samples.columns for col in beta_cols):
                    pair_data = pd.DataFrame()
                    for col in beta_cols:
                        pair_data[col] = self.posterior_samples[col]
                    
                    pd.plotting.scatter_matrix(pair_data, figsize=(10, 10))
                    plt.savefig(os.path.join(pairs_dir, f'beta_dimension{d+1}_pairs.png'))
                    plt.close()


if __name__ == "__main__":
    # Example usage (requires a data file)
    # estimation = ModelEstimation(data_path="/path/to/data.json")
    # estimation.run()
    pass
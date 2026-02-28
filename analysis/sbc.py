"""
Simulation-Based Calibration Analysis for Epistemic Agents

This module provides tools for performing simulation-based calibration (SBC) on Bayesian Decision Theory models.
SBC is a diagnostic method to assess whether a Bayesian inference algorithm can recover the true posterior
distribution, which is crucial for:

1. Model validation - ensuring the inference algorithm correctly samples from the posterior distribution
2. Prior-posterior consistency checking - verifying that parameters drawn from the prior can be recovered
3. Computational faithfulness testing - confirming that the Stan implementation correctly implements the model
4. Identifiability assessment - determining whether the model's parameters are identifiable given the data

The module works by:
1. Loading or generating a study design
2. For multiple iterations:
   a. Drawing "true" parameters from the prior
   b. Simulating data using these parameters
   c. Fitting the model to recover the parameters
   d. Computing ranks of true parameter values within posterior samples
3. Analyzing the distribution of ranks, which should be uniform if the model is calibrated
4. Producing diagnostic visualizations and statistical tests for uniformity

Examples:
    # Basic usage with defaults
    sbc = SimulationBasedCalibration()
    sbc.run()
    
    # Custom usage with specific parameters
    from utils.study_design import StudyDesign
    study = StudyDesign.load("path/to/design.json")
    sbc = SimulationBasedCalibration(
        sbc_model_path="models/custom_model_sbc.stan",
        study_design=study,
        output_dir="results/my_sbc_analysis",
        n_sbc_sims=100,
        n_mcmc_samples=1000,
        n_mcmc_chains=4,
        thin=10
    )
    ranks, true_params = sbc.run()
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
from cmdstanpy import CmdStanModel
import sys
from tqdm import tqdm
from scipy import stats

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design import StudyDesign
from utils import detect_model_name, get_model_scalar_parameters, has_risky_data

class SimulationBasedCalibration:
    """
    A class for performing simulation-based calibration (SBC) analysis on Bayesian Decision Theory models.
    
    This class handles:
    - Drawing parameters from the prior distribution
    - Simulating data based on these parameters
    - Performing Bayesian inference to recover the parameters
    - Computing rank statistics to assess calibration
    - Analyzing and visualizing rank distributions
    - Testing the uniformity of rank distributions
    - Saving outputs for later reference
    
    Attributes:
        sbc_model_path (str): Path to the Stan SBC model file
        study_design (StudyDesign): Study design object containing experiment structure
        output_dir (str): Directory where results will be saved
        n_sbc_sims (int): Number of SBC simulations to run
        n_mcmc_samples (int): Number of posterior samples to draw per chain
        n_mcmc_chains (int): Number of MCMC chains to run
        thin (int): Thinning factor for posterior samples
        sbc_model (CmdStanModel): Compiled Stan model for SBC
    """
    def __init__(
        self,
        sbc_model_path=None,
        study_design=None,
        output_dir=None,
        n_sbc_sims=100,
        n_mcmc_samples=1000,
        n_mcmc_chains=4,
        thin=1
    ):
        """
        Initialize the simulation-based calibration analysis.
        
        Parameters:
            sbc_model_path (str, optional): Path to the Stan SBC model file. 
                If None, uses default m_0_sbc.stan.
            study_design (StudyDesign, optional): Study design object. 
                If None, a new one will be generated.
            output_dir (str, optional): Directory to save results. 
                If None, creates a timestamped directory.
            n_sbc_sims (int): Number of SBC simulations to run.
            n_mcmc_samples (int): Number of posterior samples to draw per chain.
            n_mcmc_chains (int): Number of MCMC chains to run.
            thin (int): Thinning factor to reduce autocorrelation in posterior samples.
                A value of 1 means no thinning.
        """
        # Set default model path if not provided
        if sbc_model_path is None:
            sbc_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", 
                "m_0_sbc.stan"
            )
            
        self.sbc_model_path = sbc_model_path
        self.study_design = study_design
        self.n_sbc_sims = n_sbc_sims
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_chains = n_mcmc_chains
        self.thin = thin
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "sbc",
                f"run_{timestamp}"
            )
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
        # Compile the model
        self.sbc_model = CmdStanModel(stan_file=self.sbc_model_path)
        
    def run(self):
        """
        Run the simulation-based calibration analysis.
        
        This method:
        1. Ensures a study design exists or creates one
        2. For each SBC simulation:
           a. Draws parameters from the prior and simulates data
           b. Performs MCMC inference on the simulated data
           c. Computes rank statistics for each parameter
           d. Stores true parameters and ranks
        3. Analyzes rank distributions across all simulations
        
        Returns:
            tuple: (all_ranks, all_true_params) containing arrays of rank statistics
                  and true parameter values for each simulation
        """
        # Generate study design if not provided
        if self.study_design is None:
            self.study_design = StudyDesign()
            self.study_design.generate()
            
        # Save the study design
        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)
        
        # Get data dictionary for Stan
        data = self.study_design.get_data_dict()
        
        # Detect model name from the SBC model path
        model_name = detect_model_name(self.sbc_model_path)
        
        # Create parameter name list dynamically based on model
        K = self.study_design.K
        D = self.study_design.D
        
        # Start with scalar parameters for this model
        scalar_params = get_model_scalar_parameters(model_name)
        param_names = list(scalar_params)  # e.g. ["alpha"] or ["alpha", "omega"] or ["alpha", "kappa"]
        
        # Add beta parameter names
        for k in range(1, K+1):
            for d in range(1, D+1):
                param_names.append(f"beta[{k},{d}]")
                
        # Add delta parameter names
        for k in range(1, K):
            param_names.append(f"delta[{k}]")
        
        # Store all ranks and true parameters
        all_ranks = []
        all_true_params = []
        
        # Create output directory for simulations
        sims_dir = os.path.join(self.output_dir, "simulations")
        os.makedirs(sims_dir, exist_ok=True)
        
        # Save configuration information
        with open(os.path.join(self.output_dir, "config_info.json"), 'w') as f:
            config_info = {
                "n_sbc_sims": self.n_sbc_sims,
                "n_mcmc_samples": self.n_mcmc_samples,
                "n_mcmc_chains": self.n_mcmc_chains,
                "thin": self.thin,
                "effective_sample_size": self.n_mcmc_samples // self.thin
            }
            json.dump(config_info, f, indent=2)
        
        print(f"Running {self.n_sbc_sims} SBC simulations with thinning factor {self.thin}...")
        
        for i in tqdm(range(self.n_sbc_sims)):
            # Sample from the SBC model
            sbc_fit = self.sbc_model.sample(
                data=data,
                seed=123 + i,
                iter_sampling=self.n_mcmc_samples // self.n_mcmc_chains,
                iter_warmup=self.n_mcmc_samples // (2 * self.n_mcmc_chains),
                chains=self.n_mcmc_chains
            )
            
            # Extract true parameters
            true_params = sbc_fit.stan_variable("pars_")[0]
            
            # Calculate ranks of true parameters within posterior samples using thinning
            ranks = []
            for j, param in enumerate(param_names):
                # Extract binary ranks calculated in Stan with thinning
                binary_ranks = sbc_fit.stan_variable("ranks_")[::self.thin, j]
                
                # Calculate full rank: number of posterior samples LESS THAN true value
                # This is total samples MINUS number of samples GREATER THAN true value
                total_thinned_samples = len(binary_ranks)
                rank = total_thinned_samples - np.sum(binary_ranks)
                
                ranks.append(rank)
            
            # Save true parameters and ranks
            all_true_params.append(true_params)
            all_ranks.append(ranks)
            
            # Save simulation results if needed
            if i < 10:  # Save first 10 for detailed inspection
                sim_dir = os.path.join(sims_dir, f"sim_{i+1}")
                os.makedirs(sim_dir, exist_ok=True)
                
                # Save diagnostic plots and summary
                csv_file = os.path.join(sim_dir, "posterior_summary.csv")
                sbc_fit.summary().to_csv(csv_file)
                
                # Save true parameters
                with open(os.path.join(sim_dir, "true_parameters.json"), 'w') as f:
                    param_dict = {name: float(true_params[j]) for j, name in enumerate(param_names)}
                    json.dump(param_dict, f, indent=2)
        
        # Convert lists to numpy arrays
        all_ranks = np.array(all_ranks)
        all_true_params = np.array(all_true_params)
        
        # Analyze SBC results
        self._analyze_sbc(all_ranks, all_true_params, param_names)
        
        return all_ranks, all_true_params
        
    def _analyze_sbc(self, ranks, true_params, param_names):
        """
        Analyze the SBC results and create diagnostic visualizations.
        
        This method creates several diagnostic tools:
        1. Rank histograms for each parameter - should be approximately uniform if calibrated
        2. Empirical Cumulative Distribution Function (ECDF) plots - should follow the diagonal line
        3. Statistical tests for uniformity using Chi-square and Kolmogorov-Smirnov tests
        4. Summary tables and figures highlighting potential calibration issues
        
        The results help diagnose problems with the model, priors, or inference algorithm.
        
        Parameters:
            ranks (np.ndarray): Array of rank statistics for each parameter and simulation
            true_params (np.ndarray): Array of true parameter values for each simulation
            param_names (list): List of parameter names corresponding to the columns in ranks
            
        Returns:
            None: Results are saved to the output directory
        """
        # Create directory for SBC results
        sbc_dir = os.path.join(self.output_dir, "sbc_results")
        os.makedirs(sbc_dir, exist_ok=True)
        
        # Save raw rank and parameter data
        np.save(os.path.join(sbc_dir, "ranks.npy"), ranks)
        np.save(os.path.join(sbc_dir, "true_params.npy"), true_params)
        
        # Calculate expected number of ranks in each bin for uniform distribution
        n_sims = ranks.shape[0]
        # Use thinned samples count for bins
        total_thinned_samples = self.n_mcmc_samples // self.thin
        
        # Number of bins for histograms - adjusted for thinning
        n_bins = min(20, total_thinned_samples // 5)  # Rule of thumb: don't use too many bins
        bin_edges = np.linspace(0, total_thinned_samples, n_bins + 1)
        
        # Create rank histograms for each parameter
        plt.figure(figsize=(15, 10))
        num_params = len(param_names)
        ncols = 3
        nrows = (num_params + ncols - 1) // ncols
        
        # Store chi-square test results
        chi2_results = {}
        
        for i, param_name in enumerate(param_names):
            plt.subplot(nrows, ncols, i+1)
            
            # Get ranks for this parameter
            param_ranks = ranks[:, i]
            
            # Create histogram
            counts, edges, patches = plt.hist(param_ranks, bins=bin_edges, 
                                             alpha=0.7, color='skyblue', 
                                             edgecolor='black')
            
            # Calculate expected frequencies - ensuring the sum matches observed
            total_observed = np.sum(counts)
            expected_per_bin = total_observed / n_bins  # Use observed total
            expected = np.ones(n_bins) * expected_per_bin
            
            # Add horizontal line for expected counts
            plt.axhline(expected_per_bin, color='red', linestyle='--', 
                       label=f'Expected: {expected_per_bin:.1f}')
            
            # Perform chi-square goodness of fit test with corrected expectations
            try:
                chi2_stat, p_value = stats.chisquare(counts, expected)
            except ValueError:
                # If there's still an issue, force exact matching
                expected = expected * (total_observed / np.sum(expected))
                chi2_stat, p_value = stats.chisquare(counts, expected)
            
            # Store results
            chi2_results[param_name] = {
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
                "n_bins": n_bins,
                "n_simulations": n_sims
            }
            
            plt.title(f"{param_name}\np={p_value:.3f}")
            plt.xlabel("Rank")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sbc_dir, "rank_histograms.png"))
        plt.close()
        
        # Create ECDFs for each parameter
        plt.figure(figsize=(15, 10))
        
        for i, param_name in enumerate(param_names):
            plt.subplot(nrows, ncols, i+1)
            
            # Get ranks for this parameter and normalize to [0,1]
            param_ranks = ranks[:, i] / total_thinned_samples
            
            # Sort ranks
            sorted_ranks = np.sort(param_ranks)
            
            # ECDF
            ecdf = np.arange(1, n_sims + 1) / n_sims
            
            # Plot ECDF
            plt.step(sorted_ranks, ecdf, where='post', label='ECDF')
            
            # Add diagonal line representing uniform CDF
            plt.plot([0, 1], [0, 1], 'r--', label='Uniform')
            
            # Calculate Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(param_ranks, 'uniform')
            
            plt.title(f"{param_name} ECDF\nKS p={ks_pvalue:.3f}")
            plt.xlabel("Normalized Rank")
            plt.ylabel("Cumulative Probability")
            plt.grid(alpha=0.3)
            
            # Add KS statistic to chi2_results
            chi2_results[param_name]["ks_statistic"] = float(ks_stat)
            chi2_results[param_name]["ks_p_value"] = float(ks_pvalue)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sbc_dir, "rank_ecdfs.png"))
        plt.close()
        
        # Save chi-square test results
        with open(os.path.join(sbc_dir, "sbc_summary.json"), 'w') as f:
            json.dump(chi2_results, f, indent=2)
            
        # Create overall SBC diagnostic plot
        plt.figure(figsize=(12, 8))
        
        # Sort parameters by p-value for better visualization
        sorted_params = sorted(chi2_results.items(), 
                              key=lambda x: x[1]["p_value"])
        
        param_names_sorted = [x[0] for x in sorted_params]
        p_values = [x[1]["p_value"] for x in sorted_params]
        
        # Plot p-values
        plt.barh(param_names_sorted, p_values, color='skyblue')
        plt.axvline(0.05, color='red', linestyle='--', 
                   label='p=0.05 threshold')
        plt.xlabel('Chi-square p-value')
        plt.title('SBC Rank Test p-values by Parameter')
        plt.xlim(0, min(1.0, max(p_values) * 1.1))  # Max at 1.0
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sbc_dir, "sbc_pvalues.png"))
        plt.close()
        
        # Count parameters with potential issues
        n_significant = sum(1 for p in p_values if p < 0.05)
        with open(os.path.join(sbc_dir, "summary.txt"), 'w') as f:
            f.write(f"SBC Analysis Summary\n")
            f.write(f"-------------------\n\n")
            f.write(f"Total parameters tested: {len(param_names)}\n")
            f.write(f"Parameters with p < 0.05: {n_significant}\n")
            f.write(f"Thinning factor: {self.thin}\n\n")
            
            if n_significant > 0:
                f.write("Parameters with potential issues:\n")
                for param, p in zip(param_names_sorted, p_values):
                    if p < 0.05:
                        f.write(f"  - {param}: p = {p:.4f}\n")
            else:
                f.write("No significant calibration issues detected.\n")

if __name__ == "__main__":
    # Example usage
    sbc = SimulationBasedCalibration(n_sbc_sims=20)  # Use a small number for testing
    sbc.run()
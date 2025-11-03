"""
Parameter Recovery Analysis for Epistemic Agents

This module provides tools for performing parameter recovery analysis on Bayesian Decision Theory models.
It allows researchers to evaluate how well model parameters can be recovered from simulated data, which
is crucial for:

1. Model validation - ensuring that the model is identifiable and well-specified
2. Experimental design evaluation - determining if a study design provides sufficient information
   to accurately estimate parameters
3. Statistical power analysis - understanding the precision with which parameters can be estimated
   given a particular experimental design
4. Robustness assessment - evaluating how sensitive parameter estimation is to noise in the data

The module works by:
1. Loading or generating a study design
2. Simulating data multiple times using the generative model with different parameter values
3. Fitting the inference model to each simulated dataset to recover the parameters
4. Analyzing recovery performance through bias, RMSE, coverage, and credible interval width
5. Visualizing the relationship between true and estimated parameter values

Examples:
    # Basic usage with defaults
    recovery = ParameterRecovery()
    recovery.run()
    
    # Custom usage with specific parameters
    from utils.study_design import StudyDesign
    study = StudyDesign.load("path/to/design.json")
    recovery = ParameterRecovery(
        inference_model_path="models/custom_model.stan",
        sim_model_path="models/custom_sim_model.stan",
        study_design=study,
        output_dir="results/my_recovery_analysis",
        n_mcmc_samples=2000,
        n_mcmc_chains=4,
        n_iterations=20
    )
    true_params, posterior_summaries = recovery.run()
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import datetime
from tqdm import tqdm

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design import StudyDesign

class ParameterRecovery:
    """
    A class for performing parameter recovery analysis on Bayesian Decision Theory models.
    
    This class handles:
    - Simulating choice data from known parameter values
    - Fitting the inference model to recover these parameters
    - Evaluating recovery performance across multiple iterations
    - Analyzing and visualizing parameter recovery quality
    - Saving outputs for later reference
    
    Attributes:
        inference_model_path (str): Path to the Stan inference model file
        sim_model_path (str): Path to the Stan simulation model file
        study_design (StudyDesign): Study design object containing experiment structure
        output_dir (str): Directory where results will be saved
        n_mcmc_samples (int): Number of posterior samples to draw per chain
        n_mcmc_chains (int): Number of MCMC chains to run
        n_iterations (int): Number of simulation-recovery iterations to perform
        inference_model (CmdStanModel): Compiled Stan inference model
        sim_model (CmdStanModel): Compiled Stan simulation model
    """
    def __init__(
        self,
        inference_model_path=None,
        sim_model_path=None,
        study_design=None,
        output_dir=None,
        n_mcmc_samples=2000,
        n_mcmc_chains=4,
        n_iterations=20  # Number of simulation-recovery iterations
    ):
        """
        Initialize the parameter recovery analysis.
        
        Parameters:
            inference_model_path (str, optional): Path to the Stan inference model file. 
                If None, uses default m_0.stan.
            sim_model_path (str, optional): Path to the Stan simulation model file. 
                If None, uses default m_0_sim.stan.
            study_design (StudyDesign, optional): Study design object. 
                If None, a new one will be generated.
            output_dir (str, optional): Directory to save results. 
                If None, creates a timestamped directory.
            n_mcmc_samples (int): Number of posterior samples to draw per chain.
            n_mcmc_chains (int): Number of MCMC chains to run.
            n_iterations (int): Number of simulation-recovery iterations to perform.
        """
        # Set default model paths if not provided
        if inference_model_path is None:
            inference_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", 
                "m_0.stan"
            )
            
        if sim_model_path is None:
            sim_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", 
                "m_0_sim.stan"
            )
            
        self.inference_model_path = inference_model_path
        self.sim_model_path = sim_model_path
        self.study_design = study_design
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_chains = n_mcmc_chains
        self.n_iterations = n_iterations
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "parameter_recovery",
                f"run_{timestamp}"
            )
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
        # Compile the models
        self.inference_model = CmdStanModel(stan_file=self.inference_model_path)
        self.sim_model = CmdStanModel(stan_file=self.sim_model_path)
        
    def run(self):
        """
        Run the parameter recovery analysis over multiple iterations.
        
        This method:
        1. Ensures a study design exists or creates one
        2. For each iteration:
           a. Simulates choice data with randomly generated parameters
           b. Fits the inference model to the simulated data
           c. Stores true parameters and posterior summaries
        3. Analyzes parameter recovery across all iterations
        
        Returns:
            tuple: (all_true_params, all_posterior_summaries) containing lists of true parameter
                  values and corresponding posterior summaries for each iteration
        """
        # Generate study design if not provided
        if self.study_design is None:
            self.study_design = StudyDesign()
            self.study_design.generate()
            
        # Save the study design
        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)
        
        # Get data dictionary for Stan
        sim_data = self.study_design.get_data_dict()
        
        # Create structures to store results across iterations
        all_true_params = []
        all_posterior_summaries = []
        
        print(f"Running {self.n_iterations} iterations of parameter recovery...")
        
        for iteration in tqdm(range(self.n_iterations)):
            # Create iteration directory
            iter_dir = os.path.join(self.output_dir, f"iteration_{iteration+1}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Sample from simulation model to generate data with different parameter values
            sim_fit = self.sim_model.sample(
                data=sim_data,
                seed=12345 + iteration,  # Different seed for each iteration
                iter_sampling=1,         # Always just one sample
                iter_warmup=0,
                chains=1,
                fixed_param=True,
                adapt_engaged=False
            )
            
            # Extract the simulated data and true parameters
            sim_samples = sim_fit.draws_pd().iloc[0]
            
            # Extract choices
            y = np.array([sim_samples[f'y[{i+1}]'] for i in range(self.study_design.M)], dtype=int)
            
            # Create inference data by adding the choices to the study design
            inference_data = sim_data.copy()
            inference_data["y"] = y.tolist()
            
            # Extract true parameter values
            true_params = {
                "alpha": float(sim_samples["sim_alpha"]),
                "beta": [[float(sim_samples[f"sim_beta[{k+1},{d+1}]"]) 
                        for d in range(self.study_design.D)] 
                        for k in range(self.study_design.K)],
                "delta": [float(sim_samples[f"sim_delta[{k+1}]"]) 
                        for k in range(self.study_design.K-1)],
                "upsilon": [float(sim_samples[f"sim_upsilon[{k+1}]"]) 
                          for k in range(self.study_design.K)]
            }
            
            # Save the true parameters and data for this iteration
            with open(os.path.join(iter_dir, "true_parameters.json"), 'w') as f:
                json.dump(true_params, f, indent=2)
            
            # Fit the inference model to recover the parameters
            inference_fit = self.inference_model.sample(
                data=inference_data,
                seed=54321 + iteration,  # Different seed for each iteration
                iter_sampling=self.n_mcmc_samples,
                iter_warmup=self.n_mcmc_samples // 2,
                chains=self.n_mcmc_chains
            )
            
            # Get posterior summary statistics (no need to save all samples)
            summary = inference_fit.summary()
            summary.to_csv(os.path.join(iter_dir, "posterior_summary.csv"))
            
            # Save diagnostics for this iteration
            diagnostics = inference_fit.diagnose()
            with open(os.path.join(iter_dir, "diagnostics.txt"), 'w') as f:
                f.write(diagnostics)
            
            # Store results for aggregate analysis
            all_true_params.append(true_params)
            all_posterior_summaries.append(summary)
        
        # Save all true parameters
        with open(os.path.join(self.output_dir, "all_true_parameters.json"), 'w') as f:
            json.dump(all_true_params, f, indent=2)
        
        # Analyze recovery across iterations
        self._analyze_recovery(all_true_params, all_posterior_summaries)
        
        return all_true_params, all_posterior_summaries
        
    def _analyze_recovery(self, all_true_params, all_posterior_summaries):
        """
        Analyze parameter recovery across all iterations.
        
        This method evaluates recovery performance using several metrics:
        1. Bias - Average difference between true and estimated values
        2. RMSE - Root mean squared error between true and estimated values
        3. Coverage - Proportion of times true value falls within 90% credible interval
        4. CI Width - Average width of 90% credible intervals
        
        It creates visualizations and summary tables for all parameters, and saves
        the results to the output directory.
        
        Parameters:
            all_true_params (list): List of dictionaries containing true parameter values
            all_posterior_summaries (list): List of dataframes with posterior summaries
            
        Returns:
            dict: Recovery statistics for all parameters
        """
        # Create recovery plots directory
        recovery_dir = os.path.join(self.output_dir, "recovery_summary")
        os.makedirs(recovery_dir, exist_ok=True)
        
        # Create a summary dictionary for all parameters
        recovery_stats = {}
        
        # Analyze alpha recovery
        alpha_true = [params["alpha"] for params in all_true_params]
        alpha_mean = [summary.loc["alpha", "Mean"] for summary in all_posterior_summaries]
        alpha_lower = [summary.loc["alpha", "5%"] for summary in all_posterior_summaries]
        alpha_upper = [summary.loc["alpha", "95%"] for summary in all_posterior_summaries]
        
        # Calculate recovery metrics
        alpha_bias = np.mean(np.array(alpha_mean) - np.array(alpha_true))
        alpha_rmse = np.sqrt(np.mean((np.array(alpha_mean) - np.array(alpha_true))**2))
        alpha_coverage = np.mean([(t >= l) and (t <= u) for t, l, u in zip(alpha_true, alpha_lower, alpha_upper)])
        alpha_ci_width = np.mean(np.array(alpha_upper) - np.array(alpha_lower))
        
        recovery_stats["alpha"] = {
            "bias": float(alpha_bias),
            "rmse": float(alpha_rmse),
            "coverage": float(alpha_coverage),
            "ci_width": float(alpha_ci_width)
        }
        
        # Plot alpha recovery
        plt.figure(figsize=(12, 8))
        plt.scatter(alpha_true, alpha_mean, alpha=0.7)
        plt.plot([min(alpha_true), max(alpha_true)], [min(alpha_true), max(alpha_true)], 'r--')
        plt.xlabel("True Alpha")
        plt.ylabel("Estimated Alpha")
        plt.title(f"Alpha Recovery (Bias={alpha_bias:.3f}, RMSE={alpha_rmse:.3f}, Coverage={alpha_coverage:.1%})")
        plt.savefig(os.path.join(recovery_dir, "alpha_recovery.png"))
        plt.close()
        
        # Process beta parameters
        K, D = self.study_design.K, self.study_design.D
        beta_stats = {}
        
        for k in range(K):
            for d in range(D):
                param_name = f"beta[{k+1},{d+1}]"
                beta_true = [params["beta"][k][d] for params in all_true_params]
                beta_mean = [summary.loc[param_name, "Mean"] for summary in all_posterior_summaries]
                beta_lower = [summary.loc[param_name, "5%"] for summary in all_posterior_summaries]
                beta_upper = [summary.loc[param_name, "95%"] for summary in all_posterior_summaries]
                
                # Calculate recovery metrics
                beta_bias = np.mean(np.array(beta_mean) - np.array(beta_true))
                beta_rmse = np.sqrt(np.mean((np.array(beta_mean) - np.array(beta_true))**2))
                beta_coverage = np.mean([(t >= l) and (t <= u) for t, l, u in zip(beta_true, beta_lower, beta_upper)])
                beta_ci_width = np.mean(np.array(beta_upper) - np.array(beta_lower))
                
                beta_stats[f"beta_{k+1}_{d+1}"] = {
                    "bias": float(beta_bias),
                    "rmse": float(beta_rmse),
                    "coverage": float(beta_coverage),
                    "ci_width": float(beta_ci_width)
                }
        
        recovery_stats["beta"] = beta_stats
        
        # Process delta parameters
        delta_stats = {}
        
        for k in range(K - 1):
            param_name = f"delta[{k+1}]"
            delta_true = [params["delta"][k] for params in all_true_params]
            delta_mean = [summary.loc[param_name, "Mean"] for summary in all_posterior_summaries]
            delta_lower = [summary.loc[param_name, "5%"] for summary in all_posterior_summaries]
            delta_upper = [summary.loc[param_name, "95%"] for summary in all_posterior_summaries]
            
            # Calculate recovery metrics
            delta_bias = np.mean(np.array(delta_mean) - np.array(delta_true))
            delta_rmse = np.sqrt(np.mean((np.array(delta_mean) - np.array(delta_true))**2))
            delta_coverage = np.mean([(t >= l) and (t <= u) for t, l, u in zip(delta_true, delta_lower, delta_upper)])
            delta_ci_width = np.mean(np.array(delta_upper) - np.array(delta_lower))
            
            delta_stats[f"delta_{k+1}"] = {
                "bias": float(delta_bias),
                "rmse": float(delta_rmse),
                "coverage": float(delta_coverage),
                "ci_width": float(delta_ci_width)
            }
            
            # Plot delta recovery
            plt.figure(figsize=(10, 6))
            plt.scatter(delta_true, delta_mean, alpha=0.7)
            plt.plot([min(delta_true), max(delta_true)], [min(delta_true), max(delta_true)], 'r--')
            plt.xlabel(f"True Delta[{k+1}]")
            plt.ylabel(f"Estimated Delta[{k+1}]")
            plt.title(f"Delta[{k+1}] Recovery (Coverage={delta_coverage:.1%})")
            plt.savefig(os.path.join(recovery_dir, f"delta_{k+1}_recovery.png"))
            plt.close()
        
        recovery_stats["delta"] = delta_stats
        
        # Process upsilon parameters
        upsilon_stats = {}
        
        for k in range(K):
            param_name = f"upsilon[{k+1}]"
            if param_name in all_posterior_summaries[0].index:
                upsilon_true = [params["upsilon"][k] for params in all_true_params]
                upsilon_mean = [summary.loc[param_name, "Mean"] for summary in all_posterior_summaries]
                upsilon_lower = [summary.loc[param_name, "5%"] for summary in all_posterior_summaries]
                upsilon_upper = [summary.loc[param_name, "95%"] for summary in all_posterior_summaries]
                
                # Calculate recovery metrics
                upsilon_bias = np.mean(np.array(upsilon_mean) - np.array(upsilon_true))
                upsilon_rmse = np.sqrt(np.mean((np.array(upsilon_mean) - np.array(upsilon_true))**2))
                upsilon_coverage = np.mean([(t >= l) and (t <= u) for t, l, u in zip(upsilon_true, upsilon_lower, upsilon_upper)])
                upsilon_ci_width = np.mean(np.array(upsilon_upper) - np.array(upsilon_lower))
                
                upsilon_stats[f"upsilon_{k+1}"] = {
                    "bias": float(upsilon_bias),
                    "rmse": float(upsilon_rmse),
                    "coverage": float(upsilon_coverage),
                    "ci_width": float(upsilon_ci_width)
                }
        
        recovery_stats["upsilon"] = upsilon_stats
        
        # Save recovery stats
        with open(os.path.join(recovery_dir, "recovery_statistics.json"), 'w') as f:
            json.dump(recovery_stats, f, indent=2)
        
        # Generate summary table
        summary_table = {
            "Parameter": [],
            "Bias": [],
            "RMSE": [],
            "Coverage": [],
            "CI Width": []
        }
        
        # Add alpha
        summary_table["Parameter"].append("alpha")
        summary_table["Bias"].append(f"{alpha_bias:.3f}")
        summary_table["RMSE"].append(f"{alpha_rmse:.3f}")
        summary_table["Coverage"].append(f"{alpha_coverage:.1%}")
        summary_table["CI Width"].append(f"{alpha_ci_width:.3f}")
        
        # Add beta summaries (average)
        beta_biases = [stats["bias"] for stats in beta_stats.values()]
        beta_rmses = [stats["rmse"] for stats in beta_stats.values()]
        beta_coverages = [stats["coverage"] for stats in beta_stats.values()]
        beta_widths = [stats["ci_width"] for stats in beta_stats.values()]
        
        summary_table["Parameter"].append("beta (avg)")
        summary_table["Bias"].append(f"{np.mean(beta_biases):.3f}")
        summary_table["RMSE"].append(f"{np.mean(beta_rmses):.3f}")
        summary_table["Coverage"].append(f"{np.mean(beta_coverages):.1%}")
        summary_table["CI Width"].append(f"{np.mean(beta_widths):.3f}")
        
        # Add delta summaries (average)
        if delta_stats:
            delta_biases = [stats["bias"] for stats in delta_stats.values()]
            delta_rmses = [stats["rmse"] for stats in delta_stats.values()]
            delta_coverages = [stats["coverage"] for stats in delta_stats.values()]
            delta_widths = [stats["ci_width"] for stats in delta_stats.values()]
            
            summary_table["Parameter"].append("delta (avg)")
            summary_table["Bias"].append(f"{np.mean(delta_biases):.3f}")
            summary_table["RMSE"].append(f"{np.mean(delta_rmses):.3f}")
            summary_table["Coverage"].append(f"{np.mean(delta_coverages):.1%}")
            summary_table["CI Width"].append(f"{np.mean(delta_widths):.3f}")
        
        # Add upsilon summaries (average) if present
        if upsilon_stats:
            upsilon_biases = [stats["bias"] for stats in upsilon_stats.values()]
            upsilon_rmses = [stats["rmse"] for stats in upsilon_stats.values()]
            upsilon_coverages = [stats["coverage"] for stats in upsilon_stats.values()]
            upsilon_widths = [stats["ci_width"] for stats in upsilon_stats.values()]
            
            summary_table["Parameter"].append("upsilon (avg)")
            summary_table["Bias"].append(f"{np.mean(upsilon_biases):.3f}")
            summary_table["RMSE"].append(f"{np.mean(upsilon_rmses):.3f}")
            summary_table["Coverage"].append(f"{np.mean(upsilon_coverages):.1%}")
            summary_table["CI Width"].append(f"{np.mean(upsilon_widths):.3f}")
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_table)
        summary_df.to_csv(os.path.join(recovery_dir, "summary_table.csv"), index=False)
        
        # Print summary
        print("\nParameter Recovery Summary:")
        print(summary_df.to_string(index=False))
        
        return recovery_stats


if __name__ == "__main__":
    # Example usage
    recovery = ParameterRecovery(n_mcmc_samples=2000, n_iterations=10)
    recovery.run()
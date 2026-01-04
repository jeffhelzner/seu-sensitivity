"""
Sample Size Estimation Workflow for Epistemic Agents

This module provides tools for estimating the required sample size (number of decision problems, M)
to achieve a desired precision in estimating the sensitivity parameter alpha in Bayesian Decision Theory models.

Workflow:
1. For a grid of study designs (varying M, given K, D, R), simulate parameter recovery.
2. For each design, record the width of the (.05, .95) posterior interval for alpha.
3. Summarize and visualize how alpha precision varies with M.
4. Provide guidance on selecting M for a target interval width.

Configurable parameters:
- K, D, R: Model dimensions
- M_grid: List of candidate values for M
- n_iterations: Number of recovery simulations per design
- n_mcmc_samples, n_mcmc_chains: MCMC settings
- output_dir: Where to save results

Outputs:
- Table of interval widths for alpha vs. M
- Plots of interval width vs. M
- Recommendation for M given target precision

Example usage:
    from analysis.sample_size_estimation import SampleSizeEstimator
    estimator = SampleSizeEstimator(config_path="configs/sample_size_config.json")
    estimator.run()
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import datetime
from tqdm import tqdm

from utils.study_design import StudyDesign

class SampleSizeEstimator:
    """
    Estimate required sample size (M) for desired alpha precision.

    Parameters:
        config_path (str): Path to config file specifying grid and model parameters.
    """
    def __init__(self, config_path):
        """
        Initialize the estimator from a config file.

        Args:
            config_path (str): Path to JSON config file.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.K = config["K"]
        self.D = config["D"]
        self.R = config["R"]
        self.M_grid = config["M_grid"]
        self.base_design_path = config.get("base_design_path", None)
        self.n_iterations = config.get("n_iterations", 10)
        self.n_mcmc_samples = config.get("n_mcmc_samples", 1000)
        self.n_mcmc_chains = config.get("n_mcmc_chains", 4)
        self.output_dir = config.get("output_dir", "results/sample_size_estimation/custom_run")
        self.inference_model_path = config.get("inference_model_path", "models/m_0.stan")
        self.sim_model_path = config.get("sim_model_path", "models/m_0_sim.stan")
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Run sample size estimation over the grid of study designs.

        For each M in M_grid:
            - Use FIXED alternatives (w) from base design
            - Run parameter recovery n_iterations times
            - Record posterior interval width for alpha
            - Save coverage diagnostic plots
        Summarize and plot results.
        """
        # Load or generate the base design with fixed alternatives
        print("Loading base study design for fixed alternatives...")
        if self.base_design_path:
            # Load existing design
            base_design = StudyDesign.load(self.base_design_path)
            print(f"Loaded base design from: {self.base_design_path}")
        else:
            # Generate a new base design
            print("No base design path provided, generating new alternatives...")
            base_design = StudyDesign(
                M=1,  # Dummy value
                K=self.K, 
                D=self.D, 
                R=self.R,
                min_alts_per_problem=2,
                max_alts_per_problem=min(self.R, 6),
                feature_dist="uniform",
                feature_params={"low": -2, "high": 2},
                design_name="base_design"
            )
            base_design.generate()
        
        # Extract the fixed alternatives
        fixed_w = base_design.w
        
        # Save the base design with the fixed alternatives
        base_design_save_path = os.path.join(self.output_dir, "base_design.json")
        base_design.save(base_design_save_path, include_metadata=True, include_plots=False)
        print(f"Base design saved to: {base_design_save_path}")
        print(f"Using fixed alternatives (w) with {self.R} alternatives in {self.D} dimensions")
        
        results = []
        
        # Track all parameters for comparison across M
        all_recovery_data = {
            "M_values": [],
            "alpha_data": [],
            "beta_data": {f"beta_{k}_{d}": [] for k in range(1, self.K+1) for d in range(1, self.D+1)},
            "delta_data": {f"delta_{k}": [] for k in range(1, self.K)}
        }
        
        for M in self.M_grid:
            print(f"\nEvaluating M={M}...")
            # Generate study design with FIXED alternatives but varying M
            design = StudyDesign(
                M=M, 
                K=self.K, 
                D=self.D, 
                R=self.R,
                min_alts_per_problem=2,
                max_alts_per_problem=min(self.R, 6),
                feature_dist="uniform",
                feature_params={"low": -2, "high": 2},
                design_name=f"M_{M}_K_{self.K}_D_{self.D}_R_{self.R}"
            )
            
            # Override the alternatives with our fixed set
            design.w = fixed_w
            
            # Generate only the indicator matrix (I)
            design.I = design._generate_indicator_array()
            
            # Regenerate metadata with the fixed alternatives
            design.metadata = design._generate_metadata()
            
            # Create M-specific output directory
            m_dir = os.path.join(self.output_dir, f"M_{M}")
            os.makedirs(m_dir, exist_ok=True)
            
            design_path = os.path.join(m_dir, f"design_M{M}.json")
            design.save(design_path, include_metadata=True, include_plots=False)

            # Compile models
            inference_model = CmdStanModel(stan_file=self.inference_model_path)
            sim_model = CmdStanModel(stan_file=self.sim_model_path)

            # Storage for this M
            all_true_params = []
            all_posterior_summaries = []
            interval_widths = []
            
            for i in tqdm(range(self.n_iterations), desc=f"M={M}"):
                # Simulate data
                sim_fit = sim_model.sample(
                    data=design.get_data_dict(),
                    seed=12345 + i,
                    iter_sampling=1,
                    iter_warmup=0,
                    chains=1,
                    fixed_param=True,
                    adapt_engaged=False
                )
                sim_samples = sim_fit.draws_pd().iloc[0]
                y = [int(sim_samples[f'y[{m+1}]']) for m in range(M)]
                
                # Extract true parameters
                true_params = {
                    "alpha": float(sim_samples["alpha"]),
                    "beta": [[float(sim_samples[f"beta[{k+1},{d+1}]"]) 
                            for d in range(self.D)] 
                            for k in range(self.K)],
                    "delta": [float(sim_samples[f"delta[{k+1}]"]) 
                            for k in range(self.K-1)]
                }
                all_true_params.append(true_params)
                
                inference_data = design.get_data_dict().copy()
                inference_data["y"] = y

                # Fit inference model
                fit = inference_model.sample(
                    data=inference_data,
                    seed=54321 + i,
                    iter_sampling=self.n_mcmc_samples,
                    iter_warmup=self.n_mcmc_samples // 2,
                    chains=self.n_mcmc_chains
                )
                summary = fit.summary()
                all_posterior_summaries.append(summary)
                
                alpha_lower = summary.loc["alpha", "5%"]
                alpha_upper = summary.loc["alpha", "95%"]
                interval_widths.append(alpha_upper - alpha_lower)

            # Generate coverage plots for this M
            self._create_coverage_plots(M, all_true_params, all_posterior_summaries, m_dir)
            
            # Store data for cross-M comparison
            all_recovery_data["M_values"].append(M)
            self._extract_recovery_data(M, all_true_params, all_posterior_summaries, all_recovery_data)

            # Summarize for this M
            mean_width = float(np.mean(interval_widths))
            std_width = float(np.std(interval_widths))
            results.append({
                "M": M,
                "mean_interval_width": mean_width,
                "std_interval_width": std_width,
                "interval_widths": interval_widths
            })
            print(f"  Mean interval width for alpha: {mean_width:.3f} (std: {std_width:.3f})")

        # Save results
        results_df = pd.DataFrame([{
            "M": r["M"],
            "mean_interval_width": r["mean_interval_width"],
            "std_interval_width": r["std_interval_width"]
        } for r in results])
        results_df.to_csv(os.path.join(self.output_dir, "alpha_interval_widths.csv"), index=False)

        # Create cross-M comparison plots
        self._create_comparison_plots(all_recovery_data)

        # Plot alpha precision vs M
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            results_df["M"], results_df["mean_interval_width"],
            yerr=results_df["std_interval_width"], fmt='o-', capsize=5
        )
        plt.xlabel("Number of Decision Problems (M)")
        plt.ylabel("Mean 90% Posterior Interval Width for Alpha")
        plt.title("Alpha Precision vs. Sample Size (M)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "alpha_precision_vs_M.png"))
        plt.close()

        print("\nSample Size Estimation Complete.")
        print("Results saved to:", self.output_dir)
        print(results_df)
        return results_df
    
    def _create_coverage_plots(self, M, all_true_params, all_posterior_summaries, output_dir):
        """Create coverage interval plots for alpha, beta, and delta parameters."""
        
        # Alpha coverage plot
        alpha_true = [p["alpha"] for p in all_true_params]
        alpha_mean = [s.loc["alpha", "Mean"] for s in all_posterior_summaries]
        alpha_lower = [s.loc["alpha", "5%"] for s in all_posterior_summaries]
        alpha_upper = [s.loc["alpha", "95%"] for s in all_posterior_summaries]
        alpha_coverage = np.mean([(t >= l and t <= u) for t, l, u in zip(alpha_true, alpha_lower, alpha_upper)])
        
        plt.figure(figsize=(12, 6))
        for i in range(len(alpha_true)):
            color = 'green' if (alpha_true[i] >= alpha_lower[i] and alpha_true[i] <= alpha_upper[i]) else 'red'
            plt.plot([i, i], [alpha_lower[i], alpha_upper[i]], color=color, linewidth=2, alpha=0.6)
            plt.scatter(i, alpha_mean[i], color=color, s=30, zorder=3)
        plt.scatter(range(len(alpha_true)), alpha_true, color='black', s=50, marker='x', label='True Value', zorder=4)
        plt.xlabel("Iteration")
        plt.ylabel("Alpha Value")
        plt.title(f"M={M}: Alpha Coverage (Coverage = {alpha_coverage:.1%})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "alpha_coverage.png"))
        plt.close()
        
        # Beta coverage plots
        for k in range(self.K):
            for d in range(self.D):
                param_name = f"beta[{k+1},{d+1}]"
                beta_true = [p["beta"][k][d] for p in all_true_params]
                beta_mean = [s.loc[param_name, "Mean"] for s in all_posterior_summaries]
                beta_lower = [s.loc[param_name, "5%"] for s in all_posterior_summaries]
                beta_upper = [s.loc[param_name, "95%"] for s in all_posterior_summaries]
                beta_coverage = np.mean([(t >= l and t <= u) for t, l, u in zip(beta_true, beta_lower, beta_upper)])
                
                plt.figure(figsize=(12, 6))
                for i in range(len(beta_true)):
                    color = 'green' if (beta_true[i] >= beta_lower[i] and beta_true[i] <= beta_upper[i]) else 'red'
                    plt.plot([i, i], [beta_lower[i], beta_upper[i]], color=color, linewidth=2, alpha=0.6)
                    plt.scatter(i, beta_mean[i], color=color, s=30, zorder=3)
                plt.scatter(range(len(beta_true)), beta_true, color='black', s=50, marker='x', label='True Value', zorder=4)
                plt.xlabel("Iteration")
                plt.ylabel(f"Beta[{k+1},{d+1}] Value")
                plt.title(f"M={M}: Beta[{k+1},{d+1}] Coverage (Coverage = {beta_coverage:.1%})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"beta_{k+1}_{d+1}_coverage.png"))
                plt.close()
        
        # Delta coverage plots
        for k in range(self.K - 1):
            param_name = f"delta[{k+1}]"
            delta_true = [p["delta"][k] for p in all_true_params]
            delta_mean = [s.loc[param_name, "Mean"] for s in all_posterior_summaries]
            delta_lower = [s.loc[param_name, "5%"] for s in all_posterior_summaries]
            delta_upper = [s.loc[param_name, "95%"] for s in all_posterior_summaries]
            delta_coverage = np.mean([(t >= l and t <= u) for t, l, u in zip(delta_true, delta_lower, delta_upper)])
            
            plt.figure(figsize=(12, 6))
            for i in range(len(delta_true)):
                color = 'green' if (delta_true[i] >= delta_lower[i] and delta_true[i] <= delta_upper[i]) else 'red'
                plt.plot([i, i], [delta_lower[i], delta_upper[i]], color=color, linewidth=2, alpha=0.6)
                plt.scatter(i, delta_mean[i], color=color, s=30, zorder=3)
            plt.scatter(range(len(delta_true)), delta_true, color='black', s=50, marker='x', label='True Value', zorder=4)
            plt.xlabel("Iteration")
            plt.ylabel(f"Delta[{k+1}] Value")
            plt.title(f"M={M}: Delta[{k+1}] Coverage (Coverage = {delta_coverage:.1%})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"delta_{k+1}_coverage.png"))
            plt.close()
    
    def _extract_recovery_data(self, M, all_true_params, all_posterior_summaries, all_recovery_data):
        """Extract recovery statistics for comparison across M values."""
        
        # Alpha
        alpha_true = [p["alpha"] for p in all_true_params]
        alpha_mean = [s.loc["alpha", "Mean"] for s in all_posterior_summaries]
        alpha_lower = [s.loc["alpha", "5%"] for s in all_posterior_summaries]
        alpha_upper = [s.loc["alpha", "95%"] for s in all_posterior_summaries]
        
        # Calculate errors (posterior mean - true value)
        alpha_errors = np.array(alpha_mean) - np.array(alpha_true)
        
        all_recovery_data["alpha_data"].append({
            "M": M,
            "mean_ci_width": np.mean(np.array(alpha_upper) - np.array(alpha_lower)),
            "coverage": np.mean([(t >= l and t <= u) for t, l, u in zip(alpha_true, alpha_lower, alpha_upper)]),
            "mean_error": float(np.mean(alpha_errors)),
            "rmse": float(np.sqrt(np.mean(alpha_errors**2))),
            "mae": float(np.mean(np.abs(alpha_errors)))
        })
        
        # Beta parameters
        for k in range(self.K):
            for d in range(self.D):
                param_name = f"beta[{k+1},{d+1}]"
                key = f"beta_{k+1}_{d+1}"
                
                beta_true = [p["beta"][k][d] for p in all_true_params]
                beta_mean = [s.loc[param_name, "Mean"] for s in all_posterior_summaries]
                beta_lower = [s.loc[param_name, "5%"] for s in all_posterior_summaries]
                beta_upper = [s.loc[param_name, "95%"] for s in all_posterior_summaries]
                
                beta_errors = np.array(beta_mean) - np.array(beta_true)
                
                all_recovery_data["beta_data"][key].append({
                    "M": M,
                    "mean_ci_width": np.mean(np.array(beta_upper) - np.array(beta_lower)),
                    "coverage": np.mean([(t >= l and t <= u) for t, l, u in zip(beta_true, beta_lower, beta_upper)]),
                    "mean_error": float(np.mean(beta_errors)),
                    "rmse": float(np.sqrt(np.mean(beta_errors**2))),
                    "mae": float(np.mean(np.abs(beta_errors)))
                })
        
        # Delta parameters
        for k in range(self.K - 1):
            param_name = f"delta[{k+1}]"
            key = f"delta_{k+1}"
            
            delta_true = [p["delta"][k] for p in all_true_params]
            delta_mean = [s.loc[param_name, "Mean"] for s in all_posterior_summaries]
            delta_lower = [s.loc[param_name, "5%"] for s in all_posterior_summaries]
            delta_upper = [s.loc[param_name, "95%"] for s in all_posterior_summaries]
            
            delta_errors = np.array(delta_mean) - np.array(delta_true)
            
            all_recovery_data["delta_data"][key].append({
                "M": M,
                "mean_ci_width": np.mean(np.array(delta_upper) - np.array(delta_lower)),
                "coverage": np.mean([(t >= l and t <= u) for t, l, u in zip(delta_true, delta_lower, delta_upper)]),
                "mean_error": float(np.mean(delta_errors)),
                "rmse": float(np.sqrt(np.mean(delta_errors**2))),
                "mae": float(np.mean(np.abs(delta_errors)))
            })
    
    def _create_comparison_plots(self, all_recovery_data):
        """Create plots comparing recovery quality across M values."""
        
        M_values = all_recovery_data["M_values"]
        
        # Plot 1: CI Width vs M for all parameter types
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Alpha
        alpha_widths = [d["mean_ci_width"] for d in all_recovery_data["alpha_data"]]
        axes[0].plot(M_values, alpha_widths, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Mean 90% CI Width")
        axes[0].set_title("Alpha: CI Width vs. M")
        axes[0].grid(True, alpha=0.3)
        
        # Beta (all parameters)
        axes[1].set_xlabel("Number of Problems (M)")
        axes[1].set_ylabel("Mean 90% CI Width")
        axes[1].set_title("Beta Parameters: CI Width vs. M")
        for key, data in all_recovery_data["beta_data"].items():
            widths = [d["mean_ci_width"] for d in data]
            axes[1].plot(M_values, widths, 'o-', label=key, alpha=0.7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # Delta
        axes[2].set_xlabel("Number of Problems (M)")
        axes[2].set_ylabel("Mean 90% CI Width")
        axes[2].set_title("Delta Parameters: CI Width vs. M")
        for key, data in all_recovery_data["delta_data"].items():
            widths = [d["mean_ci_width"] for d in data]
            # Add more distinct line styles to differentiate delta_1 and delta_2
            linestyle = '-' if '1' in key else '--'
            marker = 'o' if '1' in key else 's'
            axes[2].plot(M_values, widths, marker=marker, linestyle=linestyle, 
                        label=key, alpha=0.7, linewidth=2, markersize=8)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ci_width_vs_M_comparison.png"))
        plt.close()
        
        # Plot 2: RMSE vs M for all parameter types
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Alpha
        alpha_rmse = [d["rmse"] for d in all_recovery_data["alpha_data"]]
        axes[0].plot(M_values, alpha_rmse, 'o-', linewidth=2, markersize=8, color='darkblue')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("Alpha: RMSE vs. M")
        axes[0].grid(True, alpha=0.3)
        
        # Beta (all parameters)
        axes[1].set_xlabel("Number of Problems (M)")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Beta Parameters: RMSE vs. M")
        for key, data in all_recovery_data["beta_data"].items():
            rmse = [d["rmse"] for d in data]
            axes[1].plot(M_values, rmse, 'o-', label=key, alpha=0.7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # Delta - with distinct styles
        axes[2].set_xlabel("Number of Problems (M)")
        axes[2].set_ylabel("RMSE")
        axes[2].set_title("Delta Parameters: RMSE vs. M")
        for key, data in all_recovery_data["delta_data"].items():
            rmse = [d["rmse"] for d in data]
            linestyle = '-' if '1' in key else '--'
            marker = 'o' if '1' in key else 's'
            axes[2].plot(M_values, rmse, marker=marker, linestyle=linestyle,
                        label=key, alpha=0.7, linewidth=2, markersize=8)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rmse_vs_M_comparison.png"))
        plt.close()
        
        # Plot 3: Mean Absolute Error vs M for all parameter types
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Alpha
        alpha_mae = [d["mae"] for d in all_recovery_data["alpha_data"]]
        axes[0].plot(M_values, alpha_mae, 'o-', linewidth=2, markersize=8, color='darkgreen')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Mean Absolute Error")
        axes[0].set_title("Alpha: MAE vs. M")
        axes[0].grid(True, alpha=0.3)
        
        # Beta (all parameters)
        axes[1].set_xlabel("Number of Problems (M)")
        axes[1].set_ylabel("Mean Absolute Error")
        axes[1].set_title("Beta Parameters: MAE vs. M")
        for key, data in all_recovery_data["beta_data"].items():
            mae = [d["mae"] for d in data]
            axes[1].plot(M_values, mae, 'o-', label=key, alpha=0.7)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # Delta - with distinct styles
        axes[2].set_xlabel("Number of Problems (M)")
        axes[2].set_ylabel("Mean Absolute Error")
        axes[2].set_title("Delta Parameters: MAE vs. M")
        for key, data in all_recovery_data["delta_data"].items():
            mae = [d["mae"] for d in data]
            linestyle = '-' if '1' in key else '--'
            marker = 'o' if '1' in key else 's'
            axes[2].plot(M_values, mae, marker=marker, linestyle=linestyle,
                        label=key, alpha=0.7, linewidth=2, markersize=8)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mae_vs_M_comparison.png"))
        plt.close()
        
        # Plot 4: Coverage vs M for all parameter types
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Alpha
        alpha_coverage = [d["coverage"] for d in all_recovery_data["alpha_data"]]
        axes[0].plot(M_values, alpha_coverage, 'o-', linewidth=2, markersize=8)
        axes[0].axhline(0.9, color='red', linestyle='--', label='Nominal 90%')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Coverage Rate")
        axes[0].set_title("Alpha: Coverage vs. M")
        axes[0].set_ylim([0, 1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Beta (all parameters)
        axes[1].set_xlabel("Number of Problems (M)")
        axes[1].set_ylabel("Coverage Rate")
        axes[1].set_title("Beta Parameters: Coverage vs. M")
        axes[1].axhline(0.9, color='red', linestyle='--', alpha=0.5)
        for key, data in all_recovery_data["beta_data"].items():
            coverage = [d["coverage"] for d in data]
            axes[1].plot(M_values, coverage, 'o-', label=key, alpha=0.7)
        axes[1].set_ylim([0, 1])
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # Delta - with distinct styles
        axes[2].set_xlabel("Number of Problems (M)")
        axes[2].set_ylabel("Coverage Rate")
        axes[2].set_title("Delta Parameters: Coverage vs. M")
        axes[2].axhline(0.9, color='red', linestyle='--', alpha=0.5)
        for key, data in all_recovery_data["delta_data"].items():
            coverage = [d["coverage"] for d in data]
            linestyle = '-' if '1' in key else '--'
            marker = 'o' if '1' in key else 's'
            axes[2].plot(M_values, coverage, marker=marker, linestyle=linestyle,
                        label=key, alpha=0.7, linewidth=2, markersize=8)
        axes[2].set_ylim([0, 1])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "coverage_vs_M_comparison.png"))
        plt.close()
        
        # Save summary statistics
        summary_data = {
            "M": M_values,
            "alpha_ci_width": alpha_widths,
            "alpha_rmse": alpha_rmse,
            "alpha_mae": alpha_mae,
            "alpha_coverage": alpha_coverage
        }
        
        for key in all_recovery_data["beta_data"].keys():
            summary_data[f"{key}_ci_width"] = [d["mean_ci_width"] for d in all_recovery_data["beta_data"][key]]
            summary_data[f"{key}_rmse"] = [d["rmse"] for d in all_recovery_data["beta_data"][key]]
            summary_data[f"{key}_mae"] = [d["mae"] for d in all_recovery_data["beta_data"][key]]
            summary_data[f"{key}_coverage"] = [d["coverage"] for d in all_recovery_data["beta_data"][key]]
        
        for key in all_recovery_data["delta_data"].keys():
            summary_data[f"{key}_ci_width"] = [d["mean_ci_width"] for d in all_recovery_data["delta_data"][key]]
            summary_data[f"{key}_rmse"] = [d["rmse"] for d in all_recovery_data["delta_data"][key]]
            summary_data[f"{key}_mae"] = [d["mae"] for d in all_recovery_data["delta_data"][key]]
            summary_data[f"{key}_coverage"] = [d["coverage"] for d in all_recovery_data["delta_data"][key]]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "recovery_summary_vs_M.csv"), index=False)


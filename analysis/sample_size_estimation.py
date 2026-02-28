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
from utils.study_design_m1 import StudyDesignM1
from utils import (detect_model_name, get_model_sim_hyperparams,
                   get_model_scalar_parameters, has_risky_data,
                   DEFAULT_PARAM_GENERATION)

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

        # Model detection for m_2/m_3 support
        self.model_name = detect_model_name(self.inference_model_path)
        self.is_risky = has_risky_data(self.model_name)
        self.has_omega = self.model_name in ("m_2", "m_3")
        self.has_kappa = self.model_name == "m_3"

        # Risky problem parameters (for m_1/m_2/m_3)
        self.N_risky = config.get("N", 20)
        self.S = config.get("S", 8)

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
            if self.is_risky:
                base_design = StudyDesignM1(
                    M=1, N=1,
                    K=self.K, D=self.D, R=self.R, S=self.S,
                    min_alts_per_problem=2,
                    max_alts_per_problem=min(self.R, 6),
                    feature_dist="uniform",
                    feature_params={"low": -2, "high": 2},
                    design_name="base_design"
                )
            else:
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
        fixed_x = base_design.x if self.is_risky and hasattr(base_design, 'x') else None
        
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
            "delta_data": {f"delta_{k}": [] for k in range(1, self.K)},
            "omega_data": [] if self.has_omega else None,
            "kappa_data": [] if self.has_kappa else None,
        }
        
        for M in self.M_grid:
            print(f"\nEvaluating M={M}...")
            # Generate study design with FIXED alternatives but varying M
            if self.is_risky:
                design = StudyDesignM1(
                    M=M, N=self.N_risky,
                    K=self.K, D=self.D, R=self.R, S=self.S,
                    min_alts_per_problem=2,
                    max_alts_per_problem=min(self.R, 6),
                    feature_dist="uniform",
                    feature_params={"low": -2, "high": 2},
                    design_name=f"M_{M}_K_{self.K}_D_{self.D}_R_{self.R}"
                )
            else:
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
            
            # For risky models, override risky alternatives and generate J
            if self.is_risky and fixed_x is not None:
                design.x = fixed_x
                design.J = design._generate_risky_indicator_array()
            
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
            omega_interval_widths = [] if self.has_omega else None
            kappa_interval_widths = [] if self.has_kappa else None
            
            for i in tqdm(range(self.n_iterations), desc=f"M={M}"):
                # Simulate data
                # Prepare simulation data with hyperparameters
                sim_data = design.get_data_dict().copy()
                sim_hyperparams = get_model_sim_hyperparams(self.model_name)
                for key in sim_hyperparams:
                    if key not in sim_data:
                        sim_data[key] = DEFAULT_PARAM_GENERATION[key]

                sim_fit = sim_model.sample(
                    data=sim_data,
                    seed=12345 + i,
                    iter_sampling=1,
                    iter_warmup=0,
                    chains=1,
                    fixed_param=True,
                    adapt_engaged=False
                )
                sim_samples = sim_fit.draws_pd().iloc[0]
                y = [int(sim_samples[f'y[{m+1}]']) for m in range(M)]
                z = None
                if self.is_risky:
                    z = [int(sim_samples[f'z[{n+1}]']) for n in range(self.N_risky)]
                
                # Extract true parameters
                true_params = {
                    "alpha": float(sim_samples["alpha"]),
                    "beta": [[float(sim_samples[f"beta[{k+1},{d+1}]"]) 
                            for d in range(self.D)] 
                            for k in range(self.K)],
                    "delta": [float(sim_samples[f"delta[{k+1}]"]) 
                            for k in range(self.K-1)]
                }
                if self.has_omega and "omega" in sim_samples.index:
                    true_params["omega"] = float(sim_samples["omega"])
                if self.has_kappa and "kappa" in sim_samples.index:
                    true_params["kappa"] = float(sim_samples["kappa"])
                inference_data["y"] = y
                if self.is_risky and z is not None:
                    inference_data["z"] = z
                # Remove simulation hyperparams not needed for inference
                for key in list(inference_data.keys()):
                    if key.endswith("_mean") or key.endswith("_sd"):
                        del inference_data[key]

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

                if self.has_omega and "omega" in summary.index:
                    omega_interval_widths.append(
                        summary.loc["omega", "95%"] - summary.loc["omega", "5%"])
                if self.has_kappa and "kappa" in summary.index:
                    kappa_interval_widths.append(
                        summary.loc["kappa", "95%"] - summary.loc["kappa", "5%"])

            # Generate coverage plots for this M
            self._create_coverage_plots(M, all_true_params, all_posterior_summaries, m_dir)
            
            # Store data for cross-M comparison
            all_recovery_data["M_values"].append(M)
            self._extract_recovery_data(M, all_true_params, all_posterior_summaries, all_recovery_data)

            # Summarize for this M
            mean_width = float(np.mean(interval_widths))
            std_width = float(np.std(interval_widths))
            result_entry = {
                "M": M,
                "mean_interval_width": mean_width,
                "std_interval_width": std_width,
                "interval_widths": interval_widths
            }
            if self.has_omega and omega_interval_widths:
                result_entry["omega_mean_width"] = float(np.mean(omega_interval_widths))
                result_entry["omega_std_width"] = float(np.std(omega_interval_widths))
            if self.has_kappa and kappa_interval_widths:
                result_entry["kappa_mean_width"] = float(np.mean(kappa_interval_widths))
                result_entry["kappa_std_width"] = float(np.std(kappa_interval_widths))
            results.append(result_entry)
            print(f"  Mean interval width for alpha: {mean_width:.3f} (std: {std_width:.3f})")
            if self.has_omega and omega_interval_widths:
                print(f"  Mean interval width for omega: {result_entry['omega_mean_width']:.3f}")
            if self.has_kappa and kappa_interval_widths:
                print(f"  Mean interval width for kappa: {result_entry['kappa_mean_width']:.3f}")

        # Save results
        rows = []
        for r in results:
            row = {
                "M": r["M"],
                "mean_interval_width": r["mean_interval_width"],
                "std_interval_width": r["std_interval_width"]
            }
            if self.has_omega:
                row["omega_mean_width"] = r.get("omega_mean_width", np.nan)
                row["omega_std_width"] = r.get("omega_std_width", np.nan)
            if self.has_kappa:
                row["kappa_mean_width"] = r.get("kappa_mean_width", np.nan)
                row["kappa_std_width"] = r.get("kappa_std_width", np.nan)
            rows.append(row)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(os.path.join(self.output_dir, "alpha_interval_widths.csv"), index=False)

        # Create cross-M comparison plots
        self._create_comparison_plots(all_recovery_data)

        # Plot alpha precision vs M
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            results_df["M"], results_df["mean_interval_width"],
            yerr=results_df["std_interval_width"], fmt='o-', capsize=5,
            label='alpha'
        )
        if self.has_omega and "omega_mean_width" in results_df.columns:
            plt.errorbar(
                results_df["M"], results_df["omega_mean_width"],
                yerr=results_df["omega_std_width"], fmt='s--', capsize=5,
                label='omega', color='orange'
            )
        if self.has_kappa and "kappa_mean_width" in results_df.columns:
            plt.errorbar(
                results_df["M"], results_df["kappa_mean_width"],
                yerr=results_df["kappa_std_width"], fmt='^:', capsize=5,
                label='kappa', color='green'
            )
        plt.xlabel("Number of Decision Problems (M)")
        plt.ylabel("Mean 90% Posterior Interval Width")
        plt.title("Scalar Parameter Precision vs. Sample Size (M)")
        if self.has_omega or self.has_kappa:
            plt.legend()
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
        
        # Omega coverage plot (m_2 and m_3)
        if self.has_omega:
            omega_true = [p["omega"] for p in all_true_params if "omega" in p]
            if omega_true:
                omega_mean = [s.loc["omega", "Mean"] for s in all_posterior_summaries]
                omega_lower = [s.loc["omega", "5%"] for s in all_posterior_summaries]
                omega_upper = [s.loc["omega", "95%"] for s in all_posterior_summaries]
                omega_coverage = np.mean([(t >= l and t <= u) for t, l, u in zip(omega_true, omega_lower, omega_upper)])
                
                plt.figure(figsize=(12, 6))
                for i in range(len(omega_true)):
                    color = 'green' if (omega_true[i] >= omega_lower[i] and omega_true[i] <= omega_upper[i]) else 'red'
                    plt.plot([i, i], [omega_lower[i], omega_upper[i]], color=color, linewidth=2, alpha=0.6)
                    plt.scatter(i, omega_mean[i], color=color, s=30, zorder=3)
                plt.scatter(range(len(omega_true)), omega_true, color='black', s=50, marker='x', label='True Value', zorder=4)
                plt.xlabel("Iteration")
                plt.ylabel("Omega Value")
                plt.title(f"M={M}: Omega Coverage (Coverage = {omega_coverage:.1%})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, "omega_coverage.png"))
                plt.close()
        
        # Kappa coverage plot (m_3 only)
        if self.has_kappa:
            kappa_true = [p["kappa"] for p in all_true_params if "kappa" in p]
            if kappa_true:
                kappa_mean = [s.loc["kappa", "Mean"] for s in all_posterior_summaries]
                kappa_lower = [s.loc["kappa", "5%"] for s in all_posterior_summaries]
                kappa_upper = [s.loc["kappa", "95%"] for s in all_posterior_summaries]
                kappa_coverage = np.mean([(t >= l and t <= u) for t, l, u in zip(kappa_true, kappa_lower, kappa_upper)])
                
                plt.figure(figsize=(12, 6))
                for i in range(len(kappa_true)):
                    color = 'green' if (kappa_true[i] >= kappa_lower[i] and kappa_true[i] <= kappa_upper[i]) else 'red'
                    plt.plot([i, i], [kappa_lower[i], kappa_upper[i]], color=color, linewidth=2, alpha=0.6)
                    plt.scatter(i, kappa_mean[i], color=color, s=30, zorder=3)
                plt.scatter(range(len(kappa_true)), kappa_true, color='black', s=50, marker='x', label='True Value', zorder=4)
                plt.xlabel("Iteration")
                plt.ylabel("Kappa Value")
                plt.title(f"M={M}: Kappa Coverage (Coverage = {kappa_coverage:.1%})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, "kappa_coverage.png"))
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
        
        # Omega (m_2 and m_3)
        if self.has_omega and all_recovery_data["omega_data"] is not None:
            omega_true = [p["omega"] for p in all_true_params if "omega" in p]
            if omega_true:
                omega_mean = [s.loc["omega", "Mean"] for s in all_posterior_summaries]
                omega_lower = [s.loc["omega", "5%"] for s in all_posterior_summaries]
                omega_upper = [s.loc["omega", "95%"] for s in all_posterior_summaries]
                omega_errors = np.array(omega_mean) - np.array(omega_true)
                
                all_recovery_data["omega_data"].append({
                    "M": M,
                    "mean_ci_width": np.mean(np.array(omega_upper) - np.array(omega_lower)),
                    "coverage": np.mean([(t >= l and t <= u) for t, l, u in zip(omega_true, omega_lower, omega_upper)]),
                    "mean_error": float(np.mean(omega_errors)),
                    "rmse": float(np.sqrt(np.mean(omega_errors**2))),
                    "mae": float(np.mean(np.abs(omega_errors)))
                })
        
        # Kappa (m_3 only)
        if self.has_kappa and all_recovery_data["kappa_data"] is not None:
            kappa_true = [p["kappa"] for p in all_true_params if "kappa" in p]
            if kappa_true:
                kappa_mean = [s.loc["kappa", "Mean"] for s in all_posterior_summaries]
                kappa_lower = [s.loc["kappa", "5%"] for s in all_posterior_summaries]
                kappa_upper = [s.loc["kappa", "95%"] for s in all_posterior_summaries]
                kappa_errors = np.array(kappa_mean) - np.array(kappa_true)
                
                all_recovery_data["kappa_data"].append({
                    "M": M,
                    "mean_ci_width": np.mean(np.array(kappa_upper) - np.array(kappa_lower)),
                    "coverage": np.mean([(t >= l and t <= u) for t, l, u in zip(kappa_true, kappa_lower, kappa_upper)]),
                    "mean_error": float(np.mean(kappa_errors)),
                    "rmse": float(np.sqrt(np.mean(kappa_errors**2))),
                    "mae": float(np.mean(np.abs(kappa_errors)))
                })
    
    def _create_comparison_plots(self, all_recovery_data):
        """Create plots comparing recovery quality across M values."""
        
        M_values = all_recovery_data["M_values"]
        
        # Plot 1: CI Width vs M for all parameter types
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Alpha
        alpha_widths = [d["mean_ci_width"] for d in all_recovery_data["alpha_data"]]
        axes[0].plot(M_values, alpha_widths, 'o-', linewidth=2, markersize=8, label='alpha')
        if all_recovery_data["omega_data"]:
            omega_widths = [d["mean_ci_width"] for d in all_recovery_data["omega_data"]]
            axes[0].plot(M_values, omega_widths, 's--', linewidth=2, markersize=8, label='omega', color='orange')
        if all_recovery_data["kappa_data"]:
            kappa_widths = [d["mean_ci_width"] for d in all_recovery_data["kappa_data"]]
            axes[0].plot(M_values, kappa_widths, '^:', linewidth=2, markersize=8, label='kappa', color='green')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Mean 90% CI Width")
        axes[0].set_title("Scalar Parameters: CI Width vs. M")
        if all_recovery_data["omega_data"] or all_recovery_data["kappa_data"]:
            axes[0].legend()
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
        axes[0].plot(M_values, alpha_rmse, 'o-', linewidth=2, markersize=8, color='darkblue', label='alpha')
        if all_recovery_data["omega_data"]:
            omega_rmse = [d["rmse"] for d in all_recovery_data["omega_data"]]
            axes[0].plot(M_values, omega_rmse, 's--', linewidth=2, markersize=8, label='omega', color='orange')
        if all_recovery_data["kappa_data"]:
            kappa_rmse = [d["rmse"] for d in all_recovery_data["kappa_data"]]
            axes[0].plot(M_values, kappa_rmse, '^:', linewidth=2, markersize=8, label='kappa', color='green')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("Scalar Parameters: RMSE vs. M")
        if all_recovery_data["omega_data"] or all_recovery_data["kappa_data"]:
            axes[0].legend()
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
        axes[0].plot(M_values, alpha_mae, 'o-', linewidth=2, markersize=8, color='darkgreen', label='alpha')
        if all_recovery_data["omega_data"]:
            omega_mae = [d["mae"] for d in all_recovery_data["omega_data"]]
            axes[0].plot(M_values, omega_mae, 's--', linewidth=2, markersize=8, label='omega', color='orange')
        if all_recovery_data["kappa_data"]:
            kappa_mae = [d["mae"] for d in all_recovery_data["kappa_data"]]
            axes[0].plot(M_values, kappa_mae, '^:', linewidth=2, markersize=8, label='kappa', color='green')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Mean Absolute Error")
        axes[0].set_title("Scalar Parameters: MAE vs. M")
        if all_recovery_data["omega_data"] or all_recovery_data["kappa_data"]:
            axes[0].legend()
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
        axes[0].plot(M_values, alpha_coverage, 'o-', linewidth=2, markersize=8, label='alpha')
        if all_recovery_data["omega_data"]:
            omega_coverage = [d["coverage"] for d in all_recovery_data["omega_data"]]
            axes[0].plot(M_values, omega_coverage, 's--', linewidth=2, markersize=8, label='omega', color='orange')
        if all_recovery_data["kappa_data"]:
            kappa_coverage = [d["coverage"] for d in all_recovery_data["kappa_data"]]
            axes[0].plot(M_values, kappa_coverage, '^:', linewidth=2, markersize=8, label='kappa', color='green')
        axes[0].axhline(0.9, color='red', linestyle='--', label='Nominal 90%')
        axes[0].set_xlabel("Number of Problems (M)")
        axes[0].set_ylabel("Coverage Rate")
        axes[0].set_title("Scalar Parameters: Coverage vs. M")
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
        
        if all_recovery_data["omega_data"]:
            summary_data["omega_ci_width"] = [d["mean_ci_width"] for d in all_recovery_data["omega_data"]]
            summary_data["omega_rmse"] = [d["rmse"] for d in all_recovery_data["omega_data"]]
            summary_data["omega_mae"] = [d["mae"] for d in all_recovery_data["omega_data"]]
            summary_data["omega_coverage"] = [d["coverage"] for d in all_recovery_data["omega_data"]]
        
        if all_recovery_data["kappa_data"]:
            summary_data["kappa_ci_width"] = [d["mean_ci_width"] for d in all_recovery_data["kappa_data"]]
            summary_data["kappa_rmse"] = [d["rmse"] for d in all_recovery_data["kappa_data"]]
            summary_data["kappa_mae"] = [d["mae"] for d in all_recovery_data["kappa_data"]]
            summary_data["kappa_coverage"] = [d["coverage"] for d in all_recovery_data["kappa_data"]]
        
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


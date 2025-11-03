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
            - Generate study design
            - Run parameter recovery n_iterations times
            - Record posterior interval width for alpha
        Summarize and plot results.
        """
        results = []
        for M in self.M_grid:
            print(f"Evaluating M={M}...")
            # Generate study design
            design = StudyDesign(
                M=M, K=self.K, D=self.D, R=self.R,
                min_alts_per_problem=2,
                max_alts_per_problem=min(self.R, 6),
                feature_dist="uniform",
                feature_params={"low": -2, "high": 2},
                design_name=f"M_{M}_K_{self.K}_D_{self.D}_R_{self.R}"
            )
            design.generate()
            design_path = os.path.join(self.output_dir, f"design_M{M}.json")
            design.save(design_path, include_metadata=True, include_plots=False)

            # Compile models
            inference_model = CmdStanModel(stan_file=self.inference_model_path)
            sim_model = CmdStanModel(stan_file=self.sim_model_path)

            # Run parameter recovery
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
                alpha_lower = summary.loc["alpha", "5%"]
                alpha_upper = summary.loc["alpha", "95%"]
                interval_widths.append(alpha_upper - alpha_lower)

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

        # Plot results
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

        # Print recommendation
        target_width = None
        if "target_interval_width" in locals():
            target_width = locals()["target_interval_width"]
        print("\nSample Size Estimation Complete.")
        print("Results saved to:", self.output_dir)
        print(results_df)
        return results_df


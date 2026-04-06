"""
Parameter recovery for hierarchical SEU sensitivity models.

Extends the base ParameterRecovery pattern to handle:
- Vector-valued alpha (J cells)
- Regression parameters (gamma0, gamma[P], sigma_cell)
- Shared delta
- Per-cell beta (not individually tracked; aggregate RMSE only)
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design_hierarchical import HierarchicalStudyDesign


class HierarchicalParameterRecovery:
    """
    Parameter recovery analysis for hierarchical SEU sensitivity models.

    Performs simulate-and-recover iterations using h_m01_sim and h_m01,
    tracking regression parameters (gamma0, gamma, sigma_cell),
    cell-level alphas, and shared delta.
    """

    def __init__(
        self,
        inference_model_path: str = None,
        sim_model_path: str = None,
        study_design: HierarchicalStudyDesign = None,
        output_dir: str = None,
        n_mcmc_samples: int = 2000,
        n_mcmc_chains: int = 4,
        n_iterations: int = 20,
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if inference_model_path is None:
            inference_model_path = os.path.join(project_root, "models", "h_m01.stan")
        if sim_model_path is None:
            sim_model_path = os.path.join(project_root, "models", "h_m01_sim.stan")

        self.inference_model_path = inference_model_path
        self.sim_model_path = sim_model_path
        self.study_design = study_design
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_chains = n_mcmc_chains
        self.n_iterations = n_iterations

        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                project_root, "results", "parameter_recovery", f"h_m01_run_{timestamp}"
            )
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Compile models
        self.inference_model = CmdStanModel(stan_file=self.inference_model_path)
        self.sim_model = CmdStanModel(stan_file=self.sim_model_path)

    def run(self):
        """
        Main recovery loop.

        For each iteration:
        1. Sample from h_m01_sim with fixed_param=True.
        2. Build inference data dict: design data + generated y.
        3. Fit h_m01.stan.
        4. Extract posteriors for gamma0, gamma[1..P], sigma_cell,
           alpha[1..J], delta[1..K-1].
        5. Compare to true values.

        Returns
        -------
        tuple : (all_true_params, all_posterior_summaries)
        """
        # Generate study design if not provided
        if self.study_design is None:
            X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=float)
            self.study_design = HierarchicalStudyDesign(
                J=6, K=3, D=2, R=10, P=2, M_per_cell=20, X=X[:6],
            )
            self.study_design.generate()

        # Save the study design
        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)

        # Get data dictionary for simulation
        sim_data = self.study_design.get_data_dict()

        J = self.study_design.J
        K = self.study_design.K
        D = self.study_design.D
        P = self.study_design.P
        M_total = self.study_design.M_total

        all_true_params = []
        all_posterior_summaries = []

        # Save config
        with open(os.path.join(self.output_dir, "config_info.json"), "w") as f:
            json.dump(
                {
                    "n_iterations": self.n_iterations,
                    "n_mcmc_samples": self.n_mcmc_samples,
                    "n_mcmc_chains": self.n_mcmc_chains,
                    "J": J, "K": K, "D": D, "P": P, "M_total": M_total,
                },
                f,
                indent=2,
            )

        print(f"Running {self.n_iterations} iterations of hierarchical parameter recovery...")

        for iteration in tqdm(range(self.n_iterations)):
            iter_dir = os.path.join(self.output_dir, f"iteration_{iteration+1}")
            os.makedirs(iter_dir, exist_ok=True)

            # 1. Simulate data
            sim_fit = self.sim_model.sample(
                data=sim_data,
                seed=12345 + iteration,
                iter_sampling=1,
                iter_warmup=0,
                chains=1,
                fixed_param=True,
                adapt_engaged=False,
            )

            sim_samples = sim_fit.draws_pd().iloc[0]

            # 2. Extract true parameters
            true_params = self._extract_true_params(sim_samples, J, K, D, P)

            with open(os.path.join(iter_dir, "true_parameters.json"), "w") as f:
                json.dump(true_params, f, indent=2)

            # 3. Build inference data
            y = [int(sim_samples[f"y[{m+1}]"]) for m in range(M_total)]

            inference_data = {
                k: v
                for k, v in sim_data.items()
                if k not in ("gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd", "beta_sd")
            }
            inference_data["y"] = y

            # 4. Fit inference model
            try:
                fit = self.inference_model.sample(
                    data=inference_data,
                    seed=54321 + iteration,
                    iter_sampling=self.n_mcmc_samples,
                    iter_warmup=self.n_mcmc_samples // 2,
                    chains=self.n_mcmc_chains,
                    show_console=False,
                )
            except RuntimeError as e:
                print(f"\n  Warning: Iteration {iteration+1} sampling failed: {str(e)[:200]}")
                with open(os.path.join(iter_dir, "error.txt"), "w") as f:
                    f.write(f"Sampling error: {str(e)}\n")
                continue

            # 5. Store results
            try:
                summary = fit.summary()
                summary.to_csv(os.path.join(iter_dir, "posterior_summary.csv"))

                diagnostics = fit.diagnose()
                with open(os.path.join(iter_dir, "diagnostics.txt"), "w") as f:
                    f.write(diagnostics)

                all_true_params.append(true_params)
                all_posterior_summaries.append(summary)
            except Exception as e:
                print(f"\n  Warning: Iteration {iteration+1} failed: {str(e)}")
                with open(os.path.join(iter_dir, "error.txt"), "w") as f:
                    f.write(f"Error: {str(e)}\n")
                continue

        # Save all true parameters
        with open(os.path.join(self.output_dir, "all_true_parameters.json"), "w") as f:
            json.dump(all_true_params, f, indent=2)

        if len(all_true_params) == 0:
            print("\nWarning: No iterations completed successfully!")
            return all_true_params, all_posterior_summaries

        print(f"\nCompleted {len(all_true_params)} out of {self.n_iterations} iterations successfully")

        # Analyze recovery
        self._analyze_recovery(all_true_params, all_posterior_summaries)

        return all_true_params, all_posterior_summaries

    def _extract_true_params(self, sim_samples, J, K, D, P) -> dict:
        """Extract true parameter values from simulation output."""
        return {
            "gamma0": float(sim_samples["gamma0"]),
            "gamma": [float(sim_samples[f"gamma[{p+1}]"]) for p in range(P)],
            "sigma_cell": float(sim_samples["sigma_cell"]),
            "alpha": [float(sim_samples[f"alpha[{j+1}]"]) for j in range(J)],
            "delta": [float(sim_samples[f"delta[{k+1}]"]) for k in range(K - 1)],
            "upsilon": [float(sim_samples[f"upsilon[{k+1}]"]) for k in range(K)],
        }

    def _analyze_recovery(self, all_true_params, all_posterior_summaries):
        """Compute bias, RMSE, coverage, CI width for each parameter."""
        recovery_dir = os.path.join(self.output_dir, "recovery_summary")
        os.makedirs(recovery_dir, exist_ok=True)

        recovery_stats = {}
        J = len(all_true_params[0]["alpha"])
        P = len(all_true_params[0]["gamma"])
        K_minus_1 = len(all_true_params[0]["delta"])

        # === Regression parameters ===
        regression_params = [("gamma0", "gamma0")]
        for p in range(P):
            regression_params.append((f"gamma[{p+1}]", f"gamma_{p+1}"))
        regression_params.append(("sigma_cell", "sigma_cell"))

        fig, axes = plt.subplots(1, len(regression_params), figsize=(5 * len(regression_params), 5))
        if len(regression_params) == 1:
            axes = [axes]

        for idx, (stan_name, label) in enumerate(regression_params):
            if stan_name == "gamma0":
                true_vals = [p["gamma0"] for p in all_true_params]
            elif stan_name == "sigma_cell":
                true_vals = [p["sigma_cell"] for p in all_true_params]
            else:
                p_idx = int(stan_name.split("[")[1].rstrip("]")) - 1
                true_vals = [p["gamma"][p_idx] for p in all_true_params]

            mean_vals = [s.loc[stan_name, "Mean"] for s in all_posterior_summaries]
            lower_vals = [s.loc[stan_name, "5%"] for s in all_posterior_summaries]
            upper_vals = [s.loc[stan_name, "95%"] for s in all_posterior_summaries]

            stats = self._compute_metrics(true_vals, mean_vals, lower_vals, upper_vals)
            recovery_stats[label] = stats

            ax = axes[idx]
            ax.scatter(true_vals, mean_vals, alpha=0.7)
            lims = [min(min(true_vals), min(mean_vals)), max(max(true_vals), max(mean_vals))]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Estimated {label}")
            ax.set_title(f"{label}\nBias={stats['bias']:.3f} RMSE={stats['rmse']:.3f} Cov={stats['coverage']:.0%}")

        plt.tight_layout()
        plt.savefig(os.path.join(recovery_dir, "regression_recovery.png"), dpi=150)
        plt.close()

        # === Alpha recovery ===
        ncols = min(J, 4)
        nrows = (J + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

        for j in range(J):
            stan_name = f"alpha[{j+1}]"
            true_vals = [p["alpha"][j] for p in all_true_params]
            mean_vals = [s.loc[stan_name, "Mean"] for s in all_posterior_summaries]
            lower_vals = [s.loc[stan_name, "5%"] for s in all_posterior_summaries]
            upper_vals = [s.loc[stan_name, "95%"] for s in all_posterior_summaries]

            stats = self._compute_metrics(true_vals, mean_vals, lower_vals, upper_vals)
            recovery_stats[f"alpha_{j+1}"] = stats

            ax = axes[j // ncols][j % ncols]
            ax.scatter(true_vals, mean_vals, alpha=0.7)
            lims = [min(min(true_vals), min(mean_vals)), max(max(true_vals), max(mean_vals))]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel(f"True alpha[{j+1}]")
            ax.set_ylabel(f"Estimated alpha[{j+1}]")
            ax.set_title(f"alpha[{j+1}]\nRMSE={stats['rmse']:.3f} Cov={stats['coverage']:.0%}")

        # Hide unused axes
        for idx in range(J, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(recovery_dir, "alpha_recovery.png"), dpi=150)
        plt.close()

        # === Delta recovery ===
        fig, axes = plt.subplots(1, K_minus_1, figsize=(5 * K_minus_1, 5))
        if K_minus_1 == 1:
            axes = [axes]

        for k in range(K_minus_1):
            stan_name = f"delta[{k+1}]"
            true_vals = [p["delta"][k] for p in all_true_params]
            mean_vals = [s.loc[stan_name, "Mean"] for s in all_posterior_summaries]
            lower_vals = [s.loc[stan_name, "5%"] for s in all_posterior_summaries]
            upper_vals = [s.loc[stan_name, "95%"] for s in all_posterior_summaries]

            stats = self._compute_metrics(true_vals, mean_vals, lower_vals, upper_vals)
            recovery_stats[f"delta_{k+1}"] = stats

            ax = axes[k]
            ax.scatter(true_vals, mean_vals, alpha=0.7)
            lims = [min(min(true_vals), min(mean_vals)), max(max(true_vals), max(mean_vals))]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel(f"True delta[{k+1}]")
            ax.set_ylabel(f"Estimated delta[{k+1}]")
            ax.set_title(f"delta[{k+1}]\nRMSE={stats['rmse']:.3f} Cov={stats['coverage']:.0%}")

        plt.tight_layout()
        plt.savefig(os.path.join(recovery_dir, "delta_recovery.png"), dpi=150)
        plt.close()

        # Save stats
        with open(os.path.join(recovery_dir, "recovery_statistics.json"), "w") as f:
            json.dump(recovery_stats, f, indent=2)

        print(f"Recovery results saved to {recovery_dir}")

    @staticmethod
    def _compute_metrics(true_vals, mean_vals, lower_vals, upper_vals) -> dict:
        """Compute bias, RMSE, coverage, CI width."""
        true_arr = np.array(true_vals)
        mean_arr = np.array(mean_vals)
        lower_arr = np.array(lower_vals)
        upper_arr = np.array(upper_vals)

        bias = float(np.mean(mean_arr - true_arr))
        rmse = float(np.sqrt(np.mean((mean_arr - true_arr) ** 2)))
        coverage = float(np.mean((true_arr >= lower_arr) & (true_arr <= upper_arr)))
        ci_width = float(np.mean(upper_arr - lower_arr))

        return {"bias": bias, "rmse": rmse, "coverage": coverage, "ci_width": ci_width}

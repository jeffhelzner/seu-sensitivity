"""
Simulation-Based Calibration for hierarchical SEU sensitivity models.

Follows the same pattern as analysis/sbc.py but uses:
- h_m01_sbc.stan (combined sim + fit)
- HierarchicalStudyDesign for data
- Custom rank extraction for gamma0, gamma, sigma_cell, alpha, delta
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cmdstanpy import CmdStanModel
from tqdm import tqdm
from scipy import stats

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design_hierarchical import HierarchicalStudyDesign


class HierarchicalSBC:
    """
    Simulation-Based Calibration for hierarchical SEU sensitivity models.

    Uses h_m01_sbc.stan which draws true parameters in transformed data,
    generates choice data, then fits and produces rank statistics.
    """

    def __init__(
        self,
        sbc_model_path: str = None,
        study_design: HierarchicalStudyDesign = None,
        output_dir: str = None,
        n_sbc_sims: int = 100,
        n_mcmc_samples: int = 1000,
        n_mcmc_chains: int = 1,
        thin: int = 3,
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if sbc_model_path is None:
            sbc_model_path = os.path.join(project_root, "models", "h_m01_sbc.stan")

        self.sbc_model_path = sbc_model_path
        self.study_design = study_design
        self.n_sbc_sims = n_sbc_sims
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_chains = n_mcmc_chains
        self.thin = thin

        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                project_root, "results", "sbc", f"h_m01_sbc_{timestamp}"
            )
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Compile model
        self.sbc_model = CmdStanModel(stan_file=self.sbc_model_path)

    def run(self):
        """
        SBC loop.

        For each simulation:
        1. Run h_m01_sbc.stan (draws true params, generates y, fits).
        2. Extract ranks_ and sum across posterior draws.
        3. After all simulations: analyze rank distributions.

        Returns
        -------
        tuple : (all_ranks, all_true_params)
        """
        # Generate study design if not provided
        if self.study_design is None:
            X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=float)
            self.study_design = HierarchicalStudyDesign(
                J=6, K=3, D=2, R=10, P=2, M_per_cell=20, X=X[:6],
            )
            self.study_design.generate()

        # Save design
        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)

        # Get data (SBC model doesn't need sim hyperparams or y)
        data = self.study_design.get_data_dict()
        # Remove sim hyperparams (SBC model draws its own)
        for key in ("gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd", "beta_sd"):
            data.pop(key, None)

        J = self.study_design.J
        K = self.study_design.K
        P = self.study_design.P

        # Build parameter names matching the pars_/ranks_ vector in h_m01_sbc.stan
        param_names = self._get_param_names(J, K, P)

        all_ranks = []
        all_true_params = []

        sims_dir = os.path.join(self.output_dir, "simulations")
        os.makedirs(sims_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.output_dir, "config_info.json"), "w") as f:
            json.dump(
                {
                    "n_sbc_sims": self.n_sbc_sims,
                    "n_mcmc_samples": self.n_mcmc_samples,
                    "n_mcmc_chains": self.n_mcmc_chains,
                    "thin": self.thin,
                    "effective_sample_size": self.n_mcmc_samples // self.thin,
                    "J": J, "K": K, "P": P,
                    "param_names": param_names,
                },
                f,
                indent=2,
            )

        print(f"Running {self.n_sbc_sims} SBC simulations with thinning factor {self.thin}...")

        for i in tqdm(range(self.n_sbc_sims)):
            sbc_fit = self.sbc_model.sample(
                data=data,
                seed=123 + i,
                iter_sampling=self.n_mcmc_samples // self.n_mcmc_chains,
                iter_warmup=self.n_mcmc_samples // (2 * self.n_mcmc_chains),
                chains=self.n_mcmc_chains,
            )

            # Extract true parameters
            true_params = sbc_fit.stan_variable("pars_")[0]

            # Calculate ranks with thinning
            ranks = []
            for j, param in enumerate(param_names):
                binary_ranks = sbc_fit.stan_variable("ranks_")[::self.thin, j]
                total_thinned = len(binary_ranks)
                rank = total_thinned - np.sum(binary_ranks)
                ranks.append(rank)

            all_true_params.append(true_params)
            all_ranks.append(ranks)

            # Save first few for inspection
            if i < 10:
                sim_dir = os.path.join(sims_dir, f"sim_{i+1}")
                os.makedirs(sim_dir, exist_ok=True)
                sbc_fit.summary().to_csv(os.path.join(sim_dir, "posterior_summary.csv"))
                with open(os.path.join(sim_dir, "true_parameters.json"), "w") as f:
                    param_dict = {name: float(true_params[j]) for j, name in enumerate(param_names)}
                    json.dump(param_dict, f, indent=2)

        all_ranks = np.array(all_ranks)
        all_true_params = np.array(all_true_params)

        # Analyze
        self._analyze_sbc(all_ranks, all_true_params, param_names)

        return all_ranks, all_true_params

    def _get_param_names(self, J, K, P) -> list:
        """
        Return ordered list of parameter names matching pars_/ranks_ in h_m01_sbc.stan.

        Order: gamma0, gamma[1..P], sigma_cell, alpha[1..J], delta[1..K-1]
        """
        names = ["gamma0"]
        for p in range(1, P + 1):
            names.append(f"gamma[{p}]")
        names.append("sigma_cell")
        for j in range(1, J + 1):
            names.append(f"alpha[{j}]")
        for k in range(1, K):
            names.append(f"delta[{k}]")
        return names

    def _analyze_sbc(self, ranks, true_params, param_names):
        """Create rank histograms, ECDF plots, chi-square and KS tests."""
        sbc_dir = os.path.join(self.output_dir, "sbc_results")
        os.makedirs(sbc_dir, exist_ok=True)

        np.save(os.path.join(sbc_dir, "ranks.npy"), ranks)
        np.save(os.path.join(sbc_dir, "true_params.npy"), true_params)

        n_sims = ranks.shape[0]
        total_thinned_samples = self.n_mcmc_samples // self.thin
        n_bins = min(20, total_thinned_samples // 5)
        bin_edges = np.linspace(0, total_thinned_samples, n_bins + 1)

        J = self.study_design.J
        K = self.study_design.K
        P = self.study_design.P

        # Identify parameter groups
        regression_names = ["gamma0"] + [f"gamma[{p}]" for p in range(1, P + 1)] + ["sigma_cell"]
        alpha_names = [f"alpha[{j}]" for j in range(1, J + 1)]
        delta_names = [f"delta[{k}]" for k in range(1, K)]

        chi2_results = {}
        ks_results = {}

        # Helper to plot a group of parameters
        def plot_group(group_names, filename_prefix):
            n_params = len(group_names)
            ncols = min(3, n_params)
            nrows = (n_params + ncols - 1) // ncols

            # Rank histograms
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            for idx, name in enumerate(group_names):
                col_idx = param_names.index(name)
                param_ranks = ranks[:, col_idx]

                ax = axes[idx // ncols][idx % ncols]
                counts, edges, patches = ax.hist(
                    param_ranks, bins=bin_edges, alpha=0.7, color="skyblue", edgecolor="black"
                )

                total_observed = np.sum(counts)
                expected_per_bin = total_observed / n_bins
                ax.axhline(expected_per_bin, color="red", linestyle="--")

                expected = np.ones(n_bins) * expected_per_bin
                try:
                    chi2_stat, p_value = stats.chisquare(counts, expected)
                except ValueError:
                    expected = expected * (total_observed / np.sum(expected))
                    chi2_stat, p_value = stats.chisquare(counts, expected)

                chi2_results[name] = {
                    "chi2_statistic": float(chi2_stat),
                    "p_value": float(p_value),
                }

                ax.set_title(f"{name}\np={p_value:.3f}")
                ax.set_xlabel("Rank")
                ax.set_ylabel("Count")
                ax.grid(alpha=0.3)

            for idx in range(n_params, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(sbc_dir, f"{filename_prefix}_ranks.png"), dpi=150)
            plt.close()

            # ECDF plots
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            for idx, name in enumerate(group_names):
                col_idx = param_names.index(name)
                param_ranks = ranks[:, col_idx] / total_thinned_samples

                sorted_ranks = np.sort(param_ranks)
                ecdf = np.arange(1, n_sims + 1) / n_sims

                ax = axes[idx // ncols][idx % ncols]
                ax.step(sorted_ranks, ecdf, where="post", label="ECDF")
                ax.plot([0, 1], [0, 1], "r--", label="Uniform")

                ks_stat, ks_pvalue = stats.kstest(param_ranks, "uniform")
                ks_results[name] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(ks_pvalue),
                }

                ax.set_title(f"{name} ECDF\nKS p={ks_pvalue:.3f}")
                ax.set_xlabel("Normalized Rank")
                ax.set_ylabel("ECDF")
                ax.grid(alpha=0.3)

            for idx in range(n_params, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(sbc_dir, f"{filename_prefix}_ecdf.png"), dpi=150)
            plt.close()

        plot_group(regression_names, "regression")
        plot_group(alpha_names, "alpha")
        plot_group(delta_names, "delta")

        # Save summary
        summary = {
            "n_simulations": n_sims,
            "n_mcmc_samples": self.n_mcmc_samples,
            "thin": self.thin,
            "chi2_tests": chi2_results,
            "ks_tests": ks_results,
        }
        with open(os.path.join(sbc_dir, "sbc_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"SBC results saved to {sbc_dir}")

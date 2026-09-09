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
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design_hierarchical import HierarchicalStudyDesign
from utils.cmdstan_artifacts import gzip_csv_files
from analysis.hierarchical_power import fit_diagnostics


def _rejected_proposal_counts(paths) -> list[int]:
    marker = "categorical_logit_lpmf: log odds parameter[1] is inf"
    return [Path(path).read_text().count(marker) for path in paths]


def _load_completed_iteration(iter_dir: Path):
    true_path = iter_dir / "true_parameters.json"
    summary_path = iter_dir / "posterior_summary.csv"
    diagnostics_path = iter_dir / "diagnostics.json"
    if not all(path.exists() for path in (true_path, summary_path, diagnostics_path)):
        return None
    with true_path.open() as handle:
        true_params = json.load(handle)
    return true_params, pd.read_csv(summary_path, index_col=0)


def _summarize_sampler_diagnostics(output_dir: Path) -> dict:
    records = []
    for diagnostics_path in sorted(output_dir.glob("iteration_*/diagnostics.json")):
        with diagnostics_path.open() as handle:
            diagnostics = json.load(handle)
        records.append(
            {
                "iteration": int(diagnostics_path.parent.name.split("_")[-1]),
                "seconds": diagnostics["seconds"],
                "max_rhat": diagnostics["max_rhat"],
                "min_ess_bulk": diagnostics["min_ess_bulk"],
                "min_ebfmi": diagnostics["min_ebfmi"],
                "divergences": diagnostics["divergences"],
                "treedepth_saturated_share": diagnostics[
                    "treedepth_saturated_share"
                ],
                "nonfinite_proposals_total": diagnostics[
                    "nonfinite_proposals_total"
                ],
            }
        )
    thresholds = {
        "max_rhat": 1.01,
        "min_ess_bulk": 400.0,
        "min_ebfmi": 0.3,
        "max_divergences": 0,
        "max_treedepth_saturated_share": 0.0,
    }
    for record in records:
        record["passes"] = (
            record["max_rhat"] < thresholds["max_rhat"]
            and record["min_ess_bulk"] >= thresholds["min_ess_bulk"]
            and record["min_ebfmi"] >= thresholds["min_ebfmi"]
            and record["divergences"] <= thresholds["max_divergences"]
            and record["treedepth_saturated_share"]
            <= thresholds["max_treedepth_saturated_share"]
        )
    return {
        "thresholds": thresholds,
        "iterations": records,
        "completed": len(records),
        "passed": sum(record["passes"] for record in records),
        "all_passed": bool(records) and all(record["passes"] for record in records),
        "total_fit_seconds": sum(record["seconds"] for record in records),
    }


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
        n_mcmc_warmup: int = None,
        n_mcmc_chains: int = 4,
        n_iterations: int = 20,
        alpha_var: str = "alpha",
        extra_scalar_params: tuple = (),
        sim_only_keys: tuple = (),
        sim_overrides: dict = None,
        fixed_eta_config: dict = None,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
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
        self.n_mcmc_warmup = (
            n_mcmc_warmup
            if n_mcmc_warmup is not None
            else n_mcmc_samples // 2
        )
        self.n_mcmc_chains = n_mcmc_chains
        self.n_iterations = n_iterations
        # h_m01_size renames alpha -> alpha_cell (alpha is observation-varying
        # there) and adds gamma_size. Keeping these as parameters lets one
        # recovery implementation drive both models rather than forking it.
        self.alpha_var = alpha_var
        self.extra_scalar_params = tuple(extra_scalar_params)
        self.sim_only_keys = tuple(sim_only_keys)
        # Overrides applied to the simulation hyperparameters. This is what
        # turns a recovery run into the §8.5(e) NULL-CALIBRATION check:
        # {"gamma_size_sd": 0} pins the true slope at zero, so the reported
        # bias and coverage for gamma_size become exactly the null statistics.
        self.sim_overrides = dict(sim_overrides or {})
        self.fixed_eta_config = dict(fixed_eta_config or {})
        self.adapt_delta = adapt_delta
        self.max_treedepth = max_treedepth

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
        sim_data.update(self.sim_overrides)
        if self.fixed_eta_config:
            sim_data.update(
                _generate_fixed_eta(
                    self.fixed_eta_config,
                    J=self.study_design.J,
                    K=self.study_design.K,
                    R=self.study_design.R,
                )
            )

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
                    "n_mcmc_warmup": self.n_mcmc_warmup,
                    "n_mcmc_chains": self.n_mcmc_chains,
                    "adapt_delta": self.adapt_delta,
                    "max_treedepth": self.max_treedepth,
                    "sim_overrides": self.sim_overrides,
                    "fixed_eta_config": self.fixed_eta_config,
                    "J": J, "K": K, "D": D, "P": P, "M_total": M_total,
                },
                f,
                indent=2,
            )

        print(f"Running {self.n_iterations} iterations of hierarchical parameter recovery...")

        for iteration in tqdm(range(self.n_iterations)):
            iter_dir = Path(self.output_dir) / f"iteration_{iteration+1}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            completed = _load_completed_iteration(iter_dir)
            if completed is not None:
                true_params, summary = completed
                all_true_params.append(true_params)
                all_posterior_summaries.append(summary)
                continue

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

            with open(iter_dir / "true_parameters.json", "w") as f:
                json.dump(true_params, f, indent=2)

            # 3. Build inference data
            y = [int(sim_samples[f"y[{m+1}]"]) for m in range(M_total)]

            inference_data = {
                k: v
                for k, v in sim_data.items()
                if k not in (
                    "gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd", "beta_sd",
                ) + self.sim_only_keys
            }
            inference_data["y"] = y

            # 4. Fit inference model
            try:
                chain_dir = iter_dir / "chains" / "main"
                chain_dir.mkdir(parents=True, exist_ok=True)
                fit_started = time.time()
                fit = self.inference_model.sample(
                    data=inference_data,
                    seed=54321 + iteration,
                    iter_sampling=self.n_mcmc_samples,
                    iter_warmup=self.n_mcmc_warmup,
                    chains=self.n_mcmc_chains,
                    adapt_delta=self.adapt_delta,
                    max_treedepth=self.max_treedepth,
                    show_console=False,
                    output_dir=str(chain_dir),
                )
                fit_seconds = time.time() - fit_started
            except RuntimeError as e:
                print(f"\n  Warning: Iteration {iteration+1} sampling failed: {str(e)[:200]}")
                with open(iter_dir / "error.txt", "w") as f:
                    f.write(f"Sampling error: {str(e)}\n")
                continue

            # 5. Store results
            try:
                summary = fit.summary()
                summary.to_csv(iter_dir / "posterior_summary.csv")

                diagnostics = fit.diagnose()
                with open(iter_dir / "diagnostics.txt", "w") as f:
                    f.write(diagnostics)
                chain_files = gzip_csv_files(fit.runset.csv_files)
                filtered_diagnostics = fit_diagnostics(
                    fit,
                    seconds=fit_seconds,
                    max_treedepth=self.max_treedepth,
                )
                filtered_diagnostics["chain_files"] = [
                    str(path) for path in chain_files
                ]
                rejection_counts = _rejected_proposal_counts(
                    fit.runset.stdout_files
                )
                filtered_diagnostics["nonfinite_proposals_by_chain"] = (
                    rejection_counts
                )
                filtered_diagnostics["nonfinite_proposals_total"] = sum(
                    rejection_counts
                )
                with open(iter_dir / "diagnostics.json", "w") as f:
                    json.dump(filtered_diagnostics, f, indent=2)
                for path in fit.runset.csv_files:
                    Path(path).unlink()

                all_true_params.append(true_params)
                all_posterior_summaries.append(summary)
            except Exception as e:
                print(f"\n  Warning: Iteration {iteration+1} failed: {str(e)}")
                with open(iter_dir / "error.txt", "w") as f:
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
        sampler_summary = _summarize_sampler_diagnostics(Path(self.output_dir))
        with open(
            Path(self.output_dir) / "recovery_summary" / "sampler_diagnostics.json",
            "w",
        ) as handle:
            json.dump(sampler_summary, handle, indent=2)

        return all_true_params, all_posterior_summaries

    def _extract_true_params(self, sim_samples, J, K, D, P) -> dict:
        """Extract true parameter values from simulation output."""
        params = {
            "gamma0": float(sim_samples["gamma0"]),
            "gamma": [float(sim_samples[f"gamma[{p+1}]"]) for p in range(P)],
            "sigma_cell": float(sim_samples["sigma_cell"]),
            # Stored under the generic key "alpha" whatever the model calls it,
            # so the downstream analysis stays model-agnostic.
            "alpha": [float(sim_samples[f"{self.alpha_var}[{j+1}]"]) for j in range(J)],
            "delta": [float(sim_samples[f"delta[{k+1}]"]) for k in range(K - 1)],
            "upsilon": [float(sim_samples[f"upsilon[{k+1}]"]) for k in range(K)],
        }
        params["extras"] = {
            name: float(sim_samples[name]) for name in self.extra_scalar_params
        }
        return params

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
        for name in self.extra_scalar_params:
            regression_params.append((name, name))
        regression_params.append(("sigma_cell", "sigma_cell"))

        fig, axes = plt.subplots(1, len(regression_params), figsize=(5 * len(regression_params), 5))
        if len(regression_params) == 1:
            axes = [axes]

        for idx, (stan_name, label) in enumerate(regression_params):
            if stan_name == "gamma0":
                true_vals = [p["gamma0"] for p in all_true_params]
            elif stan_name == "sigma_cell":
                true_vals = [p["sigma_cell"] for p in all_true_params]
            elif stan_name in self.extra_scalar_params:
                true_vals = [p["extras"][stan_name] for p in all_true_params]
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
            stan_name = f"{self.alpha_var}[{j+1}]"
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
            ax.set_xlabel(f"True {self.alpha_var}[{j+1}]")
            ax.set_ylabel(f"Estimated {self.alpha_var}[{j+1}]")
            ax.set_title(f"{self.alpha_var}[{j+1}]\nRMSE={stats['rmse']:.3f} Cov={stats['coverage']:.0%}")

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


def _generate_fixed_eta(config: dict, *, J: int, K: int, R: int) -> dict:
    """Generate reproducible expected utilities, optionally shared across cells."""
    utility_values = list(config.get("utility_values", np.linspace(0, 1, K)))
    if len(utility_values) != K:
        raise ValueError(f"fixed_eta utility_values must have length K={K}")

    assessment_files = config.get("assessment_files")
    if assessment_files is not None:
        if len(assessment_files) != J:
            raise ValueError(f"fixed_eta assessment_files must have length J={J}")
        pool = json.loads(open(config["pool_path"]).read())
        item_ids = [item["id"] for item in pool["items"]]
        if len(item_ids) != R:
            raise ValueError(f"fixed_eta pool has {len(item_ids)} items; expected R={R}")
        eta = []
        for assessment_file in assessment_files:
            payload = json.loads(open(assessment_file).read())
            probabilities = {
                record["item_id"]: record["probabilities"]
                for record in payload["assessments"]
                if record.get("parse_ok") and record.get("probabilities") is not None
            }
            if set(probabilities) != set(item_ids):
                raise ValueError(
                    f"Assessment items do not match pool items: {assessment_file}"
                )
            eta.append(
                [
                    float(np.asarray(probabilities[item_id]) @ utility_values)
                    for item_id in item_ids
                ]
            )
        return {"eta": eta, "utility_values": utility_values}

    row_groups = list(config.get("row_groups", range(J)))
    if len(row_groups) != J:
        raise ValueError(f"fixed_eta row_groups must have length J={J}")
    group_order = list(dict.fromkeys(row_groups))
    group_index = {group: index for index, group in enumerate(group_order)}

    distribution = config.get("distribution", "beta")
    rng = np.random.default_rng(config.get("seed", 20260907))
    if distribution == "beta":
        shape1 = float(config.get("shape1", 2.0))
        shape2 = float(config.get("shape2", 2.0))
        if shape1 <= 0 or shape2 <= 0:
            raise ValueError("fixed_eta beta shape parameters must be positive")
        group_eta = rng.beta(shape1, shape2, size=(len(group_order), R))
    elif distribution == "uniform":
        group_eta = rng.uniform(0.0, 1.0, size=(len(group_order), R))
    else:
        raise ValueError(f"Unsupported fixed_eta distribution {distribution!r}")

    eta = np.asarray([group_eta[group_index[group]] for group in row_groups])
    return {"eta": eta.tolist(), "utility_values": utility_values}

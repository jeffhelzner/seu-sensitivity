"""
Prior Predictive Analysis for hierarchical SEU sensitivity models.

Mirrors ``analysis/prior_predictive.py`` but is built around
``HierarchicalStudyDesign`` and the ``h_m01_sim.stan`` simulator, whose
generated quantities include regression parameters (``gamma0``, ``gamma[P]``,
``sigma_cell``), cell-level sensitivities (``log_alpha[J]``, ``alpha[J]``),
per-cell feature weights (``beta[J][K,D]``), shared utilities
(``delta[K-1]``, ``upsilon[K]``), and stacked choices (``y[M_total]``).

The analysis:
1. Loads/generates a ``HierarchicalStudyDesign``.
2. Samples ``n_param_samples`` parameter configurations from the prior by
   invoking ``h_m01_sim.stan`` with ``fixed_param=True``, ``n_choice_samples``
   draws per configuration.
3. Produces plots/summaries for regression params, per-cell alphas, shared
   utilities, and the cell-level choice distributions.

Examples
--------
    from utils.study_design_hierarchical import HierarchicalStudyDesign
    from analysis.hierarchical_prior_predictive import (
        HierarchicalPriorPredictiveAnalysis,
    )

    design = HierarchicalStudyDesign(J=6, K=3, D=2, R=10, P=2, M_per_cell=20)
    design.generate()
    analysis = HierarchicalPriorPredictiveAnalysis(
        study_design=design,
        output_dir="results/prior_predictive/h_m01_prior_analysis",
        n_param_samples=200,
        n_choice_samples=5,
    )
    analysis.run()
"""
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design_hierarchical import HierarchicalStudyDesign


# Default hyperparameter values for the h_m01_sim prior. These mirror the
# defaults used elsewhere in the hierarchical pipeline (see
# HierarchicalStudyDesign.get_data_dict).
DEFAULT_H_M01_HYPERPARAMS = {
    "gamma0_mean": 2.5,
    "gamma0_sd": 0.5,
    "gamma_sd": 0.5,
    "sigma_cell_sd": 0.3,
    "beta_sd": 1.0,
}


class HierarchicalPriorPredictiveAnalysis:
    """
    Prior predictive analysis for ``h_m01``.

    Attributes
    ----------
    sim_model_path : str
        Path to the hierarchical simulation Stan model (``h_m01_sim.stan``).
    study_design : HierarchicalStudyDesign
        Study design providing ``J, K, D, R, P, M_per_cell, X`` and the
        stacked ``cell`` / ``I`` arrays.
    output_dir : str
        Directory where plots and summaries are written.
    n_param_samples : int
        Number of independent prior draws of model parameters.
    n_choice_samples : int
        Number of choice realisations per parameter draw.
    hyperparams : dict
        Override values for the simulator's prior hyperparameters
        (``gamma0_mean``, ``gamma0_sd``, ``gamma_sd``, ``sigma_cell_sd``,
        ``beta_sd``).
    samples : pandas.DataFrame
        Combined ``draws_pd()`` output across all parameter draws, with an
        added ``param_set`` column.
    """

    def __init__(
        self,
        sim_model_path: str = None,
        study_design: HierarchicalStudyDesign = None,
        output_dir: str = None,
        n_param_samples: int = 100,
        n_choice_samples: int = 5,
        hyperparams: dict = None,
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if sim_model_path is None:
            sim_model_path = os.path.join(project_root, "models", "h_m01_sim.stan")

        self.sim_model_path = sim_model_path
        self.study_design = study_design
        self.n_param_samples = n_param_samples
        self.n_choice_samples = n_choice_samples
        self.hyperparams = {**DEFAULT_H_M01_HYPERPARAMS, **(hyperparams or {})}

        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                project_root,
                "results",
                "prior_predictive",
                f"h_m01_prior_analysis_{timestamp}",
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = CmdStanModel(stan_file=self.sim_model_path)
        self.samples: pd.DataFrame | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> pd.DataFrame:
        """Generate prior samples and run all downstream analyses."""
        if self.study_design is None:
            self.study_design = HierarchicalStudyDesign()
            self.study_design.generate()

        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)

        # Build Stan data dict, overriding hyperparams with user values.
        data = self.study_design.get_data_dict()
        for key, val in self.hyperparams.items():
            data[key] = val

        # Persist the hyperparams actually used.
        with open(os.path.join(self.output_dir, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparams, f, indent=2)

        all_samples = []
        print(
            f"Generating {self.n_param_samples} prior samples from h_m01_sim "
            f"({self.n_choice_samples} choice draws each)..."
        )
        for i in range(self.n_param_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{self.n_param_samples}")

            fit = self.model.sample(
                data=data,
                seed=12345 + i,
                iter_sampling=self.n_choice_samples,
                iter_warmup=0,
                chains=1,
                fixed_param=True,
                adapt_engaged=False,
            )

            draws = fit.draws_pd()
            draws["param_set"] = i
            all_samples.append(draws)

        self.samples = pd.concat(all_samples, ignore_index=True)
        self.samples.to_csv(
            os.path.join(self.output_dir, "prior_samples.csv"), index=False
        )

        self._analyze_regression_parameters()
        self._analyze_cell_alphas()
        self._analyze_shared_utilities()
        self._analyze_betas()
        self._analyze_choices()
        self._analyze_seu_maximizer_selection()
        self._write_summary()

        print(f"Prior predictive analysis complete. Results in {self.output_dir}")
        return self.samples

    # ------------------------------------------------------ regression params
    def _analyze_regression_parameters(self):
        """Plot priors on gamma0, gamma[p], and sigma_cell."""
        reg_dir = os.path.join(self.output_dir, "regression_parameters")
        os.makedirs(reg_dir, exist_ok=True)

        P = self.study_design.P

        # Gamma0
        self._plot_hist(
            self.samples["gamma0"],
            title="Prior Distribution of gamma0 (log-alpha intercept)",
            xlabel="gamma0",
            filepath=os.path.join(reg_dir, "gamma0_dist.png"),
        )

        # gamma[p] — one subplot per predictor
        fig, axes = plt.subplots(1, P, figsize=(5 * P, 4), squeeze=False)
        for p in range(P):
            col = f"gamma[{p+1}]"
            if col in self.samples.columns:
                vals = self.samples[col]
                axes[0][p].hist(vals, bins=30, alpha=0.7)
                axes[0][p].axvline(
                    np.median(vals),
                    color="red",
                    linestyle="--",
                    label=f"Median: {np.median(vals):.2f}",
                )
                axes[0][p].set_title(f"gamma[{p+1}]")
                axes[0][p].set_xlabel(f"gamma[{p+1}]")
                axes[0][p].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(reg_dir, "gamma_dist.png"))
        plt.close()

        # sigma_cell
        self._plot_hist(
            self.samples["sigma_cell"],
            title="Prior Distribution of sigma_cell (between-cell SD of log-alpha)",
            xlabel="sigma_cell",
            filepath=os.path.join(reg_dir, "sigma_cell_dist.png"),
        )

    # ---------------------------------------------------------- cell alphas
    def _analyze_cell_alphas(self):
        """Plot the prior distribution of alpha[j] for each cell j."""
        alpha_dir = os.path.join(self.output_dir, "cell_alphas")
        os.makedirs(alpha_dir, exist_ok=True)

        J = self.study_design.J

        ncols = min(J, 3)
        nrows = (J + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )

        alpha_summary = {}
        for j in range(J):
            col = f"alpha[{j+1}]"
            ax = axes[j // ncols][j % ncols]
            if col not in self.samples.columns:
                ax.set_visible(False)
                continue

            vals = self.samples[col]
            # Clip extreme values for visualisation; summary stats use raw.
            q_hi = np.quantile(vals, 0.99)
            ax.hist(vals.clip(upper=q_hi), bins=40, alpha=0.7)
            ax.axvline(
                np.median(vals),
                color="red",
                linestyle="--",
                label=f"Median: {np.median(vals):.2f}",
            )
            ax.set_title(f"alpha[{j+1}] (cell {j+1})")
            ax.set_xlabel("alpha")
            ax.legend()

            alpha_summary[col] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
            }

        # Hide unused axes
        for idx in range(J, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(alpha_dir, "alpha_by_cell.png"))
        plt.close()

        # log_alpha boxplot for between-cell variability
        log_alpha_cols = [f"log_alpha[{j+1}]" for j in range(J)]
        present = [c for c in log_alpha_cols if c in self.samples.columns]
        if present:
            plt.figure(figsize=(max(6, 1.1 * len(present)), 5))
            plt.boxplot(
                [self.samples[c] for c in present],
                labels=[c.replace("log_alpha", "cell ").replace("[", "").replace("]", "") for c in present],
                showfliers=False,
            )
            plt.ylabel("log_alpha")
            plt.title("Prior distribution of log_alpha across cells")
            plt.tight_layout()
            plt.savefig(os.path.join(alpha_dir, "log_alpha_boxplot.png"))
            plt.close()

        with open(os.path.join(alpha_dir, "alpha_summary.json"), "w") as f:
            json.dump(alpha_summary, f, indent=2)

    # ---------------------------------------------------- shared utilities
    def _analyze_shared_utilities(self):
        """Plot prior for delta (simplex) and upsilon (cumulative utility)."""
        util_dir = os.path.join(self.output_dir, "utilities")
        os.makedirs(util_dir, exist_ok=True)

        delta_cols = [c for c in self.samples.columns if c.startswith("delta[")]
        if delta_cols:
            fig, axes = plt.subplots(
                1, len(delta_cols), figsize=(4 * len(delta_cols), 4), squeeze=False
            )
            for i, col in enumerate(delta_cols):
                vals = self.samples[col]
                axes[0][i].hist(vals, bins=30, alpha=0.7)
                axes[0][i].set_title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(util_dir, "delta_dist.png"))
            plt.close()

        upsilon_cols = [c for c in self.samples.columns if c.startswith("upsilon[")]
        if upsilon_cols:
            fig, axes = plt.subplots(
                1, len(upsilon_cols), figsize=(4 * len(upsilon_cols), 4), squeeze=False
            )
            for i, col in enumerate(upsilon_cols):
                vals = self.samples[col]
                axes[0][i].hist(vals, bins=30, alpha=0.7)
                axes[0][i].set_title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(util_dir, "upsilon_dist.png"))
            plt.close()

    # ------------------------------------------------------------ betas
    def _analyze_betas(self):
        """Aggregate view of beta[j][k,d] across cells."""
        beta_dir = os.path.join(self.output_dir, "betas")
        os.makedirs(beta_dir, exist_ok=True)

        J, K, D = self.study_design.J, self.study_design.K, self.study_design.D

        # In cmdstanpy draws, beta shows up as `beta[j,k,d]` for hierarchical
        # array[J] matrix[K,D] declaration.
        fig, axes = plt.subplots(K, D, figsize=(4 * D, 3 * K), squeeze=False)
        for k in range(K):
            for d in range(D):
                ax = axes[k][d]
                # Stack across all cells
                cell_vals = []
                for j in range(J):
                    col = f"beta[{j+1},{k+1},{d+1}]"
                    if col in self.samples.columns:
                        cell_vals.append(self.samples[col].to_numpy())
                if not cell_vals:
                    ax.set_visible(False)
                    continue
                stacked = np.concatenate(cell_vals)
                ax.hist(stacked, bins=40, alpha=0.7)
                ax.set_title(f"beta[·,{k+1},{d+1}] (all cells)")
        plt.tight_layout()
        plt.savefig(os.path.join(beta_dir, "beta_dist.png"))
        plt.close()

    # ------------------------------------------------------------ choices
    def _analyze_choices(self):
        """Per-cell choice distributions and overall alternative frequencies."""
        choices_dir = os.path.join(self.output_dir, "choices")
        os.makedirs(choices_dir, exist_ok=True)

        J = self.study_design.J
        R = self.study_design.R
        M_total = self.study_design.M_total
        cell = np.asarray(self.study_design.cell)  # 1-indexed cell per observation

        y_cols = [f"y[{m+1}]" for m in range(M_total) if f"y[{m+1}]" in self.samples.columns]
        if not y_cols:
            print("  [choices] No y[...] columns found — skipping choice analysis.")
            return

        # Build a long DataFrame: (draw, obs_index, cell, y)
        # Note: each observation may offer a different subset of R alternatives;
        # y[m] is an index 1..N_obs[m] into the alternatives available for that
        # observation. We record the raw index for cell-level summaries.
        Y = self.samples[y_cols].to_numpy()  # (n_draws, M_total)

        # Per-cell distribution of the raw y index (truncated at max observed).
        max_idx = int(np.nanmax(Y)) if Y.size else 0
        idx_range = np.arange(1, max(max_idx + 1, 2))

        ncols = min(J, 3)
        nrows = (J + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )

        cell_summary = {}
        for j in range(J):
            cell_id = j + 1
            obs_mask = cell == cell_id
            if not np.any(obs_mask):
                axes[j // ncols][j % ncols].set_visible(False)
                continue

            cell_y = Y[:, obs_mask].ravel()
            counts = np.bincount(cell_y.astype(int), minlength=idx_range.max() + 1)[
                idx_range
            ]
            total = counts.sum()
            probs = counts / total if total > 0 else counts

            ax = axes[j // ncols][j % ncols]
            ax.bar(idx_range, probs, alpha=0.8, edgecolor="black")
            ax.set_title(f"Cell {cell_id}: choice index distribution")
            ax.set_xlabel("Alternative index within problem")
            ax.set_ylabel("Empirical probability")
            ax.set_ylim(0, 1)

            cell_summary[f"cell_{cell_id}"] = {
                "n_observations": int(obs_mask.sum()),
                "n_draws": int(cell_y.size),
                "choice_index_probs": {
                    int(i): float(p) for i, p in zip(idx_range, probs)
                },
            }

        for idx in range(J, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(choices_dir, "choice_index_by_cell.png"), dpi=150)
        plt.close()

        # Global distribution of which underlying alternative (1..R) was chosen.
        # Resolve the chosen alternative via the stacked indicator matrix I.
        I = np.asarray(self.study_design.I)  # (M_total, R)
        chosen_alt_counts = np.zeros(R, dtype=np.int64)
        for m in range(M_total):
            col = f"y[{m+1}]"
            if col not in self.samples.columns:
                continue
            alts = np.where(I[m] == 1)[0]  # 0-indexed
            if alts.size == 0:
                continue
            y_vals = self.samples[col].to_numpy().astype(int) - 1
            # Clamp defensively
            y_vals = np.clip(y_vals, 0, alts.size - 1)
            picks = alts[y_vals]
            unique, counts = np.unique(picks, return_counts=True)
            chosen_alt_counts[unique] += counts

        total = chosen_alt_counts.sum()
        if total > 0:
            probs = chosen_alt_counts / total
            plt.figure(figsize=(max(6, 0.4 * R), 5))
            plt.bar(np.arange(1, R + 1), probs, alpha=0.8, edgecolor="black")
            plt.xlabel("Alternative (1..R)")
            plt.ylabel("Prior probability of being chosen")
            plt.title("Prior distribution over chosen alternatives (aggregated)")
            plt.tight_layout()
            plt.savefig(os.path.join(choices_dir, "chosen_alternative_dist.png"), dpi=150)
            plt.close()

            cell_summary["__global__"] = {
                "chosen_alternative_probs": {
                    int(r + 1): float(p) for r, p in enumerate(probs)
                }
            }

        with open(os.path.join(choices_dir, "choice_summary.json"), "w") as f:
            json.dump(cell_summary, f, indent=2)

    # --------------------------------------------- seu maximizer selection
    def _analyze_seu_maximizer_selection(self):
        """
        Analyze the selection of SEU maximizers across decision problems,
        overall and broken down by experimental cell.

        Uses ``selected_seu_max[m]`` and ``total_seu_max_selected`` (+
        per-cell totals ``seu_max_by_cell[j]``) emitted by h_m01_sim.stan.

        Outputs (under ``seu_maximizer_selection/``):
          - ``prob_seu_max_by_problem.png``: per-problem selection probability
          - ``prob_seu_max_by_cell.png``: mean selection probability per cell
          - ``total_seu_max_distribution.png``: overall distribution
          - ``total_seu_max_by_cell.png``: per-cell distributions
          - ``seu_maximizer_summary.json``: numeric summary
        """
        seu_dir = os.path.join(self.output_dir, "seu_maximizer_selection")
        os.makedirs(seu_dir, exist_ok=True)

        M_total = self.study_design.M_total
        J = self.study_design.J
        cell = np.asarray(self.study_design.cell)  # 1-indexed

        # Per-problem probability of SEU-max selection
        prob_by_problem: dict[int, float] = {}
        for m in range(1, M_total + 1):
            col = f"selected_seu_max[{m}]"
            if col in self.samples.columns:
                prob_by_problem[m] = float(self.samples[col].mean())

        if not prob_by_problem:
            print(
                "  [seu_max] No selected_seu_max[...] columns found — "
                "skipping SEU maximizer analysis."
            )
            return

        # Plot: probability by problem, coloured by cell
        plt.figure(figsize=(max(8, 0.2 * M_total), 5))
        problems = np.array(sorted(prob_by_problem.keys()))
        probs = np.array([prob_by_problem[m] for m in problems])
        cell_ids = cell[problems - 1]

        cmap = plt.get_cmap("tab10")
        for j in range(1, J + 1):
            mask = cell_ids == j
            if np.any(mask):
                plt.bar(
                    problems[mask],
                    probs[mask],
                    alpha=0.8,
                    edgecolor="black",
                    color=cmap((j - 1) % 10),
                    label=f"Cell {j}",
                )
        plt.axhline(
            np.mean(probs),
            color="red",
            linestyle="--",
            label=f"Overall mean: {np.mean(probs):.3f}",
        )
        plt.xlabel("Decision problem")
        plt.ylabel("Prob. of selecting SEU maximizer")
        plt.title("Prior prob. of SEU-maximizer selection by problem")
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(seu_dir, "prob_seu_max_by_problem.png"), dpi=150
        )
        plt.close()

        # Per-cell mean selection probability (averaged across problems in cell)
        cell_prob_mean: dict[int, float] = {}
        cell_prob_std: dict[int, float] = {}
        for j in range(1, J + 1):
            mask = cell_ids == j
            if np.any(mask):
                cell_prob_mean[j] = float(np.mean(probs[mask]))
                cell_prob_std[j] = float(np.std(probs[mask]))

        plt.figure(figsize=(max(6, 1.1 * J), 5))
        xs = sorted(cell_prob_mean.keys())
        ys = [cell_prob_mean[j] for j in xs]
        yerrs = [cell_prob_std[j] for j in xs]
        plt.bar(xs, ys, yerr=yerrs, capsize=5, alpha=0.8, edgecolor="black")
        plt.axhline(
            np.mean(probs),
            color="red",
            linestyle="--",
            label=f"Overall mean: {np.mean(probs):.3f}",
        )
        plt.xlabel("Cell")
        plt.ylabel("Mean prob. of SEU-max selection")
        plt.title("Prior prob. of SEU-maximizer selection by cell")
        plt.xticks(xs)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(seu_dir, "prob_seu_max_by_cell.png"), dpi=150)
        plt.close()

        # Overall distribution of total_seu_max_selected
        overall_summary = {}
        if "total_seu_max_selected" in self.samples.columns:
            total = self.samples["total_seu_max_selected"].astype(float)
            plt.figure(figsize=(10, 6))
            plt.hist(
                total,
                bins=range(0, M_total + 2),
                align="left",
                rwidth=0.8,
                alpha=0.7,
                edgecolor="black",
            )
            plt.axvline(
                total.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {total.mean():.2f}",
            )
            plt.axvline(
                total.median(),
                color="blue",
                linestyle="--",
                label=f"Median: {total.median():.0f}",
            )
            plt.xlabel(
                f"# problems where SEU maximizer selected (out of {M_total})"
            )
            plt.ylabel("Frequency")
            plt.title("Prior distribution of total SEU maximizers selected")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(seu_dir, "total_seu_max_distribution.png"),
                dpi=150,
            )
            plt.close()

            overall_summary = {
                "mean": float(total.mean()),
                "std": float(total.std()),
                "median": float(total.median()),
                "min": int(total.min()),
                "max": int(total.max()),
                "q25": float(total.quantile(0.25)),
                "q75": float(total.quantile(0.75)),
            }

        # Per-cell distributions (seu_max_by_cell[j])
        cell_total_summary: dict[int, dict] = {}
        cell_cols = [
            f"seu_max_by_cell[{j}]"
            for j in range(1, J + 1)
            if f"seu_max_by_cell[{j}]" in self.samples.columns
        ]
        if cell_cols:
            ncols = min(J, 3)
            nrows = (J + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )
            for j in range(1, J + 1):
                col = f"seu_max_by_cell[{j}]"
                ax = axes[(j - 1) // ncols][(j - 1) % ncols]
                if col not in self.samples.columns:
                    ax.set_visible(False)
                    continue
                vals = self.samples[col].astype(float)
                M_j = int(np.sum(cell == j))
                ax.hist(
                    vals,
                    bins=range(0, M_j + 2),
                    align="left",
                    rwidth=0.8,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.axvline(
                    vals.mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {vals.mean():.2f}",
                )
                ax.set_title(f"Cell {j} (M_j = {M_j})")
                ax.set_xlabel("# SEU-max selections")
                ax.set_ylabel("Frequency")
                ax.legend(fontsize=8)
                cell_total_summary[j] = {
                    "M_j": M_j,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "mean_prob": float(vals.mean() / M_j) if M_j > 0 else None,
                }
            for idx in range(J, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(seu_dir, "total_seu_max_by_cell.png"), dpi=150
            )
            plt.close()

        # JSON summary
        summary = {
            "total_problems": M_total,
            "overall_prob_seu_max": float(np.mean(probs)),
            "prob_seu_max_by_problem": {
                int(m): float(p) for m, p in prob_by_problem.items()
            },
            "prob_seu_max_by_cell": {
                int(j): {
                    "mean": cell_prob_mean[j],
                    "std": cell_prob_std[j],
                }
                for j in cell_prob_mean
            },
            "total_seu_max_selected": overall_summary,
            "seu_max_by_cell": cell_total_summary,
        }
        with open(os.path.join(seu_dir, "seu_maximizer_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Console summary
        print("\n" + "=" * 60)
        print("SEU MAXIMIZER SELECTION SUMMARY (h_m01 prior predictive)")
        print("=" * 60)
        print(f"Total problems: {M_total}")
        print(
            f"Overall prob. of SEU-max selection: "
            f"{summary['overall_prob_seu_max']:.3f}"
        )
        if overall_summary:
            print(
                f"Total SEU max selected: mean={overall_summary['mean']:.2f}, "
                f"median={overall_summary['median']:.0f}, "
                f"range=[{overall_summary['min']}, {overall_summary['max']}]"
            )
        for j in sorted(cell_total_summary.keys()):
            s = cell_total_summary[j]
            mp = s["mean_prob"]
            mp_str = f"{mp:.3f}" if mp is not None else "NA"
            print(
                f"  Cell {j}: mean={s['mean']:.2f} / M_j={s['M_j']}  "
                f"(mean prob={mp_str})"
            )
        print("=" * 60 + "\n")

    # --------------------------------------------------------------- summary
    def _write_summary(self):
        """Write a top-level summary JSON with quantiles for key parameters."""
        summary = {
            "n_param_samples": self.n_param_samples,
            "n_choice_samples": self.n_choice_samples,
            "J": self.study_design.J,
            "K": self.study_design.K,
            "D": self.study_design.D,
            "R": self.study_design.R,
            "P": self.study_design.P,
            "M_total": self.study_design.M_total,
            "hyperparameters": self.hyperparams,
        }

        key_cols = ["gamma0", "sigma_cell"]
        key_cols += [f"gamma[{p+1}]" for p in range(self.study_design.P)]
        key_cols += [f"alpha[{j+1}]" for j in range(self.study_design.J)]
        key_cols += [f"delta[{k+1}]" for k in range(self.study_design.K - 1)]
        key_cols += [f"upsilon[{k+1}]" for k in range(self.study_design.K)]

        param_summary = {}
        for col in key_cols:
            if col in self.samples.columns:
                vals = self.samples[col]
                param_summary[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "q05": float(np.quantile(vals, 0.05)),
                    "q95": float(np.quantile(vals, 0.95)),
                }
        summary["parameter_summary"] = param_summary

        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _plot_hist(values, title, xlabel, filepath, bins=30):
        vals = np.asarray(values)
        plt.figure(figsize=(10, 6))
        plt.hist(vals, bins=bins, alpha=0.7)
        plt.axvline(
            np.median(vals),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(vals):.2f}",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    analysis = HierarchicalPriorPredictiveAnalysis(
        n_param_samples=50, n_choice_samples=5
    )
    analysis.run()

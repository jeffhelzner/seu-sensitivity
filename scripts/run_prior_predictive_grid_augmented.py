"""
Prior Predictive Grid Search for Augmented Models (m_1, m_2, m_3)

Sweeps over grids of lognormal prior hyperparameters for the sensitivity
parameters in each augmented model, using the corresponding _sim.stan
simulation model and the actual augmented study design (w, I, x, J).

For each grid point the script draws n_param_samples parameter sets from
the prior, simulates choices, and reports:

  SEU-max rate (uncertain) = mean(total_seu_max_selected_uncertain) / M
  SEU-max rate (risky)     = mean(total_seu_max_selected_risky)     / N
  SEU-max rate (combined)  = mean(total_seu_max_selected)           / (M+N)

This gives the prior-implied probability that the agent selects the
alternative with the highest subjective expected utility — a directly
interpretable behavioural diagnostic, separately for each decision context.

Usage:
    python scripts/run_prior_predictive_grid_augmented.py \\
        [--stan-data PATH] [--n-samples N] [--output-dir DIR] \\
        [--models m_1 m_2 m_3]

The script uses m_1_sim.stan, m_2_sim.stan, m_3_sim.stan which accept
prior hyperparameters as data inputs.
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from cmdstanpy import CmdStanModel
import datetime
import logging

# Suppress verbose cmdstanpy logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Grids of candidate priors ───────────────────────────────────────
#
# Recall: if X ~ lognormal(mu, sigma)  then
#   median = exp(mu),  mode = exp(mu - sigma^2)
#   mean   = exp(mu + sigma^2/2)

# Alpha grid — shared across all three models (sensitivity for uncertain choices)
ALPHA_GRID = [
    # --- m_0 baseline ---
    (0.0, 1.0),       # median=1,    95th≈5

    # --- wider tails, same centre ---
    (0.0, 1.5),       # median=1,    95th≈20
    (0.0, 2.0),       # median=1,    95th≈39

    # --- shifted centre ---
    (2.0, 1.0),       # median=7.4,  95th≈38
    (2.5, 1.0),       # median=12.2, 95th≈63
    (3.0, 0.75),      # median=20.1, 95th≈68   ← m_01 prior
    (3.0, 1.0),       # median=20.1, 95th≈104
    (3.5, 0.5),       # median=33.1, 95th≈72
    (3.5, 0.75),      # median=33.1, 95th≈112
    (3.5, 1.0),       # median=33.1, 95th≈172
    (4.0, 0.5),       # median=54.6, 95th≈119
    (4.0, 0.75),      # median=54.6, 95th≈185
]

# Omega grid — for m_2 (independent sensitivity for risky choices)
OMEGA_GRID = [
    (0.0, 1.0),       # median=1
    (2.0, 1.0),       # median=7.4
    (3.0, 0.75),      # median=20.1
    (3.0, 1.0),       # median=20.1
    (3.5, 0.75),      # median=33.1
    (4.0, 0.5),       # median=54.6
    (4.0, 0.75),      # median=54.6
    (4.5, 0.5),       # median=90.0
    (5.0, 0.5),       # median=148.4
]

# Kappa grid — for m_3 (proportionality parameter: omega = kappa * alpha)
KAPPA_GRID = [
    (0.0, 0.25),      # median=1.0, tight around 1 (~ m_1)
    (0.0, 0.5),       # median=1.0, moderate spread ← m_3 default
    (0.0, 0.75),      # median=1.0, wider
    (0.0, 1.0),       # median=1.0, very wide
    (0.5, 0.25),      # median=1.65, risky slightly more sensitive
    (0.5, 0.5),       # median=1.65, moderate spread
    (-0.5, 0.25),     # median=0.61, risky slightly less sensitive
    (-0.5, 0.5),      # median=0.61, moderate spread
]


def load_augmented_stan_data(path: str) -> dict:
    """Load an augmented stan_data JSON and prepare it for the sim model."""
    with open(path) as f:
        data = json.load(f)
    # The sim models don't need y or z — remove if present
    data.pop("y", None)
    data.pop("z", None)
    return data


def run_grid_point_m1(
    model: CmdStanModel,
    base_data: dict,
    alpha_mean: float,
    alpha_sd: float,
    n_param_samples: int,
    beta_sd: float = 1.0,
) -> dict:
    """
    Run prior predictive simulation for one (alpha_mean, alpha_sd) point
    using the m_1_sim model (shared alpha for uncertain + risky).
    """
    data = dict(base_data)
    data["alpha_mean"] = alpha_mean
    data["alpha_sd"] = alpha_sd
    data["beta_sd"] = beta_sd

    M = data["M"]
    N = data["N"]

    try:
        fit = model.sample(
            data=data,
            seed=12345,
            iter_sampling=n_param_samples,
            iter_warmup=0,
            chains=1,
            fixed_param=True,
            adapt_engaged=False,
            show_progress=False,
        )
        draws = fit.draws_pd()
        alphas = draws["alpha"].values.astype(float)
        seu_uncertain = draws["total_seu_max_selected_uncertain"].values.astype(float)
        seu_risky = draws["total_seu_max_selected_risky"].values.astype(float)
        seu_total = draws["total_seu_max_selected"].values.astype(float)

        # Filter out NaN rows (softmax overflow)
        valid = ~(np.isnan(alphas) | np.isnan(seu_uncertain) | np.isnan(seu_risky))
        n_failed = int((~valid).sum())
        alphas = alphas[valid]
        seu_rates_uncertain = seu_uncertain[valid] / M
        seu_rates_risky = seu_risky[valid] / N
        seu_rates_total = seu_total[valid] / (M + N)
    except RuntimeError:
        return _failed_result_m1(alpha_mean, alpha_sd, n_param_samples)

    n_ok = len(alphas)
    if n_ok == 0:
        return _failed_result_m1(alpha_mean, alpha_sd, n_ok + n_failed)

    return {
        "model": "m_1",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "prior_label": f"lognormal({alpha_mean}, {alpha_sd})",
        "alpha_median": float(np.median(alphas)),
        "alpha_q05": float(np.quantile(alphas, 0.05)),
        "alpha_q95": float(np.quantile(alphas, 0.95)),
        "seu_rate_uncertain_mean": float(np.mean(seu_rates_uncertain)),
        "seu_rate_uncertain_sd": float(np.std(seu_rates_uncertain)),
        "seu_rate_uncertain_q05": float(np.quantile(seu_rates_uncertain, 0.05)),
        "seu_rate_uncertain_q95": float(np.quantile(seu_rates_uncertain, 0.95)),
        "seu_rate_risky_mean": float(np.mean(seu_rates_risky)),
        "seu_rate_risky_sd": float(np.std(seu_rates_risky)),
        "seu_rate_risky_q05": float(np.quantile(seu_rates_risky, 0.05)),
        "seu_rate_risky_q95": float(np.quantile(seu_rates_risky, 0.95)),
        "seu_rate_combined_mean": float(np.mean(seu_rates_total)),
        "seu_rate_combined_sd": float(np.std(seu_rates_total)),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "fail_rate": n_failed / (n_ok + n_failed),
    }


def _failed_result_m1(alpha_mean, alpha_sd, n_failed):
    return {
        "model": "m_1",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "prior_label": f"lognormal({alpha_mean}, {alpha_sd})",
        "alpha_median": float("nan"),
        "alpha_q05": float("nan"),
        "alpha_q95": float("nan"),
        "seu_rate_uncertain_mean": float("nan"),
        "seu_rate_uncertain_sd": float("nan"),
        "seu_rate_uncertain_q05": float("nan"),
        "seu_rate_uncertain_q95": float("nan"),
        "seu_rate_risky_mean": float("nan"),
        "seu_rate_risky_sd": float("nan"),
        "seu_rate_risky_q05": float("nan"),
        "seu_rate_risky_q95": float("nan"),
        "seu_rate_combined_mean": float("nan"),
        "seu_rate_combined_sd": float("nan"),
        "n_ok": 0,
        "n_failed": n_failed,
        "fail_rate": 1.0,
    }


def run_grid_point_m2(
    model: CmdStanModel,
    base_data: dict,
    alpha_mean: float,
    alpha_sd: float,
    omega_mean: float,
    omega_sd: float,
    n_param_samples: int,
    beta_sd: float = 1.0,
) -> dict:
    """
    Run prior predictive simulation for one
    (alpha_mean, alpha_sd, omega_mean, omega_sd) point
    using the m_2_sim model (separate alpha and omega).
    """
    data = dict(base_data)
    data["alpha_mean"] = alpha_mean
    data["alpha_sd"] = alpha_sd
    data["omega_mean"] = omega_mean
    data["omega_sd"] = omega_sd
    data["beta_sd"] = beta_sd

    M = data["M"]
    N = data["N"]

    try:
        fit = model.sample(
            data=data,
            seed=12345,
            iter_sampling=n_param_samples,
            iter_warmup=0,
            chains=1,
            fixed_param=True,
            adapt_engaged=False,
            show_progress=False,
        )
        draws = fit.draws_pd()
        alphas = draws["alpha"].values.astype(float)
        omegas = draws["omega"].values.astype(float)
        seu_uncertain = draws["total_seu_max_selected_uncertain"].values.astype(float)
        seu_risky = draws["total_seu_max_selected_risky"].values.astype(float)
        seu_total = draws["total_seu_max_selected"].values.astype(float)

        valid = ~(np.isnan(alphas) | np.isnan(omegas) |
                  np.isnan(seu_uncertain) | np.isnan(seu_risky))
        n_failed = int((~valid).sum())
        alphas = alphas[valid]
        omegas = omegas[valid]
        seu_rates_uncertain = seu_uncertain[valid] / M
        seu_rates_risky = seu_risky[valid] / N
        seu_rates_total = seu_total[valid] / (M + N)
    except RuntimeError:
        return _failed_result_m2(alpha_mean, alpha_sd, omega_mean, omega_sd, n_param_samples)

    n_ok = len(alphas)
    if n_ok == 0:
        return _failed_result_m2(alpha_mean, alpha_sd, omega_mean, omega_sd, n_ok + n_failed)

    return {
        "model": "m_2",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "omega_mean": omega_mean,
        "omega_sd": omega_sd,
        "prior_label": f"α~LN({alpha_mean},{alpha_sd}), ω~LN({omega_mean},{omega_sd})",
        "alpha_median": float(np.median(alphas)),
        "alpha_q05": float(np.quantile(alphas, 0.05)),
        "alpha_q95": float(np.quantile(alphas, 0.95)),
        "omega_median": float(np.median(omegas)),
        "omega_q05": float(np.quantile(omegas, 0.05)),
        "omega_q95": float(np.quantile(omegas, 0.95)),
        "seu_rate_uncertain_mean": float(np.mean(seu_rates_uncertain)),
        "seu_rate_uncertain_sd": float(np.std(seu_rates_uncertain)),
        "seu_rate_uncertain_q05": float(np.quantile(seu_rates_uncertain, 0.05)),
        "seu_rate_uncertain_q95": float(np.quantile(seu_rates_uncertain, 0.95)),
        "seu_rate_risky_mean": float(np.mean(seu_rates_risky)),
        "seu_rate_risky_sd": float(np.std(seu_rates_risky)),
        "seu_rate_risky_q05": float(np.quantile(seu_rates_risky, 0.05)),
        "seu_rate_risky_q95": float(np.quantile(seu_rates_risky, 0.95)),
        "seu_rate_combined_mean": float(np.mean(seu_rates_total)),
        "seu_rate_combined_sd": float(np.std(seu_rates_total)),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "fail_rate": n_failed / (n_ok + n_failed),
    }


def _failed_result_m2(alpha_mean, alpha_sd, omega_mean, omega_sd, n_failed):
    return {
        "model": "m_2",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "omega_mean": omega_mean,
        "omega_sd": omega_sd,
        "prior_label": f"α~LN({alpha_mean},{alpha_sd}), ω~LN({omega_mean},{omega_sd})",
        "alpha_median": float("nan"),
        "alpha_q05": float("nan"),
        "alpha_q95": float("nan"),
        "omega_median": float("nan"),
        "omega_q05": float("nan"),
        "omega_q95": float("nan"),
        "seu_rate_uncertain_mean": float("nan"),
        "seu_rate_uncertain_sd": float("nan"),
        "seu_rate_uncertain_q05": float("nan"),
        "seu_rate_uncertain_q95": float("nan"),
        "seu_rate_risky_mean": float("nan"),
        "seu_rate_risky_sd": float("nan"),
        "seu_rate_risky_q05": float("nan"),
        "seu_rate_risky_q95": float("nan"),
        "seu_rate_combined_mean": float("nan"),
        "seu_rate_combined_sd": float("nan"),
        "n_ok": 0,
        "n_failed": n_failed,
        "fail_rate": 1.0,
    }


def run_grid_point_m3(
    model: CmdStanModel,
    base_data: dict,
    alpha_mean: float,
    alpha_sd: float,
    kappa_mean: float,
    kappa_sd: float,
    n_param_samples: int,
    beta_sd: float = 1.0,
) -> dict:
    """
    Run prior predictive simulation for one
    (alpha_mean, alpha_sd, kappa_mean, kappa_sd) point
    using the m_3_sim model (omega = kappa * alpha).
    """
    data = dict(base_data)
    data["alpha_mean"] = alpha_mean
    data["alpha_sd"] = alpha_sd
    data["kappa_mean"] = kappa_mean
    data["kappa_sd"] = kappa_sd
    data["beta_sd"] = beta_sd

    M = data["M"]
    N = data["N"]

    try:
        fit = model.sample(
            data=data,
            seed=12345,
            iter_sampling=n_param_samples,
            iter_warmup=0,
            chains=1,
            fixed_param=True,
            adapt_engaged=False,
            show_progress=False,
        )
        draws = fit.draws_pd()
        alphas = draws["alpha"].values.astype(float)
        kappas = draws["kappa"].values.astype(float)
        omegas = draws["omega"].values.astype(float)
        seu_uncertain = draws["total_seu_max_selected_uncertain"].values.astype(float)
        seu_risky = draws["total_seu_max_selected_risky"].values.astype(float)
        seu_total = draws["total_seu_max_selected"].values.astype(float)

        valid = ~(np.isnan(alphas) | np.isnan(kappas) | np.isnan(omegas) |
                  np.isnan(seu_uncertain) | np.isnan(seu_risky))
        n_failed = int((~valid).sum())
        alphas = alphas[valid]
        kappas = kappas[valid]
        omegas = omegas[valid]
        seu_rates_uncertain = seu_uncertain[valid] / M
        seu_rates_risky = seu_risky[valid] / N
        seu_rates_total = seu_total[valid] / (M + N)
    except RuntimeError:
        return _failed_result_m3(alpha_mean, alpha_sd, kappa_mean, kappa_sd, n_param_samples)

    n_ok = len(alphas)
    if n_ok == 0:
        return _failed_result_m3(alpha_mean, alpha_sd, kappa_mean, kappa_sd, n_ok + n_failed)

    return {
        "model": "m_3",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "kappa_mean": kappa_mean,
        "kappa_sd": kappa_sd,
        "prior_label": f"α~LN({alpha_mean},{alpha_sd}), κ~LN({kappa_mean},{kappa_sd})",
        "alpha_median": float(np.median(alphas)),
        "alpha_q05": float(np.quantile(alphas, 0.05)),
        "alpha_q95": float(np.quantile(alphas, 0.95)),
        "kappa_median": float(np.median(kappas)),
        "kappa_q05": float(np.quantile(kappas, 0.05)),
        "kappa_q95": float(np.quantile(kappas, 0.95)),
        "omega_median": float(np.median(omegas)),
        "omega_q05": float(np.quantile(omegas, 0.05)),
        "omega_q95": float(np.quantile(omegas, 0.95)),
        "seu_rate_uncertain_mean": float(np.mean(seu_rates_uncertain)),
        "seu_rate_uncertain_sd": float(np.std(seu_rates_uncertain)),
        "seu_rate_uncertain_q05": float(np.quantile(seu_rates_uncertain, 0.05)),
        "seu_rate_uncertain_q95": float(np.quantile(seu_rates_uncertain, 0.95)),
        "seu_rate_risky_mean": float(np.mean(seu_rates_risky)),
        "seu_rate_risky_sd": float(np.std(seu_rates_risky)),
        "seu_rate_risky_q05": float(np.quantile(seu_rates_risky, 0.05)),
        "seu_rate_risky_q95": float(np.quantile(seu_rates_risky, 0.95)),
        "seu_rate_combined_mean": float(np.mean(seu_rates_total)),
        "seu_rate_combined_sd": float(np.std(seu_rates_total)),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "fail_rate": n_failed / (n_ok + n_failed),
    }


def _failed_result_m3(alpha_mean, alpha_sd, kappa_mean, kappa_sd, n_failed):
    return {
        "model": "m_3",
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "kappa_mean": kappa_mean,
        "kappa_sd": kappa_sd,
        "prior_label": f"α~LN({alpha_mean},{alpha_sd}), κ~LN({kappa_mean},{kappa_sd})",
        "alpha_median": float("nan"),
        "alpha_q05": float("nan"),
        "alpha_q95": float("nan"),
        "kappa_median": float("nan"),
        "kappa_q05": float("nan"),
        "kappa_q95": float("nan"),
        "omega_median": float("nan"),
        "omega_q05": float("nan"),
        "omega_q95": float("nan"),
        "seu_rate_uncertain_mean": float("nan"),
        "seu_rate_uncertain_sd": float("nan"),
        "seu_rate_uncertain_q05": float("nan"),
        "seu_rate_uncertain_q95": float("nan"),
        "seu_rate_risky_mean": float("nan"),
        "seu_rate_risky_sd": float("nan"),
        "seu_rate_risky_q05": float("nan"),
        "seu_rate_risky_q95": float("nan"),
        "seu_rate_combined_mean": float("nan"),
        "seu_rate_combined_sd": float("nan"),
        "n_ok": 0,
        "n_failed": n_failed,
        "fail_rate": 1.0,
    }


# ── Display helpers ─────────────────────────────────────────────────

def print_results_table_m1(results: list[dict]) -> str:
    """Format m_1 results as a readable table."""
    lines = []
    header = (
        f"{'Prior α':>24s}  "
        f"{'α med':>7s}  {'α 90%CI':>16s}  "
        f"{'SEU unc':>8s}  {'SEU risk':>8s}  {'SEU comb':>8s}  "
        f"{'fail%':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r["n_ok"] == 0:
            line = f"{r['prior_label']:>24s}  {'  ALL FAILED':>56s}  {r['fail_rate']*100:5.1f}"
        else:
            line = (
                f"{r['prior_label']:>24s}  "
                f"{r['alpha_median']:7.1f}  "
                f"[{r['alpha_q05']:6.1f}, {r['alpha_q95']:6.1f}]  "
                f"{r['seu_rate_uncertain_mean']:8.3f}  "
                f"{r['seu_rate_risky_mean']:8.3f}  "
                f"{r['seu_rate_combined_mean']:8.3f}  "
                f"{r['fail_rate']*100:5.1f}"
            )
        lines.append(line)
    return "\n".join(lines)


def print_results_table_m2(results: list[dict]) -> str:
    """Format m_2 results as a readable table."""
    lines = []
    header = (
        f"{'Prior (α, ω)':>48s}  "
        f"{'α med':>7s}  {'ω med':>7s}  "
        f"{'SEU unc':>8s}  {'SEU risk':>8s}  {'SEU comb':>8s}  "
        f"{'fail%':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r["n_ok"] == 0:
            line = f"{r['prior_label']:>48s}  {'  ALL FAILED':>40s}  {r['fail_rate']*100:5.1f}"
        else:
            line = (
                f"{r['prior_label']:>48s}  "
                f"{r['alpha_median']:7.1f}  "
                f"{r['omega_median']:7.1f}  "
                f"{r['seu_rate_uncertain_mean']:8.3f}  "
                f"{r['seu_rate_risky_mean']:8.3f}  "
                f"{r['seu_rate_combined_mean']:8.3f}  "
                f"{r['fail_rate']*100:5.1f}"
            )
        lines.append(line)
    return "\n".join(lines)


def print_results_table_m3(results: list[dict]) -> str:
    """Format m_3 results as a readable table."""
    lines = []
    header = (
        f"{'Prior (α, κ)':>48s}  "
        f"{'α med':>7s}  {'κ med':>7s}  {'ω med':>7s}  "
        f"{'SEU unc':>8s}  {'SEU risk':>8s}  {'SEU comb':>8s}  "
        f"{'fail%':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r["n_ok"] == 0:
            line = f"{r['prior_label']:>48s}  {'  ALL FAILED':>48s}  {r['fail_rate']*100:5.1f}"
        else:
            line = (
                f"{r['prior_label']:>48s}  "
                f"{r['alpha_median']:7.1f}  "
                f"{r['kappa_median']:7.1f}  "
                f"{r['omega_median']:7.1f}  "
                f"{r['seu_rate_uncertain_mean']:8.3f}  "
                f"{r['seu_rate_risky_mean']:8.3f}  "
                f"{r['seu_rate_combined_mean']:8.3f}  "
                f"{r['fail_rate']*100:5.1f}"
            )
        lines.append(line)
    return "\n".join(lines)


# ── Per-model sweep runners ─────────────────────────────────────────

def sweep_m1(base_data: dict, n_samples: int, output_dir: Path):
    """Grid search for m_1 (shared alpha)."""
    sim_path = str(PROJECT_ROOT / "models" / "m_1_sim.stan")
    print(f"\n{'='*80}")
    print("MODEL m_1  (shared α for uncertain + risky)")
    print(f"{'='*80}")
    print(f"Compiling: {sim_path}")
    model = CmdStanModel(stan_file=sim_path)

    results = []
    n_grid = len(ALPHA_GRID)
    print(f"Running {n_grid} grid points × {n_samples} samples each ...\n")

    for idx, (alpha_mean, alpha_sd) in enumerate(ALPHA_GRID):
        label = f"lognormal({alpha_mean}, {alpha_sd})"
        print(f"  [{idx+1}/{n_grid}] {label} ...", end=" ", flush=True)
        r = run_grid_point_m1(model, base_data, alpha_mean, alpha_sd, n_samples)
        results.append(r)
        if r["n_ok"] == 0:
            print("ALL FAILED")
        else:
            print(
                f"SEU unc={r['seu_rate_uncertain_mean']:.3f}  "
                f"risk={r['seu_rate_risky_mean']:.3f}  "
                f"(α med={r['alpha_median']:.1f})"
                + (f"  [{r['n_failed']} failures]" if r["n_failed"] > 0 else "")
            )

    table = print_results_table_m1(results)
    print(f"\n{table}\n")

    _save_results("m_1", results, table, output_dir, base_data, sim_path, n_samples)
    return results


def sweep_m2(base_data: dict, n_samples: int, output_dir: Path):
    """Grid search for m_2 (separate alpha, omega)."""
    sim_path = str(PROJECT_ROOT / "models" / "m_2_sim.stan")
    print(f"\n{'='*80}")
    print("MODEL m_2  (separate α for uncertain, ω for risky)")
    print(f"{'='*80}")
    print(f"Compiling: {sim_path}")
    model = CmdStanModel(stan_file=sim_path)

    # 2D grid: alpha × omega
    grid = list(itertools.product(ALPHA_GRID, OMEGA_GRID))
    results = []
    n_grid = len(grid)
    print(f"Running {n_grid} grid points × {n_samples} samples each ...\n")

    for idx, ((am, asd), (om, osd)) in enumerate(grid):
        label = f"α~LN({am},{asd}), ω~LN({om},{osd})"
        print(f"  [{idx+1}/{n_grid}] {label} ...", end=" ", flush=True)
        r = run_grid_point_m2(model, base_data, am, asd, om, osd, n_samples)
        results.append(r)
        if r["n_ok"] == 0:
            print("ALL FAILED")
        else:
            print(
                f"SEU unc={r['seu_rate_uncertain_mean']:.3f}  "
                f"risk={r['seu_rate_risky_mean']:.3f}  "
                f"(α med={r['alpha_median']:.1f}, ω med={r['omega_median']:.1f})"
                + (f"  [{r['n_failed']} failures]" if r["n_failed"] > 0 else "")
            )

    table = print_results_table_m2(results)
    print(f"\n{table}\n")

    _save_results("m_2", results, table, output_dir, base_data, sim_path, n_samples)
    return results


def sweep_m3(base_data: dict, n_samples: int, output_dir: Path):
    """Grid search for m_3 (alpha, kappa with omega = kappa*alpha)."""
    sim_path = str(PROJECT_ROOT / "models" / "m_3_sim.stan")
    print(f"\n{'='*80}")
    print("MODEL m_3  (α for uncertain, ω = κ·α for risky)")
    print(f"{'='*80}")
    print(f"Compiling: {sim_path}")
    model = CmdStanModel(stan_file=sim_path)

    # 2D grid: alpha × kappa
    grid = list(itertools.product(ALPHA_GRID, KAPPA_GRID))
    results = []
    n_grid = len(grid)
    print(f"Running {n_grid} grid points × {n_samples} samples each ...\n")

    for idx, ((am, asd), (km, ksd)) in enumerate(grid):
        label = f"α~LN({am},{asd}), κ~LN({km},{ksd})"
        print(f"  [{idx+1}/{n_grid}] {label} ...", end=" ", flush=True)
        r = run_grid_point_m3(model, base_data, am, asd, km, ksd, n_samples)
        results.append(r)
        if r["n_ok"] == 0:
            print("ALL FAILED")
        else:
            print(
                f"SEU unc={r['seu_rate_uncertain_mean']:.3f}  "
                f"risk={r['seu_rate_risky_mean']:.3f}  "
                f"(α med={r['alpha_median']:.1f}, κ med={r['kappa_median']:.1f}, ω med={r['omega_median']:.1f})"
                + (f"  [{r['n_failed']} failures]" if r["n_failed"] > 0 else "")
            )

    table = print_results_table_m3(results)
    print(f"\n{table}\n")

    _save_results("m_3", results, table, output_dir, base_data, sim_path, n_samples)
    return results


# ── Save helpers ────────────────────────────────────────────────────

def _save_results(model_name, results, table, output_dir, base_data, sim_path, n_samples):
    """Save grid search results to CSV, JSON, and TXT."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = model_dir / "grid_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV:  {csv_path}")

    json_path = model_dir / "grid_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "model": model_name,
                    "M": base_data["M"],
                    "N": base_data["N"],
                    "n_param_samples": n_samples,
                    "sim_model": sim_path,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"  Saved JSON: {json_path}")

    txt_path = model_dir / "grid_results.txt"
    txt_path.write_text(table)
    print(f"  Saved TXT:  {txt_path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prior predictive grid search for augmented models (m_1, m_2, m_3)"
    )
    parser.add_argument(
        "--stan-data",
        type=str,
        default=str(
            PROJECT_ROOT
            / "applications"
            / "temperature_study_with_risky_alts"
            / "results"
            / "stan_data_augmented_T0_7.json"
        ),
        help="Path to an augmented stan_data JSON (default: T=0.7)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of parameter draws per grid point (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "results"
            / "prior_predictive"
            / "augmented_grid_search"
        ),
        help="Directory to save results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["m_1", "m_2", "m_3"],
        default=["m_1", "m_2", "m_3"],
        help="Which models to sweep (default: all three)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Loading augmented Stan data from: {args.stan_data}")
    base_data = load_augmented_stan_data(args.stan_data)
    print(
        f"  M={base_data['M']}  N={base_data['N']}  "
        f"K={base_data['K']}  D={base_data['D']}  "
        f"R={base_data['R']}  S={base_data['S']}"
    )

    # ── Sweep each requested model ──
    if "m_1" in args.models:
        sweep_m1(base_data, args.n_samples, output_dir)

    if "m_2" in args.models:
        sweep_m2(base_data, args.n_samples, output_dir)

    if "m_3" in args.models:
        sweep_m3(base_data, args.n_samples, output_dir)

    print(f"\n{'='*80}")
    print("All requested grid searches complete.")
    print(f"Results saved under: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

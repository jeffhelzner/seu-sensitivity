"""
Prior Predictive Grid Search for Alpha Prior Calibration

Sweeps over a grid of lognormal(alpha_mean, alpha_sd) prior hyperparameters
using the m_0_sim.stan simulation model and the actual temperature-study
design (w, I).  For each grid point the script draws n_param_samples
parameter sets from the prior, simulates choices, and reports:

  SEU-max rate  =  mean(total_seu_max_selected) / M

This gives the prior-implied probability that the agent selects the
alternative with the highest subjective expected utility — a directly
interpretable behavioural diagnostic.

Usage:
    python scripts/run_prior_predictive_grid.py [--stan-data PATH]
                                                 [--n-samples N]
                                                 [--output-dir DIR]

The script uses m_0_sim.stan (== m_01_sim.stan structurally) because the
sim model accepts alpha_mean / alpha_sd as data inputs.
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


# ── Grid of candidate priors ────────────────────────────────────────
# Each entry: (alpha_mean, alpha_sd) for lognormal(alpha_mean, alpha_sd)
#
# Recall: if X ~ lognormal(mu, sigma)  then
#   median = exp(mu),  mode = exp(mu - sigma^2)
#   mean   = exp(mu + sigma^2/2)

PRIOR_GRID = [
    # --- m_0 baseline ---
    (0.0, 1.0),       # median=1,    95th≈5

    # --- wider tails, same centre ---
    (0.0, 1.5),       # median=1,    95th≈20
    (0.0, 2.0),       # median=1,    95th≈39

    # --- shifted centre ---
    (2.0, 1.0),       # median=7.4,  95th≈38
    (2.5, 1.0),       # median=12.2, 95th≈63
    (3.0, 0.75),      # median=20.1, 95th≈68
    (3.0, 1.0),       # median=20.1, 95th≈104
    (3.5, 0.5),       # median=33.1, 95th≈72
    (3.5, 0.75),      # median=33.1, 95th≈112
    (3.5, 1.0),       # median=33.1, 95th≈172
    (4.0, 0.5),       # median=54.6, 95th≈119
    (4.0, 0.75),      # median=54.6, 95th≈185
]


def load_stan_data(path: str) -> dict:
    """Load a temperature-study stan_data JSON and prepare it for the sim model."""
    with open(path) as f:
        data = json.load(f)
    # The sim model doesn't need y — remove it if present
    data.pop("y", None)
    return data


def run_grid_point(
    model: CmdStanModel,
    base_data: dict,
    alpha_mean: float,
    alpha_sd: float,
    n_param_samples: int,
    beta_sd: float = 1.0,
) -> dict:
    """
    Run prior predictive simulation for one (alpha_mean, alpha_sd) point.

    Uses a single Stan call with iter_sampling=n_param_samples in
    fixed_param mode, which is far more efficient than N separate calls.

    Returns a dict with summary statistics.
    """
    data = dict(base_data)
    data["alpha_mean"] = alpha_mean
    data["alpha_sd"] = alpha_sd
    data["beta_sd"] = beta_sd

    M = data["M"]

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
        seu_counts = draws["total_seu_max_selected"].values.astype(float)
        seu_rates = seu_counts / M

        # Filter out any NaN rows (softmax overflow)
        valid = ~(np.isnan(alphas) | np.isnan(seu_rates))
        n_failed = int((~valid).sum())
        alphas = alphas[valid]
        seu_rates = seu_rates[valid]
    except RuntimeError:
        # All chains crashed
        return {
            "alpha_mean": alpha_mean,
            "alpha_sd": alpha_sd,
            "prior_label": f"lognormal({alpha_mean}, {alpha_sd})",
            "alpha_median": float("nan"),
            "alpha_q05": float("nan"),
            "alpha_q25": float("nan"),
            "alpha_q75": float("nan"),
            "alpha_q95": float("nan"),
            "seu_rate_mean": float("nan"),
            "seu_rate_sd": float("nan"),
            "seu_rate_q05": float("nan"),
            "seu_rate_q95": float("nan"),
            "n_ok": 0,
            "n_failed": n_param_samples,
            "fail_rate": 1.0,
        }

    n_ok = len(alphas)
    if n_ok == 0:
        return {
            "alpha_mean": alpha_mean,
            "alpha_sd": alpha_sd,
            "prior_label": f"lognormal({alpha_mean}, {alpha_sd})",
            "alpha_median": float("nan"),
            "alpha_q05": float("nan"),
            "alpha_q25": float("nan"),
            "alpha_q75": float("nan"),
            "alpha_q95": float("nan"),
            "seu_rate_mean": float("nan"),
            "seu_rate_sd": float("nan"),
            "seu_rate_q05": float("nan"),
            "seu_rate_q95": float("nan"),
            "n_ok": 0,
            "n_failed": n_failed,
            "fail_rate": 1.0,
        }

    return {
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "prior_label": f"lognormal({alpha_mean}, {alpha_sd})",
        "alpha_median": float(np.median(alphas)),
        "alpha_q05": float(np.quantile(alphas, 0.05)),
        "alpha_q25": float(np.quantile(alphas, 0.25)),
        "alpha_q75": float(np.quantile(alphas, 0.75)),
        "alpha_q95": float(np.quantile(alphas, 0.95)),
        "seu_rate_mean": float(np.mean(seu_rates)),
        "seu_rate_sd": float(np.std(seu_rates)),
        "seu_rate_q05": float(np.quantile(seu_rates, 0.05)),
        "seu_rate_median": float(np.median(seu_rates)),
        "seu_rate_q95": float(np.quantile(seu_rates, 0.95)),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "fail_rate": n_failed / (n_ok + n_failed),
    }


def print_results_table(results: list[dict]) -> str:
    """Format results as a readable table and return as string."""
    lines = []
    header = (
        f"{'Prior':>24s}  "
        f"{'α med':>7s}  {'α 90%CI':>16s}  "
        f"{'SEU rate':>8s}  {'± sd':>6s}  {'SEU 90%CI':>16s}  "
        f"{'fail%':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r["n_ok"] == 0:
            line = (
                f"{r['prior_label']:>24s}  "
                f"{'  ALL FAILED':40s}  "
                f"{r['fail_rate']*100:5.1f}"
            )
        else:
            line = (
                f"{r['prior_label']:>24s}  "
                f"{r['alpha_median']:7.1f}  "
                f"[{r['alpha_q05']:6.1f}, {r['alpha_q95']:6.1f}]  "
                f"{r['seu_rate_mean']:8.3f}  "
                f"{r['seu_rate_sd']:6.3f}  "
                f"[{r['seu_rate_q05']:6.3f}, {r['seu_rate_q95']:6.3f}]  "
                f"{r['fail_rate']*100:5.1f}"
            )
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Prior predictive grid search for alpha prior calibration"
    )
    parser.add_argument(
        "--stan-data",
        type=str,
        default=str(
            PROJECT_ROOT
            / "applications"
            / "temperature_study"
            / "results"
            / "stan_data_T0_7.json"
        ),
        help="Path to a stan_data JSON from the temperature study (default: T=0.7)",
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
        default=str(PROJECT_ROOT / "results" / "prior_predictive" / "grid_search"),
        help="Directory to save results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Loading Stan data from: {args.stan_data}")
    base_data = load_stan_data(args.stan_data)
    print(f"  M={base_data['M']}  K={base_data['K']}  D={base_data['D']}  R={base_data['R']}")

    # ── Compile sim model ──
    sim_model_path = str(PROJECT_ROOT / "models" / "m_0_sim.stan")
    print(f"Compiling sim model: {sim_model_path}")
    model = CmdStanModel(stan_file=sim_model_path)

    # ── Sweep ──
    results = []
    n_grid = len(PRIOR_GRID)
    print(f"\nRunning {n_grid} grid points × {args.n_samples} samples each ...\n")

    for idx, (alpha_mean, alpha_sd) in enumerate(PRIOR_GRID):
        label = f"lognormal({alpha_mean}, {alpha_sd})"
        print(f"  [{idx+1}/{n_grid}] {label} ...", end=" ", flush=True)
        r = run_grid_point(model, base_data, alpha_mean, alpha_sd, args.n_samples)
        results.append(r)
        if r["n_ok"] == 0:
            print(f"ALL FAILED ({r['n_failed']} failures)")
        elif r["n_failed"] > 0:
            print(
                f"SEU rate = {r['seu_rate_mean']:.3f}  "
                f"(α median = {r['alpha_median']:.1f})  "
                f"[{r['n_failed']} failures]"
            )
        else:
            print(
                f"SEU rate = {r['seu_rate_mean']:.3f}  "
                f"(α median = {r['alpha_median']:.1f})"
            )

    # ── Print table ──
    table = print_results_table(results)
    print(f"\n{'='*80}")
    print("PRIOR PREDICTIVE GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Stan data:     {args.stan_data}")
    print(f"Samples/point: {args.n_samples}")
    print(f"{'='*80}\n")
    print(table)
    print()

    # ── Save ──
    df = pd.DataFrame(results)
    csv_path = output_dir / "grid_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV:  {csv_path}")

    json_path = output_dir / "grid_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "stan_data": args.stan_data,
                    "n_param_samples": args.n_samples,
                    "sim_model": sim_model_path,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved JSON: {json_path}")

    txt_path = output_dir / "grid_results.txt"
    txt_path.write_text(table)
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()

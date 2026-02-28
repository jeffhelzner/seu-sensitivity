"""
Sample Size Estimation Workflow Script

This script estimates the required number of decision problems (M) to achieve
a desired precision in estimating the sensitivity parameter alpha in Bayesian
Decision Theory models.

Purpose:
    - Evaluate estimation precision across a grid of sample sizes (M values)
    - For each M, simulate data and run parameter recovery multiple times
    - Report credible interval widths, RMSE, MAE, and coverage vs. M
    - Help determine the minimum M needed for a given study's precision target

Usage:
    python scripts/run_sample_size_estimation.py [--config CONFIG_PATH]

    The config file should be a JSON file with the following structure:
    {
        "K": 3,                    # Number of possible consequences
        "D": 2,                    # Dimensions of alternative features
        "R": 10,                   # Number of distinct alternatives
        "M_grid": [10, 20, 30, 40, 50],  # Sample sizes to evaluate
        "n_iterations": 10,        # Recovery iterations per M value
        "n_mcmc_samples": 1000,    # Posterior samples per chain
        "n_mcmc_chains": 4,        # Number of MCMC chains
        "output_dir": "results/sample_size_estimation/custom_run",
        "inference_model_path": "models/m_0.stan",
        "sim_model_path": "models/m_0_sim.stan"
    }

Examples:
    # Run with default configuration
    python scripts/run_sample_size_estimation.py

    # Run with a custom configuration file
    python scripts/run_sample_size_estimation.py --config configs/sample_size_config.json

Outputs:
    - alpha_interval_widths.csv     — CI widths for each M
    - recovery_summary_vs_M.csv     — RMSE, MAE, coverage by M
    - alpha_precision_vs_M.png      — Precision plot
    - ci_width_vs_M_comparison.png  — CI width comparison plot
    - rmse_vs_M_comparison.png      — RMSE comparison plot
    - coverage_vs_M_comparison.png  — Coverage comparison plot
"""

import argparse
import os
import sys

# Add project root to Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sample_size_estimation import SampleSizeEstimator

def main():
    parser = argparse.ArgumentParser(description="Run sample size estimation workflow")
    parser.add_argument('--config', type=str, default='configs/sample_size_config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    estimator = SampleSizeEstimator(config_path=args.config)
    estimator.run()

if __name__ == "__main__":
    main()

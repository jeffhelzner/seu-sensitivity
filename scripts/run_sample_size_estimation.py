"""
Sample Size Estimation Workflow Script

This script estimates the required number of decision problems (M) to achieve a desired precision
in estimating the sensitivity parameter alpha in Bayesian Decision Theory models.

Usage:
    python scripts/run_sample_size_estimation.py --config configs/sample_size_config.json

Config file structure:
{
    "K": 3,
    "D": 2,
    "R": 10,
    "M_grid": [10, 20, 30, 40, 50],
    "n_iterations": 10,
    "n_mcmc_samples": 1000,
    "n_mcmc_chains": 4,
    "output_dir": "results/sample_size_estimation/custom_run",
    "inference_model_path": "models/m_0.stan",
    "sim_model_path": "models/m_0_sim.stan"
}
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

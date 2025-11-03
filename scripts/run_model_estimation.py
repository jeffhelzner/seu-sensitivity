"""
Model Estimation Workflow Script

This script fits the Bayesian Decision Theory model to observed choice data using a configuration file.

Purpose:
    - Load model and data paths from a config file
    - Run MCMC sampling to estimate model parameters
    - Save posterior samples, summaries, and diagnostics
    - Generate plots and posterior predictive checks

Config file structure:
{
    "data_path": "applications/llm_rationality/results/run_20251021_165854/stan_data_GPT-4.json",
    "model_path": "models/m_0.stan",
    "output_dir": "results/estimation/custom_run",
    "n_mcmc_samples": 2000,
    "n_mcmc_chains": 4
}

Usage:
    python scripts/run_model_estimation.py --config configs/model_estimation_config.json

Outputs:
    - Results saved to output_dir (posterior samples, summary, diagnostics, plots)

See analysis/model_estimation.py for details on the workflow.
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.model_estimation import ModelEstimation

def run_from_config(config_path):
    """
    Run model estimation from a configuration file.

    Args:
        config_path (str): Path to JSON config file specifying model/data/params.

    Returns:
        CmdStanMCMC: Fitted Stan model object
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_path = config.get('model_path', None)
    data_path = config.get('data_path', None)
    output_dir = config.get('output_dir', None)
    n_mcmc_samples = config.get('n_mcmc_samples', 2000)
    n_mcmc_chains = config.get('n_mcmc_chains', 4)

    estimation = ModelEstimation(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        n_mcmc_samples=n_mcmc_samples,
        n_mcmc_chains=n_mcmc_chains
    )
    fit = estimation.run()
    print(f"Model estimation completed. Results saved to: {estimation.output_dir}")
    return fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model estimation workflow")
    parser.add_argument('--config', type=str, default='configs/model_estimation_config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    run_from_config(args.config)

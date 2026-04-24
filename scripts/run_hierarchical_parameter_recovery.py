"""
Hierarchical Parameter Recovery Analysis Script

Runs parameter recovery for the h_m01 hierarchical model, using
``h_m01_sim.stan`` to simulate data and ``h_m01.stan`` to recover
parameters, with a ``HierarchicalStudyDesign``.

Usage:
    python scripts/run_hierarchical_parameter_recovery.py \
        [--config configs/h_m01_parameter_recovery_config.json]

The config file is a JSON object with the following optional keys::

    {
      "inference_model_path": "models/h_m01.stan",
      "sim_model_path": "models/h_m01_sim.stan",
      "study_design_path": "path/to/design.json",       # existing design, or
      "study_design_config": {                           # generate a new one
        "J": 6, "K": 3, "D": 2, "R": 10, "P": 2,
        "M_per_cell": 20,
        "min_alts_per_problem": 2,
        "max_alts_per_problem": 4,
        "feature_dist": "normal",
        "feature_params": {"loc": 0, "scale": 1},
        "X": [[0,0],[1,0],[0,1],[1,1],[1,0],[0,1]]       # optional explicit X
      },
      "output_dir": "results/parameter_recovery/h_m01_recovery",
      "n_mcmc_samples": 2000,
      "n_mcmc_chains": 4,
      "n_iterations": 20
    }
"""
import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hierarchical_parameter_recovery import HierarchicalParameterRecovery
from utils.study_design_hierarchical import HierarchicalStudyDesign


def _build_study_design(design_config: dict) -> HierarchicalStudyDesign:
    """Instantiate and generate a HierarchicalStudyDesign from config."""
    X = design_config.get("X")
    if X is not None:
        X = np.asarray(X, dtype=float)

    design = HierarchicalStudyDesign(
        J=design_config.get("J", 6),
        K=design_config.get("K", 3),
        D=design_config.get("D", 2),
        R=design_config.get("R", 10),
        P=design_config.get("P", 2),
        M_per_cell=design_config.get("M_per_cell", 20),
        X=X,
        min_alts_per_problem=design_config.get("min_alts_per_problem", 2),
        max_alts_per_problem=design_config.get("max_alts_per_problem", 4),
        feature_dist=design_config.get("feature_dist", "normal"),
        feature_params=design_config.get(
            "feature_params", {"loc": 0, "scale": 1}
        ),
        design_name=design_config.get("design_name", "h_m01_parameter_recovery"),
    )
    design.generate()
    return design


def run_from_config(config_path: str) -> HierarchicalParameterRecovery:
    """Run hierarchical parameter recovery analysis from a JSON config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    inference_model_path = config.get("inference_model_path")
    sim_model_path = config.get("sim_model_path")
    output_dir = config.get("output_dir")
    n_mcmc_samples = config.get("n_mcmc_samples", 2000)
    n_mcmc_chains = config.get("n_mcmc_chains", 4)
    n_iterations = config.get("n_iterations", 20)

    study_design = None
    if config.get("study_design_path"):
        study_design = HierarchicalStudyDesign.load(config["study_design_path"])
        print(f"Loaded study design from: {config['study_design_path']}")
    elif "study_design_config" in config:
        study_design = _build_study_design(config["study_design_config"])
        print(
            f"Created hierarchical study design: J={study_design.J}, "
            f"K={study_design.K}, D={study_design.D}, R={study_design.R}, "
            f"P={study_design.P}, M_total={study_design.M_total}"
        )

    recovery = HierarchicalParameterRecovery(
        inference_model_path=inference_model_path,
        sim_model_path=sim_model_path,
        study_design=study_design,
        output_dir=output_dir,
        n_mcmc_samples=n_mcmc_samples,
        n_mcmc_chains=n_mcmc_chains,
        n_iterations=n_iterations,
    )

    true_params, posterior_summaries = recovery.run()

    print("Hierarchical parameter recovery analysis completed.")
    print(f"Results saved to: {recovery.output_dir}")
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hierarchical parameter recovery analysis for h_m01"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h_m01_parameter_recovery_config.json",
        help="Path to configuration JSON",
    )
    args = parser.parse_args()
    run_from_config(args.config)

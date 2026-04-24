"""
Hierarchical Prior Predictive Analysis Script

Runs prior predictive analysis for the h_m01 hierarchical model, using
``h_m01_sim.stan`` together with a ``HierarchicalStudyDesign``.

Usage:
    python scripts/run_hierarchical_prior_predictive.py \
        [--config configs/h_m01_prior_analysis_config.json]

The config file is a JSON object with the following optional keys::

    {
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
      "output_dir": "results/prior_predictive/h_m01_prior_analysis",
      "n_param_samples": 200,
      "n_choice_samples": 5,
      "hyperparameters": {
        "gamma0_mean": 3.0,
        "gamma0_sd": 1.0,
        "gamma_sd": 1.0,
        "sigma_cell_sd": 0.5,
        "beta_sd": 1.0
      }
    }
"""
import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hierarchical_prior_predictive import (
    HierarchicalPriorPredictiveAnalysis,
)
from utils.study_design_hierarchical import HierarchicalStudyDesign


def _build_study_design(design_config: dict) -> HierarchicalStudyDesign:
    """Instantiate and generate a HierarchicalStudyDesign from config.

    Supports two modes:
    - Factorial: specify ``factors: [k1, k2, ...]`` (optionally
      ``reference_indices`` and ``include_interactions``). J and P are
      derived automatically via treatment coding.
    - Manual: specify J and P directly; X is taken from ``X`` if present,
      otherwise a default bit-pattern design is used.
    """
    if "factors" in design_config:
        design = HierarchicalStudyDesign.from_factorial(
            factors=design_config["factors"],
            K=design_config.get("K", 3),
            D=design_config.get("D", 2),
            R=design_config.get("R", 10),
            M_per_cell=design_config.get("M_per_cell", 20),
            reference_indices=design_config.get("reference_indices"),
            include_interactions=design_config.get(
                "include_interactions", False
            ),
            min_alts_per_problem=design_config.get("min_alts_per_problem", 2),
            max_alts_per_problem=design_config.get("max_alts_per_problem", 4),
            feature_dist=design_config.get("feature_dist", "normal"),
            feature_params=design_config.get(
                "feature_params", {"loc": 0, "scale": 1}
            ),
            design_name=design_config.get("design_name", "h_m01_prior_analysis"),
        )
        design.generate()
        return design

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
        design_name=design_config.get("design_name", "h_m01_prior_analysis"),
    )
    design.generate()
    return design


def run_from_config(config_path: str) -> HierarchicalPriorPredictiveAnalysis:
    """Run hierarchical prior predictive analysis from a JSON config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    sim_model_path = config.get("sim_model_path")
    output_dir = config.get("output_dir")
    n_param_samples = config.get("n_param_samples", 100)
    n_choice_samples = config.get("n_choice_samples", 5)
    hyperparams = config.get("hyperparameters")

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

    analysis = HierarchicalPriorPredictiveAnalysis(
        sim_model_path=sim_model_path,
        study_design=study_design,
        output_dir=output_dir,
        n_param_samples=n_param_samples,
        n_choice_samples=n_choice_samples,
        hyperparams=hyperparams,
    )
    analysis.run()

    print("Hierarchical prior predictive analysis completed.")
    print(f"Results saved to: {analysis.output_dir}")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hierarchical prior predictive analysis for h_m01"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h_m01_prior_analysis_config.json",
        help="Path to configuration JSON",
    )
    args = parser.parse_args()
    run_from_config(args.config)

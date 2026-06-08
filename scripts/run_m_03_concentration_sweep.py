"""
Concentration Sweep Driver for Route 2 Evaluation (Report 13)

Runs a sweep of matched-prior parameter recovery and (optionally) SBC across
a grid of Dirichlet concentration values alpha0 on the delta simplex, using
the parameterized m_03 model trio.

Each grid point uses Dirichlet(alpha0 * 1_{K-1}) for BOTH data generation
(via m_03_sim) and inference (via m_03), so each run is a calibration-faithful
recovery study under a different prior.

Usage:
    # Recovery sweep only
    python scripts/run_m_03_concentration_sweep.py \\
        --config configs/m_03_concentration_sweep_recovery_config.json

    # SBC sweep
    python scripts/run_m_03_concentration_sweep.py \\
        --config configs/m_03_concentration_sweep_sbc_config.json \\
        --mode sbc

Both config files must include a ``concentration_grid`` list and a
``study_design_config`` block; per-alpha0 results are written under
``output_dir/alpha0=<value>/``.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.parameter_recovery import ParameterRecovery
from analysis.sbc import SimulationBasedCalibration
from utils.study_design import StudyDesign


def _build_study_design(config):
    design_config = config["study_design_config"]
    study = StudyDesign(
        M=design_config.get("M", 25),
        K=design_config.get("K", 3),
        D=design_config.get("D", 5),
        R=design_config.get("R", 15),
        min_alts_per_problem=design_config.get("min_alts_per_problem", 2),
        max_alts_per_problem=design_config.get("max_alts_per_problem", 5),
        feature_dist=design_config.get("feature_dist", "normal"),
        feature_params=design_config.get("feature_params", {"loc": 0, "scale": 1}),
        design_name="m_03_concentration_sweep",
    )
    study.generate()
    return study


def _output_subdir(base_output_dir, alpha0):
    # Use a filesystem-friendly suffix; treat 1.0 as "1" etc. for readability.
    alpha0_str = (f"{alpha0:.1f}").rstrip("0").rstrip(".") or "0"
    subdir = os.path.join(base_output_dir, f"alpha0={alpha0_str}")
    os.makedirs(subdir, exist_ok=True)
    return subdir


def run_recovery_sweep(config):
    base_output_dir = config["output_dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    study = _build_study_design(config)
    # Persist a single canonical study design at the sweep root so every alpha0
    # is evaluated on the same alternatives + problems.
    study.save(os.path.join(base_output_dir, "study_design.json"))

    for alpha0 in config["concentration_grid"]:
        alpha0 = float(alpha0)
        out_dir = _output_subdir(base_output_dir, alpha0)
        print(f"\n=== Recovery sweep: delta_concentration = {alpha0} -> {out_dir} ===")

        recovery = ParameterRecovery(
            inference_model_path=config.get("inference_model_path", "models/m_03.stan"),
            sim_model_path=config.get("sim_model_path", "models/m_03_sim.stan"),
            study_design=study,
            output_dir=out_dir,
            n_mcmc_samples=config.get("n_mcmc_samples", 2000),
            n_mcmc_chains=config.get("n_mcmc_chains", 4),
            n_iterations=config.get("n_iterations", 50),
            sim_hyperparams={"delta_concentration": alpha0},
            inference_hyperparams={"delta_concentration": alpha0},
        )
        recovery.run()


def run_sbc_sweep(config):
    base_output_dir = config["output_dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    study = _build_study_design(config)
    study.save(os.path.join(base_output_dir, "study_design.json"))

    for alpha0 in config["concentration_grid"]:
        alpha0 = float(alpha0)
        out_dir = _output_subdir(base_output_dir, alpha0)
        print(f"\n=== SBC sweep: delta_concentration = {alpha0} -> {out_dir} ===")

        sbc = SimulationBasedCalibration(
            sbc_model_path=config.get("sbc_model_path", "models/m_03_sbc.stan"),
            study_design=study,
            output_dir=out_dir,
            n_sbc_sims=config.get("n_sbc_sims", 200),
            n_mcmc_samples=config.get("n_mcmc_samples", 1000),
            n_mcmc_chains=config.get("n_mcmc_chains", 1),
            thin=config.get("thin", 3),
            extra_data={"delta_concentration": alpha0},
        )
        sbc.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to sweep config JSON")
    parser.add_argument("--mode", choices=["recovery", "sbc"], default="recovery",
                        help="Which sweep to run (default: recovery).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if "concentration_grid" not in config:
        raise ValueError(f"Config {args.config} must include a 'concentration_grid' list")

    if args.mode == "recovery":
        run_recovery_sweep(config)
    else:
        run_sbc_sweep(config)


if __name__ == "__main__":
    main()

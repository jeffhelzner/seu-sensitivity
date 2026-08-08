#!/usr/bin/env python
"""
Run the Phase D power analysis (§8.5), regime (a): pseudo-replication.

Usage
-----
    python scripts/run_hierarchical_power.py --config configs/<file>.json
    python scripts/run_hierarchical_power.py --config <file> --measure

``--measure`` overrides ``n_iterations`` to a small number so ONE cell's real
per-iteration cost can be measured before any grid is launched. Phase C's
a-priori estimate was wrong by 1.8x; the rule is now measure-then-size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.hierarchical_power import HierarchicalPowerAnalysis
from utils.study_design_hierarchical import HierarchicalStudyDesign


def _compile(path: str):
    from cmdstanpy import CmdStanModel

    return CmdStanModel(stan_file=path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Short run to measure per-iteration cost before sizing a grid.",
    )
    parser.add_argument("--measure-iterations", type=int, default=2)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iteration count for a REAL run (writes to --output-dir directly, "
        "unlike --measure which sandboxes into a _measure/ subdirectory).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rho-copy", type=float, default=None)
    parser.add_argument("--menus-per-cell", type=int, default=None)
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=None,
        help="Override sampler draws. SMOKE USE ONLY -- timings taken under an "
        "override are not comparable to the config's real settings.",
    )
    parser.add_argument("--iter-warmup", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        config = json.load(fh)

    design_cfg = dict(config["study_design_config"])
    if args.menus_per_cell is not None:
        design_cfg["M_per_cell"] = args.menus_per_cell

    design = HierarchicalStudyDesign.from_factorial(**design_cfg)
    design.generate()

    output_dir = args.output_dir or config["output_dir"]
    if args.measure:
        output_dir = os.path.join(output_dir, "_measure")

    n_iterations = (
        args.measure_iterations
        if args.measure
        else (args.iterations or config.get("n_iterations", 20))
    )
    rho_copy = args.rho_copy if args.rho_copy is not None else config.get("rho_copy", 0.0)

    print(f"Compiling {config['sim_model_path']} ...")
    sim_model = _compile(config["sim_model_path"])
    print(f"Compiling {config['inference_model_path']} ...")
    inference_model = _compile(config["inference_model_path"])

    n_samples = args.iter_sampling or config.get("n_mcmc_samples", 2000)
    n_warmup = args.iter_warmup or config.get("n_mcmc_warmup")
    n_chains = args.chains or config.get("n_mcmc_chains", 4)
    overridden = any(
        v is not None for v in (args.iter_sampling, args.iter_warmup, args.chains)
    )
    if overridden:
        print(
            "WARNING: sampler settings overridden on the command line. Any timing "
            "from this run is NOT a valid basis for sizing a grid -- that was "
            "exactly the Phase C 1.8x error."
        )

    analysis = HierarchicalPowerAnalysis(
        sim_model=sim_model,
        inference_model=inference_model,
        study_design=design,
        output_dir=output_dir,
        n_iterations=n_iterations,
        n_mcmc_samples=n_samples,
        n_mcmc_warmup=n_warmup,
        n_mcmc_chains=n_chains,
        adapt_delta=config.get("adapt_delta", 0.95),
        num_presentations=config.get("num_presentations", 2),
        rho_copy=rho_copy,
        sim_overrides=config.get("sim_overrides"),
        sim_only_keys=config.get("sim_only_keys", []),
    )

    summary = analysis.run()

    print("\n" + "=" * 70)
    print(f"regime (a)  rho_copy={summary['rho_copy']}  "
          f"menus/cell={summary['menus_per_cell']:.0f}  "
          f"M_total={summary['M_total']}")
    print("=" * 70)
    for key in (
        "power_excludes_zero",
        "power_outside_rope",
        "correct_sign_rate",
        "type_s_rate",
        "coverage",
        "mean_ci_width",
        "mean_agreement_rate",
        "seconds_per_iteration",
    ):
        value = summary.get(key)
        print(f"  {key:<26}{value if value is None else round(value, 4)}")
    print(f"\n  nominal coverage {summary['nominal_coverage']}")
    print(f"  wrote {output_dir}/summary.json  and  timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

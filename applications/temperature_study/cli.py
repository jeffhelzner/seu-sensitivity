"""
Command-line interface for the Temperature Study.

Usage:
    python -m applications.temperature_study run [options]
    python -m applications.temperature_study estimate-cost [options]
    python -m applications.temperature_study validate [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── subcommands ─────────────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> None:
    """Validate configuration and claim pool without making API calls."""
    from .config import StudyConfig

    config = StudyConfig.from_yaml(args.config)
    print(f"✅  Config valid  ({len(config.temperatures)} temperatures, "
          f"{config.num_problems} problems, P={config.num_presentations})")

    claims_path = Path(config.claims_file)
    if claims_path.exists():
        with open(claims_path) as f:
            pool = json.load(f)
        n = len(pool.get("claims", []))
        print(f"✅  Claim pool loaded  ({n} claims from {claims_path.name})")
    else:
        print(f"⚠️   Claim pool not found at {claims_path}")


def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """Print estimated API cost breakdown."""
    from .config import StudyConfig

    config = StudyConfig.from_yaml(args.config)

    avg_alts = (config.min_alternatives + config.max_alternatives) / 2
    n_delib = config.num_problems * avg_alts * len(config.temperatures)
    n_choices = (
        config.num_problems
        * config.num_presentations
        * len(config.temperatures)
    )
    n_embed = n_delib

    print("═══ Estimated API Calls ═══")
    print(f"  Deliberations : {n_delib:,.0f}")
    print(f"  Choices       : {n_choices:,.0f}")
    print(f"  Embeddings    : {n_embed:,.0f}")
    print(f"  Total         : {n_delib + n_choices + n_embed:,.0f}")
    print()
    print("(Detailed token-level cost estimation will be added with study_runner)")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full study pipeline."""
    from .config import StudyConfig
    from .study_runner import TemperatureStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = TemperatureStudyRunner(config)
    summary = runner.run(
        skip_collection=getattr(args, "skip_collection", False),
        skip_model_fitting=getattr(args, "skip_fitting", False),
    )
    print(f"✅  Run complete.  Summary saved to {summary['phases'].get('phase3_data_prep', {}).get('pca_summary', {})}")
    print(f"    Duration: {summary.get('duration_seconds', 0):.1f}s")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Re-run data preparation (PCA + Stan data) from saved collection data."""
    from .config import StudyConfig
    from .study_runner import TemperatureStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = TemperatureStudyRunner(config)
    summary = runner.run(skip_collection=True, skip_model_fitting=True)
    print(f"✅  Data preparation complete.")
    for temp_str, info in summary["phases"]["phase3_data_prep"]["per_temperature"].items():
        print(f"    T={temp_str}: M={info['M']}, R={info['R']}, D={info['D']}, NAs removed={info['na_removed']}")


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit Stan models on existing Stan data files."""
    from .config import StudyConfig
    from .study_runner import TemperatureStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = TemperatureStudyRunner(config)
    fit_results = runner.fit_only()

    if fit_results.get("skipped"):
        print(f"⚠️  Model fitting skipped: {fit_results['reason']}")
        return

    print("✅  Model fitting complete.")
    for temp_str, info in fit_results.items():
        if isinstance(info, dict) and "alpha_median" in info:
            print(
                f"    T={temp_str}: α median={info['alpha_median']:.3f}  "
                f"90% CI=[{info['alpha_q05']:.3f}, {info['alpha_q95']:.3f}]"
            )


# ── main entry point ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="temperature_study",
        description="Temperature Study: LLM sensitivity × temperature",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging"
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    p_val = sub.add_parser("validate", help="Validate config and claim pool")
    p_val.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_val.set_defaults(func=cmd_validate)

    # estimate-cost
    p_cost = sub.add_parser("estimate-cost", help="Estimate API costs")
    p_cost.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_cost.set_defaults(func=cmd_estimate_cost)

    # run
    p_run = sub.add_parser("run", help="Run full study pipeline")
    p_run.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_run.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip API collection; load saved data instead",
    )
    p_run.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Stop after data preparation (no model fitting)",
    )
    p_run.set_defaults(func=cmd_run)

    # prepare (re-run Phase 3 only)
    p_prep = sub.add_parser(
        "prepare",
        help="Re-run data preparation from saved collection data",
    )
    p_prep.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_prep.set_defaults(func=cmd_prepare)

    # fit (Phase 4 only — fit Stan models on existing data)
    p_fit = sub.add_parser(
        "fit",
        help="Fit Stan models on existing stan_data_T*.json files",
    )
    p_fit.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_fit.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)

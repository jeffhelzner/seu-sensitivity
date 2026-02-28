"""
Command-line interface for the Temperature Study with Risky Alternatives.

Usage:
    python -m applications.temperature_study_with_risky_alts run [options]
    python -m applications.temperature_study_with_risky_alts estimate-cost [options]
    python -m applications.temperature_study_with_risky_alts validate [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── subcommands ─────────────────────────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate configuration and risky alternatives pool."""
    from .config import StudyConfig

    config = StudyConfig.from_yaml(args.config)
    print(
        f"✅  Config valid  ({len(config.temperatures)} temperatures, "
        f"{config.num_problems} problems, P={config.num_presentations})"
    )

    alts_path = Path(config.risky_alternatives_file)
    if alts_path.exists():
        with open(alts_path) as f:
            pool = json.load(f)
        n = len(pool.get("risky_alternatives", []))
        print(f"✅  Risky alternatives pool loaded  ({n} alternatives)")

        # Validate simplexes
        invalid = []
        for alt in pool["risky_alternatives"]:
            probs = alt["probabilities"]
            total = sum(probs)
            if abs(total - 1.0) > 1e-6:
                invalid.append(f"{alt['id']}: sums to {total}")
        if invalid:
            print(f"⚠️   Invalid simplexes: {invalid}")
        else:
            print("✅  All probability simplexes valid (sum to 1.0)")
    else:
        print(f"⚠️   Risky alternatives file not found at {alts_path}")

    ts_dir = Path(config.temperature_study_results_dir)
    if ts_dir.exists():
        # Check for stan_data files
        missing = []
        for temp in config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            if not (ts_dir / f"stan_data_T{ts}.json").exists():
                missing.append(f"T={temp}")
        if missing:
            print(
                f"⚠️   Missing uncertain Stan data for: {', '.join(missing)}"
            )
        else:
            print(
                f"✅  All {len(config.temperatures)} uncertain Stan data "
                f"files found in {ts_dir}"
            )
    else:
        print(f"⚠️   Temperature study results dir not found: {ts_dir}")


def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """Print estimated API cost breakdown."""
    from .config import StudyConfig
    from applications.temperature_study.llm_client import OPENAI_PRICING, ANTHROPIC_PRICING

    config = StudyConfig.from_yaml(args.config)

    n_choices = (
        config.num_problems
        * config.num_presentations
        * len(config.temperatures)
    )

    # Token estimates per call:
    # Risky choice: ~500 input tokens (system + user + alternative descriptions),
    #               ~10 output tokens (single number)
    choice_input_tok = 500 * n_choices
    choice_output_tok = 10 * n_choices

    provider = config.provider
    if provider == "anthropic":
        pricing = ANTHROPIC_PRICING.get(
            config.llm_model, {"input": 0, "output": 0}
        )
    else:
        pricing = OPENAI_PRICING.get(
            config.llm_model, {"input": 0, "output": 0}
        )

    choice_cost = (
        (choice_input_tok / 1_000_000) * pricing["input"]
        + (choice_output_tok / 1_000_000) * pricing["output"]
    )

    print("═══ Estimated API Calls ═══")
    print(f"  Risky choices : {n_choices:,.0f}")
    print(f"  (No assessments or embeddings needed for risky alternatives)")
    print()
    print(f"═══ Estimated Cost ({config.llm_model}) ═══")
    print(
        f"  Risky choices : ${choice_cost:,.2f}  "
        f"({choice_input_tok / 1e6:.2f}M in / {choice_output_tok / 1e6:.2f}M out)"
    )
    print()
    print("Note: Token counts are estimates based on typical prompt/response sizes.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full risky alternatives pipeline."""
    from .config import StudyConfig
    from .study_runner import RiskyStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = RiskyStudyRunner(config)
    summary = runner.run(
        skip_collection=getattr(args, "skip_collection", False),
        skip_model_fitting=getattr(args, "skip_fitting", False),
    )
    print("✅  Run complete.")
    print(f"    Duration: {summary.get('duration_seconds', 0):.1f}s")
    if "phase3_data_prep" in summary.get("phases", {}):
        for temp_str, info in summary["phases"]["phase3_data_prep"][
            "per_temperature"
        ].items():
            print(
                f"    T={temp_str}: M={info['M']}, N={info['N']}, "
                f"R={info['R']}, S={info['S']}, D={info['D']}, "
                f"risky NAs={info['risky_na_removed']}"
            )


def cmd_prepare(args: argparse.Namespace) -> None:
    """Re-run data preparation from saved risky choice data."""
    from .config import StudyConfig
    from .study_runner import RiskyStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = RiskyStudyRunner(config)
    summary = runner.run(skip_collection=True, skip_model_fitting=True)
    print("✅  Data preparation complete.")
    for temp_str, info in summary["phases"]["phase3_data_prep"][
        "per_temperature"
    ].items():
        print(
            f"    T={temp_str}: M={info['M']}, N={info['N']}, "
            f"R={info['R']}, S={info['S']}, D={info['D']}, "
            f"risky NAs={info['risky_na_removed']}"
        )


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit Stan models on existing augmented Stan data files."""
    from .config import StudyConfig
    from .study_runner import RiskyStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = RiskyStudyRunner(config)
    models = getattr(args, "models", None) or None
    fit_results = runner.fit_only(models=models)

    if fit_results.get("skipped"):
        print(f"⚠️  Model fitting skipped: {fit_results['reason']}")
        return

    print("✅  Model fitting complete.\n")

    for model_name, model_results in fit_results.items():
        if not isinstance(model_results, dict):
            continue
        print(f"══ {model_name} ══")
        for temp_str, info in model_results.items():
            if not isinstance(info, dict) or "alpha_median" not in info:
                continue
            print(
                f"  T={temp_str}: α median={info['alpha_median']:.3f}, "
                f"mean={info['alpha_mean']:.3f}, "
                f"90% CI=[{info['alpha_q05']:.3f}, {info['alpha_q95']:.3f}]"
            )
        print()


# ── main entry point ────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="temperature_study_with_risky_alts",
        description="Temperature Study: Risky alternatives extension for m_1/m_2/m_3",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging"
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    p_val = sub.add_parser(
        "validate", help="Validate config, alternatives, and uncertain data"
    )
    p_val.add_argument(
        "-c", "--config", default=None, help="Path to YAML config"
    )
    p_val.set_defaults(func=cmd_validate)

    # estimate-cost
    p_cost = sub.add_parser("estimate-cost", help="Estimate API costs")
    p_cost.add_argument(
        "-c", "--config", default=None, help="Path to YAML config"
    )
    p_cost.set_defaults(func=cmd_estimate_cost)

    # run
    p_run = sub.add_parser("run", help="Run full risky alternatives pipeline")
    p_run.add_argument(
        "-c", "--config", default=None, help="Path to YAML config"
    )
    p_run.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip API collection; load saved risky choices instead",
    )
    p_run.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Stop after data preparation (no model fitting)",
    )
    p_run.set_defaults(func=cmd_run)

    # prepare
    p_prep = sub.add_parser(
        "prepare",
        help="Re-run data preparation from saved risky choice data",
    )
    p_prep.add_argument(
        "-c", "--config", default=None, help="Path to YAML config"
    )
    p_prep.set_defaults(func=cmd_prepare)

    # fit
    p_fit = sub.add_parser(
        "fit",
        help="Fit models on existing augmented Stan data files (default: m_11, m_21, m_31)",
    )
    p_fit.add_argument(
        "-c", "--config", default=None, help="Path to YAML config"
    )
    p_fit.add_argument(
        "-m", "--models",
        nargs="+",
        default=None,
        help="Model names to fit (default: m_11 m_21 m_31)",
    )
    p_fit.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)

"""
Command-line interface for the GPT-4o × Ellsberg Study (Cell 1,2).

Usage:
    python -m applications.gpt4o_ellsberg_study run [options]
    python -m applications.gpt4o_ellsberg_study estimate-cost [options]
    python -m applications.gpt4o_ellsberg_study validate [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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


# -- subcommands --

def cmd_validate(args: argparse.Namespace) -> None:
    """Validate configuration, alternative pool, and API keys."""
    from .config import StudyConfig

    config = StudyConfig.from_yaml(args.config)
    print(
        f"  Config valid  ({len(config.temperatures)} temperatures, "
        f"{config.num_problems} problems, P={config.num_presentations}, K={config.K})"
    )

    alt_path = Path(config.alternatives_file)
    if alt_path.exists():
        with open(alt_path) as f:
            pool = json.load(f)
        n = len(pool.get("alternatives", []))
        print(f"  Alternative pool loaded  ({n} alternatives from {alt_path.name})")

        alts = pool["alternatives"]
        ids = [a["id"] for a in alts]
        if len(ids) != len(set(ids)):
            print("  WARNING: Duplicate alternative IDs found")
    else:
        print(f"  WARNING: Alternative pool not found at {alt_path}")

    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("  OPENAI_API_KEY is set")
    else:
        print("  WARNING: OPENAI_API_KEY not set (needed for LLM calls and embeddings)")


def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """Print estimated API cost breakdown."""
    from .config import StudyConfig
    from applications.temperature_study.llm_client import (
        OPENAI_PRICING,
        EMBEDDING_PRICING,
    )

    config = StudyConfig.from_yaml(args.config)

    alt_path = Path(config.alternatives_file)
    if alt_path.exists():
        with open(alt_path) as f:
            pool = json.load(f)
        n_alts = len(pool.get("alternatives", []))
    else:
        n_alts = 30
        print(f"  Alternative pool not found, assuming {n_alts} alternatives")

    n_assess = n_alts * len(config.temperatures)
    n_choices = (
        config.num_problems
        * config.num_presentations
        * len(config.temperatures)
    )
    n_embed = n_assess

    assess_input_tok = 500 * n_assess
    assess_output_tok = 150 * n_assess
    choice_input_tok = 800 * n_choices
    choice_output_tok = 10 * n_choices
    embed_tok = 200 * n_embed

    pricing = OPENAI_PRICING.get(config.llm_model, {"input": 0, "output": 0})
    embed_pricing = EMBEDDING_PRICING.get(config.embedding_model, 0.0)

    assess_cost = (
        (assess_input_tok / 1_000_000) * pricing["input"]
        + (assess_output_tok / 1_000_000) * pricing["output"]
    )
    choice_cost = (
        (choice_input_tok / 1_000_000) * pricing["input"]
        + (choice_output_tok / 1_000_000) * pricing["output"]
    )
    embed_cost = (embed_tok / 1_000_000) * embed_pricing

    total_cost = assess_cost + choice_cost + embed_cost

    print("=== Estimated API Calls ===")
    print(f"  Assessments   : {n_assess:,.0f}  (GPT-4o)")
    print(f"  Choices       : {n_choices:,.0f}  (GPT-4o)")
    print(f"  Embeddings    : {n_embed:,.0f}  (OpenAI {config.embedding_model})")
    print(f"  Total calls   : {n_assess + n_choices + n_embed:,.0f}")
    print()
    print(f"=== Estimated Cost ({config.llm_model}) ===")
    print(
        f"  Assessments   : ${assess_cost:,.2f}  "
        f"({assess_input_tok / 1e6:.2f}M in / {assess_output_tok / 1e6:.2f}M out)"
    )
    print(
        f"  Choices       : ${choice_cost:,.2f}  "
        f"({choice_input_tok / 1e6:.2f}M in / {choice_output_tok / 1e6:.2f}M out)"
    )
    print(
        f"  Embeddings    : ${embed_cost:,.2f}  "
        f"({embed_tok / 1e6:.2f}M tokens, {config.embedding_model})"
    )
    print(f"  -------------------------")
    print(f"  Total         : ${total_cost:,.2f}")
    print()
    print("Note: Token counts are estimates based on typical prompt/response sizes.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full study pipeline."""
    from .config import StudyConfig
    from .study_runner import GPT4oEllsbergStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = GPT4oEllsbergStudyRunner(config)
    summary = runner.run(
        skip_collection=getattr(args, "skip_collection", False),
        skip_model_fitting=getattr(args, "skip_fitting", False),
    )
    print(f"  Run complete. Summary saved to {runner.results_dir / 'run_summary.json'}")
    print(f"    Duration: {summary.get('duration_seconds', 0):.1f}s")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Re-run data preparation from saved collection data."""
    from .config import StudyConfig
    from .study_runner import GPT4oEllsbergStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = GPT4oEllsbergStudyRunner(config)
    summary = runner.run(skip_collection=True, skip_model_fitting=True)
    print("  Data preparation complete.")
    for temp_str, info in summary["phases"]["phase3_data_prep"]["per_temperature"].items():
        print(
            f"    T={temp_str}: M={info['M']}, R={info['R']}, "
            f"D={info['D']}, NAs removed={info['na_removed']}"
        )


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit Stan models on existing Stan data files."""
    from .config import StudyConfig
    from .study_runner import GPT4oEllsbergStudyRunner

    config = StudyConfig.from_yaml(args.config)

    if args.model:
        config.stan_model = args.model

    runner = GPT4oEllsbergStudyRunner(config)
    fit_results = runner.fit_only()

    if fit_results.get("skipped"):
        print(f"  Model fitting skipped: {fit_results['reason']}")
        return

    print("  Model fitting complete.\n")

    for temp_str, info in fit_results.items():
        if not isinstance(info, dict) or "alpha_median" not in info:
            continue

        print(f"-- T={temp_str} --")
        print(
            f"  alpha  median={info['alpha_median']:.3f}  "
            f"mean={info['alpha_mean']:.3f}  "
            f"90% CI=[{info['alpha_q05']:.3f}, {info['alpha_q95']:.3f}]"
        )

        diag = info.get("diagnostics", "")
        if "no problems detected" in diag.lower():
            print("  Diagnostics: no problems detected")
        else:
            for line in diag.splitlines():
                line = line.strip()
                if line and not line.startswith("Processing"):
                    print(f"  Diagnostics: {line}")
                    break

        ppc = info.get("ppc_p_values", {})
        if ppc:
            parts = [f"{stat}={p:.3f}" for stat, p in ppc.items()]
            print(f"  PPC p-values: {', '.join(parts)}")

        print(f"  Output: {info['output_dir']}")
        print()


# -- main entry point --

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gpt4o_ellsberg_study",
        description="GPT-4o x Ellsberg Study: Cell (1,2) of the 2x2 factorial",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging"
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    p_val = sub.add_parser("validate", help="Validate config, alternative pool, and API keys")
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
        "--skip-collection", action="store_true",
        help="Skip API collection; load saved data instead",
    )
    p_run.add_argument(
        "--skip-fitting", action="store_true",
        help="Stop after data preparation (no model fitting)",
    )
    p_run.set_defaults(func=cmd_run)

    # prepare
    p_prep = sub.add_parser(
        "prepare", help="Re-run data preparation from saved collection data",
    )
    p_prep.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_prep.set_defaults(func=cmd_prepare)

    # fit
    p_fit = sub.add_parser(
        "fit", help="Fit Stan models on existing stan_data_T*.json files",
    )
    p_fit.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_fit.add_argument(
        "-m", "--model", default=None,
        help="Stan model name (e.g. m_02). Overrides config.",
    )
    p_fit.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)

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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (three levels up from this file)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables


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
    from .llm_client import OPENAI_PRICING, ANTHROPIC_PRICING, EMBEDDING_PRICING

    config = StudyConfig.from_yaml(args.config)

    # Load claim pool to count claims
    claims_path = Path(config.claims_file)
    if claims_path.exists():
        with open(claims_path) as f:
            pool = json.load(f)
        n_claims = len(pool.get("claims", []))
    else:
        # Fallback: estimate from problem parameters
        n_claims = 30  # default pool size
        print(f"⚠️   Claim pool not found at {claims_path}, assuming {n_claims} claims")

    n_assess = n_claims * len(config.temperatures)
    n_choices = (
        config.num_problems
        * config.num_presentations
        * len(config.temperatures)
    )
    n_embed = n_assess

    # Token estimates per call (based on prompt templates and typical responses)
    # Assessment: ~300 input tokens (system + user + claim description),
    #             ~100 output tokens (2-4 sentence analysis)
    # Choice:     ~400 input tokens (system + user + assessment texts),
    #             ~10 output tokens (single number)
    # Embedding:  ~200 tokens per assessment response
    assess_input_tok = 300 * n_assess
    assess_output_tok = 100 * n_assess
    choice_input_tok = 400 * n_choices
    choice_output_tok = 10 * n_choices
    embed_tok = 200 * n_embed

    # Look up pricing
    provider = getattr(config, "provider", "openai")
    if provider == "anthropic":
        pricing = ANTHROPIC_PRICING.get(config.llm_model, {"input": 0, "output": 0})
    else:
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

    print("═══ Estimated API Calls ═══")
    print(f"  Assessments   : {n_assess:,.0f}")
    print(f"  Choices       : {n_choices:,.0f}")
    print(f"  Embeddings    : {n_embed:,.0f}")
    print(f"  Total calls   : {n_assess + n_choices + n_embed:,.0f}")
    print()
    print(f"═══ Estimated Cost ({config.llm_model}) ═══")
    print(f"  Assessments   : ${assess_cost:,.2f}  "
          f"({assess_input_tok / 1e6:.2f}M in / {assess_output_tok / 1e6:.2f}M out)")
    print(f"  Choices       : ${choice_cost:,.2f}  "
          f"({choice_input_tok / 1e6:.2f}M in / {choice_output_tok / 1e6:.2f}M out)")
    print(f"  Embeddings    : ${embed_cost:,.2f}  "
          f"({embed_tok / 1e6:.2f}M tokens, {config.embedding_model})")
    print(f"  ─────────────────────────")
    print(f"  Total         : ${total_cost:,.2f}")
    print()
    print("Note: Token counts are estimates based on typical prompt/response sizes.")


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

    print("✅  Model fitting complete.\n")

    for temp_str, info in fit_results.items():
        if not isinstance(info, dict) or "alpha_median" not in info:
            continue

        print(f"── T={temp_str} ──")
        print(
            f"  α  median={info['alpha_median']:.3f}  "
            f"mean={info['alpha_mean']:.3f}  "
            f"90% CI=[{info['alpha_q05']:.3f}, {info['alpha_q95']:.3f}]"
        )

        # Diagnostics one-liner
        diag = info.get("diagnostics", "")
        if "no problems detected" in diag.lower():
            print("  Diagnostics: ✓ no problems detected")
        else:
            # Print first meaningful line of diagnostics
            for line in diag.splitlines():
                line = line.strip()
                if line and not line.startswith("Processing"):
                    print(f"  Diagnostics: {line}")
                    break

        # PPC p-values
        ppc = info.get("ppc_p_values", {})
        if ppc:
            parts = [f"{stat}={p:.3f}" for stat, p in ppc.items()]
            print(f"  PPC p-values: {', '.join(parts)}")

        print(f"  Output: {info['output_dir']}")
        print()


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

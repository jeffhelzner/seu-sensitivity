"""
Command-line interface for the Temperature Study with EU Prompt.

Usage:
    python -m applications.temperature_study_with_eu_prompt run [options]
    python -m applications.temperature_study_with_eu_prompt estimate-cost [options]
    python -m applications.temperature_study_with_eu_prompt validate [options]
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
    """Validate configuration and base study data availability."""
    from .config import StudyConfig

    config = StudyConfig.from_yaml(args.config)
    print(f"✅  Config valid  ({len(config.temperatures)} temperatures, "
          f"model={config.stan_model})")

    base_dir = Path(config.base_study_results_dir)
    if base_dir.exists():
        # Check for required files
        problems_ok = (base_dir / "problems.json").exists()
        assess_ok = all(
            (base_dir / "assessments_T{}.json".format(f"{t:.1f}".replace(".", "_"))).exists()
            for t in config.temperatures
        )
        emb_ok = all(
            (base_dir / "embeddings_reduced_T{}.npz".format(f"{t:.1f}".replace(".", "_"))).exists()
            for t in config.temperatures
        )
        print(f"✅  Base study dir: {base_dir}")
        print(f"    Problems: {'✅' if problems_ok else '❌'}")
        print(f"    Assessments: {'✅' if assess_ok else '❌'}")
        print(f"    Embeddings: {'✅' if emb_ok else '❌'}")
    else:
        print(f"⚠️   Base study results not found at {base_dir}")


def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """Print estimated API cost — only choice calls needed."""
    from .config import StudyConfig
    from applications.temperature_study.llm_client import OPENAI_PRICING, ANTHROPIC_PRICING

    config = StudyConfig.from_yaml(args.config)

    # Load problems to get exact count
    base_dir = Path(config.base_study_results_dir)
    problems_path = base_dir / "problems.json"
    if problems_path.exists():
        with open(problems_path) as f:
            data = json.load(f)
        n_problems = len(data["problems"])
        n_presentations = len(data["problems"][0]["presentations"]) if data["problems"] else 3
    else:
        n_problems = 100
        n_presentations = 3
        print(f"⚠️   Problems not found, assuming {n_problems} × {n_presentations}")

    n_choices = n_problems * n_presentations * len(config.temperatures)

    # Token estimates
    choice_input_tok = 400 * n_choices
    choice_output_tok = 10 * n_choices

    provider = config.provider
    if provider == "anthropic":
        pricing = ANTHROPIC_PRICING.get(config.llm_model, {"input": 0, "output": 0})
    else:
        pricing = OPENAI_PRICING.get(config.llm_model, {"input": 0, "output": 0})

    choice_cost = (
        (choice_input_tok / 1_000_000) * pricing["input"]
        + (choice_output_tok / 1_000_000) * pricing["output"]
    )

    print("═══ Estimated API Calls (EU-Prompt Study) ═══")
    print(f"  Choices only  : {n_choices:,.0f}")
    print(f"  (No assessments or embeddings — reused from base study)")
    print()
    print(f"═══ Estimated Cost ({config.llm_model}) ═══")
    print(f"  Choices       : ${choice_cost:,.2f}  "
          f"({choice_input_tok / 1e6:.2f}M in / {choice_output_tok / 1e6:.2f}M out)")
    print()
    print("Note: Token counts are estimates based on typical prompt/response sizes.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the EU-prompt study pipeline."""
    from .config import StudyConfig
    from .study_runner import EUPromptStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = EUPromptStudyRunner(config)
    summary = runner.run(
        skip_collection=getattr(args, "skip_collection", False),
        skip_model_fitting=getattr(args, "skip_fitting", False),
    )
    print(f"✅  Run complete.")
    print(f"    Duration: {summary.get('duration_seconds', 0):.1f}s")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Re-run data preparation from saved EU-prompt choices."""
    from .config import StudyConfig
    from .study_runner import EUPromptStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = EUPromptStudyRunner(config)
    summary = runner.run(skip_collection=True, skip_model_fitting=True)
    print("✅  Data preparation complete.")
    for temp_str, info in summary["phases"]["data_preparation"]["per_temperature"].items():
        print(f"    T={temp_str}: M={info['M']}, R={info['R']}, D={info['D']}, NAs removed={info['na_removed']}")


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit Stan models on existing Stan data files."""
    from .config import StudyConfig
    from .study_runner import EUPromptStudyRunner

    config = StudyConfig.from_yaml(args.config)
    runner = EUPromptStudyRunner(config)
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

        diag = info.get("diagnostics", "")
        if "no problems detected" in diag.lower():
            print("  Diagnostics: ✓ no problems detected")
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


# ── main entry point ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="temperature_study_with_eu_prompt",
        description="Temperature Study with EU Prompt: explicit EU-maximization instruction",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging"
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    p_val = sub.add_parser("validate", help="Validate config and base study data")
    p_val.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_val.set_defaults(func=cmd_validate)

    # estimate-cost
    p_cost = sub.add_parser("estimate-cost", help="Estimate API costs")
    p_cost.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_cost.set_defaults(func=cmd_estimate_cost)

    # run
    p_run = sub.add_parser("run", help="Run EU-prompt study pipeline")
    p_run.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_run.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip API collection; load saved choices instead",
    )
    p_run.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Stop after data preparation (no model fitting)",
    )
    p_run.set_defaults(func=cmd_run)

    # prepare (re-run data prep only)
    p_prep = sub.add_parser(
        "prepare",
        help="Re-run data preparation from saved EU-prompt choices",
    )
    p_prep.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_prep.set_defaults(func=cmd_prepare)

    # fit (model fitting only)
    p_fit = sub.add_parser(
        "fit",
        help="Fit Stan models on existing Stan data",
    )
    p_fit.add_argument("-c", "--config", default=None, help="Path to YAML config")
    p_fit.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)

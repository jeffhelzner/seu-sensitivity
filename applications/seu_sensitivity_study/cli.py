"""
Command-line interface for the SEU Sensitivity Study.

Usage:
    python -m applications.seu_sensitivity_study run [options]
    python -m applications.seu_sensitivity_study validate [options]
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


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate configuration and prompts without making API calls."""
    from .config import SEUSensitivityStudyConfig
    from . import prompts as prompts_module

    config = (
        SEUSensitivityStudyConfig.from_yaml(args.config)
        if args.config
        else SEUSensitivityStudyConfig()
    )

    print(
        f"Config valid: {len(config.cells)} cells across {len(config.pool_ids)} pool(s); "
        f"K={config.K}, menu sizes {config.menu_sizes}, "
        f"{config.num_presentations} presentations"
    )
    print(f"Expected choice calls: {config.expected_choice_calls():,}")

    for pool_id in config.pool_ids:
        X, columns, cell_ids = config.design_matrix_for_pool(pool_id)
        print(f"\n[{pool_id}] design matrix {X.shape[0]} x {X.shape[1]}")
        print(f"  columns: {columns}")
        print(f"  menus per family: {config.problems_for(pool_id)}")
        try:
            resolved = prompts_module.load_prompt_sets(pool_id)
            print(f"  prompts OK for families: {sorted(resolved)}")
        except (FileNotFoundError, ValueError) as error:
            print(f"  prompts FAILED: {error}")

    if args.cells:
        print("\nCells:")
        for cell in config.cells:
            print(
                f"  {cell.cell_id}: {cell.model_name} ({cell.provider}) "
                f"[{cell.prompt_condition}] temp={cell.temperature}"
            )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the collection pipeline."""
    from .config import SEUSensitivityStudyConfig
    from .study_runner import SEUSensitivityStudyRunner

    config = (
        SEUSensitivityStudyConfig.from_yaml(args.config)
        if args.config
        else SEUSensitivityStudyConfig()
    )
    if args.output_dir:
        config.results_dir = args.output_dir

    runner = SEUSensitivityStudyRunner(config)
    output = runner.run(
        phases=args.phases.split(",") if args.phases else None,
        pool_ids=args.pools.split(",") if args.pools else None,
        cell_ids=args.cells.split(",") if args.cells else None,
        model_names=args.models.split(",") if args.models else None,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(output, indent=2, default=str))


def cmd_manifest(args: argparse.Namespace) -> None:
    """Write the provenance manifest (§6.5)."""
    from .config import SEUSensitivityStudyConfig
    from .study_runner import SEUSensitivityStudyRunner

    config = (
        SEUSensitivityStudyConfig.from_yaml(args.config)
        if args.config
        else SEUSensitivityStudyConfig()
    )
    if args.output_dir:
        config.results_dir = args.output_dir

    manifest = SEUSensitivityStudyRunner(config).write_manifest()
    print(json.dumps(manifest, indent=2, default=str))


def main() -> None:
    from .study_runner import PHASES

    parser = argparse.ArgumentParser(
        prog="seu_sensitivity_study",
        description="6-model x 3-prompt x 3-pool SEU sensitivity study",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    p_validate = subparsers.add_parser("validate", help="Validate config and prompts")
    p_validate.add_argument("--config", type=str, default=None)
    p_validate.add_argument("--cells", action="store_true", help="List every cell")

    p_run = subparsers.add_parser("run", help="Run the collection pipeline")
    p_run.add_argument("--config", type=str, default=None)
    p_run.add_argument("--output-dir", type=str, default=None)
    p_run.add_argument(
        "--phases",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {','.join(PHASES)}",
    )
    p_run.add_argument("--pools", type=str, default=None, help="Comma-separated pool ids")
    p_run.add_argument("--cells", type=str, default=None, help="Comma-separated cell ids")
    p_run.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model names; restricts the 'assess' phase only. "
            "Produces a PARTIAL assessment set -- use for costed probes."
        ),
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned API call counts and exit without calling",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Proceed past an uncleared validation gate (recorded in the summary)",
    )

    p_manifest = subparsers.add_parser("manifest", help="Write the provenance manifest")
    p_manifest.add_argument("--config", type=str, default=None)
    p_manifest.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    else:
        parser.print_help()
        sys.exit(1)

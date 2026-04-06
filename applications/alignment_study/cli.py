"""
Command-line interface for the Alignment Study.

Usage:
    python -m applications.alignment_study run [options]
    python -m applications.alignment_study validate [options]
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
    """Validate configuration without making API calls."""
    from .config import AlignmentStudyConfig

    if args.config:
        config = AlignmentStudyConfig.from_yaml(args.config)
    else:
        config = AlignmentStudyConfig()

    print(f"Config valid: {len(config.cells)} cells, "
          f"{config.num_problems} problems, K={config.K}")

    for cell in config.cells:
        print(f"  {cell.cell_id}: {cell.model_name} ({cell.provider}) "
              f"[{cell.prompt_condition}]")

    X, col_names = config.get_design_matrix()
    print(f"\nDesign matrix: {X.shape[0]} x {X.shape[1]}")
    print(f"Columns: {col_names}")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the study pipeline."""
    from .config import AlignmentStudyConfig
    from .study_runner import AlignmentStudyRunner

    if args.config:
        config = AlignmentStudyConfig.from_yaml(args.config)
    else:
        config = AlignmentStudyConfig()

    if args.output_dir:
        config.results_dir = args.output_dir

    runner = AlignmentStudyRunner(config)
    output = runner.run(
        skip_collection=args.skip_collection,
        cells_to_run=args.cells.split(",") if args.cells else None,
    )

    print(json.dumps(output, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alignment_study",
        description="6-model × 3-prompt alignment study",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate config")
    p_validate.add_argument("--config", type=str, default=None)

    # run
    p_run = subparsers.add_parser("run", help="Run the study")
    p_run.add_argument("--config", type=str, default=None)
    p_run.add_argument("--output-dir", type=str, default=None)
    p_run.add_argument("--skip-collection", action="store_true",
                       help="Skip API calls; load existing data")
    p_run.add_argument("--cells", type=str, default=None,
                       help="Comma-separated cell IDs to run")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)

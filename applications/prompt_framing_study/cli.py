"""
Command-line interface for the prompt framing study.
"""
import argparse
import logging
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (three levels up from this file)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try current working directory
        load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_run(args):
    """Run the full study pipeline."""
    from .study_runner import StudyRunner
    
    runner = StudyRunner(config_path=args.config)
    
    if args.dry_run:
        estimate = runner.estimate_cost()
        from .cost_estimator import CostEstimator
        estimator = CostEstimator()
        estimator.print_estimate(estimate)
        print("\nDry run complete. Use without --dry-run to execute.")
        return
    
    results = runner.run()
    print(f"\nStudy complete! Results saved to: {results.get('results_dir')}")


def cmd_estimate(args):
    """Estimate costs for a study configuration."""
    from .cost_estimator import CostEstimator
    import yaml
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Try to get num_claims from claims file
    num_claims = args.num_claims
    if num_claims is None:
        claims_path = config.get("claims_file", "data/claims.json")
        base_dir = Path(args.config).parent.parent
        full_path = base_dir / claims_path
        if full_path.exists():
            import json
            with open(full_path, 'r') as f:
                claims_data = json.load(f)
            num_claims = len(claims_data.get("claims", []))
        else:
            num_claims = 20  # Default
    
    estimator = CostEstimator()
    estimate = estimator.estimate_total_study_cost(
        num_claims=num_claims,
        num_problems=config.get("num_problems", 100),
        num_variants=config.get("num_variants", 4),
        num_repetitions=config.get("num_repetitions", 1),
        avg_alternatives=(
            config.get("min_alternatives", 2) + 
            config.get("max_alternatives", 4)
        ) / 2,
        llm_model=config.get("llm_model", "gpt-4"),
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        provider=config.get("provider", "openai")
    )
    estimator.print_estimate(estimate)


def cmd_validate(args):
    """Validate configuration and data files."""
    from .validation import validate_config, validate_claims_file, validate_stan_data
    import yaml
    import json
    
    errors = []
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            warnings = validate_config(config)
            print(f"✓ Config valid: {args.config}")
            for w in warnings:
                print(f"  ⚠ {w}")
        except Exception as e:
            errors.append(f"Config: {e}")
            print(f"✗ Config invalid: {e}")
    
    if args.claims:
        try:
            validate_claims_file(args.claims)
            print(f"✓ Claims valid: {args.claims}")
        except Exception as e:
            errors.append(f"Claims: {e}")
            print(f"✗ Claims invalid: {e}")
    
    if args.stan_data:
        try:
            with open(args.stan_data, 'r') as f:
                data = json.load(f)
            validate_stan_data(data, model=args.model)
            print(f"✓ Stan data valid: {args.stan_data}")
        except Exception as e:
            errors.append(f"Stan data: {e}")
            print(f"✗ Stan data invalid: {e}")
    
    if errors:
        sys.exit(1)


def cmd_visualize(args):
    """Generate visualizations from results."""
    from .visualization import StudyVisualizer
    import json
    
    viz = StudyVisualizer(args.results_dir)
    
    metadata_path = Path(args.results_dir) / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: No metadata found at {metadata_path}")
        metadata = {}
    
    if args.alpha_comparison:
        if "model_fits" in metadata:
            output = args.output or str(Path(args.results_dir) / "alpha_comparison.png")
            viz.plot_alpha_comparison(
                metadata["model_fits"],
                save_path=output,
                show=not args.no_show
            )
            print(f"Saved alpha comparison plot to {output}")
        else:
            print("No model fits found in results. Run with fit_models=True first.")
    
    if args.choices:
        choices_path = Path(args.results_dir) / "raw_choices.json"
        if choices_path.exists():
            with open(choices_path, 'r') as f:
                choices = json.load(f)
            output = args.output or str(Path(args.results_dir) / "choice_distributions.png")
            viz.plot_choice_distribution(choices, save_path=output, show=not args.no_show)
            print(f"Saved choice distribution plot to {output}")
        else:
            print(f"No choices file found at {choices_path}")
    
    if args.report:
        output_dir = viz.create_summary_report(metadata, output_dir=args.output)
        print(f"Generated summary report in {output_dir}")


def cmd_resume(args):
    """Resume an interrupted study run."""
    from .study_runner import StudyRunner
    
    runner = StudyRunner(results_dir=args.results_dir)
    results = runner.resume()
    print(f"\nStudy resumed and complete! Results in: {results.get('results_dir')}")


def main():
    parser = argparse.ArgumentParser(
        prog="prompt_framing_study",
        description="Investigate how prompt framing affects LLM rationality"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Default config path relative to module
    default_config = Path(__file__).parent / "configs" / "study_config.yaml"
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the study pipeline")
    run_parser.add_argument(
        "-c", "--config", 
        default=str(default_config),
        help="Path to study config"
    )
    run_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show cost estimate without running"
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate study costs")
    est_parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Path to study config"
    )
    est_parser.add_argument(
        "--num-claims", 
        type=int,
        help="Number of base claims (auto-detected if not specified)"
    )
    est_parser.set_defaults(func=cmd_estimate)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate files")
    val_parser.add_argument("--config", help="Validate config file")
    val_parser.add_argument("--claims", help="Validate claims file")
    val_parser.add_argument("--stan-data", help="Validate Stan data file")
    val_parser.add_argument(
        "--model", 
        default="m_0", 
        choices=["m_0", "m_1"],
        help="Model for Stan data validation"
    )
    val_parser.set_defaults(func=cmd_validate)
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("results_dir", help="Path to results directory")
    viz_parser.add_argument(
        "--alpha-comparison", 
        action="store_true",
        help="Generate alpha comparison plot"
    )
    viz_parser.add_argument(
        "--choices", 
        action="store_true",
        help="Generate choice distribution plots"
    )
    viz_parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate full summary report"
    )
    viz_parser.add_argument(
        "-o", "--output", 
        help="Output file/directory path"
    )
    viz_parser.add_argument(
        "--no-show", 
        action="store_true",
        help="Don't display plots interactively"
    )
    viz_parser.set_defaults(func=cmd_visualize)
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume interrupted study")
    resume_parser.add_argument(
        "results_dir", 
        help="Path to results directory with checkpoint"
    )
    resume_parser.set_defaults(func=cmd_resume)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

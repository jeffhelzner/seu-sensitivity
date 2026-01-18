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


def cmd_fit(args):
    """Fit Stan models to collected data."""
    import json
    import numpy as np
    
    results_dir = Path(args.results_dir)
    
    # Load metadata to get variant names
    metadata_path = results_dir / "run_metadata.json"
    if not metadata_path.exists():
        print(f"Error: No metadata found at {metadata_path}")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    variants = metadata.get("variants", [])
    if not variants:
        print("Error: No variants found in metadata")
        sys.exit(1)
    
    try:
        import cmdstanpy
    except ImportError:
        print("Error: cmdstanpy not installed. Install with: pip install cmdstanpy")
        sys.exit(1)
    
    # Find the model file
    model_name = args.model
    model_path = Path(__file__).parent.parent.parent / "models" / f"{model_name}.stan"
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Compiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
    
    fit_results = {}
    
    for variant in variants:
        stan_data_path = results_dir / f"stan_data_{variant}.json"
        if not stan_data_path.exists():
            print(f"Warning: Stan data not found for {variant}, skipping")
            continue
        
        print(f"\nFitting model for variant: {variant}")
        
        with open(stan_data_path, 'r') as f:
            stan_data = json.load(f)
        
        try:
            fit = model.sample(
                data=stan_data,
                chains=args.chains,
                parallel_chains=args.chains,
                iter_warmup=args.warmup,
                iter_sampling=args.samples,
                seed=args.seed
            )
            
            # Extract alpha posterior
            alpha_samples = fit.stan_variable("alpha")
            
            fit_results[variant] = {
                "alpha_mean": float(np.mean(alpha_samples)),
                "alpha_std": float(np.std(alpha_samples)),
                "alpha_median": float(np.median(alpha_samples)),
                "alpha_q05": float(np.percentile(alpha_samples, 5)),
                "alpha_q95": float(np.percentile(alpha_samples, 95)),
            }
            
            print(f"  α = {fit_results[variant]['alpha_mean']:.3f} "
                  f"(95% CI: [{fit_results[variant]['alpha_q05']:.3f}, "
                  f"{fit_results[variant]['alpha_q95']:.3f}])")
            
            # Save samples
            fit.save_csvfiles(str(results_dir / f"samples_{variant}"))
            
        except Exception as e:
            print(f"  Error fitting model: {e}")
            fit_results[variant] = {"error": str(e)}
    
    # Update metadata with fit results
    metadata["model_fits"] = fit_results
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel fitting complete! Results saved to {metadata_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("ALPHA ESTIMATES BY PROMPT VARIANT")
    print("="*60)
    print(f"{'Variant':<12} {'Mean α':>10} {'Std':>8} {'95% CI':>20}")
    print("-"*60)
    for variant in variants:
        if variant in fit_results and "error" not in fit_results[variant]:
            r = fit_results[variant]
            ci = f"[{r['alpha_q05']:.2f}, {r['alpha_q95']:.2f}]"
            print(f"{variant:<12} {r['alpha_mean']:>10.3f} {r['alpha_std']:>8.3f} {ci:>20}")
    print("="*60)


def cmd_robustness(args):
    """Run robustness analysis on collected data."""
    from .robustness_analysis import RobustnessAnalyzer
    from .contextualized_embedding import ContextualizedEmbeddingManager
    import json
    
    results_dir = Path(args.results_dir)
    
    # Load embeddings
    embeddings_path = results_dir / "raw_embeddings.npz"
    if not embeddings_path.exists():
        print(f"Error: No embeddings found at {embeddings_path}")
        sys.exit(1)
    
    print("Loading embeddings...")
    embeddings = ContextualizedEmbeddingManager.load_embeddings(str(embeddings_path))
    
    # Load metadata
    metadata_path = results_dir / "run_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    analyzer = RobustnessAnalyzer(
        results_dir=str(results_dir),
        target_dims=args.dims or [16, 32, 64, 128]
    )
    
    print("\nAnalyzing PCA dimension sensitivity...")
    pca_results = analyzer.analyze_pca_sensitivity(
        embeddings=embeddings,
        claims=[],  # Not needed for basic analysis
        problems=[],
        choices={},
        fit_model=False
    )
    
    print("\nAnalyzing reduction method sensitivity...")
    reduction_results = analyzer.analyze_reduction_method_sensitivity(
        embeddings=embeddings,
        target_dim=metadata.get("config", {}).get("target_dim", 32)
    )
    
    # Save robustness results
    robustness_output = results_dir / "robustness_analysis.json"
    with open(robustness_output, 'w') as f:
        json.dump({
            "pca_sensitivity": pca_results,
            "reduction_method_sensitivity": reduction_results
        }, f, indent=2)
    
    print(f"\nRobustness analysis complete! Results saved to {robustness_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("PCA DIMENSION SENSITIVITY")
    print("="*60)
    for dim_key, dim_results in pca_results.get("analysis", {}).items():
        dim = dim_key.replace("dim_", "")
        print(f"\nTarget dimension: {dim}")
        for variant, stats in dim_results.items():
            exp_var = stats.get("explained_variance")
            if exp_var:
                print(f"  {variant}: {exp_var*100:.1f}% variance explained")


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
    
    # Fit command (post-hoc model fitting)
    fit_parser = subparsers.add_parser("fit", help="Fit Stan models to collected data")
    fit_parser.add_argument(
        "results_dir",
        help="Path to results directory"
    )
    fit_parser.add_argument(
        "--model",
        default="m_0",
        choices=["m_0", "m_1"],
        help="Stan model to fit"
    )
    fit_parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains"
    )
    fit_parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Warmup iterations per chain"
    )
    fit_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Sampling iterations per chain"
    )
    fit_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    fit_parser.set_defaults(func=cmd_fit)
    
    # Robustness command
    robust_parser = subparsers.add_parser("robustness", help="Run robustness analysis")
    robust_parser.add_argument(
        "results_dir",
        help="Path to results directory"
    )
    robust_parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        help="PCA dimensions to test (default: 16 32 64 128)"
    )
    robust_parser.set_defaults(func=cmd_robustness)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

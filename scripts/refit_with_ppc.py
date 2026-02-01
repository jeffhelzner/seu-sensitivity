"""
Refit Models with Posterior Predictive Checks

This script refits the m_0 model to existing Stan data files and generates
posterior predictive check statistics. Useful for adding PPC analysis to
previously collected pilot study data.

Usage:
    python scripts/refit_with_ppc.py --data-dir path/to/stan_data_files
    python scripts/refit_with_ppc.py --data-dir applications/prompt_framing_study/results/run_20260117_184005
    
Output:
    - Updated run_metadata.json with PPC results
    - PPC plots in ppc/ subdirectory
    - ppc_summary.json for each variant
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cmdstanpy import CmdStanModel
from analysis.posterior_predictive_checks import PosteriorPredictiveChecker


def refit_with_ppc(
    data_dir: str,
    output_dir: str = None,
    model_path: str = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 42,
    variants: list = None
):
    """
    Refit models to existing Stan data with PPC statistics.
    
    Args:
        data_dir: Directory containing stan_data_*.json files
        output_dir: Directory for output (defaults to data_dir)
        model_path: Path to Stan model (defaults to models/m_0.stan)
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations
        iter_sampling: Sampling iterations
        seed: Random seed
        variants: List of variants to fit (defaults to all found)
    
    Returns:
        Dictionary with fit results and PPC summaries
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    
    if model_path is None:
        model_path = project_root / "models" / "m_0.stan"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Find Stan data files
    stan_files = list(data_dir.glob("stan_data_*.json"))
    if not stan_files:
        raise FileNotFoundError(f"No stan_data_*.json files found in {data_dir}")
    
    # Extract variant names
    available_variants = [f.stem.replace("stan_data_", "") for f in stan_files]
    
    if variants:
        variants = [v for v in variants if v in available_variants]
    else:
        variants = available_variants
    
    print(f"Found variants: {variants}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Compile model
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=str(model_path))
    
    # Create output directories
    ppc_dir = output_dir / "ppc"
    ppc_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Fitting variant: {variant}")
        print('='*60)
        
        # Load Stan data
        stan_file = data_dir / f"stan_data_{variant}.json"
        with open(stan_file, 'r') as f:
            stan_data = json.load(f)
        
        print(f"  Problems (M): {stan_data['M']}")
        print(f"  Consequences (K): {stan_data['K']}")
        print(f"  Dimensions (D): {stan_data['D']}")
        
        # Fit model
        print(f"  Running MCMC ({chains} chains × {iter_sampling} samples)...")
        fit = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=min(chains, 4),
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed
        )
        
        # Extract alpha posterior
        alpha_samples = fit.stan_variable("alpha")
        
        fit_results = {
            "alpha_mean": float(np.mean(alpha_samples)),
            "alpha_std": float(np.std(alpha_samples)),
            "alpha_median": float(np.median(alpha_samples)),
            "alpha_q05": float(np.percentile(alpha_samples, 5)),
            "alpha_q95": float(np.percentile(alpha_samples, 95)),
            "alpha_q025": float(np.percentile(alpha_samples, 2.5)),
            "alpha_q975": float(np.percentile(alpha_samples, 97.5)),
        }
        
        print(f"  α estimate: {fit_results['alpha_mean']:.3f} ± {fit_results['alpha_std']:.3f}")
        print(f"  α 90% CI: [{fit_results['alpha_q05']:.3f}, {fit_results['alpha_q95']:.3f}]")
        
        # Run posterior predictive checks
        print(f"  Running posterior predictive checks...")
        try:
            checker = PosteriorPredictiveChecker(fit, stan_data)
            ppc_results = checker.to_dict()
            
            fit_results["ppc"] = ppc_results
            
            # Print PPC summary
            print(f"\n  PPC Results:")
            for stat, p in ppc_results["p_values"].items():
                symbol = ppc_results["interpretation"][stat]["status"]
                print(f"    {stat}: p = {p:.3f} {symbol}")
            
            # Save PPC plots
            variant_ppc_dir = ppc_dir / variant
            variant_ppc_dir.mkdir(parents=True, exist_ok=True)
            checker.plot_all_diagnostics(str(variant_ppc_dir), show=False)
            
            # Save PPC summary JSON
            ppc_json_path = variant_ppc_dir / "ppc_summary.json"
            with open(ppc_json_path, 'w') as f:
                json.dump(ppc_results, f, indent=2)
            
            print(f"  PPC plots saved to: {variant_ppc_dir}")
            
        except Exception as e:
            print(f"  Warning: PPC failed: {e}")
            fit_results["ppc"] = {"error": str(e)}
        
        # Save samples
        samples_dir = output_dir / f"samples_{variant}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        fit.save_csvfiles(str(samples_dir))
        
        results[variant] = fit_results
    
    # Update or create run_metadata.json
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata["model_fits"] = results
    metadata["ppc_analysis_completed"] = True
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Updated metadata: {metadata_path}")
    
    # Print summary table
    print("\nFit Summary:")
    print("-" * 70)
    print(f"{'Variant':<12} {'α Mean':>8} {'α Std':>8} {'90% CI':>18} {'PPC LL':>10}")
    print("-" * 70)
    for variant, res in results.items():
        ci = f"[{res['alpha_q05']:.2f}, {res['alpha_q95']:.2f}]"
        ppc_ll = res.get("ppc", {}).get("p_values", {}).get("ll", "N/A")
        if isinstance(ppc_ll, float):
            ppc_ll = f"{ppc_ll:.3f}"
        print(f"{variant:<12} {res['alpha_mean']:>8.3f} {res['alpha_std']:>8.3f} {ci:>18} {ppc_ll:>10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Refit models with posterior predictive checks"
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Directory containing stan_data_*.json files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (defaults to data-dir)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to Stan model file"
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains"
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=1000,
        help="Warmup iterations"
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=1000,
        help="Sampling iterations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+",
        default=None,
        help="Variants to fit (default: all found)"
    )
    
    args = parser.parse_args()
    
    refit_with_ppc(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        variants=args.variants
    )


if __name__ == "__main__":
    main()

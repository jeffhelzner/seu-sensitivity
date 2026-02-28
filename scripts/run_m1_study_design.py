#!/usr/bin/env python3
"""
Study Design Generation Script for m_1 Model

This script creates experimental study designs for the combined risky + uncertain
choice model (m_1). The m_1 model extends m_0 by pairing uncertain decision
problems (feature-derived probabilities) with risky decision problems (known
objective probabilities), enabling separate identification of utilities and
subjective probability mappings.

Purpose:
    - Generate study designs with both uncertain (M) and risky (N) decision problems
    - Configure feature spaces for uncertain alternatives and probability simplexes
      for risky alternatives
    - Analyze properties of the generated design
    - Save designs in Stan-compatible format for downstream analyses

Usage:
    python scripts/run_m1_study_design.py --config configs/m1_study_config.json
    python scripts/run_m1_study_design.py --M 30 --N 30 --output m1_test.json

    The config file should be a JSON file with the following structure:
    {
        "M": 20,                   # Number of uncertain decision problems
        "N": 20,                   # Number of risky decision problems
        "K": 3,                    # Number of possible consequences
        "D": 2,                    # Dimensions to describe uncertain alternatives
        "R": 10,                   # Number of distinct uncertain alternatives
        "S": 8,                    # Number of distinct risky alternatives
        "min_alts_per_problem": 2, # Minimum alternatives per problem
        "max_alts_per_problem": 5, # Maximum alternatives per problem
        "risky_probs": "fixed",    # How to generate risky probabilities
        "feature_dist": "normal",  # Distribution for uncertain features
        "feature_params": {"loc": 0, "scale": 1},
        "design_name": "m1_study"  # Name for the design
    }

Examples:
    # Generate from config file
    python scripts/run_m1_study_design.py --config configs/m1_study_config.json

    # Generate from command-line arguments
    python scripts/run_m1_study_design.py --M 30 --N 30 --K 4 --R 15 --S 10

    # Custom output location without plots
    python scripts/run_m1_study_design.py --M 20 --N 20 --output my_design.json --no-plots

Outputs:
    - Study design JSON file in results/designs/ (Stan-compatible)
    - Visualization plots in results/designs/<name>_plots/ (unless --no-plots)
    - Design summary printed to console
"""
import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.study_design_m1 import StudyDesignM1


def main():
    parser = argparse.ArgumentParser(
        description='Generate study design for m_1 model (combined risky and uncertain choice)'
    )
    parser.add_argument('--config', type=str, 
                       help='Path to configuration JSON file')
    parser.add_argument('--M', type=int, default=20,
                       help='Number of uncertain decision problems')
    parser.add_argument('--N', type=int, default=20,
                       help='Number of risky decision problems')
    parser.add_argument('--K', type=int, default=3,
                       help='Number of possible consequences')
    parser.add_argument('--D', type=int, default=2,
                       help='Dimensions of alternative features')
    parser.add_argument('--R', type=int, default=10,
                       help='Number of distinct uncertain alternatives')
    parser.add_argument('--S', type=int, default=8,
                       help='Number of distinct risky alternatives')
    parser.add_argument('--min-alts', type=int, default=2,
                       help='Minimum alternatives per problem')
    parser.add_argument('--max-alts', type=int, default=5,
                       help='Maximum alternatives per problem')
    parser.add_argument('--risky-probs', type=str, default='fixed',
                       choices=['uniform', 'fixed', 'random'],
                       help='Method for generating risky probabilities')
    parser.add_argument('--output', type=str, default='m1_study.json',
                       help='Output filename (saved to results/designs/)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        design = StudyDesignM1.from_config(args.config)
    else:
        # Create design from command-line arguments
        design = StudyDesignM1(
            M=args.M,
            N=args.N,
            K=args.K,
            D=args.D,
            R=args.R,
            S=args.S,
            min_alts_per_problem=args.min_alts,
            max_alts_per_problem=args.max_alts,
            risky_probs=args.risky_probs
        )
    
    # Generate the design
    print("\nGenerating study design...")
    design.generate()
    
    # Print summary
    print("\n" + "="*60)
    print("STUDY DESIGN SUMMARY")
    print("="*60)
    print(f"Model: m_1 (combined risky and uncertain choice)")
    print(f"\nUncertain Choice Problems:")
    print(f"  Problems (M): {design.M}")
    print(f"  Alternatives (R): {design.R}")
    print(f"  Feature dimensions (D): {design.D}")
    print(f"\nRisky Choice Problems:")
    print(f"  Problems (N): {design.N}")
    print(f"  Alternatives (S): {design.S}")
    print(f"  Probability generation: {design.risky_probs}")
    print(f"\nShared Parameters:")
    print(f"  Consequences (K): {design.K}")
    print(f"  Alternatives per problem: {args.min_alts}-{args.max_alts}")
    print(f"\nTotal decision problems: {design.M + design.N}")
    print("="*60 + "\n")
    
    # Save the design
    output_path = design.save(
        args.output, 
        include_metadata=True, 
        include_plots=not args.no_plots
    )
    
    print(f"\n✓ Study design successfully saved to: {output_path}")
    
    if not args.no_plots:
        plot_dir = output_path.parent / f"{output_path.stem}_plots"
        print(f"✓ Visualization plots saved to: {plot_dir}")
    
    # Print some statistics
    if hasattr(design, 'metadata'):
        print("\nDesign Statistics:")
        print(f"  Uncertain alternatives per problem: "
              f"{design.metadata['n_alts_per_problem_mean']:.2f} "
              f"(range: {design.metadata['n_alts_per_problem_min']}-"
              f"{design.metadata['n_alts_per_problem_max']})")
        print(f"  Risky alternatives per problem: "
              f"{design.metadata['risky_n_alts_per_problem_mean']:.2f} "
              f"(range: {design.metadata['risky_n_alts_per_problem_min']}-"
              f"{design.metadata['risky_n_alts_per_problem_max']})")


if __name__ == "__main__":
    main()

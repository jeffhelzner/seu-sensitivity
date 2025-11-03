"""
Property Claim Problem Generator

This script demonstrates how to use the ClaimDesignGenerator to create
decision problems from a set of base property claims.

Workflow:
1. Load base property claims from data/property_claims.json
2. Generate decision problems with varying numbers of alternatives
3. Save the generated problems to problems/property_problems.json
4. Print analysis of the generated problem set

Usage:
    python applications/llm_rationality/generate_property_problems.py

Requirements:
    - Base claims in data/property_claims.json

Output:
    - Generated problems saved to problems/property_problems.json
    - Printed analysis of the generated problems
"""
import os
from pathlib import Path
from claim_design import ClaimDesignGenerator

def main():
    # Get the directory where this script is located
    base_dir = Path(__file__).parent
    
    # Create necessary directories using absolute paths
    data_dir = base_dir / "data"
    problems_dir = base_dir / "problems"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(problems_dir, exist_ok=True)
    
    # Use absolute paths for files
    claims_file = data_dir / "property_claims.json"
    problems_file = problems_dir / "property_problems.json"
    
    # Initialize generator with absolute path
    generator = ClaimDesignGenerator(str(claims_file))
    
    # Generate problems with varying numbers of alternatives
    problems = generator.generate_problems(
        num_problems=10,
        min_alts=2,
        max_alts=5,
        seed=42  # For reproducibility
    )
    
    # Save problems using absolute path
    generator.save_problems(problems, str(problems_file))
    
    # Analyze and print statistics
    analysis = generator.analyze_problems(problems)
    print("\nProblem Analysis:")
    print(f"Total problems: {analysis['num_problems']}")
    print(f"Alternatives per problem: {analysis['alternatives_per_problem']['min']} to "
          f"{analysis['alternatives_per_problem']['max']} "
          f"(mean: {analysis['alternatives_per_problem']['mean']:.2f})")
    print("\nDistribution of alternatives per problem:")
    for count, freq in sorted(analysis['alternatives_per_problem']['distribution'].items()):
        print(f"  {count} alternatives: {freq} problems")
    
    print("\nClaim usage distribution:")
    for claim_id, count in sorted(analysis['claim_usage']['distribution'].items()):
        print(f"  {claim_id}: used in {count} problems")

if __name__ == "__main__":
    main()
"""
Claim-based Study Design Generator for LLM Rationality Benchmarking

This module provides tools for generating decision problems from a set of 
text-based claims. It follows a similar approach to the StudyDesign class
but is specialized for creating text-based decision problems.

Workflow:
1. Create a JSON file with base descriptions (e.g., property claims)
2. Use ClaimDesignGenerator to create decision problems by sampling from these descriptions
3. Save generated problems for use in benchmarking

Example:
    # Create decision problems from base claims
    generator = ClaimDesignGenerator("data/property_claims.json")
    problems = generator.generate_problems(num_problems=10, min_alts=2, max_alts=5)
    generator.save_problems(problems, "problems/property_problems.json")
"""
import json
import numpy as np
import os
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class ClaimDesignGenerator:
    """
    Generate decision problems from a set of base text claims.
    
    This class takes a set of base claims and generates decision problems
    by selecting subsets of these claims with varying numbers of alternatives.
    """
    
    def __init__(self, claims_file: str):
        """Load claims and context from file."""
        with open(claims_file, 'r') as f:
            data = json.load(f)
            self.claims = data.get("claims", data)  # Support both formats
            self.context = data.get("context", "Default context")
            
    def generate_problems(self, 
                          num_problems: int = 10, 
                          min_alts: int = 2, 
                          max_alts: int = 5,
                          seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate decision problems by selecting subsets of claims.
        
        Args:
            num_problems: Number of decision problems to generate
            min_alts: Minimum alternatives per problem
            max_alts: Maximum alternatives per problem
            seed: Random seed for reproducibility
            
        Returns:
            List of decision problems with context and alternatives
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        problems = []
        
        for i in range(num_problems):
            # Determine number of alternatives for this problem
            n_alts = random.randint(min_alts, min(max_alts, len(self.claims)))
            
            # Select random subset of claims
            selected_claims = random.sample(self.claims, n_alts)
            
            # Create problem
            problem = {
                "id": f"PP{i+1:03d}",
                "context": self.context,
                "alternatives": [claim["description"] for claim in selected_claims],
                "metadata": {
                    "claim_ids": [claim["id"] for claim in selected_claims],
                    "num_alternatives": n_alts,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            problems.append(problem)
            
        return problems
    
    def save_problems(self, problems: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save generated problems to a JSON file.
        
        Args:
            problems: List of generated decision problems
            filepath: Path to save the problems
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add metadata
        output = {
            "problems": problems,
            "metadata": {
                "num_problems": len(problems),
                "base_claims_file": self.claims_file,
                "generated_at": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Saved {len(problems)} problems to {filepath}")
    
    @staticmethod
    def load_problems(filepath: str) -> List[Dict[str, Any]]:
        """
        Load problems from a JSON file.
        
        Args:
            filepath: Path to the problems file
            
        Returns:
            List of decision problems
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if "problems" in data:
            return data["problems"]
        else:
            # Handle legacy format
            return data
            
    def analyze_problems(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the generated problems.
        
        Args:
            problems: List of decision problems
            
        Returns:
            Dictionary of analysis metrics
        """
        # Count occurrences of each claim
        claim_counts = {}
        for problem in problems:
            for claim_id in problem["metadata"]["claim_ids"]:
                claim_counts[claim_id] = claim_counts.get(claim_id, 0) + 1
                
        # Count alternatives per problem
        alt_counts = [problem["metadata"]["num_alternatives"] for problem in problems]
        
        return {
            "num_problems": len(problems),
            "alternatives_per_problem": {
                "min": min(alt_counts),
                "max": max(alt_counts),
                "mean": sum(alt_counts) / len(alt_counts),
                "distribution": {str(i): alt_counts.count(i) for i in set(alt_counts)}
            },
            "claim_usage": {
                "min": min(claim_counts.values()) if claim_counts else 0,
                "max": max(claim_counts.values()) if claim_counts else 0,
                "mean": sum(claim_counts.values()) / len(claim_counts) if claim_counts else 0,
                "distribution": claim_counts
            }
        }
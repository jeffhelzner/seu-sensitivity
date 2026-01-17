"""
Problem Generator Module for Prompt Framing Study

Generates decision problems from base claims.
"""
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class ProblemGenerator:
    """
    Generate decision problems by sampling from base claims.
    
    Adapted from legacy ClaimDesignGenerator with improvements.
    """
    
    def __init__(self, claims_file: Optional[str] = None, claims: Optional[List[Dict]] = None):
        """
        Initialize the problem generator.
        
        Args:
            claims_file: Path to JSON file containing claims
            claims: Direct list of claim dicts (alternative to claims_file)
        """
        if claims is not None:
            self.claims = claims
        elif claims_file is not None:
            with open(claims_file, 'r') as f:
                data = json.load(f)
            self.claims = data["claims"]
            self.consequences = data.get("consequences", ["bad", "neutral", "good"])
        else:
            raise ValueError("Must provide either claims_file or claims")
        
        self.consequences = getattr(self, 'consequences', ["bad", "neutral", "good"])
        self.K = len(self.consequences)
        
        # Build claim lookup
        self.claim_lookup = {c["id"]: c for c in self.claims}
        
        logger.info(f"Initialized ProblemGenerator with {len(self.claims)} claims, K={self.K}")
    
    def generate_problems(
        self,
        num_problems: int = 100,
        min_alternatives: int = 2,
        max_alternatives: int = 4,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate decision problems.
        
        Args:
            num_problems: Number of problems to generate
            min_alternatives: Minimum claims per problem
            max_alternatives: Maximum claims per problem
            seed: Random seed for reproducibility
            
        Returns:
            List of decision problem dicts
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Validate parameters
        max_alts = min(max_alternatives, len(self.claims))
        if max_alts < min_alternatives:
            raise ValueError(
                f"Cannot generate problems: max_alternatives ({max_alts}) < min_alternatives ({min_alternatives})"
            )
        
        problems = []
        for i in range(num_problems):
            n_alts = random.randint(min_alternatives, max_alts)
            selected = random.sample(self.claims, n_alts)
            
            problems.append({
                "id": f"P{i+1:04d}",
                "claim_ids": [c["id"] for c in selected],
                "claims": [c["description"] for c in selected],
                "num_alternatives": n_alts,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "seed": seed
                }
            })
        
        logger.info(f"Generated {num_problems} problems with {min_alternatives}-{max_alts} alternatives each")
        return problems
    
    def generate_balanced_problems(
        self,
        num_problems: int = 100,
        alternatives_per_problem: int = 3,
        ensure_coverage: bool = True,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate problems with balanced claim coverage.
        
        This ensures each claim appears roughly the same number of times
        across all problems.
        
        Args:
            num_problems: Number of problems to generate
            alternatives_per_problem: Fixed number of alternatives per problem
            ensure_coverage: If True, ensure all claims appear at least once
            seed: Random seed for reproducibility
            
        Returns:
            List of decision problem dicts
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if alternatives_per_problem > len(self.claims):
            raise ValueError(
                f"alternatives_per_problem ({alternatives_per_problem}) > num_claims ({len(self.claims)})"
            )
        
        # Calculate target appearances per claim
        total_slots = num_problems * alternatives_per_problem
        target_per_claim = total_slots // len(self.claims)
        
        # Track claim appearances
        appearances = {c["id"]: 0 for c in self.claims}
        
        problems = []
        for i in range(num_problems):
            # Prioritize claims that haven't appeared enough
            if ensure_coverage and i < len(self.claims) // alternatives_per_problem:
                # Early problems: ensure coverage
                available = [c for c in self.claims if appearances[c["id"]] < target_per_claim + 1]
                if len(available) < alternatives_per_problem:
                    available = self.claims.copy()
            else:
                # Weight selection by inverse appearance count
                weights = [1.0 / (appearances[c["id"]] + 1) for c in self.claims]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Sample without replacement according to weights
                indices = np.random.choice(
                    len(self.claims), 
                    size=alternatives_per_problem, 
                    replace=False,
                    p=weights
                )
                available = [self.claims[idx] for idx in indices]
            
            # Select claims
            if len(available) >= alternatives_per_problem:
                selected = random.sample(available, alternatives_per_problem)
            else:
                selected = available + random.sample(
                    [c for c in self.claims if c not in available],
                    alternatives_per_problem - len(available)
                )
            
            # Update appearances
            for c in selected:
                appearances[c["id"]] += 1
            
            problems.append({
                "id": f"P{i+1:04d}",
                "claim_ids": [c["id"] for c in selected],
                "claims": [c["description"] for c in selected],
                "num_alternatives": alternatives_per_problem,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "seed": seed,
                    "balanced": True
                }
            })
        
        # Log coverage statistics
        min_app = min(appearances.values())
        max_app = max(appearances.values())
        mean_app = sum(appearances.values()) / len(appearances)
        logger.info(
            f"Generated {num_problems} balanced problems. "
            f"Claim appearances: min={min_app}, max={max_app}, mean={mean_app:.1f}"
        )
        
        return problems
    
    def get_claim_descriptions(self, problem: Dict) -> List[str]:
        """
        Get claim descriptions for a problem.
        
        Args:
            problem: Problem dict
            
        Returns:
            List of claim description strings
        """
        if "claims" in problem:
            return problem["claims"]
        return [self.claim_lookup[cid]["description"] for cid in problem["claim_ids"]]
    
    def save_problems(self, problems: List[Dict], filepath: str):
        """
        Save problems to JSON file.
        
        Args:
            problems: List of problem dicts
            filepath: Output file path
        """
        output = {
            "problems": problems,
            "K": self.K,
            "consequences": self.consequences,
            "metadata": {
                "num_problems": len(problems),
                "num_claims": len(self.claims),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(problems)} problems to {filepath}")
    
    @staticmethod
    def load_problems(filepath: str) -> Tuple[List[Dict], int]:
        """
        Load problems from file.
        
        Args:
            filepath: Path to problems JSON file
            
        Returns:
            Tuple of (problems list, K value)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data['problems'])} problems from {filepath}")
        return data["problems"], data.get("K", 3)
    
    def get_coverage_stats(self, problems: List[Dict]) -> Dict[str, Any]:
        """
        Calculate claim coverage statistics for a set of problems.
        
        Args:
            problems: List of problem dicts
            
        Returns:
            Dict with coverage statistics
        """
        appearances = {c["id"]: 0 for c in self.claims}
        
        for problem in problems:
            for cid in problem["claim_ids"]:
                if cid in appearances:
                    appearances[cid] += 1
        
        counts = list(appearances.values())
        never_used = sum(1 for c in counts if c == 0)
        
        return {
            "num_problems": len(problems),
            "num_claims": len(self.claims),
            "total_appearances": sum(counts),
            "min_appearances": min(counts),
            "max_appearances": max(counts),
            "mean_appearances": np.mean(counts),
            "std_appearances": np.std(counts),
            "claims_never_used": never_used,
            "coverage_rate": (len(self.claims) - never_used) / len(self.claims)
        }

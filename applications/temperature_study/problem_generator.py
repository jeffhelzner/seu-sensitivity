"""
Problem Generator for the Temperature Study.

Generates decision problems by sampling claims from the pool and creates
P randomly-shuffled presentations per problem for position counterbalancing.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import StudyConfig

logger = logging.getLogger(__name__)

# Letters used for claim labelling in deliberation prompts (position-neutral)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ProblemGenerator:
    """
    Generate decision problems with shuffled presentations.

    Each problem randomly samples 2-4 claims from the pool.
    For each problem, P presentations are generated with different
    random orderings of the claims (position counterbalancing).
    """

    def __init__(
        self,
        claims_file: Optional[str] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            claims_file: Path to JSON file with ``{"claims": [...]}`` structure.
            claims: Direct list of claim dicts (alternative to *claims_file*).
        """
        if claims is not None:
            self.claims = claims
        elif claims_file is not None:
            with open(claims_file) as f:
                data = json.load(f)
            self.claims = data["claims"]
            self.consequences = data.get("consequences", [])
        else:
            raise ValueError("Must provide either claims_file or claims")

        self.consequences: List[str] = getattr(self, "consequences", [])
        self.claim_lookup: Dict[str, Dict[str, Any]] = {
            c["id"]: c for c in self.claims
        }
        logger.info(
            "ProblemGenerator initialised with %d claims", len(self.claims)
        )

    # ------------------------------------------------------------------
    # Problem generation
    # ------------------------------------------------------------------

    def generate_problems(
        self,
        num_problems: int = 100,
        min_alternatives: int = 2,
        max_alternatives: int = 4,
        num_presentations: int = 3,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate base problems with shuffled presentations.

        Args:
            num_problems: Number of problems to create.
            min_alternatives: Minimum claims per problem (≥ 2).
            max_alternatives: Maximum claims per problem.
            num_presentations: P — shuffled orderings per problem.
            seed: Random seed for reproducibility.

        Returns:
            List of problem dicts conforming to the ``problems.json`` schema.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        max_alts = min(max_alternatives, len(self.claims))
        if max_alts < min_alternatives:
            raise ValueError(
                f"max_alternatives ({max_alts}) < min_alternatives "
                f"({min_alternatives}) given pool of {len(self.claims)} claims"
            )

        problems: List[Dict[str, Any]] = []
        for i in range(num_problems):
            n_alts = random.randint(min_alternatives, max_alts)
            selected = random.sample(self.claims, n_alts)
            claim_ids = [c["id"] for c in selected]

            presentations = self._generate_presentations(
                claim_ids, num_presentations
            )

            problems.append(
                {
                    "id": f"P{i + 1:04d}",
                    "claim_ids": claim_ids,
                    "num_alternatives": n_alts,
                    "presentations": presentations,
                }
            )

        self._log_coverage(problems)
        return problems

    @classmethod
    def from_config(cls, config: StudyConfig) -> "ProblemGenerator":
        """Create a generator wired to the config's claim pool."""
        return cls(claims_file=config.claims_file)

    def generate_problems_from_config(
        self, config: StudyConfig
    ) -> List[Dict[str, Any]]:
        """Convenience: generate problems using all config parameters."""
        return self.generate_problems(
            num_problems=config.num_problems,
            min_alternatives=config.min_alternatives,
            max_alternatives=config.max_alternatives,
            num_presentations=config.num_presentations,
            seed=config.seed,
        )

    # ------------------------------------------------------------------
    # Presentations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_presentations(
        claim_ids: List[str], num_presentations: int
    ) -> List[Dict[str, Any]]:
        """
        Create *num_presentations* random permutations of *claim_ids*.

        The first presentation preserves the canonical ordering.
        Subsequent presentations are random shuffles.
        """
        presentations: List[Dict[str, Any]] = []
        for p in range(1, num_presentations + 1):
            if p == 1:
                order = list(claim_ids)
            else:
                order = list(claim_ids)
                random.shuffle(order)
            presentations.append(
                {"presentation_id": p, "order": order}
            )
        return presentations

    # ------------------------------------------------------------------
    # Prompt formatting helpers
    # ------------------------------------------------------------------

    def format_deliberation_claims_list(
        self, claim_ids: List[str]
    ) -> str:
        """
        Format claims with letter labels for the deliberation prompt.

        Example output::

            - Claim A: <description>
            - Claim B: <description>
        """
        lines: List[str] = []
        for idx, cid in enumerate(claim_ids):
            letter = _LETTERS[idx]
            desc = self.claim_lookup[cid]["description"]
            lines.append(f"- Claim {letter}: {desc}")
        return "\n".join(lines)

    def format_choice_claims_list(
        self, ordered_claim_ids: List[str]
    ) -> str:
        """
        Format claims with numeric labels for the choice prompt.

        Example output::

            - Claim 1: <description>
            - Claim 2: <description>
        """
        lines: List[str] = []
        for idx, cid in enumerate(ordered_claim_ids, start=1):
            desc = self.claim_lookup[cid]["description"]
            lines.append(f"- Claim {idx}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def claim_index_to_letter(index: int) -> str:
        """Map a 0-based claim index to its letter label (A, B, …)."""
        return _LETTERS[index]

    @staticmethod
    def num_range_str(n: int) -> str:
        """Return e.g. ``'1, 2, or 3'`` for *n*=3."""
        nums = list(range(1, n + 1))
        if len(nums) <= 2:
            return " or ".join(str(x) for x in nums)
        return ", ".join(str(x) for x in nums[:-1]) + f", or {nums[-1]}"

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_claim_description(self, claim_id: str) -> str:
        """Return the description string for a claim ID."""
        return self.claim_lookup[claim_id]["description"]

    def get_claim_descriptions(
        self, claim_ids: List[str]
    ) -> List[str]:
        """Return descriptions for a list of claim IDs (order-preserving)."""
        return [self.claim_lookup[cid]["description"] for cid in claim_ids]

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------

    def _log_coverage(self, problems: List[Dict[str, Any]]) -> None:
        stats = self.get_coverage_stats(problems)
        logger.info(
            "Generated %d problems. Claim appearances: "
            "min=%d, max=%d, mean=%.1f, coverage=%.0f%%",
            stats["num_problems"],
            stats["min_appearances"],
            stats["max_appearances"],
            stats["mean_appearances"],
            stats["coverage_rate"] * 100,
        )

    def get_coverage_stats(
        self, problems: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return claim coverage statistics for a set of problems."""
        appearances: Dict[str, int] = {c["id"]: 0 for c in self.claims}
        for prob in problems:
            for cid in prob["claim_ids"]:
                if cid in appearances:
                    appearances[cid] += 1

        counts = list(appearances.values())
        never_used = sum(1 for c in counts if c == 0)
        return {
            "num_problems": len(problems),
            "num_claims": len(self.claims),
            "total_appearances": sum(counts),
            "min_appearances": min(counts) if counts else 0,
            "max_appearances": max(counts) if counts else 0,
            "mean_appearances": float(np.mean(counts)) if counts else 0.0,
            "claims_never_used": never_used,
            "coverage_rate": (
                (len(self.claims) - never_used) / len(self.claims)
                if self.claims
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_problems(
        self, problems: List[Dict[str, Any]], filepath: str | Path
    ) -> None:
        """Save problems to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "problems": problems,
            "K": len(self.consequences) if self.consequences else 3,
            "consequences": self.consequences,
            "metadata": {
                "num_problems": len(problems),
                "num_claims": len(self.claims),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved %d problems to %s", len(problems), filepath)

    @staticmethod
    def load_problems(
        filepath: str | Path,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Load problems from JSON.

        Returns:
            ``(problems_list, K)``
        """
        with open(filepath) as f:
            data = json.load(f)
        problems = data["problems"]
        K = data.get("K", 3)
        logger.info("Loaded %d problems from %s", len(problems), filepath)
        return problems, K

"""
Problem Generator for the Ellsberg Study.

Generates decision problems by sampling alternatives from the pool and creates
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


class ProblemGenerator:
    """
    Generate decision problems with shuffled presentations.

    Each problem randomly samples 2-4 alternatives from the pool.
    For each problem, P presentations are generated with different
    random orderings (position counterbalancing).
    """

    def __init__(
        self,
        alternatives_file: Optional[str] = None,
        alternatives: Optional[List[Dict[str, Any]]] = None,
    ):
        if alternatives is not None:
            self.alternatives = alternatives
        elif alternatives_file is not None:
            with open(alternatives_file) as f:
                data = json.load(f)
            self.alternatives = data["alternatives"]
            self.consequences = data.get("consequences", [])
        else:
            raise ValueError("Must provide either alternatives_file or alternatives")

        self.consequences: List[str] = getattr(self, "consequences", [])
        self.alternative_lookup: Dict[str, Dict[str, Any]] = {
            a["id"]: a for a in self.alternatives
        }
        logger.info(
            "ProblemGenerator initialised with %d alternatives",
            len(self.alternatives),
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
            min_alternatives: Minimum gambles per problem (>= 2).
            max_alternatives: Maximum gambles per problem.
            num_presentations: P — shuffled orderings per problem.
            seed: Random seed for reproducibility.

        Returns:
            List of problem dicts conforming to the problems.json schema.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        max_alts = min(max_alternatives, len(self.alternatives))
        if max_alts < min_alternatives:
            raise ValueError(
                f"max_alternatives ({max_alts}) < min_alternatives "
                f"({min_alternatives}) given pool of {len(self.alternatives)} alternatives"
            )

        problems: List[Dict[str, Any]] = []
        for i in range(num_problems):
            n_alts = random.randint(min_alternatives, max_alts)
            selected = random.sample(self.alternatives, n_alts)
            alternative_ids = [a["id"] for a in selected]

            presentations = self._generate_presentations(
                alternative_ids, num_presentations
            )

            problems.append(
                {
                    "id": f"P{i + 1:04d}",
                    "alternative_ids": alternative_ids,
                    "num_alternatives": n_alts,
                    "presentations": presentations,
                }
            )

        self._log_coverage(problems)
        return problems

    @classmethod
    def from_config(cls, config: StudyConfig) -> "ProblemGenerator":
        """Create a generator wired to the config's alternative pool."""
        return cls(alternatives_file=config.alternatives_file)

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
        alternative_ids: List[str], num_presentations: int
    ) -> List[Dict[str, Any]]:
        """
        Create *num_presentations* random permutations of *alternative_ids*.

        The first presentation preserves the canonical ordering.
        Subsequent presentations are random shuffles.
        """
        presentations: List[Dict[str, Any]] = []
        for p in range(1, num_presentations + 1):
            if p == 1:
                order = list(alternative_ids)
            else:
                order = list(alternative_ids)
                random.shuffle(order)
            presentations.append(
                {"presentation_id": p, "order": order}
            )
        return presentations

    # ------------------------------------------------------------------
    # Prompt formatting helpers
    # ------------------------------------------------------------------

    def format_choice_alternatives_list(
        self, ordered_alternative_ids: List[str]
    ) -> str:
        """
        Format alternatives with numeric labels for the choice prompt.

        Example output::

            - Gamble 1: <description>
            - Gamble 2: <description>
        """
        lines: List[str] = []
        for idx, aid in enumerate(ordered_alternative_ids, start=1):
            desc = self.alternative_lookup[aid]["description"]
            lines.append(f"- Gamble {idx}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def format_choice_assessments_list(
        ordered_alternative_ids: List[str],
        assessments: Dict[str, str],
    ) -> str:
        """
        Format assessment texts with numeric labels for the choice prompt.

        Args:
            ordered_alternative_ids: Alternative IDs in presentation order.
            assessments: Mapping of alternative_id -> assessment text.

        Example output::

            - Gamble 1: <assessment text>
            - Gamble 2: <assessment text>
        """
        lines: List[str] = []
        for idx, aid in enumerate(ordered_alternative_ids, start=1):
            text = assessments[aid]
            lines.append(f"- Gamble {idx}: {text}")
        return "\n".join(lines)

    @staticmethod
    def num_range_str(n: int) -> str:
        """Return e.g. '1, 2, or 3' for n=3."""
        nums = list(range(1, n + 1))
        if len(nums) <= 2:
            return " or ".join(str(x) for x in nums)
        return ", ".join(str(x) for x in nums[:-1]) + f", or {nums[-1]}"

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_alternative_description(self, alternative_id: str) -> str:
        """Return the description string for an alternative ID."""
        return self.alternative_lookup[alternative_id]["description"]

    def get_alternative_descriptions(
        self, alternative_ids: List[str]
    ) -> List[str]:
        """Return descriptions for a list of alternative IDs (order-preserving)."""
        return [
            self.alternative_lookup[aid]["description"]
            for aid in alternative_ids
        ]

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------

    def _log_coverage(self, problems: List[Dict[str, Any]]) -> None:
        stats = self.get_coverage_stats(problems)
        logger.info(
            "Generated %d problems. Alternative appearances: "
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
        """Return alternative coverage statistics for a set of problems."""
        appearances: Dict[str, int] = {a["id"]: 0 for a in self.alternatives}
        for prob in problems:
            for aid in prob["alternative_ids"]:
                if aid in appearances:
                    appearances[aid] += 1

        counts = list(appearances.values())
        never_used = sum(1 for c in counts if c == 0)
        return {
            "num_problems": len(problems),
            "num_alternatives": len(self.alternatives),
            "total_appearances": sum(counts),
            "min_appearances": min(counts) if counts else 0,
            "max_appearances": max(counts) if counts else 0,
            "mean_appearances": float(np.mean(counts)) if counts else 0.0,
            "alternatives_never_used": never_used,
            "coverage_rate": (
                (len(self.alternatives) - never_used) / len(self.alternatives)
                if self.alternatives
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
            "K": len(self.consequences) if self.consequences else 4,
            "consequences": self.consequences,
            "metadata": {
                "num_problems": len(problems),
                "num_alternatives": len(self.alternatives),
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
            (problems_list, K)
        """
        with open(filepath) as f:
            data = json.load(f)
        problems = data["problems"]
        K = data.get("K", 4)
        logger.info("Loaded %d problems from %s", len(problems), filepath)
        return problems, K

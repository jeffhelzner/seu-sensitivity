"""
Data Preparation for the Ellsberg Study.

Adapts the temperature study's data preparation infrastructure for
Ellsberg-style alternatives (alternative_ids instead of claim_ids).

Reuses EmbeddingReducer, filter_valid_choices, and saving helpers
directly from the temperature study module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import StudyConfig

# Import reusable components directly from temperature study
from applications.temperature_study.data_preparation import (
    EmbeddingReducer,
    filter_valid_choices,
    save_stan_data,
    save_na_log,
    save_reduced_embeddings,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "EmbeddingReducer",
    "EllsbergStanDataBuilder",
    "filter_valid_choices",
    "save_stan_data",
    "save_na_log",
    "save_reduced_embeddings",
]


class EllsbergStanDataBuilder:
    """
    Build the Stan data dict for m_0 from valid choices and reduced embeddings.

    Adapted from the temperature study's StanDataBuilder to use
    alternative_ids / alternative_chosen field names instead of
    claim_ids / claim_chosen.

    Produces: {M, K, D, R, w, I, y} matching the m_0.stan data block.
    """

    def __init__(self, config: StudyConfig):
        self.K = config.K

    def build(
        self,
        valid_choices: List[Dict[str, Any]],
        reduced_embeddings: Dict[str, np.ndarray],
        problems: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble a Stan-compatible data dict for one temperature condition.

        Args:
            valid_choices: List of valid choice entries (valid==True).
            reduced_embeddings: Maps alternative_id -> reduced embedding vector.
            problems: Full problem list (for alternative_ids lookup).

        Returns:
            Dict ready for cmdstanpy.CmdStanModel.sample(data=...).
        """
        if not valid_choices:
            raise ValueError("No valid choices to build Stan data from")

        prob_lookup: Dict[str, Dict[str, Any]] = {
            p["id"]: p for p in problems
        }

        # 1. Determine R and the alternative -> index mapping
        all_alt_ids: set[str] = set()
        for entry in valid_choices:
            pid = entry["problem_id"]
            if pid not in prob_lookup:
                raise ValueError(f"Choice references unknown problem: {pid}")
            all_alt_ids.update(prob_lookup[pid]["alternative_ids"])

        alt_id_list = sorted(all_alt_ids)
        alt_to_idx: Dict[str, int] = {
            aid: i for i, aid in enumerate(alt_id_list)
        }
        R = len(alt_id_list)

        # 2. Build w[R, D]
        w: List[List[float]] = []
        D = None
        for aid in alt_id_list:
            if aid not in reduced_embeddings:
                raise ValueError(
                    f"No reduced embedding found for alternative {aid}"
                )
            vec = reduced_embeddings[aid]
            w.append(vec.tolist() if isinstance(vec, np.ndarray) else list(vec))
            if D is None:
                D = len(vec)

        if D is None:
            raise ValueError("Could not determine embedding dimension D")

        # 3. Build I[M, R] and y[M]
        M = len(valid_choices)
        I = np.zeros((M, R), dtype=int)
        y: List[int] = []

        for m, entry in enumerate(valid_choices):
            pid = entry["problem_id"]
            problem_alts = prob_lookup[pid]["alternative_ids"]

            for aid in problem_alts:
                I[m, alt_to_idx[aid]] = 1

            chosen_aid = entry["alternative_chosen"]
            active_ids_sorted = sorted(
                problem_alts, key=lambda a: alt_to_idx[a]
            )
            try:
                y_val = active_ids_sorted.index(chosen_aid) + 1
            except ValueError:
                raise ValueError(
                    f"Chosen alternative {chosen_aid} not in problem {pid}'s alternatives"
                )
            y.append(y_val)

        stan_data = {
            "M": M,
            "K": self.K,
            "D": D,
            "R": R,
            "w": w,
            "I": I.tolist(),
            "y": y,
        }

        logger.info(
            "Stan data: M=%d, K=%d, D=%d, R=%d",
            M,
            self.K,
            D,
            R,
        )
        return stan_data

    @staticmethod
    def validate_stan_data(stan_data: Dict[str, Any]) -> List[str]:
        """
        Run consistency checks on assembled Stan data.

        Returns:
            List of issue descriptions (empty if all checks pass).
        """
        issues: List[str] = []
        M = stan_data["M"]
        K = stan_data["K"]
        D = stan_data["D"]
        R = stan_data["R"]
        w = stan_data["w"]
        I = stan_data["I"]
        y = stan_data["y"]

        if len(w) != R:
            issues.append(f"len(w)={len(w)} != R={R}")
        if any(len(v) != D for v in w):
            issues.append("Not all w vectors have dimension D")
        if len(I) != M:
            issues.append(f"len(I)={len(I)} != M={M}")
        if any(len(row) != R for row in I):
            issues.append(f"I rows have inconsistent length (expected R={R})")
        if len(y) != M:
            issues.append(f"len(y)={len(y)} != M={M}")

        for m in range(M):
            N_m = sum(I[m])
            if N_m < 2:
                issues.append(f"Observation {m}: only {N_m} alternatives (need >=2)")
            if y[m] < 1 or y[m] > N_m:
                issues.append(
                    f"Observation {m}: y={y[m]} out of range [1, {N_m}]"
                )

        return issues

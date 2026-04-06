"""
Data preparation for the alignment study.

Handles the key new requirement: building stacked Stan data across
multiple cells with different NA patterns, using a cell membership
vector for the hierarchical model.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AlignmentStudyConfig, CellSpec

logger = logging.getLogger(__name__)


class HierarchicalStanDataBuilder:
    """
    Builds stacked Stan data for h_m01 from per-cell choice data.

    Takes per-cell valid choices and shared reduced embeddings,
    produces a single data dictionary matching h_m01.stan's data block.
    """

    def __init__(self, config: AlignmentStudyConfig):
        self.K = config.K
        self.config = config

    def build(
        self,
        per_cell_valid_choices: Dict[str, List[Dict[str, Any]]],
        reduced_embeddings: Dict[str, np.ndarray],
        problems: List[Dict[str, Any]],
        design_matrix: np.ndarray,
        cell_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Build the stacked data dictionary.

        Parameters
        ----------
        per_cell_valid_choices : dict
            {cell_id -> list of valid choice dicts}
            Each dict has 'problem_id', 'alternatives' (list of claim_ids),
            'choice' (1-indexed int).
        reduced_embeddings : dict
            {claim_id -> np.ndarray of dim D}
            Shared across all cells (same PCA projection).
        problems : list
            List of problem dicts (shared across cells).
        design_matrix : np.ndarray, shape (J, P)
            Cell-level design matrix for regression.
        cell_ids : list of str
            Ordered list of cell IDs matching design_matrix rows.

        Returns
        -------
        dict matching h_m01.stan data block
        """
        # 1. Build shared alternative pool
        all_claim_ids = sorted(reduced_embeddings.keys())
        claim_to_idx = {cid: i for i, cid in enumerate(all_claim_ids)}
        R = len(all_claim_ids)
        D = len(next(iter(reduced_embeddings.values())))

        # w[R][D] from reduced_embeddings
        w = [reduced_embeddings[cid].tolist() for cid in all_claim_ids]

        # 2. Stack observations across cells
        J = len(cell_ids)
        P = design_matrix.shape[1]

        stacked_I = []
        stacked_cell = []
        stacked_y = []
        M_per_cell = []

        for j, cell_id in enumerate(cell_ids):
            valid_choices = per_cell_valid_choices.get(cell_id, [])
            M_j = len(valid_choices)
            M_per_cell.append(M_j)

            for choice_entry in valid_choices:
                alternatives = choice_entry["alternatives"]  # list of claim_ids
                choice = choice_entry["choice"]  # 1-indexed

                # Build indicator row
                indicator = [0] * R
                active_indices = []
                for claim_id in alternatives:
                    idx = claim_to_idx[claim_id]
                    indicator[idx] = 1
                    active_indices.append(idx)

                stacked_I.append(indicator)
                stacked_cell.append(j + 1)  # 1-indexed for Stan
                stacked_y.append(choice)

        M_total = len(stacked_y)

        stan_data = {
            "J": J,
            "K": self.K,
            "D": D,
            "R": R,
            "P": P,
            "w": w,
            "M_total": M_total,
            "cell": stacked_cell,
            "I": stacked_I,
            "y": stacked_y,
            "M_per_cell": M_per_cell,
            "X": design_matrix.tolist(),
        }

        # Validate
        errors = self.validate_stan_data(stan_data)
        if errors:
            for err in errors:
                logger.error("Stan data validation error: %s", err)
            raise ValueError(f"Stan data validation failed with {len(errors)} errors")

        return stan_data

    @staticmethod
    def validate_stan_data(stan_data: Dict[str, Any]) -> List[str]:
        """
        Validate the stacked data structure.

        Checks:
        - Dimensions consistent
        - cell values in [1, J]
        - y[m] <= sum(I[m]) for all m
        - M_per_cell sums to M_total
        - X shape is (J, P)
        - At least 2 alternatives per observation
        """
        errors = []

        J = stan_data["J"]
        M_total = stan_data["M_total"]
        R = stan_data["R"]
        P = stan_data["P"]

        # Check M_per_cell sums
        if sum(stan_data["M_per_cell"]) != M_total:
            errors.append(
                f"sum(M_per_cell)={sum(stan_data['M_per_cell'])} != M_total={M_total}"
            )

        # Check cell values
        for m, c in enumerate(stan_data["cell"]):
            if c < 1 or c > J:
                errors.append(f"cell[{m}]={c} out of range [1, {J}]")

        # Check y vs available alternatives
        for m in range(M_total):
            n_alts = sum(stan_data["I"][m])
            if n_alts < 2:
                errors.append(f"Observation {m} has only {n_alts} alternatives (need >= 2)")
            if stan_data["y"][m] > n_alts:
                errors.append(f"y[{m}]={stan_data['y'][m]} > n_alts={n_alts}")

        # Check X shape
        X = stan_data["X"]
        if len(X) != J:
            errors.append(f"len(X)={len(X)} != J={J}")
        if X and len(X[0]) != P:
            errors.append(f"len(X[0])={len(X[0])} != P={P}")

        # Check w dimensions
        if len(stan_data["w"]) != R:
            errors.append(f"len(w)={len(stan_data['w'])} != R={R}")

        return errors

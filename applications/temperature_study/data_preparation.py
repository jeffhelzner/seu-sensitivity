"""
Data Preparation for the Temperature Study.

Handles:
1. Pooled PCA across all temperatures (DESIGN.md §8, decision #1)
2. NA filtering and removal logging (DESIGN.md §5 Phase 4)
3. Stan data assembly (M, K, D, R, w, I, y) per temperature
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .config import StudyConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# PCA: pooled across temperatures
# ──────────────────────────────────────────────────────────────────────

class EmbeddingReducer:
    """
    Fit PCA on pooled raw embeddings and project per temperature.

    Per DESIGN.md §8 decision #1: all five temperature conditions'
    raw embeddings are pooled, PCA is fit on the pooled set to learn
    a projection matrix, and that projection is applied to each
    temperature's embeddings separately.
    """

    def __init__(self, target_dim: int = 32, seed: int = 42):
        self.target_dim = target_dim
        self.pca: Optional[PCA] = None
        self.seed = seed

    def fit(self, pooled_embeddings: Dict[str, np.ndarray]) -> "EmbeddingReducer":
        """
        Fit PCA on pooled raw embeddings from all temperatures.

        Args:
            pooled_embeddings: Dict mapping arbitrary keys to 1-D raw
                embedding vectors.  Typically keyed as
                ``"{claim_id}_T{temp}"``.

        Returns:
            self (for chaining).
        """
        if not pooled_embeddings:
            raise ValueError("Cannot fit PCA on empty embeddings")

        matrix = np.stack(list(pooled_embeddings.values()))
        n_samples, raw_dim = matrix.shape

        # Clamp target_dim to the feasible range
        effective_dim = min(self.target_dim, n_samples, raw_dim)
        if effective_dim < self.target_dim:
            logger.warning(
                "Clamping target_dim from %d to %d "
                "(n_samples=%d, raw_dim=%d)",
                self.target_dim,
                effective_dim,
                n_samples,
                raw_dim,
            )

        self.pca = PCA(n_components=effective_dim, random_state=self.seed)
        self.pca.fit(matrix)
        logger.info(
            "PCA fit: %d components, explained variance = %.3f",
            effective_dim,
            self.pca.explained_variance_ratio_.sum(),
        )
        return self

    def transform(
        self, raw_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply the fitted PCA projection to a set of raw embeddings.

        Args:
            raw_embeddings: Dict mapping keys (e.g. claim IDs)
                to 1-D raw embedding vectors.

        Returns:
            Dict with the same keys, values replaced by reduced-dim arrays.
        """
        if self.pca is None:
            raise RuntimeError("Must call fit() before transform()")

        keys = list(raw_embeddings.keys())
        matrix = np.stack([raw_embeddings[k] for k in keys])
        reduced = self.pca.transform(matrix)
        return {k: reduced[i] for i, k in enumerate(keys)}

    def fit_transform_pooled(
        self,
        per_temp_raw: Dict[float, Dict[str, np.ndarray]],
    ) -> Dict[float, Dict[str, np.ndarray]]:
        """
        Convenience: pool → fit → project per temperature.

        Args:
            per_temp_raw: Maps temperature → {claim_id → raw_vector}.

        Returns:
            Maps temperature → {claim_id → reduced_vector}.
        """
        # Pool all temperatures
        pooled: Dict[str, np.ndarray] = {}
        for temp, embs in per_temp_raw.items():
            for key, vec in embs.items():
                pooled_key = f"{key}_T{temp}"
                pooled[pooled_key] = vec
        self.fit(pooled)

        # Project each temperature separately
        result: Dict[float, Dict[str, np.ndarray]] = {}
        for temp, embs in per_temp_raw.items():
            result[temp] = self.transform(embs)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Return PCA fit summary."""
        if self.pca is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                self.pca.explained_variance_ratio_.sum()
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# NA filtering
# ──────────────────────────────────────────────────────────────────────

def filter_valid_choices(
    choices_dict: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Separate valid from NA choices and build a removal log.

    Args:
        choices_dict: Full output from ``ChoiceCollector.collect_temperature()``.

    Returns:
        ``(valid_entries, na_log)`` where *na_log* records every excluded
        observation for audit (DESIGN.md §5 Phase 4, §6.4).
    """
    valid: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for entry in choices_dict["choices"]:
        if entry["valid"]:
            valid.append(entry)
        else:
            removed.append(
                {
                    "problem_id": entry["problem_id"],
                    "presentation_id": entry["presentation_id"],
                    "raw_response": entry["raw_response"],
                }
            )

    temperature = choices_dict["temperature"]
    na_log = {
        "temperature": temperature,
        "total_observations": len(choices_dict["choices"]),
        "valid_observations": len(valid),
        "removed_observations": len(removed),
        "na_rate": round(
            len(removed) / len(choices_dict["choices"]), 4
        )
        if choices_dict["choices"]
        else 0.0,
        "removed_entries": removed,
        "filtered_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "T=%.1f: %d valid, %d removed (NA rate=%.3f)",
        temperature,
        len(valid),
        len(removed),
        na_log["na_rate"],
    )
    return valid, na_log


# ──────────────────────────────────────────────────────────────────────
# Stan data assembly
# ──────────────────────────────────────────────────────────────────────

class StanDataBuilder:
    """
    Build the Stan data dict for m_0 from valid choices and reduced embeddings.

    Produces: ``{M, K, D, R, w, I, y}`` matching the m_0.stan data block.

    - **M**: number of valid observations (each presentation = 1 observation)
    - **K**: number of consequences (from config, typically 3)
    - **D**: embedding dimension after PCA
    - **R**: number of distinct claims across all problems
    - **w[R, D]**: reduced embedding for each distinct claim
    - **I[M, R]**: indicator matrix (which claims appear in which observation)
    - **y[M]**: 1-indexed chosen alternative *within the active set* of each observation
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
            valid_choices: List of valid choice entries (``valid==True``).
            reduced_embeddings: Maps claim_id → reduced embedding vector
                (D-dim).  One vector per distinct claim.
            problems: Full problem list (for claim_ids lookup).

        Returns:
            Dict ready for ``cmdstanpy.CmdStanModel.sample(data=...)``.

        Raises:
            ValueError: If any required embedding is missing or choices
                reference unknown problems.
        """
        if not valid_choices:
            raise ValueError("No valid choices to build Stan data from")

        # Build a problem lookup
        prob_lookup: Dict[str, Dict[str, Any]] = {
            p["id"]: p for p in problems
        }

        # ── 1. Determine R and the claim → index mapping ──
        # Collect ALL distinct claims that appear in any observation's problem
        all_claim_ids: set[str] = set()
        for entry in valid_choices:
            pid = entry["problem_id"]
            if pid not in prob_lookup:
                raise ValueError(f"Choice references unknown problem: {pid}")
            all_claim_ids.update(prob_lookup[pid]["claim_ids"])

        # Sorted for deterministic ordering
        claim_id_list = sorted(all_claim_ids)
        claim_to_idx: Dict[str, int] = {
            cid: i for i, cid in enumerate(claim_id_list)
        }
        R = len(claim_id_list)

        # ── 2. Build w[R, D] ──
        # One embedding per distinct claim, keyed directly by claim_id.
        w: List[List[float]] = []
        D = None
        for cid in claim_id_list:
            if cid not in reduced_embeddings:
                raise ValueError(
                    f"No reduced embedding found for claim {cid}"
                )
            vec = reduced_embeddings[cid]
            w.append(vec.tolist() if isinstance(vec, np.ndarray) else list(vec))
            if D is None:
                D = len(vec)

        if D is None:
            raise ValueError("Could not determine embedding dimension D")

        # ── 3. Build I[M, R] and y[M] ──
        M = len(valid_choices)
        I = np.zeros((M, R), dtype=int)
        y: List[int] = []

        for m, entry in enumerate(valid_choices):
            pid = entry["problem_id"]
            problem_claims = prob_lookup[pid]["claim_ids"]

            # Set indicator bits for all claims in this problem
            for cid in problem_claims:
                I[m, claim_to_idx[cid]] = 1

            # y is 1-indexed position WITHIN the active set for this observation.
            # The "active set" is the subset of R where I[m, r] == 1.
            # Stan's transformed data iterates r = 1..R and collects x where I[m,r]==1.
            # So the ordering of the active set is the sorted claim_to_idx order.
            chosen_cid = entry["claim_chosen"]
            active_ids_sorted = sorted(
                problem_claims, key=lambda c: claim_to_idx[c]
            )
            try:
                y_val = active_ids_sorted.index(chosen_cid) + 1  # 1-indexed
            except ValueError:
                raise ValueError(
                    f"Chosen claim {chosen_cid} not in problem {pid}'s claims"
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

        # Check y bounds
        for m in range(M):
            N_m = sum(I[m])
            if N_m < 2:
                issues.append(f"Observation {m}: only {N_m} alternatives (need ≥2)")
            if y[m] < 1 or y[m] > N_m:
                issues.append(
                    f"Observation {m}: y={y[m]} out of range [1, {N_m}]"
                )

        return issues


# ──────────────────────────────────────────────────────────────────────
# Saving helpers
# ──────────────────────────────────────────────────────────────────────

def save_stan_data(
    stan_data: Dict[str, Any],
    filepath: str | Path,
) -> Path:
    """Save Stan data dict to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    serializable = _make_serializable(stan_data)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved Stan data to %s (M=%d)", filepath, stan_data["M"])
    return filepath


def save_na_log(
    na_log: Dict[str, Any],
    filepath: str | Path,
) -> Path:
    """Save NA removal log to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(na_log, f, indent=2)
    logger.info("Saved NA log to %s", filepath)
    return filepath


def save_reduced_embeddings(
    reduced: Dict[str, np.ndarray],
    filepath: str | Path,
) -> Path:
    """Save reduced-dim embeddings to NPZ."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **reduced)
    logger.info("Saved reduced embeddings to %s", filepath)
    return filepath


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

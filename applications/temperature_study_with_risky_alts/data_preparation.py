"""
Data Preparation for the Temperature Study with Risky Alternatives.

Handles:
1. Loading existing uncertain Stan data from the original temperature study
2. Building risky data block (N, S, x, J, z) from collected risky choices
3. Merging uncertain + risky into augmented Stan data for m_1/m_2/m_3
4. NA filtering for risky choices
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import StudyConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# NA filtering (risky choices)
# ──────────────────────────────────────────────────────────────────────

def filter_valid_risky_choices(
    choices_dict: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Separate valid from NA risky choices and build a removal log.

    Args:
        choices_dict: Full output from ``RiskyChoiceCollector.collect_temperature()``.

    Returns:
        ``(valid_entries, na_log)``
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
        "Risky T=%.1f: %d valid, %d removed (NA rate=%.3f)",
        temperature,
        len(valid),
        len(removed),
        na_log["na_rate"],
    )
    return valid, na_log


# ──────────────────────────────────────────────────────────────────────
# Risky Stan data builder
# ──────────────────────────────────────────────────────────────────────

class RiskyStanDataBuilder:
    """
    Build the risky data block (N, S, x, J, z) from valid risky choices.

    This block is compatible with the risky data section of m_1, m_2,
    and m_3 Stan models.

    - **N**: number of valid risky observations (each presentation = 1 observation)
    - **S**: number of distinct risky alternatives
    - **x[S, K]**: objective probability simplexes for each risky alternative
    - **J[N, S]**: indicator matrix (which alternatives appear in which observation)
    - **z[N]**: 1-indexed chosen alternative within the active set
    """

    def __init__(self, config: StudyConfig):
        self.K = config.K

    def build(
        self,
        valid_choices: List[Dict[str, Any]],
        risky_alternatives: List[Dict[str, Any]],
        problems: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble the risky Stan data block from valid choices.

        Args:
            valid_choices: List of valid risky choice entries.
            risky_alternatives: Full list of risky alternative dicts
                (with ``id`` and ``probabilities`` fields).
            problems: Full risky problem list (for alternative_ids lookup).

        Returns:
            Dict with keys {N, S, x, J, z} ready for Stan.

        Raises:
            ValueError: On data inconsistency.
        """
        if not valid_choices:
            raise ValueError("No valid risky choices to build Stan data from")

        # Build problem lookup
        prob_lookup: Dict[str, Dict[str, Any]] = {
            p["id"]: p for p in problems
        }

        # Build alternative lookup
        alt_lookup: Dict[str, Dict[str, Any]] = {
            a["id"]: a for a in risky_alternatives
        }

        # ── 1. Determine S and alt → index mapping ──
        # Collect ALL distinct alternatives that appear in any observation
        all_alt_ids: set[str] = set()
        for entry in valid_choices:
            pid = entry["problem_id"]
            if pid not in prob_lookup:
                raise ValueError(
                    f"Risky choice references unknown problem: {pid}"
                )
            all_alt_ids.update(prob_lookup[pid]["alternative_ids"])

        # Sorted for deterministic ordering
        alt_id_list = sorted(all_alt_ids)
        alt_to_idx: Dict[str, int] = {
            aid: i for i, aid in enumerate(alt_id_list)
        }
        S = len(alt_id_list)

        # ── 2. Build x[S, K] — objective probability simplexes ──
        x: List[List[float]] = []
        for aid in alt_id_list:
            if aid not in alt_lookup:
                raise ValueError(
                    f"No probability data found for alternative {aid}"
                )
            probs = alt_lookup[aid]["probabilities"]
            if len(probs) != self.K:
                raise ValueError(
                    f"Alternative {aid} has {len(probs)} probabilities, "
                    f"expected K={self.K}"
                )
            # Validate simplex
            total = sum(probs)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Alternative {aid} probabilities sum to {total}, not 1.0"
                )
            x.append(probs)

        # ── 3. Build J[N, S] and z[N] ──
        N = len(valid_choices)
        J = np.zeros((N, S), dtype=int)
        z: List[int] = []

        for n, entry in enumerate(valid_choices):
            pid = entry["problem_id"]
            problem_alts = prob_lookup[pid]["alternative_ids"]

            # Set indicator bits
            for aid in problem_alts:
                J[n, alt_to_idx[aid]] = 1

            # z is 1-indexed position within the active set.
            # The active set ordering follows the sorted alt_to_idx order
            # (matching how Stan's transformed data iterates s=1..S
            # and collects x where J[n,s]==1).
            chosen_aid = entry["alternative_chosen"]
            active_ids_sorted = sorted(
                problem_alts, key=lambda a: alt_to_idx[a]
            )
            try:
                z_val = active_ids_sorted.index(chosen_aid) + 1  # 1-indexed
            except ValueError:
                raise ValueError(
                    f"Chosen alternative {chosen_aid} not in problem "
                    f"{pid}'s alternatives"
                )
            z.append(z_val)

        risky_data = {
            "N": N,
            "S": S,
            "x": x,
            "J": J.tolist(),
            "z": z,
        }

        logger.info("Risky Stan data: N=%d, S=%d, K=%d", N, S, self.K)
        return risky_data

    @staticmethod
    def validate_risky_stan_data(risky_data: Dict[str, Any]) -> List[str]:
        """
        Run consistency checks on the risky Stan data block.

        Returns:
            List of issue descriptions (empty if all checks pass).
        """
        issues: List[str] = []
        N = risky_data["N"]
        S = risky_data["S"]
        x = risky_data["x"]
        J = risky_data["J"]
        z = risky_data["z"]

        if len(x) != S:
            issues.append(f"len(x)={len(x)} != S={S}")
        if len(J) != N:
            issues.append(f"len(J)={len(J)} != N={N}")
        if any(len(row) != S for row in J):
            issues.append(f"J rows have inconsistent length (expected S={S})")
        if len(z) != N:
            issues.append(f"len(z)={len(z)} != N={N}")

        # Check z bounds
        for n in range(N):
            N_n = sum(J[n])
            if N_n < 2:
                issues.append(
                    f"Risky observation {n}: only {N_n} alternatives (need ≥2)"
                )
            if z[n] < 1 or z[n] > N_n:
                issues.append(
                    f"Risky observation {n}: z={z[n]} out of range [1, {N_n}]"
                )

        # Check x are valid simplexes
        for s, simplex in enumerate(x):
            total = sum(simplex)
            if abs(total - 1.0) > 1e-6:
                issues.append(
                    f"x[{s}] sums to {total}, not 1.0"
                )
            if any(p < 0 for p in simplex):
                issues.append(f"x[{s}] contains negative probabilities")

        return issues


# ──────────────────────────────────────────────────────────────────────
# Augmented Stan data assembly (uncertain + risky)
# ──────────────────────────────────────────────────────────────────────

def build_augmented_stan_data(
    uncertain_stan_data: Dict[str, Any],
    risky_stan_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge uncertain and risky Stan data blocks into a single dict
    compatible with m_1, m_2, and m_3.

    The uncertain block provides: {M, K, D, R, w, I, y}
    The risky block provides: {N, S, x, J, z}

    Args:
        uncertain_stan_data: Stan data from the original temperature study.
        risky_stan_data: Risky Stan data block from RiskyStanDataBuilder.

    Returns:
        Merged dict with all fields required by m_1/m_2/m_3.

    Raises:
        ValueError: If K values don't match between blocks.
    """
    # Validate K consistency
    K_uncertain = uncertain_stan_data["K"]
    # Infer K from risky x dimensions
    if risky_stan_data["x"]:
        K_risky = len(risky_stan_data["x"][0])
    else:
        K_risky = K_uncertain

    if K_uncertain != K_risky:
        raise ValueError(
            f"K mismatch: uncertain K={K_uncertain}, risky K={K_risky}"
        )

    augmented = {}

    # Uncertain block
    augmented["M"] = uncertain_stan_data["M"]
    augmented["K"] = uncertain_stan_data["K"]
    augmented["D"] = uncertain_stan_data["D"]
    augmented["R"] = uncertain_stan_data["R"]
    augmented["w"] = uncertain_stan_data["w"]
    augmented["I"] = uncertain_stan_data["I"]
    augmented["y"] = uncertain_stan_data["y"]

    # Risky block
    augmented["N"] = risky_stan_data["N"]
    augmented["S"] = risky_stan_data["S"]
    augmented["x"] = risky_stan_data["x"]
    augmented["J"] = risky_stan_data["J"]
    augmented["z"] = risky_stan_data["z"]

    logger.info(
        "Augmented Stan data: M=%d, R=%d, D=%d, N=%d, S=%d, K=%d",
        augmented["M"],
        augmented["R"],
        augmented["D"],
        augmented["N"],
        augmented["S"],
        augmented["K"],
    )
    return augmented


def validate_augmented_stan_data(
    stan_data: Dict[str, Any],
) -> List[str]:
    """
    Run full consistency checks on augmented (uncertain + risky) Stan data.

    Returns:
        List of issue descriptions (empty if all checks pass).
    """
    issues: List[str] = []

    # Check all required fields are present
    required_fields = ["M", "K", "D", "R", "w", "I", "y", "N", "S", "x", "J", "z"]
    for field in required_fields:
        if field not in stan_data:
            issues.append(f"Missing required field: {field}")

    if issues:
        return issues  # Can't validate further without required fields

    # ── Uncertain block validation ──
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
            issues.append(
                f"Uncertain obs {m}: only {N_m} alternatives (need ≥2)"
            )
        if y[m] < 1 or y[m] > N_m:
            issues.append(
                f"Uncertain obs {m}: y={y[m]} out of range [1, {N_m}]"
            )

    # ── Risky block validation ──
    N = stan_data["N"]
    S = stan_data["S"]
    x = stan_data["x"]
    J = stan_data["J"]
    z = stan_data["z"]

    if len(x) != S:
        issues.append(f"len(x)={len(x)} != S={S}")
    if any(len(simplex) != K for simplex in x):
        issues.append(f"Not all x simplexes have dimension K={K}")
    if len(J) != N:
        issues.append(f"len(J)={len(J)} != N={N}")
    if any(len(row) != S for row in J):
        issues.append(f"J rows have inconsistent length (expected S={S})")
    if len(z) != N:
        issues.append(f"len(z)={len(z)} != N={N}")

    for n in range(N):
        N_n = sum(J[n])
        if N_n < 2:
            issues.append(
                f"Risky obs {n}: only {N_n} alternatives (need ≥2)"
            )
        if z[n] < 1 or z[n] > N_n:
            issues.append(
                f"Risky obs {n}: z={z[n]} out of range [1, {N_n}]"
            )

    # Check x simplexes
    for s, simplex in enumerate(x):
        total = sum(simplex)
        if abs(total - 1.0) > 1e-6:
            issues.append(f"x[{s}] sums to {total}, not 1.0")
        if any(p < 0 for p in simplex):
            issues.append(f"x[{s}] contains negative probabilities")

    return issues


# ──────────────────────────────────────────────────────────────────────
# Saving helpers
# ──────────────────────────────────────────────────────────────────────

def save_augmented_stan_data(
    stan_data: Dict[str, Any],
    filepath: str | Path,
) -> Path:
    """Save augmented Stan data dict to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    serializable = _make_serializable(stan_data)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(
        "Saved augmented Stan data to %s (M=%d, N=%d)",
        filepath,
        stan_data["M"],
        stan_data["N"],
    )
    return filepath


def save_risky_na_log(
    na_log: Dict[str, Any],
    filepath: str | Path,
) -> Path:
    """Save risky NA removal log to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(na_log, f, indent=2)
    logger.info("Saved risky NA log to %s", filepath)
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

"""
Position Bias Analysis for the Temperature Study.

Implements DESIGN.md §6.2:
  1. Position choice rates per temperature
  2. Chi-squared test for deviation from uniform
  3. Position effect size (Cramér's V)
  4. Temperature × position interaction test
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def position_choice_rates(
    choices: List[Dict[str, Any]],
    max_positions: int = 4,
) -> Dict[int, float]:
    """
    Compute the rate at which each position was chosen (valid entries only).

    Args:
        choices: List of choice entry dicts (must have ``valid``,
            ``position_chosen``, ``claim_order``).
        max_positions: Maximum position to track (1-indexed).

    Returns:
        Dict mapping 1-indexed position → proportion.
    """
    valid = [c for c in choices if c["valid"]]
    if not valid:
        return {}

    counts: Counter = Counter()
    for c in valid:
        counts[c["position_chosen"]] += 1

    total = len(valid)
    return {
        pos: round(counts.get(pos, 0) / total, 4)
        for pos in range(1, max_positions + 1)
    }


def position_choice_counts(
    choices: List[Dict[str, Any]],
    max_positions: int = 4,
) -> Dict[int, int]:
    """Raw counts of how often each position was chosen."""
    valid = [c for c in choices if c["valid"]]
    counts: Counter = Counter()
    for c in valid:
        counts[c["position_chosen"]] += 1
    return {pos: counts.get(pos, 0) for pos in range(1, max_positions + 1)}


def chi_squared_uniformity_test(
    choices: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Chi-squared goodness-of-fit test against a uniform distribution.

    Only includes positions actually used (i.e. problems with 2 alts
    don't contribute to position 3/4 bins).

    Returns:
        Dict with ``chi2``, ``p_value``, ``df``, ``observed``, ``expected``.
    """
    # Group by problem size so we only test within each N
    by_n: Dict[int, Counter] = defaultdict(Counter)
    for c in choices:
        if not c["valid"]:
            continue
        n = len(c["claim_order"])
        by_n[n][c["position_chosen"]] += 1

    # Aggregate observed and expected
    observed: List[int] = []
    expected: List[float] = []

    for n, counter in sorted(by_n.items()):
        total = sum(counter.values())
        expected_per_pos = total / n
        for pos in range(1, n + 1):
            observed.append(counter.get(pos, 0))
            expected.append(expected_per_pos)

    if not observed or len(observed) < 2:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "df": 0,
            "observed": observed,
            "expected": expected,
            "significant": False,
        }

    obs_arr = np.array(observed, dtype=float)
    exp_arr = np.array(expected, dtype=float)

    # Degrees of freedom: one per position minus one per N-group
    n_groups = len(by_n)
    df = len(observed) - n_groups
    if df <= 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "df": 0,
            "observed": observed,
            "expected": [round(e, 2) for e in expected],
            "significant": False,
        }

    chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
    p_value = float(1 - stats.chi2.cdf(chi2, df))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "df": df,
        "observed": observed,
        "expected": [round(e, 2) for e in expected],
        "significant": p_value < 0.05,
    }


def cramers_v(
    choices: List[Dict[str, Any]],
) -> float:
    """
    Cramér's V effect size for the position–choice association.

    Measures how strongly position predicts being chosen, controlling
    for the number of categories.

    Returns:
        Cramér's V in [0, 1].  0 = no association, 1 = perfect.
    """
    result = chi_squared_uniformity_test(choices)
    chi2 = result["chi2"]
    n = sum(result["observed"])
    if n == 0 or result["df"] == 0:
        return 0.0

    # k = number of categories (max positions observed)
    k = max(
        len(c["claim_order"]) for c in choices if c["valid"]
    )
    v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0
    return round(float(v), 4)


def position_bias_by_temperature(
    per_temp_choices: Dict[float, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyse position bias across all temperatures.

    Returns a summary dict with per-temperature rates, chi-squared tests,
    effect sizes, and a cross-temperature interaction assessment.
    """
    summary: Dict[str, Any] = {"per_temperature": {}}

    for temp in sorted(per_temp_choices):
        entries = per_temp_choices[temp]["choices"]
        rates = position_choice_rates(entries)
        chi2_result = chi_squared_uniformity_test(entries)
        v = cramers_v(entries)

        summary["per_temperature"][str(temp)] = {
            "rates": rates,
            "chi_squared": chi2_result,
            "cramers_v": v,
        }

    # Cross-temperature interaction: compare Cramér's V across temps
    vs = {
        t: info["cramers_v"]
        for t, info in summary["per_temperature"].items()
    }
    summary["cramers_v_by_temp"] = vs

    # Simple trend: Spearman correlation between temp and V
    temps = sorted(float(t) for t in vs)
    v_vals = [vs[str(t)] for t in temps]
    if len(temps) >= 3:
        rho, p = stats.spearmanr(temps, v_vals)
        summary["position_bias_trend"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "interpretation": (
                "Position bias increases with temperature"
                if rho > 0
                else "Position bias decreases with temperature"
            ),
        }

    return summary


def per_claim_position_analysis(
    choices: List[Dict[str, Any]],
    claim_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each claim, compare how often it is chosen when in position 1
    versus other positions.

    This mirrors the pilot's per-claim analysis but uses proper
    counterbalanced presentations.

    Returns:
        Dict mapping claim_id → {pos1_rate, other_rate, diff, n_pos1, n_other}.
    """
    valid = [c for c in choices if c["valid"]]
    if not valid:
        return {}

    # Collect all claim IDs if not specified
    if claim_ids is None:
        claim_ids_set: set[str] = set()
        for c in valid:
            claim_ids_set.update(c["claim_order"])
        claim_ids = sorted(claim_ids_set)

    results: Dict[str, Dict[str, Any]] = {}
    for cid in claim_ids:
        pos1_shown = pos1_chosen = 0
        other_shown = other_chosen = 0

        for c in valid:
            if cid not in c["claim_order"]:
                continue
            idx = c["claim_order"].index(cid)  # 0-indexed position
            chosen_cid = c["claim_chosen"]

            if idx == 0:
                pos1_shown += 1
                if chosen_cid == cid:
                    pos1_chosen += 1
            else:
                other_shown += 1
                if chosen_cid == cid:
                    other_chosen += 1

        pos1_rate = pos1_chosen / pos1_shown if pos1_shown > 0 else 0.0
        other_rate = other_chosen / other_shown if other_shown > 0 else 0.0

        results[cid] = {
            "pos1_rate": round(pos1_rate, 4),
            "other_rate": round(other_rate, 4),
            "diff": round(pos1_rate - other_rate, 4),
            "n_pos1": pos1_shown,
            "n_other": other_shown,
        }

    return results

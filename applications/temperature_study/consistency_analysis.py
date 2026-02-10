"""
Choice Consistency Analysis for the Temperature Study.

Implements DESIGN.md §6.3:
  1. Unanimity rate — proportion of problems where all P presentations
     yield the same *claim* choice.
  2. Modal agreement rate — proportion of presentations matching the
     plurality choice.
  3. Temperature trend — test whether consistency decreases with
     increasing temperature.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _group_by_problem(
    choices: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group valid choice entries by problem_id."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in choices:
        if c["valid"]:
            groups[c["problem_id"]].append(c)
    return dict(groups)


def unanimity_rate(choices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fraction of problems where every presentation yields the same claim.

    Only includes problems with ≥2 valid presentations so we can
    actually measure agreement.

    Returns:
        Dict with ``rate``, ``n_unanimous``, ``n_problems``.
    """
    groups = _group_by_problem(choices)

    n_unanimous = 0
    n_eligible = 0

    for pid, entries in groups.items():
        if len(entries) < 2:
            continue
        n_eligible += 1
        claims = {e["claim_chosen"] for e in entries}
        if len(claims) == 1:
            n_unanimous += 1

    rate = n_unanimous / n_eligible if n_eligible > 0 else 0.0
    return {
        "rate": round(rate, 4),
        "n_unanimous": n_unanimous,
        "n_problems": n_eligible,
    }


def modal_agreement_rate(choices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Average proportion of presentations that agree with the most common
    (modal) claim choice for each problem.

    This is a softer measure than unanimity — e.g. 2-out-of-3 agreement
    yields 0.667 for that problem.

    Returns:
        Dict with ``mean_rate``, ``per_problem`` (pid → agreement ratio),
        ``n_problems``.
    """
    groups = _group_by_problem(choices)

    per_problem: Dict[str, float] = {}
    for pid, entries in groups.items():
        if len(entries) < 2:
            continue
        counter = Counter(e["claim_chosen"] for e in entries)
        modal_count = counter.most_common(1)[0][1]
        per_problem[pid] = round(modal_count / len(entries), 4)

    mean_rate = (
        round(float(np.mean(list(per_problem.values()))), 4)
        if per_problem
        else 0.0
    )
    return {
        "mean_rate": mean_rate,
        "per_problem": per_problem,
        "n_problems": len(per_problem),
    }


def consistency_by_temperature(
    per_temp_choices: Dict[float, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute unanimity and modal-agreement rates per temperature and
    test for a temperature trend.

    Returns:
        Summary with per-temperature metrics, Spearman trend test, and
        interpretation.
    """
    summary: Dict[str, Any] = {"per_temperature": {}}

    for temp in sorted(per_temp_choices):
        entries = per_temp_choices[temp]["choices"]
        una = unanimity_rate(entries)
        modal = modal_agreement_rate(entries)
        summary["per_temperature"][str(temp)] = {
            "unanimity": una,
            "modal_agreement": modal,
        }

    # Trend: Spearman correlation between temperature and unanimity rate
    temps = sorted(float(t) for t in summary["per_temperature"])
    una_rates = [
        summary["per_temperature"][str(t)]["unanimity"]["rate"]
        for t in temps
    ]
    modal_rates = [
        summary["per_temperature"][str(t)]["modal_agreement"]["mean_rate"]
        for t in temps
    ]

    result: Dict[str, Any] = {"per_temperature": summary["per_temperature"]}

    if len(temps) >= 3:
        rho_u, p_u = stats.spearmanr(temps, una_rates)
        rho_m, p_m = stats.spearmanr(temps, modal_rates)

        result["unanimity_trend"] = {
            "spearman_rho": round(float(rho_u), 4),
            "p_value": round(float(p_u), 4),
        }
        result["modal_agreement_trend"] = {
            "spearman_rho": round(float(rho_m), 4),
            "p_value": round(float(p_m), 4),
        }
        result["interpretation"] = (
            "Consistency decreases with temperature"
            if rho_u < 0
            else "No clear decrease in consistency with temperature"
        )

    return result


def entropy_of_choices(choices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shannon entropy of the claim-choice distribution for each problem,
    averaged across problems.

    Higher entropy → more dispersed choices across presentations (less
    consistency).

    Returns:
        Dict with ``mean_entropy``, ``per_problem``, ``max_possible``.
    """
    groups = _group_by_problem(choices)

    per_problem: Dict[str, float] = {}
    max_entropy = 0.0

    for pid, entries in groups.items():
        if len(entries) < 2:
            continue
        counter = Counter(e["claim_chosen"] for e in entries)
        total = len(entries)
        probs = np.array([v / total for v in counter.values()])
        h = float(-np.sum(probs * np.log2(probs + 1e-12)))
        per_problem[pid] = round(h, 4)

        # Max possible entropy for this problem
        n_claims = len(entries[0]["claim_order"])
        max_entropy = max(max_entropy, np.log2(n_claims))

    mean_h = (
        round(float(np.mean(list(per_problem.values()))), 4)
        if per_problem
        else 0.0
    )
    return {
        "mean_entropy": mean_h,
        "per_problem": per_problem,
        "max_possible_entropy": round(float(max_entropy), 4),
        "n_problems": len(per_problem),
    }

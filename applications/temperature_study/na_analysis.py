"""
NA (Non-Parseable Response) Quality Analysis for the Temperature Study.

Implements DESIGN.md §6.4:
  1. NA rate per temperature
  2. NA distribution across problems (concentration test)
  3. Effective sample size reporting
  4. Sensitivity check: worst-case NA imputation
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def na_rates_by_temperature(
    per_temp_choices: Dict[float, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    NA rate and effective sample size per temperature.

    Returns:
        Dict with per-temperature and overall NA statistics.
    """
    summary: Dict[str, Any] = {"per_temperature": {}, "overall": {}}
    total_valid = 0
    total_na = 0

    for temp in sorted(per_temp_choices):
        d = per_temp_choices[temp]
        entries = d["choices"]
        n_valid = sum(1 for c in entries if c["valid"])
        n_na = sum(1 for c in entries if not c["valid"])
        total = n_valid + n_na

        summary["per_temperature"][str(temp)] = {
            "valid": n_valid,
            "na": n_na,
            "total": total,
            "na_rate": round(n_na / total, 4) if total > 0 else 0.0,
            "effective_M": n_valid,
        }
        total_valid += n_valid
        total_na += n_na

    grand_total = total_valid + total_na
    summary["overall"] = {
        "valid": total_valid,
        "na": total_na,
        "total": grand_total,
        "na_rate": (
            round(total_na / grand_total, 4) if grand_total > 0 else 0.0
        ),
    }

    # Trend: does NA rate increase with temperature?
    temps = sorted(float(t) for t in summary["per_temperature"])
    rates = [
        summary["per_temperature"][str(t)]["na_rate"]
        for t in temps
    ]
    if len(temps) >= 3 and any(r > 0 for r in rates):
        rho, p = stats.spearmanr(temps, rates)
        summary["na_rate_trend"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "interpretation": (
                "NA rate increases with temperature"
                if rho > 0 and p < 0.10
                else "No significant trend in NA rate"
            ),
        }

    return summary


def na_concentration(
    choices: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Test whether NAs concentrate in specific problems or are uniformly
    spread.

    Uses a chi-squared test: observed NA count per problem vs. uniform
    expectation.  Concentration suggests problematic claim descriptions.

    Returns:
        Dict with per-problem NA counts, chi-squared result, and
        a flag for problematic problems (≥3× expected rate).
    """
    na_entries = [c for c in choices if not c["valid"]]
    all_pids = sorted({c["problem_id"] for c in choices})
    n_problems = len(all_pids)

    if not na_entries or n_problems == 0:
        return {
            "na_per_problem": {},
            "chi_squared": None,
            "concentrated_problems": [],
            "uniform": True,
            "total_na": 0,
            "n_problems": n_problems,
        }

    na_counts = Counter(c["problem_id"] for c in na_entries)
    total_na = len(na_entries)
    expected_per = total_na / n_problems if n_problems > 0 else 0

    # Chi-squared goodness of fit
    observed = np.array([na_counts.get(pid, 0) for pid in all_pids], dtype=float)
    expected = np.full(n_problems, expected_per)

    # Only run test if we have enough NAs
    if total_na >= 5 and n_problems > 1:
        chi2 = float(np.sum((observed - expected) ** 2 / (expected + 1e-12)))
        df = n_problems - 1
        p_value = float(1 - stats.chi2.cdf(chi2, df))
    else:
        chi2 = 0.0
        df = 0
        p_value = 1.0

    # Flag problems with ≥3× expected NA rate
    threshold = 3 * expected_per if expected_per > 0 else 1
    concentrated = [
        pid for pid in all_pids if na_counts.get(pid, 0) >= threshold
    ]

    return {
        "na_per_problem": {pid: na_counts.get(pid, 0) for pid in all_pids},
        "chi_squared": {
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 4),
            "df": df,
        },
        "concentrated_problems": concentrated,
        "uniform": p_value >= 0.05,
        "total_na": total_na,
        "n_problems": n_problems,
    }


def effective_sample_summary(
    per_temp_choices: Dict[float, Dict[str, Any]],
    nominal_M: int = 300,
) -> Dict[str, Any]:
    """
    Report effective M (after NA removal) per temperature alongside
    the nominal M.

    Args:
        per_temp_choices: Temperature → choice data dict.
        nominal_M: Expected number of observations (problems × presentations).

    Returns:
        Dict mapping temp → {nominal, effective, retention_rate}.
    """
    result: Dict[str, Any] = {}
    for temp in sorted(per_temp_choices):
        entries = per_temp_choices[temp]["choices"]
        n_valid = sum(1 for c in entries if c["valid"])
        result[str(temp)] = {
            "nominal_M": nominal_M,
            "effective_M": n_valid,
            "retention_rate": round(n_valid / nominal_M, 4) if nominal_M > 0 else 0.0,
        }
    return result


def worst_case_imputation(
    choices: List[Dict[str, Any]],
    imputed_position: int = 1,
) -> List[Dict[str, Any]]:
    """
    Create an augmented choice list where all NAs are imputed as the
    given position (default: position 1, the worst-case for position bias).

    The imputed entries are copies with ``valid=True``, ``imputed=True``,
    and ``position_chosen`` / ``claim_chosen`` filled in.

    Returns:
        New list of choice entries (originals unchanged).
    """
    augmented: List[Dict[str, Any]] = []
    for c in choices:
        if c["valid"]:
            augmented.append(c)
        else:
            imputed = dict(c)
            imputed["valid"] = True
            imputed["imputed"] = True
            imputed["position_chosen"] = imputed_position
            # Map back to claim
            order = imputed.get("claim_order", [])
            if order and 1 <= imputed_position <= len(order):
                imputed["claim_chosen"] = order[imputed_position - 1]
            else:
                imputed["claim_chosen"] = None
            augmented.append(imputed)
    return augmented


def na_quality_report(
    per_temp_choices: Dict[float, Dict[str, Any]],
    nominal_M: int = 300,
) -> Dict[str, Any]:
    """
    Full §6.4 report: NA rates, concentration, effective M, and
    a flag for whether sensitivity imputation is needed.

    Returns:
        Comprehensive NA quality dict.
    """
    rates = na_rates_by_temperature(per_temp_choices)

    # Per-temperature concentration analysis
    concentration: Dict[str, Any] = {}
    for temp in sorted(per_temp_choices):
        entries = per_temp_choices[temp]["choices"]
        concentration[str(temp)] = na_concentration(entries)

    eff = effective_sample_summary(per_temp_choices, nominal_M)

    # Determine if sensitivity check is warranted
    max_na_rate = max(
        (info["na_rate"] for info in rates["per_temperature"].values()),
        default=0.0,
    )
    sensitivity_needed = max_na_rate > 0.05

    return {
        "na_rates": rates,
        "concentration": concentration,
        "effective_sample_size": eff,
        "sensitivity_imputation_recommended": sensitivity_needed,
        "max_na_rate": max_na_rate,
    }

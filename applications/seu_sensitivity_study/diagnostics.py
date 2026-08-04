"""
Collection diagnostics (study plan §3.1, §6.4, §8.8).

Three reported quantities live here.

**NA accounting (§6.4).**  Refusal is signal for RQ5, not a nuisance, so NA rate
is tabulated by cell x difficulty stratum x menu size x resolution path.  The
resolution-path axis is what separates parser-induced NA from genuine refusal.

**Position-flip rate (§3.1).**  Residual across-presentation variability at
temperature 0 is predominantly position bias -- a real incoherence, but a
different construct from SEU-insensitivity, which the softmax would otherwise
absorb into alpha.

**The R6 stability subset (§3.1, v0.5).**  This is the one that needed a design
decision rather than a formula.  With ``num_presentations = 2`` a menu either
flips or does not, so "stable" is binary and a *stratified* criterion would
select exactly the same menus as an absolute zero-flip rule.  Stratifying only
helps if it also **rebalances**: because P(flip) rises with menu size even
under a size-invariant softmax, the stable set is dominated by small menus, and
re-estimating the RQ6 slope on it would use a design whose size spread has
collapsed.  :func:`size_balanced_stability_subset` therefore downsamples each
size stratum to the smallest stable count, deterministically, so the subset
keeps a balanced size margin and gamma_size stays estimable.  Retention counts
before and after balancing are reported either way (§8.8).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "na_table",
    "menu_stability",
    "position_flip_summary",
    "size_balanced_stability_subset",
]


# ---------------------------------------------------------------------------
# NA accounting (§6.4)
# ---------------------------------------------------------------------------


def na_table(choice_sets: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    One row per cell x stratum x menu size, plus resolution-path counts.

    Rows are emitted even when the NA count is zero, so a downstream table has
    a complete grid rather than a ragged one.
    """
    rows: List[Dict[str, Any]] = []
    for cell_id, choice_set in sorted(choice_sets.items()):
        buckets: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for record in choice_set["choices"]:
            key = (record["difficulty_stratum"], record["menu_size"])
            bucket = buckets.setdefault(
                key,
                {
                    "cell_id": cell_id,
                    "pool_id": choice_set["pool_id"],
                    "model_name": choice_set["model_name"],
                    "prompt_condition": choice_set["prompt_condition"],
                    "difficulty_stratum": key[0],
                    "menu_size": key[1],
                    "total": 0,
                    "na_count": 0,
                    "answer_token": 0,
                    "fallback_parse": 0,
                    "unresolved": 0,
                },
            )
            bucket["total"] += 1
            bucket[record["resolution_path"]] += 1
            if record["chosen_position"] is None:
                bucket["na_count"] += 1

        for bucket in buckets.values():
            bucket["na_rate"] = bucket["na_count"] / bucket["total"]
            # Share of resolutions that needed the weaker parser: a rising
            # value with menu size is the §6.4 parser-degradation signature.
            bucket["fallback_share"] = (
                bucket["fallback_parse"] / bucket["total"] if bucket["total"] else 0.0
            )
            rows.append(bucket)

    rows.sort(key=lambda row: (row["cell_id"], row["difficulty_stratum"], row["menu_size"]))
    return rows


# ---------------------------------------------------------------------------
# Position bias (§3.1)
# ---------------------------------------------------------------------------


def menu_stability(choice_set: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Per-menu position stability for one cell.

    ``stable`` is ``None`` when fewer than two presentations resolved: with no
    second observation there is nothing to compare, and counting that as
    "stable" would quietly inflate the subset with menus we never actually
    tested for position bias.
    """
    by_problem: Dict[str, Dict[str, Any]] = {}
    for record in choice_set["choices"]:
        entry = by_problem.setdefault(
            record["problem_id"],
            {
                "problem_id": record["problem_id"],
                "menu_size": record["menu_size"],
                "difficulty_stratum": record["difficulty_stratum"],
                "chosen_items": [],
                "n_observations": 0,
            },
        )
        entry["n_observations"] += 1
        if record["chosen_item_id"] is not None:
            entry["chosen_items"].append(record["chosen_item_id"])

    for entry in by_problem.values():
        resolved = entry["chosen_items"]
        entry["n_resolved"] = len(resolved)
        entry["stable"] = len(set(resolved)) == 1 if len(resolved) >= 2 else None
    return by_problem


def position_flip_summary(choice_set: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Flip rate for one cell, overall and by menu size.

    The by-size breakdown is not decoration: even under a size-invariant
    softmax the probability that two presentations disagree is 1 - sum(p_i^2),
    which rises with menu size as choice mass spreads over more near-tied
    alternatives.  A flip rate that rises with size is therefore the *expected*
    baseline, not evidence of size-dependent position bias, and the two must
    not be read as the same thing.
    """
    stability = menu_stability(choice_set)
    comparable = [entry for entry in stability.values() if entry["stable"] is not None]
    flipped = [entry for entry in comparable if not entry["stable"]]

    by_size: Dict[int, Dict[str, Any]] = {}
    for entry in comparable:
        bucket = by_size.setdefault(
            entry["menu_size"], {"menu_size": entry["menu_size"], "comparable": 0, "flipped": 0}
        )
        bucket["comparable"] += 1
        if not entry["stable"]:
            bucket["flipped"] += 1
    for bucket in by_size.values():
        bucket["flip_rate"] = bucket["flipped"] / bucket["comparable"]

    return {
        "cell_id": choice_set["cell_id"],
        "pool_id": choice_set["pool_id"],
        "menus_total": len(stability),
        "menus_comparable": len(comparable),
        "menus_flipped": len(flipped),
        "flip_rate": (len(flipped) / len(comparable)) if comparable else 0.0,
        "by_menu_size": [by_size[size] for size in sorted(by_size)],
    }


def size_balanced_stability_subset(
    choice_sets: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 42,
    balance: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    The R6 robustness subset: menus stable in **every** cell, size-balanced.

    A menu enters only if no cell flipped on it, so the subset is a property of
    the design rather than of one arm -- otherwise each cell's contrasts would
    be computed on a different set of menus.

    With *balance* true (the default) each size stratum is downsampled to the
    smallest stable count, which is what keeps the RQ6 slope estimable on the
    subset.  Set it false to inspect the raw stable set.

    Returns ``(problem_ids, report)``; the report carries retention by size
    before and after balancing (§8.8).
    """
    per_menu: Dict[str, Dict[str, Any]] = {}
    flipped_anywhere: set[str] = set()
    untested: set[str] = set()

    for choice_set in choice_sets.values():
        for problem_id, entry in menu_stability(choice_set).items():
            per_menu.setdefault(
                problem_id,
                {"menu_size": entry["menu_size"], "difficulty_stratum": entry["difficulty_stratum"]},
            )
            if entry["stable"] is None:
                untested.add(problem_id)
            elif not entry["stable"]:
                flipped_anywhere.add(problem_id)

    stable = sorted(set(per_menu) - flipped_anywhere - untested)

    by_size_before: Dict[int, List[str]] = {}
    for problem_id in stable:
        by_size_before.setdefault(per_menu[problem_id]["menu_size"], []).append(problem_id)

    selected = list(stable)
    if balance and by_size_before:
        smallest = min(len(ids) for ids in by_size_before.values())
        rng = random.Random(seed)
        selected = []
        for size in sorted(by_size_before):
            ids = sorted(by_size_before[size])
            rng.shuffle(ids)
            selected.extend(sorted(ids[:smallest]))
        selected.sort()

    by_size_after: Dict[int, int] = {}
    for problem_id in selected:
        size = per_menu[problem_id]["menu_size"]
        by_size_after[size] = by_size_after.get(size, 0) + 1

    total_menus = len(per_menu)
    report = {
        "menus_total": total_menus,
        "menus_untested": len(untested),
        "menus_flipped_somewhere": len(flipped_anywhere),
        "menus_stable_everywhere": len(stable),
        "retention_before_balance": {
            size: len(ids) for size, ids in sorted(by_size_before.items())
        },
        "retention_after_balance": dict(sorted(by_size_after.items())),
        "balanced": balance,
        "seed": seed,
        "retention_rate": (len(selected) / total_menus) if total_menus else 0.0,
    }

    if not selected:
        logger.warning(
            "Stability subset is empty: %d menu(s) flipped somewhere, %d untested. "
            "The R6 robustness read cannot be computed for this pool.",
            len(flipped_anywhere),
            len(untested),
        )
    elif balance and len(set(report["retention_after_balance"].values())) > 1:
        logger.warning("Balanced subset is still uneven: %s", report["retention_after_balance"])

    return selected, report

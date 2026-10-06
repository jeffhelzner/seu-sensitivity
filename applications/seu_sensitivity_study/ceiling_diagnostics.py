"""Retained-choice diagnostics and conditional likelihood slices, not gates."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy.special import logsumexp

from .ceiling_prior import POLICY_VERSION


def quantiles(values, *, extremes=False):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if not np.all(np.isfinite(values)):
        raise ValueError("Quantiles require finite values")
    probabilities = [0, .05, .5, .95, 1] if extremes else [.05, .5, .95, .99]
    names = ("min", "q05", "q50", "q95", "max") if extremes else ("q05", "q50", "q95", "q99")
    return dict(zip(names, map(float, np.quantile(values, probabilities))))


def _validate_observation_universe(data, preparation):
    from .assessment_scale import validate_reference
    from .config import build_cells
    from .data_preparation import build_observation_reference, filter_resolved_choices

    try:
        if preparation.get("observation_metadata_version") != 2:
            raise ValueError("A3 requires version-2 complete frozen observation evidence")
        group = preparation["pool_id"]
        reference = validate_reference(preparation["assessment_scale_reference"], group=group)
        frozen = preparation["frozen_observation_reference"]
        if frozen != build_observation_reference(reference, frozen["menus"]):
            raise ValueError("A3 frozen menu/presentation mapping is malformed")
        pool_ids = ["venture", "hiring"] if group == "matched_rq5" else [group]
        canonical = {cell.cell_id: cell for cell in build_cells(pool_ids)}
        retained = preparation["cell_ids"]
        excluded = preparation["excluded_cells"]
        if (len(set(retained)) != len(retained) or len(set(excluded)) != len(excluded)
                or set(retained) & set(excluded) or set(retained) | set(excluded) != set(canonical)
                or set(preparation["na_logs"]) != set(canonical)):
            raise ValueError("A3 retained/excluded cells and mandatory NA audits must partition canonical cells")
        selected = preparation["presentation_id"]
        if selected is not None and (type(selected) is not int or selected not in (1, 2)):
            raise ValueError("A3 invalid selected presentation")
        expected = {
            (cell_id, menu["id"], presentation["presentation_id"]): (menu, presentation["order"])
            for cell_id, cell in canonical.items() for menu in frozen["menus"]
            if menu["pool_id"] == cell.pool_id for presentation in menu["presentations"]
            if selected is None or presentation["presentation_id"] == selected
        }
        by_cell = {cell_id: [] for cell_id in canonical}
        seen = set()
        excluded_keys = set()
        for is_excluded, rows in ((False, preparation["observations"]), (True, preparation["exclusions"])):
            for row in rows:
                key = (row["cell_id"], row["problem_id"], row["presentation_id"])
                if key not in expected or key in seen:
                    raise ValueError("A3 retained/excluded keys must be a disjoint frozen universe")
                seen.add(key)
                menu, order = expected[key]
                position = row["chosen_position"]
                unresolved = position is None
                if (row["item_order"] != order or row["menu_size"] != len(order)
                        or row["difficulty_stratum"] != menu["difficulty_stratum"]
                        or row["resolution_path"] not in ("unresolved", "answer_token", "fallback_parse")
                        or unresolved != (row["resolution_path"] == "unresolved")
                        or (unresolved and row["chosen_item_id"] is not None)
                        or (not unresolved and (type(position) is not int or position not in range(1, len(order) + 1)
                                                or row["chosen_item_id"] != order[position - 1]))
                        or (not is_excluded and (unresolved or row["cell_id"] not in retained))):
                    raise ValueError("A3 row disagrees with frozen presentation mapping or resolution")
                by_cell[row["cell_id"]].append(row)
                if is_excluded:
                    excluded_keys.add(key)
        if seen != set(expected):
            raise ValueError("A3 retained/excluded keys do not cover the complete frozen universe")
        total_na = 0
        for cell_id, rows in by_cell.items():
            _, audit = filter_resolved_choices({"cell_id": cell_id, "pool_id": group, "choices": rows})
            supplied = preparation["na_logs"][cell_id]
            audit["removed_observations"].sort(key=lambda row: (row["problem_id"], row["presentation_id"]))
            supplied = {**supplied, "removed_observations": sorted(
                supplied["removed_observations"], key=lambda row: (row["problem_id"], row["presentation_id"]))}
            if supplied != audit:
                raise ValueError("A3 NA audit disagrees with retained/excluded records")
            whole_cell = audit["na_rate"] > .30
            if whole_cell != (cell_id in excluded):
                raise ValueError("A3 excluded cell membership disagrees with NA threshold")
            for row in rows:
                key = (cell_id, row["problem_id"], row["presentation_id"])
                should_exclude = whole_cell or row["chosen_position"] is None
                reason = "cell_na_exclusion" if whole_cell else "unresolved_choice"
                if (should_exclude != (key in excluded_keys)
                        or (should_exclude and row.get("reason") != reason)):
                    raise ValueError("A3 exclusion reason/membership disagrees with NA audit")
            total_na += audit["na_count"]
        if preparation["overall_na_rate"] != total_na / len(expected):
            raise ValueError("A3 overall NA rate disagrees with complete audit")
        counts = Counter(row["cell_id"] for row in preparation["observations"])
        if data["M_per_cell"] != [counts[cell_id] for cell_id in retained]:
            raise ValueError("A3 retained counts disagree with M_per_cell")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("A3 missing or malformed complete observation evidence") from error


def validate_observations(data, preparation):
    observations = preparation.get("observations")
    items = preparation.get("item_ids", [])
    cells = preparation.get("cell_ids", [])
    if (preparation.get("observation_metadata_version") not in (1, 2)
            or not isinstance(observations, list) or len(observations) != data["M_total"]
            or len(items) != data["R"] or len(set(items)) != len(items)
            or len(cells) != data["J"] or "exclusions" not in preparation):
        raise ValueError("A3 requires bound observation metadata and exclusions")
    seen = set()
    menus = {}
    reference_menus = {menu["id"]: tuple(sorted(menu["item_ids"]))
                       for menu in preparation.get("assessment_scale_reference", {}).get("menus", [])}
    for index, observation in enumerate(observations):
        cell_id = cells[data["cell"][index] - 1]
        active = [item for item, included in zip(items, data["I"][index]) if included]
        order = observation.get("item_order", [])
        key = (cell_id, observation.get("problem_id"), observation.get("presentation_id"))
        if (observation.get("cell_id") != cell_id or not isinstance(key[1], str)
                or not key[1] or key[2] not in (1, 2) or key in seen
                or len(order) != len(active) or set(order) != set(active)
                or observation.get("menu_size") != len(active)
                or observation.get("chosen_item_id") != active[data["y"][index] - 1]
                or observation.get("chosen_position") not in range(1, len(order) + 1)
                or order[observation["chosen_position"] - 1] != observation["chosen_item_id"]):
            raise ValueError("Observation mapping disagrees with retained I/y/cell")
        composition = tuple(sorted(active))
        if reference_menus and reference_menus.get(key[1]) != composition:
            raise ValueError("Observation menu ID disagrees with bound reference composition")
        if key[1] in menus and menus[key[1]] != composition:
            raise ValueError("Menu ID has inconsistent compositions")
        menus[key[1]] = composition
        seen.add(key)
    exclusions = preparation["exclusions"]
    if not isinstance(exclusions, list):
        raise ValueError("Exclusions must be explicit records")
    for exclusion in exclusions:
        key = tuple(exclusion.get(name) for name in ("cell_id", "problem_id", "presentation_id"))
        if (key in seen or not all(isinstance(value, str) and value for value in key[:2])
                or key[2] not in (1, 2) or exclusion.get("menu_size") not in (2, 4, 6, 8)
            or (reference_menus and len(reference_menus.get(key[1], ())) != exclusion.get("menu_size"))
                or exclusion.get("reason") not in ("cell_na_exclusion", "unresolved_choice")):
            raise ValueError("Invalid or duplicated exclusion record")
        seen.add(key)
    _validate_observation_universe(data, preparation)
    return observations


def conditional_log_likelihood(eta_menus, choices, sizes, levels, slope):
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    result = np.zeros(len(levels))
    if (not np.isfinite(slope) or not np.all(np.isfinite(levels))
            or not np.all(np.isfinite(sizes)) or any(not np.all(np.isfinite(eta)) for eta in eta_menus)):
        return np.full(len(levels), np.nan)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for eta, choice, size in zip(eta_menus, choices, sizes):
            centered = np.asarray(eta, dtype=float) - np.max(eta)
            log_scale = levels + slope * size
            logits = np.zeros((len(levels), len(centered)))
            negative = centered < 0
            logits[:, negative] = -np.exp(log_scale[:, None] + np.log(-centered[negative]))
            result += logits[:, choice - 1] - logsumexp(logits, axis=1)
    return result


def likelihood_slices(eta_menus, choices, sizes, level_draws, slope_draws):
    if (not eta_menus or np.asarray(level_draws).size == 0 or np.asarray(slope_draws).size == 0
            or not np.all(np.isfinite(level_draws)) or not np.all(np.isfinite(slope_draws))
            or not np.all(np.isfinite(sizes)) or any(not np.all(np.isfinite(eta)) for eta in eta_menus)):
        return {"status": "unresolved", "reason": "Nonfinite or empty diagnostic input", "slices": []}
    equal = [bool(np.all(eta == np.max(eta))) for eta in eta_menus]
    maximizers = [int(np.count_nonzero(eta == np.max(eta))) for eta in eta_menus]
    all_max = all(eta[choice - 1] == np.max(eta) for eta, choice in zip(eta_menus, choices))
    classification = "constant" if all(equal) else "finite_supremum_at_infinity" if all_max else "eventual_upper_tail_decay"
    limit = -sum(math.log(count) for count in maximizers) if all_max else None
    result = {"status": "resolved", "classification": classification,
              "analytic_limit": limit, "limit_is_negative_infinity": not all_max,
              "fixed_finite_slope_only": True, "slices": [],
              "finite_likelihood_maximizer": False if classification == "finite_supremum_at_infinity" else None,
              "qualification": "Finite posterior upper quantiles reflect prior regularization and pooling, not likelihood-only bounds."
              if classification == "finite_supremum_at_infinity" else None}
    posterior = quantiles(level_draws)
    locations = {**posterior, "q95_plus_log2": posterior["q95"] + math.log(2)}
    lower, upper = min(-5.0, posterior["q05"] - 2), max(10.0, posterior["q99"] + 2)
    for label, slope in zip(("q05", "q50", "q95"), np.quantile(slope_draws, [.05, .5, .95])):
        grid = np.linspace(lower, upper, 401)
        likelihood = conditional_log_likelihood(eta_menus, choices, sizes, grid, slope)
        extra = conditional_log_likelihood(eta_menus, choices, sizes, list(locations.values()), slope)
        row = {"slope_quantile": label, "slope": float(slope), "initial_endpoints": [lower, upper],
               "initial_grid_count": 401, "refined": False}
        if not np.all(np.isfinite(likelihood)) or not np.all(np.isfinite(extra)):
            row.update(status="unresolved", reason="Nonfinite likelihood evaluation")
            result["status"] = "unresolved"
        else:
            if extra.max() > likelihood.max():
                grid = np.unique(np.concatenate((grid, list(locations.values()))))
                likelihood = conditional_log_likelihood(eta_menus, choices, sizes, grid, slope)
                row.update(refined=True, refinement="Added posterior evaluation locations to grid")
            maximum = float(likelihood.max())
            row.update(status="resolved", grid=grid.tolist(), endpoints=[float(grid[0]), float(grid[-1])],
                       grid_maximum=maximum, grid_maximum_t=float(grid[np.argmax(likelihood)]),
                       lower_endpoint_max=bool(likelihood[0] == maximum),
                       upper_endpoint_max=bool(likelihood[-1] == maximum),
                       log_likelihood_difference=(likelihood - maximum).tolist(),
                       evaluations={name: {"t": location, "log_likelihood": float(value),
                                           "difference_from_grid_maximum": float(value - maximum),
                                           "supremum_gap": float(limit - value) if classification == "finite_supremum_at_infinity" else None}
                                    for (name, location), value in zip(locations.items(), extra)},
                       q95_doubling_signed_change=float(extra[-1] - extra[2]))
        result["slices"].append(row)
    return result


def _counts(rows, exclusions):
    return {"observations": len(rows), "distinct_menu_ids": len({row["problem_id"] for row in rows}),
            "exclusions": len(exclusions)}


def retained_ceiling_report(data, preparation, level_draws, slope_draws):
    observations = validate_observations(data, preparation)
    eta = np.asarray(data["eta"])
    cells = preparation["cell_ids"]
    if np.asarray(level_draws).ndim != 2 or np.asarray(level_draws).shape[1] != len(cells):
        raise ValueError("Cell level draws must follow retained cell order")
    report = {"policy_version": POLICY_VERSION, "decision_count": 0, "gate": None,
              "interpretation": "Conditional likelihood slices, not profile likelihoods or confidence intervals; fixed eta and independent-choice likelihood only.",
              "excluded_cells": preparation.get("excluded_cells", []),
              "exclusions": preparation["exclusions"], "cells": {}}
    for cell_index, cell_id in enumerate(cells):
        indices = [index for index, row in enumerate(observations) if row["cell_id"] == cell_id]
        selected = [observations[index] for index in indices]
        exclusions = [row for row in preparation["exclusions"] if row["cell_id"] == cell_id]
        menus, choices, individual = [], [], []
        for index, row in zip(indices, selected):
            values = eta[cell_index, np.asarray(data["I"][index], dtype=bool)]
            choice = data["y"][index]
            exact = values == values.max()
            near = np.isclose(values, values.max(), atol=1e-12, rtol=0)
            menus.append(values)
            choices.append(choice)
            individual.append({**row, "regret": float(values.max() - values[choice - 1]),
                               "top_two_gap": float(np.sort(values)[-1] - np.sort(values)[-2]),
                               "exact_maximizers": int(exact.sum()), "near_maximizers": int(near.sum()),
                               "chosen_exact_maximizer": bool(exact[choice - 1]),
                               "chosen_near_maximizer": bool(near[choice - 1]),
                               "all_equal_exact": bool(exact.all()), "all_equal_near": bool(near.all())})

        def summary(rows):
            return {"regret": quantiles([row["regret"] for row in rows], extremes=True),
                    "top_two_gap": quantiles([row["top_two_gap"] for row in rows], extremes=True),
                    **{mode: {"maximizer_choices": sum(row[f"chosen_{mode}_maximizer"] for row in rows),
                              "below_maximum_choices": sum(not row[f"chosen_{mode}_maximizer"] for row in rows),
                              "all_equal_menus_observations": sum(row[f"all_equal_{mode}"] for row in rows),
                              "maximizer_count_distribution": dict(Counter(str(row[f"{mode}_maximizers"]) for row in rows))}
                       for mode in ("exact", "near")}}

        report["cells"][cell_id] = {
            **_counts(selected, exclusions), **summary(individual), "individual": individual,
            "near_tie_tolerance": {"atol": 1e-12, "rtol": 0},
            "menus": {row["problem_id"]: {name: row[name] for name in (
                "exact_maximizers", "near_maximizers", "all_equal_exact", "all_equal_near", "top_two_gap")}
                for row in individual},
            "by_size": {str(size): {**_counts([row for row in selected if row["menu_size"] == size],
                                               [row for row in exclusions if row["menu_size"] == size]),
                                    **summary([row for row in individual if row["menu_size"] == size])}
                        for size in (2, 4, 6, 8)},
            "by_presentation": {str(presentation): _counts(
                [row for row in selected if row["presentation_id"] == presentation],
                [row for row in exclusions if row["presentation_id"] == presentation]) for presentation in (1, 2)},
            "by_size_and_presentation": {f"{size}/{presentation}": _counts(
                [row for row in selected if (row["menu_size"], row["presentation_id"]) == (size, presentation)],
                [row for row in exclusions if (row["menu_size"], row["presentation_id"]) == (size, presentation)])
                for size in (2, 4, 6, 8) for presentation in (1, 2)},
            "likelihood": likelihood_slices(menus, choices, [data["s"][index] for index in indices],
                                             np.asarray(level_draws)[:, cell_index], slope_draws),
        }
    return report
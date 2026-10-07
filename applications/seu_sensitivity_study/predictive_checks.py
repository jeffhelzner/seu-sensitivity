"""Amendment 8 descriptive checks of saved joint posterior replicas."""

from __future__ import annotations

from math import fsum

import numpy as np
from scipy.special import logsumexp


POLICY_ID = "amendment8_predictive_checks_v1"
SIZES = (2, 4, 6, 8)


def validate_evidence(data, preparation):
    from .assessment_scale import validate_retained_data
    from .data_preparation import build_predictive_reference

    if preparation.get("observation_metadata_version") != 2:
        raise ValueError("A4 requires version-2 complete observation evidence")
    validate_retained_data(preparation.get("assessment_scale_reference"), data, preparation,
                           group=preparation["pool_id"])
    try:
        reference = preparation["predictive_reference"]
        rebuilt = build_predictive_reference(preparation["assessment_scale_reference"],
                                             preparation["frozen_observation_reference"], reference["items"])
        if reference != rebuilt:
            raise ValueError("A4 predictive_reference differs from canonical roles/frozen recipes")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("A4 requires predictive_reference; regenerate preparation reports from frozen inputs "
                         "and refresh manifest SHA256 bindings (no refit)") from error
    return reference


def validate_sibling_evidence(primary, sibling, variant):
    for field in ("predictive_reference", "frozen_observation_reference"):
        if sibling.get(field) != primary.get(field):
            raise ValueError("A4 frozen role/menu evidence differs across sibling fit variants")
    if variant.startswith("utility_") and sibling != primary:
        raise ValueError("A4 utility variants must retain exactly the primary observations and exclusions")
    if sibling["presentation_id"] is not None:
        fields = ("cell_id", "problem_id", "presentation_id", "chosen_item_id", "chosen_position", "resolution_path")
        original = {tuple(row[field] for field in fields) for row in primary["observations"] + primary["exclusions"]
                    if row["presentation_id"] == sibling["presentation_id"]}
        selected = {tuple(row[field] for field in fields) for row in sibling["observations"] + sibling["exclusions"]}
        if original != selected:
            raise ValueError("A4 presentation fit changes original collection outcomes")


def quantiles(values):
    return dict(zip(("q05", "q50", "q95"), map(float, np.quantile(values, [0.05, 0.5, 0.95]))))


def _observation_mean(values):
    values = np.asarray(values, dtype=float)
    means = np.fromiter((fsum(row) for row in np.atleast_2d(values).tolist()), dtype=float) / values.shape[-1]
    return means[0] if values.ndim == 1 else means


def scalar_summary(observed, replicated, *, structural_zero=False):
    """Summarize a scalar, preserving joint-draw discrepancies and exact ties."""
    replicated = np.asarray(replicated, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if (replicated.ndim != 1 or not replicated.size
            or observed.shape not in ((), replicated.shape)
            or not np.all(np.isfinite(replicated)) or not np.all(np.isfinite(observed))):
        raise ValueError("A4 scalar statistics require finite matched draws")
    dependent = observed.ndim == 1
    predictive = quantiles(replicated)
    difference = quantiles(observed - replicated)
    flagged = (difference["q05"] > 0 or difference["q95"] < 0) if dependent else (
        observed < predictive["q05"] or observed > predictive["q95"])
    return {
        "status": "available", "draw_dependent": dependent,
        "observed": quantiles(observed) if dependent else float(observed),
        "replicated": predictive, "observed_minus_replicated": difference,
        "difference_from_predictive_median": None if dependent else float(observed - predictive["q50"]),
        "tails": {"less": float(np.mean(replicated < observed)),
                  "equal": float(np.mean(replicated == observed)),
                  "greater": float(np.mean(replicated > observed))},
        "descriptive_review_flag": bool(flagged), "structural_zero": structural_zero,
    }


def unavailable(reason):
    return {"status": "unavailable", "reason": reason, "descriptive_review_flag": None}


def interpretation(report):
    """Keep nonexclusive qualifications alongside RQ6 without changing decisions."""
    if report["status"] != "descriptive":
        return {"policy_version": POLICY_ID, "status": "unavailable", "reason": report["reason"],
                "primary_decision_unchanged": True}

    def flagged(section, fields):
        return [{key: row[key] for key in ("scope", "id", "pool_id", "family", "difficulty_stratum", "menu_size",
                                          "presentation_id", "position", "item_id", "observation_count",
                                          "counts_by_size", "complete_pair_count") if key in row} | {"statistic": name}
                for row in report[section] for name in fields
                if (row.get("statistics", row).get(name) or {}).get("descriptive_review_flag")]

    trends = flagged("trends", ("maximizer_fraction", "mean_regret", "filler_fraction"))
    patterns = flagged("groups", ("maximizer_fraction", "mean_regret", "filler_fraction"))
    patterns += flagged("items", ("conditional_rate", "unconditional_share"))
    positions = flagged("positions", ("fraction",))
    repetition = [row for row in flagged("pairs", ("same_item",))
                  if any(pair["scope"] == row["scope"] and pair["id"] == row["id"]
                         and all(pair.get(key) == row.get(key) for key in ("family", "difficulty_stratum", "menu_size"))
                         and pair["same_item"]["difference_from_predictive_median"] > 0 for pair in report["pairs"])]
    missing = [row for row in report["missingness"]["groups"]
               if row["scope"] == "pool_task" and not any(key in row for key in ("family", "menu_size", "presentation_id"))]
    return {"policy_version": POLICY_ID, "status": "available", "primary_decision_unchanged": True,
            "qualifications": {
                "behavioral_trend_not_reproduced": {"applies": bool(trends), "rows": trends,
                    "meaning": "The fitted model does not reproduce flagged trends at the descriptive band; gamma_size is not a complete explanation of size-related choices."},
                "choice_pattern_discrepancies": {"applies": bool(patterns), "rows": patterns,
                    "meaning": "Report patterns and counts; one discrepancy does not identify an omitted mechanism."},
                "excess_same_item_repetition": {"applies": bool(repetition), "rows": repetition,
                    "meaning": "Qualifies independent choices; compare presentation-specific medians and intervals. An interval including zero in a half-sized fit alone is not failure."},
                "display_position_discrepancies": {"applies": bool(positions), "rows": positions,
                    "meaning": "Qualifies the position-insensitive model, not a causal position claim."},
                "unavailable_sizes_or_pairs": {"applies": any(stat["status"] == "unavailable" for row in report["trends"] for stat in row["statistics"].values())
                    or any(row["same_item"]["status"] == "unavailable" for row in report["pairs"]),
                    "meaning": "Absent sizes or complete pairs leave their diagnostics unavailable."},
                "conditional_on_retention": {"applies": any(row["excluded_total"] for row in missing),
                    "counts": missing, "differential_missingness": any(row["range"] > 0 for row in report["missingness"]["size_unresolved"]),
                    "meaning": "Missingness and whole-cell removal limit retained-choice interpretation; no MNAR robustness or equivalence claim."}},
            "always": "Sampler gates, no flags, or predictive agreement do not establish model truth, precise upper-tail sensitivity, or unrestricted interpretation. Read A3 and A4 jointly; no decision cancellation or replacement fit."}


def size_trend(observed, replicated):
    """Use all four sizes with equal weight, including structural filler zeros."""
    if set(observed) != set(SIZES) or set(replicated) != set(SIZES):
        return unavailable("All four retained sizes required; no imputation or reweighting")
    return scalar_summary(sum((size - 5) * observed[size] for size in SIZES) / 20,
                          sum((size - 5) * np.asarray(replicated[size]) for size in SIZES) / 20)


def _counts(rows):
    return {"observation_count": len(rows),
            "distinct_menu_count": len({row["problem_id"] for row in rows}),
            "contributing_cell_count": len({row["cell_id"] for row in rows})}


def _missing_counts(rows, retained_keys):
    unresolved = sum(row["chosen_position"] is None for row in rows)
    retained = sum((row["cell_id"], row["problem_id"], row["presentation_id"]) in retained_keys for row in rows)
    removed = sum(row["chosen_position"] is not None and row.get("reason") == "cell_na_exclusion" for row in rows)
    if len(rows) != retained + unresolved + removed:
        raise ValueError("A4 missingness exclusions do not reconcile")
    return {"eligible": len(rows), "resolved": len(rows) - unresolved, "retained": retained,
            "unresolved": unresolved, "resolved_whole_cell_removed": removed,
            "excluded_total": unresolved + removed,
            "whole_cell_excluded_total": sum(row.get("reason") == "cell_na_exclusion" for row in rows),
            "unresolved_fraction": unresolved / len(rows) if rows else None}


def build_predictive_report(data, preparation, predicted, alpha_obs):
    """Validate bound inputs and summarize saved replicas without generating draws."""
    reference = validate_evidence(data, preparation)
    return _calculate(data, preparation, reference, predicted, alpha_obs)


def _calculate(data, preparation, reference, predicted, alpha_obs):
    from .config import build_cells

    observations = preparation["observations"]
    universe = observations + preparation["exclusions"]
    menus = {menu["id"]: menu for menu in reference["menus"]}
    item_ids = preparation["item_ids"]
    items = {item["id"]: item for item in reference["items"]}
    canonical = {cell.cell_id: cell.pool_id for cell in build_cells(["venture", "hiring"])}
    pool_ids = sorted({item["pool_id"] for item in reference["items"]})
    eligible_cells = sorted(cell_id for cell_id, pool in canonical.items() if pool in pool_ids)
    predicted = np.asarray(predicted)
    alpha_obs = np.asarray(alpha_obs, dtype=float)
    sizes = np.asarray(data["I"]).sum(axis=1)
    if (predicted.ndim != 2 or predicted.shape[1] != len(observations) or not len(predicted)
            or alpha_obs.shape != predicted.shape or not np.all(np.isfinite(alpha_obs))
            or np.any(alpha_obs <= 0) or not np.all(np.isfinite(predicted))
            or np.any(predicted != np.floor(predicted))
            or np.any(predicted < 1) or np.any(predicted > sizes)):
        raise ValueError("A4 requires valid same-row saved y_pred and alpha_obs arrays")
    predicted = predicted.astype(int) - 1
    draw_count, observation_count = predicted.shape
    choices = np.asarray(data["y"], dtype=int) - 1
    eta = np.asarray(data["eta"], dtype=float)
    cells = np.asarray(data["cell"], dtype=int) - 1
    fixed_names = ("maximizer_fraction", "mean_regret", "filler_fraction")
    score_names = ("mean_selected_probability", "mean_log_score")
    observed = {name: np.empty(observation_count) for name in fixed_names}
    observed.update({name: np.empty(predicted.shape) for name in score_names})
    replicated = {name: np.empty(predicted.shape) for name in (*fixed_names, *score_names)}
    replicated_items = np.empty(predicted.shape, dtype=int)
    replicated_positions = np.empty(predicted.shape, dtype=int)
    observed_items = np.empty(observation_count, dtype=int)
    observed_positions = np.empty(observation_count, dtype=int)
    filler_probability = np.empty(predicted.shape)
    probabilities = []
    for index, row in enumerate(observations):
        active = np.flatnonzero(data["I"][index])
        values = eta[cells[index], active]
        logits = alpha_obs[:, index, None] * (values - values.max())
        logs = logits - logsumexp(logits, axis=1)[:, None]
        probability = np.exp(logs)
        probabilities.append(probability)
        predicted_index = predicted[:, index]
        observed_log = logs[:, choices[index]]
        replicated_log = logs[np.arange(draw_count), predicted_index]
        observed["mean_log_score"][:, index] = observed_log
        replicated["mean_log_score"][:, index] = replicated_log
        observed["mean_selected_probability"][:, index] = np.exp(observed_log)
        replicated["mean_selected_probability"][:, index] = np.exp(replicated_log)
        fillers = np.isin([item_ids[item] for item in active], menus[row["problem_id"]]["filler_item_ids"])
        filler_probability[:, index] = probability[:, fillers].sum(axis=1)
        for name, values_by_item in (("maximizer_fraction", values == values.max()),
                                     ("mean_regret", values.max() - values), ("filler_fraction", fillers)):
            observed[name][index] = values_by_item[choices[index]]
            replicated[name][:, index] = values_by_item[predicted_index]
        positions = np.asarray([row["item_order"].index(item_ids[item]) + 1 for item in active])
        observed_items[index] = active[choices[index]]
        replicated_items[:, index] = active[predicted_index]
        observed_positions[index] = positions[choices[index]]
        replicated_positions[:, index] = positions[predicted_index]

    retained_keys = {(row["cell_id"], row["problem_id"], row["presentation_id"]) for row in observations}
    report = {"policy_version": POLICY_ID, "status": "descriptive", "decision_count": 0,
              "gate": None, "draw_count": draw_count,
              "replication": "saved y_pred; same joint posterior row across all observations and statistics; no RNG",
              "weighting": "equal retained observations, not Amendment 5 equal-cell estimands",
              "scope": "in-sample conditional independent-choice likelihood; not held-out forecasts",
              "flags": "descriptive central 90% review flags, not p-values, discoveries or a multiplicity gate",
              "exact_maximizers": True, "presentation_id": preparation["presentation_id"],
              "groups": [], "items": [], "positions": [], "pairs": [], "trends": [],
              "missingness": {"groups": [], "size_unresolved": [],
                              "scope": "original eligible observations for the selected presentation scope",
                              "partition": "eligible = retained + unresolved + resolved_whole_cell_removed"}}

    def summarize(indices, name):
        if not indices:
            return unavailable("No retained observations")
        observed_values = observed[name]
        observed_mean = _observation_mean(observed_values[..., indices])
        return scalar_summary(observed_mean, _observation_mean(replicated[name][:, indices]),
                              structural_zero=name == "filler_fraction" and bool(np.all(sizes[indices] == 2)))

    scopes = [("cell", cell_id, canonical[cell_id]) for cell_id in eligible_cells]
    scopes += [("pool_task", pool, pool) for pool in pool_ids]
    for scope, identity, pool in scopes:
        base = {"scope": scope, "id": identity, "pool_id": pool}

        def belongs(row):
            return row["cell_id"] == identity if scope == "cell" else canonical[row["cell_id"]] == pool

        scope_indices = [index for index, row in enumerate(observations) if belongs(row)]
        scope_universe = [row for row in universe if belongs(row)]
        families = sorted({item["family"] for item in items.values() if item["pool_id"] == pool})
        partitions = [{}] + [{"menu_size": size} for size in SIZES]
        partitions += [{"family": family, "difficulty_stratum": stratum, "menu_size": size}
                       for family in families for stratum in ("strong", "ambiguous", "weak") for size in SIZES]

        def matches(row, partition):
            return all(menus[row["problem_id"]][key] == value for key, value in partition.items())

        per_size = {}
        for partition in partitions:
            descriptor = {**base, **partition}
            indices = [index for index in scope_indices if matches(observations[index], partition)]
            selected = [observations[index] for index in indices]
            original = [row for row in scope_universe if matches(row, partition)]
            stats = {name: summarize(indices, name) for name in (*fixed_names, *score_names)}
            report["groups"].append({**descriptor, **_counts(selected), "statistics": stats,
                                     "observations_containing_fillers": sum(row["menu_size"] > 2 for row in selected),
                                     "conditional_filler_probability": quantiles(_observation_mean(filler_probability[:, indices])) if indices else None})
            report["missingness"]["groups"].append({**descriptor, **_missing_counts(original, retained_keys)})
            for presentation in (1, 2):
                report["missingness"]["groups"].append({**descriptor, "presentation_id": presentation,
                    **_missing_counts([row for row in original if row["presentation_id"] == presentation], retained_keys)})
            if "family" in partition:
                per_size[partition["family"], partition["difficulty_stratum"], partition["menu_size"]] = indices

        failure_rates = {}
        for size in SIZES:
            original = [row for row in scope_universe if row["menu_size"] == size]
            if not original:
                raise ValueError("A4 missing original eligible size denominators in frozen design")
            failure_rates[str(size)] = _missing_counts(original, retained_keys)
        rates = [row["unresolved_fraction"] for row in failure_rates.values()]
        report["missingness"]["size_unresolved"].append({**base, "by_size": failure_rates,
                                                       "range": max(rates) - min(rates)})

        for family in families:
            for stratum in ("strong", "ambiguous", "weak"):
                indices_by_size = {size: per_size[family, stratum, size] for size in SIZES}
                stats = {}
                for name in fixed_names:
                    actual = {size: _observation_mean(observed[name][indices]) for size, indices in indices_by_size.items() if indices}
                    replicas = {size: _observation_mean(replicated[name][:, indices]) for size, indices in indices_by_size.items() if indices}
                    stats[name] = size_trend(actual, replicas)
                report["trends"].append({**base, "family": family, "difficulty_stratum": stratum,
                    "counts_by_size": {str(size): len(indices) for size, indices in indices_by_size.items()},
                    "weights": {str(size): (size - 5) / 20 for size in SIZES}, "statistics": stats})

        for item_index, item_id in enumerate(item_ids):
            if items[item_id]["pool_id"] != pool:
                continue
            count = len(scope_indices)
            exposure = sum(data["I"][index][item_index] for index in scope_indices)
            actual = int(np.sum(observed_items[scope_indices] == item_index))
            replica = np.sum(replicated_items[:, scope_indices] == item_index, axis=1)
            report["items"].append({**base, "item_id": item_id, **_counts([observations[index] for index in scope_indices]),
                "exposure_count": exposure, "observed_selection_count": actual,
                "selection_count": scalar_summary(actual, replica) if count else unavailable("No retained observations"),
                "conditional_rate": scalar_summary(actual / exposure, replica / exposure) if exposure else unavailable("Zero exposure"),
                "unconditional_share": scalar_summary(actual / count, replica / count) if count else unavailable("No retained observations")})

        for size in SIZES:
            for presentation in (1, 2):
                indices = [index for index in scope_indices if observations[index]["menu_size"] == size
                           and observations[index]["presentation_id"] == presentation]
                for position in range(1, size + 1):
                    actual = int(np.sum(observed_positions[indices] == position))
                    replica = np.sum(replicated_positions[:, indices] == position, axis=1)
                    report["positions"].append({**base, "menu_size": size, "presentation_id": presentation,
                        "position": position, **_counts([observations[index] for index in indices]),
                        "count": scalar_summary(actual, replica) if indices else unavailable("No retained observations"),
                        "fraction": scalar_summary(actual / len(indices), replica / len(indices)) if indices else unavailable("No retained observations")})

        retained_pairs = {}
        for index in scope_indices:
            row = observations[index]
            retained_pairs.setdefault((row["cell_id"], row["problem_id"]), {})[row["presentation_id"]] = index
        pair_universe = [(cell_id, menu["id"]) for cell_id in eligible_cells
                         if (cell_id == identity if scope == "cell" else canonical[cell_id] == pool)
                         for menu in reference["menus"] if menu["pool_id"] == canonical[cell_id]]
        for partition in [{}] + [part for part in partitions if "family" in part]:
            keys = [key for key in pair_universe if all(menus[key[1]][name] == value for name, value in partition.items())]
            complete = [retained_pairs[key] for key in keys if len(retained_pairs.get(key, {})) == 2]
            single = sum(len(retained_pairs.get(key, {})) == 1 for key in keys)
            neither = sum(key not in retained_pairs for key in keys)
            descriptor = {**base, **partition, "eligible_pair_count": len(keys), "complete_pair_count": len(complete),
                          "single_retained_presentation_count": single, "neither_retained_pair_count": neither,
                          "distinct_menu_count": len({key[1] for key in keys if len(retained_pairs.get(key, {})) == 2}),
                          "contributing_cell_count": len({key[0] for key in keys if len(retained_pairs.get(key, {})) == 2})}
            if preparation["presentation_id"] is not None or not complete:
                reason = "Presentation-only fit: unavailable by design" if preparation["presentation_id"] is not None else "No complete retained pairs"
                report["pairs"].append({**descriptor, "same_item": unavailable(reason), "same_position": unavailable(reason),
                                        "conditional_same_item": None, "conditional_same_position": None})
                continue
            first = [pair[1] for pair in complete]
            second = [pair[2] for pair in complete]
            same_item = np.mean(replicated_items[:, first] == replicated_items[:, second], axis=1)
            same_position = np.mean(replicated_positions[:, first] == replicated_positions[:, second], axis=1)
            conditional_item = np.zeros(draw_count)
            conditional_position = np.zeros(draw_count)
            for left, right in zip(first, second):
                active_left = np.flatnonzero(data["I"][left])
                active_right = np.flatnonzero(data["I"][right])
                right_by_item = {item: index for index, item in enumerate(active_right)}
                aligned = [right_by_item[item] for item in active_left]
                order_left = observations[left]["item_order"]
                order_right = observations[right]["item_order"]
                position_aligned = [right_by_item[item_ids.index(order_right[order_left.index(item_ids[item])])]
                                    for item in active_left]
                conditional_item += np.sum(probabilities[left] * probabilities[right][:, aligned], axis=1)
                conditional_position += np.sum(probabilities[left] * probabilities[right][:, position_aligned], axis=1)
            report["pairs"].append({**descriptor,
                "same_item": scalar_summary(np.mean(observed_items[first] == observed_items[second]), same_item),
                "same_position": scalar_summary(np.mean(observed_positions[first] == observed_positions[second]), same_position),
                "conditional_same_item": quantiles(conditional_item / len(complete)),
                "conditional_same_position": quantiles(conditional_position / len(complete)),
                "interpretation": "Same-item repetition is not position invariance or a causal repeated-exposure effect"})
    return report
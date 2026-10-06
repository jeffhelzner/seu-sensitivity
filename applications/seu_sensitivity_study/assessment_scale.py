"""Fixed-input descriptive assessment-scale policy and reference validation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from .config import MODELS, build_cells
from .pools import load_pool


POLICY_VERSION = "amendment6_assessment_scale_v1"
TIE_ATOL = 1e-12


def policy() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "utility_middle": 0.5,
        "fit_variant": "primary",
        "included_in_primary_family": False,
        "transformation": "alpha_star = alpha * sd; logalpha_star = logalpha + log(sd)",
        "reference": "all 60 fixed pool items per arm; matched own 24 items per task",
        "weighting": "equal fixed items, never observation or exposure weighted",
        "ddof": 0,
        "centering": "conceptual only; no change to likelihood or priors",
        "interval_quantiles": [0.05, 0.95],
        "zero_sd": "affected standardized rows unavailable; no epsilon",
        "missing_provenance": "fail",
        "menu_reference": "full fixed menu set before choices and exclusions",
        "tie_atol": TIE_ATOL,
        "tie_rtol": 0.0,
        "ppc_tie_policy_changed": False,
        "rq2": "within-model offsets cancel exactly",
        "rq6": "gamma_size unchanged",
    }


def build_reference(*, group, items, problems, probabilities):
    """Build from complete fixed inputs, without accepting any choice records."""
    from .data_preparation import assessment_expected_utilities

    if group not in ("venture", "hiring", "matched_rq5"):
        raise ValueError("assessment_scale reference requires a canonical fit group")
    pool_ids = ["venture", "hiring"] if group == "matched_rq5" else [group]
    canonical = {}
    for pool_id in pool_ids:
        family = {"venture": "procurement", "hiring": "matched"}[pool_id]
        selected = [item for item in load_pool(pool_id)["items"]
                    if group != "matched_rq5" or item["family"] == family]
        expected_count = 24 if group == "matched_rq5" else 60
        if len(selected) != expected_count:
            raise ValueError("assessment_scale frozen item count changed")
        for item in selected:
            canonical[item["id"]] = {"id": item["id"], "pool_id": pool_id, "family": item["family"]}
    if (len(items) != len(canonical) or {item["id"] for item in items} != set(canonical)
            or any(item["family"] != canonical[item["id"]]["family"] for item in items)):
        raise ValueError("assessment_scale requires all frozen item IDs and proper families (60 or own 24)")
    fixed_items = [canonical[item_id] for item_id in sorted(canonical)]
    menus = []
    for problem in problems:
        item_ids = problem["item_ids"]
        if (not isinstance(problem["id"], str) or not problem["id"]
                or len(item_ids) < 2 or len(set(item_ids)) != len(item_ids)
                or not set(item_ids).issubset(canonical)):
            raise ValueError("assessment_scale fixed menu has invalid item IDs")
        menu_pools = {canonical[item_id]["pool_id"] for item_id in item_ids}
        menu_families = {canonical[item_id]["family"] for item_id in item_ids}
        if len(menu_pools) != 1 or len(menu_families) != 1:
            raise ValueError("assessment_scale menus must stay within source pool and family")
        if any(len(presentation["order"]) != len(item_ids) or set(presentation["order"]) != set(item_ids)
               for presentation in problem.get("presentations", [])):
            raise ValueError("assessment_scale fixed presentations must use the full declared menu items")
        menus.append({"id": problem["id"], "pool_id": next(iter(menu_pools)),
                      "item_ids": sorted(item_ids)})
    if len({menu["id"] for menu in menus}) != len(menus):
        raise ValueError("assessment_scale fixed menu IDs must be unique")
    menus.sort(key=lambda menu: menu["id"])
    arms = []
    for pool_id in pool_ids:
        item_ids = [item["id"] for item in fixed_items if item["pool_id"] == pool_id]
        pool_menus = [menu for menu in menus if menu["pool_id"] == pool_id]
        matched_family = {"venture": "procurement", "hiring": "matched"}[pool_id]
        expected_counts = {(matched_family, size): 10 for size in (2, 4, 6, 8)}
        if group != "matched_rq5":
            primary_family = {"venture": "startup", "hiring": "candidates"}[pool_id]
            expected_counts.update({(primary_family, size): 25 for size in (2, 4, 6, 8)})
        menu_counts = Counter((canonical[menu["item_ids"][0]]["family"], len(menu["item_ids"]))
                              for menu in pool_menus)
        if menu_counts != expected_counts:
            raise ValueError(
                f"assessment_scale requires full fixed menus for {pool_id}: "
                f"expected family-by-size counts {expected_counts}, got {dict(menu_counts)}"
            )
        for model in MODELS:
            try:
                source = probabilities[model.name]
                eta = np.asarray(assessment_expected_utilities(
                    source, item_ids=item_ids, utilities=[0.0, 0.5, 1.0]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("assessment_scale requires complete finite assessment probabilities") from error
            indices = {item_id: index for index, item_id in enumerate(item_ids)}
            gaps = []
            for menu in pool_menus:
                ordered = np.sort(eta[[indices[item_id] for item_id in menu["item_ids"]]])
                gaps.append(float(ordered[-1] - ordered[-2]))
            sd = float(np.std(eta, ddof=0)) if np.any(eta != eta[0]) else 0.0
            arms.append({
                "pool_id": pool_id, "model_name": model.name,
                "item_ids": item_ids, "item_count": len(item_ids),
                "probabilities": [list(map(float, source[item_id])) for item_id in item_ids],
                "eta": eta.tolist(), "mean": float(eta.mean()), "sd": sd,
                "log_sd": math.log(sd) if sd > 0 else None,
                "status": "available" if sd > 0 else "unavailable_zero_sd",
                "menu_ids": [menu["id"] for menu in pool_menus],
                "menu_count": len(pool_menus), "top_two_gaps": gaps,
                "top_two_gap_quantiles": dict(zip(("q05", "q50", "q95"), map(float, np.quantile(gaps, [0.05, 0.5, 0.95])))),
                "tie_count": int(np.count_nonzero(np.isclose(gaps, 0.0, atol=TIE_ATOL, rtol=0))),
                "tie_prevalence": float(np.mean(np.isclose(gaps, 0.0, atol=TIE_ATOL, rtol=0))),
            })
    return {"policy": policy(), "group": group, "items": fixed_items, "menus": menus, "arms": arms}


def validate_reference(reference, *, group):
    """Recompute all geometry from the bound full-input reference, failing closed."""
    try:
        if not isinstance(reference, Mapping) or reference.get("group") != group or reference.get("policy") != policy():
            raise ValueError("missing, stale or wrong-midpoint assessment_scale reference")
        probabilities = defaultdict(dict)
        for arm in reference["arms"]:
            if len(arm["item_ids"]) != len(arm["probabilities"]):
                raise ValueError("assessment_scale probability dimensions mismatch")
            probabilities[arm["model_name"]].update(zip(arm["item_ids"], arm["probabilities"]))
        rebuilt = build_reference(group=group, items=reference["items"],
                                  problems=reference["menus"], probabilities=probabilities)
        if reference != rebuilt:
            raise ValueError("assessment_scale reference geometry or full item mapping is stale/malformed")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("missing or malformed assessment_scale reference") from error
    return reference


def validate_retained_data(reference, data, preparation, *, group):
    """Check item indexing, sibling arms, eta, and active menu support."""
    from .data_preparation import assessment_expected_utilities

    validate_reference(reference, group=group)
    if preparation.get("observation_metadata_version") == 2:
        from .ceiling_diagnostics import validate_observations

        validate_observations(data, preparation)
    item_ids = preparation.get("item_ids")
    if (not isinstance(item_ids, list) or len(item_ids) != data["R"]
            or len(set(item_ids)) != len(item_ids)
            or set(item_ids) != {item["id"] for item in reference["items"]}):
        raise ValueError("assessment_scale preparation must map every full item ID to eta columns")
    arms = {(arm["pool_id"], arm["model_name"]): arm for arm in reference["arms"]}
    canonical_cells = {cell.cell_id: cell for cell in build_cells(["venture", "hiring"])}
    for cell_index, cell_id in enumerate(preparation["cell_ids"]):
        cell = canonical_cells[cell_id]
        arm = arms[cell.pool_id, cell.model_name]
        indices = [item_ids.index(item_id) for item_id in arm["item_ids"]]
        probabilities = dict(zip(arm["item_ids"], arm["probabilities"]))
        expected = assessment_expected_utilities(probabilities, item_ids=arm["item_ids"], utilities=data["utility_values"])
        if not np.allclose(np.asarray(data["eta"])[cell_index, indices], expected, atol=1e-12, rtol=0):
            raise ValueError("assessment_scale retained eta disagrees with bound reference probabilities")
        allowed = {tuple(menu["item_ids"]) for menu in reference["menus"] if menu["pool_id"] == cell.pool_id}
        for observation in np.flatnonzero(np.asarray(data["cell"]) == cell_index + 1):
            active = tuple(sorted(item_ids[index] for index in np.flatnonzero(data["I"][observation])))
            if active not in allowed:
                raise ValueError("assessment_scale observation outside fixed menu or matched family support")


def contrast_offset(reference, weights):
    """Combine arm-level weights first so within-model prompt offsets cancel exactly."""
    cells = {cell.cell_id: cell for cell in build_cells(["venture", "hiring"])}
    arms = {(arm["pool_id"], arm["model_name"]): arm for arm in reference["arms"]}
    grouped = defaultdict(list)
    zero_cells = []
    for cell_id, weight in weights.items():
        cell = cells[cell_id]
        key = (cell.pool_id, cell.model_name)
        if arms[key]["sd"] == 0:
            zero_cells.append(cell_id)
        grouped[key].append(weight)
    if zero_cells:
        return None, zero_cells
    return math.fsum(math.fsum(weights) * arms[key]["log_sd"] for key, weights in grouped.items()), []
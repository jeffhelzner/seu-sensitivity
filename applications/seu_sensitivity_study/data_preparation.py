"""
Embedding, reduction, and Stan-data assembly (study plan §5, §6.1 steps 4-5).

Three things here differ from the methodology-paper pipeline, each deliberately.

**Item texts are embedded, not assessments (§5).**  ``h_m01`` assumes a single
shared alternative pool ``w`` within a fit.  One embedding per item satisfies
that exactly, and the cell-specific belief map beta_j absorbs the assessment
step as a cell-specific reading of a fixed item description.  Embedding
per-cell assessments would instead give a *different* ``w`` per cell.

**PCA is fit per pool.**  A D=32 axis means something different for claims than
for candidate profiles, so pools are never projected into a shared space
(§8.1).

**``y`` is re-indexed into the active set.**  This is the easiest thing in the
whole pipeline to get silently wrong.  The collectors record
``chosen_position`` as a position in the *presentation order*, because that is
what the model was shown.  Stan enumerates each menu's alternatives by
ascending pool index ``r`` (see the ``x_flat`` loop in ``h_m01.stan``), so
``y[m]`` must be the chosen item's rank among the menu's *sorted* pool indices.
Passing the presentation position straight through would silently scramble
every observation whose menu was not already in sorted order -- which, after
the reversal counterbalancing, is most of them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from . import schemas

logger = logging.getLogger(__name__)

__all__ = [
    "embed_pool_items",
    "reduce_embeddings",
    "filter_resolved_choices",
    "assessment_expected_utilities",
    "build_matched_rq5_stan_data",
    "build_stan_data",
]


# ---------------------------------------------------------------------------
# Embedding (§6.1 step 4)
# ---------------------------------------------------------------------------


def embed_pool_items(
    pool: Mapping[str, Any], embedding_client: Any
) -> Dict[str, np.ndarray]:
    """
    Embed each item's text once.  Shared across every cell in the pool.

    The returned mapping has exactly one vector per item -- nothing is "pooled
    across cells", which was a leftover concept from the assessment-embedding
    pipeline.
    """
    items = list(pool["items"])
    texts = [item["text"] for item in items]
    vectors = embedding_client.embed(texts)
    if len(vectors) != len(items):
        raise ValueError(
            f"Embedding client returned {len(vectors)} vector(s) for {len(items)} item(s)"
        )
    logger.info("Embedded %d item texts for pool %r", len(items), pool["pool_id"])
    return {
        item["id"]: np.asarray(vector, dtype=float)
        for item, vector in zip(items, vectors)
    }


def reduce_embeddings(
    raw_embeddings: Mapping[str, np.ndarray],
    *,
    target_dim: int = 32,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Project one pool's item embeddings to ``target_dim`` via PCA.

    Returns ``(reduced, info)`` where *info* records the realised dimension and
    explained variance for the provenance manifest (§6.5).
    """
    if not raw_embeddings:
        raise ValueError("Cannot fit PCA on an empty embedding set")

    item_ids = sorted(raw_embeddings)
    matrix = np.stack([raw_embeddings[item_id] for item_id in item_ids])
    n_samples, raw_dim = matrix.shape

    effective_dim = min(target_dim, n_samples, raw_dim)
    if effective_dim < target_dim:
        logger.warning(
            "Clamping PCA target_dim %d -> %d (n_items=%d, raw_dim=%d)",
            target_dim,
            effective_dim,
            n_samples,
            raw_dim,
        )

    pca = PCA(n_components=effective_dim, random_state=seed)
    projected = pca.fit_transform(matrix)

    info = {
        "target_dim": target_dim,
        "effective_dim": int(effective_dim),
        "n_items": int(n_samples),
        "raw_dim": int(raw_dim),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "seed": seed,
    }
    logger.info(
        "PCA: %d components, explained variance %.3f",
        effective_dim,
        info["explained_variance_ratio"],
    )
    return {item_id: projected[i] for i, item_id in enumerate(item_ids)}, info


# ---------------------------------------------------------------------------
# NA filtering (§6.4)
# ---------------------------------------------------------------------------


def filter_resolved_choices(
    choice_set: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split a choice set into usable observations and an NA audit log.

    The log is stratified by difficulty, menu size, and resolution path,
    because §6.4 treats differential refusal as signal and needs
    parser-induced NA separable from genuine refusal.
    """
    resolved: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for record in choice_set["choices"]:
        if record["chosen_position"] is None:
            removed.append(record)
        else:
            resolved.append(record)

    total = len(choice_set["choices"])
    log = {
        "cell_id": choice_set["cell_id"],
        "pool_id": choice_set["pool_id"],
        "total_observations": total,
        "resolved": len(resolved),
        "na_count": len(removed),
        "na_rate": (len(removed) / total) if total else 0.0,
        "na_by_stratum": _tally(removed, "difficulty_stratum"),
        "na_by_menu_size": _tally(removed, "menu_size"),
        "resolution_paths": _tally(choice_set["choices"], "resolution_path"),
        "removed_observations": [
            {
                "problem_id": record["problem_id"],
                "presentation_id": record["presentation_id"],
                "menu_size": record["menu_size"],
                "difficulty_stratum": record["difficulty_stratum"],
                "resolution_path": record["resolution_path"],
                "raw_response": record.get("raw_response"),
            }
            for record in removed
        ],
    }
    return resolved, log


def _tally(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Stan data assembly (§6.1 step 5)
# ---------------------------------------------------------------------------


def assessment_expected_utilities(
    probabilities: Mapping[str, Sequence[float]],
    *,
    item_ids: Sequence[str],
    utilities: Sequence[float],
) -> List[float]:
    """Return item-level expected utilities from elicited probabilities."""
    utility_vector = np.asarray(utilities, dtype=float)
    if utility_vector.ndim != 1 or len(utility_vector) < 2:
        raise ValueError("utilities must be a one-dimensional consequence vector")

    expected_utilities: List[float] = []
    for item_id in item_ids:
        if item_id not in probabilities:
            raise KeyError(f"No parsed assessment probabilities for item {item_id!r}")
        probability_vector = np.asarray(probabilities[item_id], dtype=float)
        if probability_vector.shape != utility_vector.shape:
            raise ValueError(
                f"Assessment probabilities for {item_id!r} have length "
                f"{probability_vector.size}; expected {utility_vector.size}"
            )
        if not np.all(np.isfinite(probability_vector)) or np.any(probability_vector < 0):
            raise ValueError(
                f"Assessment probabilities for {item_id!r} must be finite and nonnegative"
            )
        total = float(probability_vector.sum())
        if not np.isclose(total, 1.0, atol=0.05):
            raise ValueError(
                f"Assessment probabilities for {item_id!r} sum to {total}, expected 1"
            )
        probability_vector = probability_vector / total
        expected_utilities.append(float(probability_vector @ utility_vector))
    return expected_utilities


def build_stan_data(
    *,
    pool: Mapping[str, Any],
    problem_set: Mapping[str, Any],
    choice_sets: Mapping[str, Mapping[str, Any]],
    reduced_embeddings: Mapping[str, np.ndarray],
    design_matrix: np.ndarray,
    cell_ids: Sequence[str],
    K: int,
    include_menu_size: bool = False,
    assessment_probabilities: Optional[
        Mapping[str, Mapping[str, Sequence[float]]]
    ] = None,
    cell_model_names: Optional[Sequence[str]] = None,
    utility_values: Optional[Sequence[float]] = None,
    design_column_names: Optional[Sequence[str]] = None,
    presentation_id: Optional[int] = None,
    include_assessment_scale_reference: bool = False,
    validate: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Assemble one pool's stacked Stan payload.

    Parameters
    ----------
    choice_sets:
        ``{cell_id: choice_set}``.  Cells absent from the mapping contribute no
        observations; that is an error rather than a silent zero-row cell,
        since ``h_m01`` declares ``M_per_cell`` as strictly positive.
    include_menu_size:
        Emit the centered per-observation covariate ``s`` for ``h_m01_size``
        (RQ6, §4).
    assessment_probabilities:
        Optional ``{model_name: {item_id: probabilities}}`` mapping. When
        supplied, emit a cell-by-item matrix of fixed expected utilities
        instead of embedding features for the assessment-anchored model.
    design_column_names:
        When supplied, require the retained intercept-plus-design matrix to
        remain full rank after whole-cell NA exclusions.
    presentation_id:
        When supplied, retain only that frozen presentation index. This creates
        a one-observation-per-menu sensitivity payload without conditioning on
        whether the two observed choices agree.

    Returns
    -------
    (stan_data, report)
        *report* carries the per-cell NA logs and the index maps needed to
        trace any observation back to its menu.
    """
    scale_reference = None
    if include_assessment_scale_reference:
        from .assessment_scale import build_reference

        scale_reference = build_reference(
            group=problem_set["pool_id"], items=pool["items"],
            problems=problem_set["problems"], probabilities=assessment_probabilities,
        )
    item_ids = sorted(reduced_embeddings)
    item_index = {item_id: position for position, item_id in enumerate(item_ids)}
    R = len(item_ids)
    D = len(next(iter(reduced_embeddings.values())))

    problems = {problem["id"]: problem for problem in problem_set["problems"]}
    presentation_orders = {
        (problem["id"], presentation["presentation_id"]): presentation["order"]
        for problem in problem_set["problems"]
        for presentation in problem["presentations"]
    }
    available_presentations = {
        presentation["presentation_id"]
        for problem in problem_set["problems"]
        for presentation in problem["presentations"]
    }
    if presentation_id is not None and presentation_id not in available_presentations:
        raise ValueError(
            f"presentation_id {presentation_id!r} is not in the problem design"
        )

    stacked_I: List[List[int]] = []
    stacked_cell: List[int] = []
    stacked_y: List[int] = []
    menu_sizes: List[int] = []
    M_per_cell: List[int] = []
    na_logs: Dict[str, Any] = {}

    resolved_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    retained_indices: List[int] = []
    excluded_cells: List[str] = []
    for index, cell_id in enumerate(cell_ids):
        if cell_id not in choice_sets:
            raise KeyError(
                f"No choice set supplied for cell {cell_id!r}; every cell in the "
                f"design matrix must contribute observations"
            )
        choice_set = choice_sets[cell_id]
        if presentation_id is not None:
            choice_set = {
                **choice_set,
                "choices": [
                    record
                    for record in choice_set["choices"]
                    if record["presentation_id"] == presentation_id
                ],
            }
        resolved, na_log = filter_resolved_choices(choice_set)
        na_logs[cell_id] = na_log
        if na_log["na_rate"] > 0.30:
            excluded_cells.append(cell_id)
            logger.warning(
                "Excluding cell %r from Stan data: NA rate %.1f%% exceeds 30%%",
                cell_id,
                100.0 * na_log["na_rate"],
            )
            continue
        if not resolved:
            raise ValueError(
                f"Cell {cell_id!r} has no resolved observations (NA rate "
                f"{na_log['na_rate']:.1%}); h_m01 requires M_per_cell >= 1"
            )
        resolved_by_cell[cell_id] = resolved
        retained_indices.append(index)

    if not retained_indices:
        raise ValueError("Every cell exceeds the 30% NA exclusion threshold")

    retained_cell_ids = [cell_ids[index] for index in retained_indices]
    retained_design_matrix = np.asarray(design_matrix)[retained_indices]
    design_rank = None
    required_design_rank = None
    if design_column_names is not None:
        if len(design_column_names) != retained_design_matrix.shape[1]:
            raise ValueError(
                "design_column_names must contain one name per design-matrix column"
            )
        with_intercept = np.column_stack(
            [np.ones(len(retained_design_matrix)), retained_design_matrix]
        )
        design_rank = int(np.linalg.matrix_rank(with_intercept))
        required_design_rank = with_intercept.shape[1]
        if design_rank < required_design_rank:
            raise ValueError(
                "Cell exclusions make the confirmatory design inestimable: "
                f"rank {design_rank} < {required_design_rank}; excluded cells: "
                f"{excluded_cells}"
            )
    retained_model_names = (
        [cell_model_names[index] for index in retained_indices]
        if cell_model_names is not None
        else None
    )

    for position, cell_id in enumerate(retained_cell_ids, start=1):
        resolved = resolved_by_cell[cell_id]
        for record in resolved:
            key = (record["problem_id"], record["presentation_id"])
            order = presentation_orders.get(key)
            if order is None:
                raise KeyError(f"Observation {key} is not in the problem design")

            indicator = [0] * R
            active = []
            for menu_item in order:
                index = item_index[menu_item]
                indicator[index] = 1
                active.append(index)
            active.sort()

            chosen_index = item_index[record["chosen_item_id"]]
            # Rank within the SORTED active set -- see module docstring.
            stacked_y.append(active.index(chosen_index) + 1)
            stacked_I.append(indicator)
            stacked_cell.append(position)
            menu_sizes.append(record["menu_size"])

        M_per_cell.append(len(resolved))

    stan_data: Dict[str, Any] = {
        "J": len(retained_cell_ids),
        "K": K,
        "R": R,
        "P": int(design_matrix.shape[1]),
        "M_total": len(stacked_y),
        "cell": stacked_cell,
        "I": stacked_I,
        "y": stacked_y,
        "X": np.asarray(retained_design_matrix, dtype=float).tolist(),
        "M_per_cell": M_per_cell,
    }

    anchored = assessment_probabilities is not None
    if anchored:
        if cell_model_names is None or len(cell_model_names) != len(cell_ids):
            raise ValueError("cell_model_names must contain one model name per cell")
        if utility_values is None or len(utility_values) != K:
            raise ValueError(f"utility_values must contain K={K} entries")
        stan_data["eta"] = [
            assessment_expected_utilities(
                assessment_probabilities[model_name],
                item_ids=item_ids,
                utilities=utility_values,
            )
            for model_name in retained_model_names
        ]
        stan_data["utility_values"] = [float(value) for value in utility_values]
    else:
        stan_data["D"] = D
        stan_data["w"] = [
            reduced_embeddings[item_id].tolist() for item_id in item_ids
        ]

    mean_menu_size = float(np.mean(menu_sizes)) if menu_sizes else 0.0
    if include_menu_size:
        # Centering is per pool, matching the per-pool fit (§4, §8.2).
        stan_data["s"] = [float(size) - mean_menu_size for size in menu_sizes]

    if validate:
        if anchored:
            model = "h_m01_size_assessment_anchored"
        else:
            model = "h_m01_size" if include_menu_size else "h_m01"
        schemas.check(
            schemas.validate_stan_data(stan_data, model=model),
            context=f"stan data for pool {problem_set['pool_id']!r}",
        )

    report = {
        "pool_id": problem_set["pool_id"],
        "cell_ids": list(retained_cell_ids),
        "excluded_cells": excluded_cells,
        "confirmatory_design_rank": design_rank,
        "confirmatory_design_required_rank": required_design_rank,
        "item_ids": item_ids,
        "design_columns": list(design_column_names) if design_column_names is not None else None,
        "mean_menu_size": mean_menu_size,
        "menu_sizes": menu_sizes,
        "presentation_id": presentation_id,
        "na_logs": na_logs,
        "overall_na_rate": _overall_na_rate(na_logs),
    }
    if scale_reference is not None:
        from .assessment_scale import validate_retained_data

        report["assessment_scale_reference"] = scale_reference
        validate_retained_data(scale_reference, stan_data, report, group=problem_set["pool_id"])
    logger.info(
        "Built Stan data for pool %r: J=%d, R=%d, D=%s, M_total=%d (overall NA %.1f%%)",
        problem_set["pool_id"],
        stan_data["J"],
        R,
        D if not anchored else "assessment-anchored",
        stan_data["M_total"],
        100.0 * report["overall_na_rate"],
    )
    return stan_data, report


def build_matched_rq5_stan_data(
    *,
    venture_pool: Mapping[str, Any],
    hiring_pool: Mapping[str, Any],
    venture_problem_set: Mapping[str, Any],
    hiring_problem_set: Mapping[str, Any],
    venture_choice_sets: Mapping[str, Mapping[str, Any]],
    hiring_choice_sets: Mapping[str, Mapping[str, Any]],
    assessment_probabilities: Mapping[str, Mapping[str, Sequence[float]]],
    design_matrix: np.ndarray,
    cell_ids: Sequence[str],
    cell_model_names: Sequence[str],
    design_column_names: Sequence[str],
    utility_values: Sequence[float],
    K: int,
    presentation_id: Optional[int] = None,
    include_assessment_scale_reference: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the dedicated assessment-anchored matched RQ5 re-slice."""
    family_by_pool = {"venture": "procurement", "hiring": "matched"}
    pools_by_id = {"venture": venture_pool, "hiring": hiring_pool}
    problems_by_id = {
        "venture": venture_problem_set,
        "hiring": hiring_problem_set,
    }
    choices_by_id = {
        "venture": venture_choice_sets,
        "hiring": hiring_choice_sets,
    }

    selected_items: Dict[str, List[Mapping[str, Any]]] = {}
    matched_keys: Dict[str, Dict[str, str]] = {}
    for pool_id, family in family_by_pool.items():
        items = [item for item in pools_by_id[pool_id]["items"] if item["family"] == family]
        index = {item.get("matched_key"): item["id"] for item in items}
        if None in index or len(index) != len(items):
            raise ValueError(f"{pool_id}/{family} must have unique non-null matched keys")
        selected_items[pool_id] = items
        matched_keys[pool_id] = index
    if set(matched_keys["venture"]) != set(matched_keys["hiring"]):
        raise ValueError("RQ5 matched keys differ between procurement and hiring")

    selected_problems: Dict[str, List[Mapping[str, Any]]] = {}
    key_menus: Dict[str, List[List[str]]] = {}
    for pool_id, family in family_by_pool.items():
        problems = [
            problem for problem in problems_by_id[pool_id]["problems"]
            if problem["family"] == family
        ]
        item_to_key = {item_id: key for key, item_id in matched_keys[pool_id].items()}
        selected_problems[pool_id] = problems
        key_menus[pool_id] = [
            [item_to_key[item_id] for item_id in problem["item_ids"]]
            for problem in problems
        ]
    if key_menus["venture"] != key_menus["hiring"]:
        raise ValueError("RQ5 procurement and hiring menus are not paired by matched key")

    problem_ids = {
        pool_id: {problem["id"] for problem in problems}
        for pool_id, problems in selected_problems.items()
    }
    if problem_ids["venture"] & problem_ids["hiring"]:
        raise ValueError("RQ5 problem IDs must be unique across source pools")

    sliced_choices: Dict[str, Dict[str, Any]] = {}
    for pool_id, source in choices_by_id.items():
        for cell_id, choice_set in source.items():
            sliced = [
                record for record in choice_set["choices"]
                if record["problem_id"] in problem_ids[pool_id]
            ]
            sliced_choices[cell_id] = {
                **choice_set,
                "pool_id": "matched_rq5",
                "choices": sliced,
            }

    combined_items = selected_items["venture"] + selected_items["hiring"]
    placeholder_vectors = {
        item["id"]: np.zeros(1, dtype=float) for item in combined_items
    }
    combined_pool = {
        "pool_id": "matched_rq5",
        "items": combined_items,
    }
    combined_problem_set = {
        "pool_id": "matched_rq5",
        "problems": selected_problems["venture"] + selected_problems["hiring"],
    }
    stan_data, report = build_stan_data(
        pool=combined_pool,
        problem_set=combined_problem_set,
        choice_sets=sliced_choices,
        reduced_embeddings=placeholder_vectors,
        design_matrix=design_matrix,
        cell_ids=cell_ids,
        K=K,
        include_menu_size=True,
        assessment_probabilities=assessment_probabilities,
        cell_model_names=cell_model_names,
        utility_values=utility_values,
        design_column_names=design_column_names,
        presentation_id=presentation_id,
        include_assessment_scale_reference=include_assessment_scale_reference,
    )
    report.update(
        {
            "matched_item_pairs": len(matched_keys["venture"]),
            "paired_menus_per_task": len(key_menus["venture"]),
            "representation": "assessment_anchored_no_pca",
            "source_families": family_by_pool,
        }
    )
    return stan_data, report


def _overall_na_rate(na_logs: Mapping[str, Mapping[str, Any]]) -> float:
    total = sum(log["total_observations"] for log in na_logs.values())
    na = sum(log["na_count"] for log in na_logs.values())
    return (na / total) if total else 0.0

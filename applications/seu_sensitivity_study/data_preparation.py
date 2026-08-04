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

    Returns
    -------
    (stan_data, report)
        *report* carries the per-cell NA logs and the index maps needed to
        trace any observation back to its menu.
    """
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

    stacked_I: List[List[int]] = []
    stacked_cell: List[int] = []
    stacked_y: List[int] = []
    menu_sizes: List[int] = []
    M_per_cell: List[int] = []
    na_logs: Dict[str, Any] = {}

    for position, cell_id in enumerate(cell_ids, start=1):
        if cell_id not in choice_sets:
            raise KeyError(
                f"No choice set supplied for cell {cell_id!r}; every cell in the "
                f"design matrix must contribute observations"
            )
        resolved, na_log = filter_resolved_choices(choice_sets[cell_id])
        na_logs[cell_id] = na_log

        if not resolved:
            raise ValueError(
                f"Cell {cell_id!r} has no resolved observations (NA rate "
                f"{na_log['na_rate']:.1%}); h_m01 requires M_per_cell >= 1"
            )

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
        "J": len(cell_ids),
        "K": K,
        "D": D,
        "R": R,
        "P": int(design_matrix.shape[1]),
        "w": [reduced_embeddings[item_id].tolist() for item_id in item_ids],
        "M_total": len(stacked_y),
        "cell": stacked_cell,
        "I": stacked_I,
        "y": stacked_y,
        "X": np.asarray(design_matrix, dtype=float).tolist(),
        "M_per_cell": M_per_cell,
    }

    mean_menu_size = float(np.mean(menu_sizes)) if menu_sizes else 0.0
    if include_menu_size:
        # Centering is per pool, matching the per-pool fit (§4, §8.2).
        stan_data["s"] = [float(size) - mean_menu_size for size in menu_sizes]

    if validate:
        model = "h_m01_size" if include_menu_size else "h_m01"
        schemas.check(
            schemas.validate_stan_data(stan_data, model=model),
            context=f"stan data for pool {problem_set['pool_id']!r}",
        )

    report = {
        "pool_id": problem_set["pool_id"],
        "cell_ids": list(cell_ids),
        "item_ids": item_ids,
        "mean_menu_size": mean_menu_size,
        "menu_sizes": menu_sizes,
        "na_logs": na_logs,
        "overall_na_rate": _overall_na_rate(na_logs),
    }
    logger.info(
        "Built Stan data for pool %r: J=%d, R=%d, D=%d, M_total=%d (overall NA %.1f%%)",
        problem_set["pool_id"],
        stan_data["J"],
        R,
        D,
        stan_data["M_total"],
        100.0 * report["overall_na_rate"],
    )
    return stan_data, report


def _overall_na_rate(na_logs: Mapping[str, Mapping[str, Any]]) -> float:
    total = sum(log["total_observations"] for log in na_logs.values())
    na = sum(log["na_count"] for log in na_logs.values())
    return (na / total) if total else 0.0

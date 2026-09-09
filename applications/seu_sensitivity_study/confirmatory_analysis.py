"""Executable confirmatory contrast and decision contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .config import MODELS, PROMPT_CONDITIONS, REFERENCE_MODEL, REFERENCE_PROMPT
from .config import get_model_spec

__all__ = [
    "BULK_ESS_MINIMUM",
    "CENTRAL_INTERVAL_MASS",
    "LOG_ALPHA_ROPE",
    "MENU_SIZE_ROPE",
    "TAIL_ESS_MINIMUM",
    "ContrastSpec",
    "classify_interval",
    "contract_manifest",
    "interaction_aliasing_report",
    "matched_rq5_contract",
    "matched_rq5_design",
    "primary_contrasts",
]


CENTRAL_INTERVAL_MASS = 0.90
LOG_ALPHA_ROPE = math.log(1.25)
MENU_SIZE_ROPE = math.log(1.05)
BULK_ESS_MINIMUM = 400
TAIL_ESS_MINIMUM = 400


@dataclass(frozen=True)
class ContrastSpec:
    """One predeclared linear contrast of primary regression coefficients."""

    contrast_id: str
    research_question: str
    label: str
    coefficients: Tuple[float, ...]
    column_names: Tuple[str, ...]
    rope_half_width: float = LOG_ALPHA_ROPE
    expected_direction: str = "two_sided"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coefficient_vector(
    column_names: Sequence[str], weights: Mapping[str, float]
) -> Tuple[float, ...]:
    unknown = sorted(set(weights) - set(column_names))
    if unknown:
        raise ValueError(f"Contrast references unknown design columns: {unknown}")
    return tuple(float(weights.get(name, 0.0)) for name in column_names)


def _model_column(model_name: str) -> str:
    return f"model_{get_model_spec(model_name).slug}"


def primary_contrasts(column_names: Sequence[str]) -> Tuple[ContrastSpec, ...]:
    """Return the approved seven RQ1 and two RQ2 contrasts for one pool."""
    names = tuple(column_names)
    contrasts = []

    for model in MODELS:
        if model.name == REFERENCE_MODEL:
            continue
        column = _model_column(model.name)
        contrasts.append(
            ContrastSpec(
                contrast_id=f"rq1_{model.slug}_minus_{get_model_spec(REFERENCE_MODEL).slug}",
                research_question="RQ1",
                label=f"{model.name} minus {REFERENCE_MODEL}",
                coefficients=_coefficient_vector(names, {column: 1.0}),
                column_names=names,
            )
        )

    for vendor in ("openai", "anthropic"):
        flagship = next(
            model for model in MODELS
            if model.vendor == vendor and model.tier == "flagship"
        )
        small = next(
            model for model in MODELS
            if model.vendor == vendor and model.tier == "small"
        )
        weights = {}
        if flagship.name != REFERENCE_MODEL:
            weights[_model_column(flagship.name)] = 1.0
        if small.name != REFERENCE_MODEL:
            weights[_model_column(small.name)] = -1.0
        contrasts.append(
            ContrastSpec(
                contrast_id=f"rq1_{vendor}_flagship_minus_small",
                research_question="RQ1",
                label=f"{vendor} flagship minus small",
                coefficients=_coefficient_vector(names, weights),
                column_names=names,
                expected_direction="positive",
            )
        )

    for prompt in PROMPT_CONDITIONS:
        if prompt == REFERENCE_PROMPT:
            continue
        contrasts.append(
            ContrastSpec(
                contrast_id=f"rq2_{prompt}_minus_{REFERENCE_PROMPT}",
                research_question="RQ2",
                label=f"{prompt} minus {REFERENCE_PROMPT}",
                coefficients=_coefficient_vector(
                    names, {f"prompt_{prompt}": 1.0}
                ),
                column_names=names,
                expected_direction=(
                    "positive" if prompt == "seu_maximizing" else "two_sided"
                ),
            )
        )
    return tuple(contrasts)


def classify_interval(
    *, lower: float, median: float, upper: float, rope_half_width: float
) -> str:
    """Apply the approved central-90%-interval-plus-ROPE decision rule."""
    if not all(math.isfinite(value) for value in (lower, median, upper)):
        raise ValueError("Contrast interval values must be finite")
    if lower > median or median > upper:
        raise ValueError("Contrast interval must satisfy lower <= median <= upper")
    if lower > 0 and median > rope_half_width:
        return "detected_positive"
    if upper < 0 and median < -rope_half_width:
        return "detected_negative"
    return "not_detected"


def interaction_aliasing_report(
    design_matrix: Sequence[Sequence[float]], column_names: Sequence[str]
) -> Dict[str, Any]:
    """Verify that the full interaction spans the additive residual space."""
    design = np.asarray(design_matrix, dtype=float)
    names = tuple(column_names)
    if design.ndim != 2 or design.shape[1] != len(names):
        raise ValueError("Design matrix columns must match column_names")

    model_indices = [i for i, name in enumerate(names) if name.startswith("model_")]
    prompt_indices = [i for i, name in enumerate(names) if name.startswith("prompt_")]
    interactions = np.column_stack(
        [design[:, model] * design[:, prompt]
         for model in model_indices for prompt in prompt_indices]
    )
    additive = np.column_stack([np.ones(design.shape[0]), design])
    projection, *_ = np.linalg.lstsq(additive, interactions, rcond=None)
    residual_interactions = interactions - additive @ projection

    additive_rank = int(np.linalg.matrix_rank(additive))
    interaction_rank = int(np.linalg.matrix_rank(interactions))
    residual_interaction_rank = int(np.linalg.matrix_rank(residual_interactions))
    full_rank = int(np.linalg.matrix_rank(np.column_stack([additive, interactions])))
    residual_cell_dimension = int(design.shape[0] - additive_rank)
    exactly_aliased = (
        residual_interaction_rank == residual_cell_dimension
        and full_rank == design.shape[0]
    )
    if not exactly_aliased:
        raise ValueError(
            "Full model-by-prompt interaction does not span the residual cell space"
        )
    return {
        "n_cells": int(design.shape[0]),
        "additive_rank": additive_rank,
        "interaction_columns": int(interactions.shape[1]),
        "interaction_rank": interaction_rank,
        "residual_cell_dimension": residual_cell_dimension,
        "residual_interaction_rank": residual_interaction_rank,
        "full_rank": full_rank,
        "exactly_aliased_with_cell_residuals": True,
    }


def matched_rq5_design(cells: Sequence[Any]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Build the joint task-by-model design for the matched RQ5 re-slice."""
    model_names = [model.name for model in MODELS if model.name != REFERENCE_MODEL]
    prompt_names = [prompt for prompt in PROMPT_CONDITIONS if prompt != REFERENCE_PROMPT]
    columns = [f"model_{get_model_spec(name).slug}" for name in model_names]
    columns += [f"prompt_{prompt}" for prompt in prompt_names]
    columns.append("task_hiring")
    columns += [
        f"task_hiring_x_model_{get_model_spec(name).slug}" for name in model_names
    ]

    design = np.zeros((len(cells), len(columns)), dtype=float)
    for row, cell in enumerate(cells):
        if cell.pool_id not in {"venture", "hiring"}:
            raise ValueError(f"RQ5 cell has unsupported pool {cell.pool_id!r}")
        if cell.model_name in model_names:
            model_index = model_names.index(cell.model_name)
            design[row, model_index] = 1.0
        if cell.prompt_condition in prompt_names:
            design[row, len(model_names) + prompt_names.index(cell.prompt_condition)] = 1.0
        if cell.pool_id == "hiring":
            task_index = len(model_names) + len(prompt_names)
            design[row, task_index] = 1.0
            if cell.model_name in model_names:
                design[row, task_index + 1 + model_names.index(cell.model_name)] = 1.0

    with_intercept = np.column_stack([np.ones(len(design)), design])
    if np.linalg.matrix_rank(with_intercept) != with_intercept.shape[1]:
        raise ValueError("Matched RQ5 design is rank deficient")
    return design, tuple(columns)


def matched_rq5_contract(cells: Sequence[Any]) -> Dict[str, Any]:
    """Return the six within-model hiring-minus-procurement contrasts."""
    design, columns = matched_rq5_design(cells)
    task_index = columns.index("task_hiring")
    contrasts = []
    for model in MODELS:
        coefficients = np.zeros(len(columns), dtype=float)
        coefficients[task_index] = 1.0
        interaction = f"task_hiring_x_model_{model.slug}"
        if interaction in columns:
            coefficients[columns.index(interaction)] = 1.0
        contrasts.append(
            ContrastSpec(
                contrast_id=f"rq5_{model.slug}_hiring_minus_procurement",
                research_question="RQ5",
                label=f"{model.name}: hiring minus procurement",
                coefficients=tuple(coefficients),
                column_names=columns,
            ).to_dict()
        )
    return {
        "status": "confirmatory_fit_validation_pending",
        "representation": "assessment_anchored_no_pca",
        "n_cells": int(design.shape[0]),
        "design_columns": list(columns),
        "design_rank_with_intercept": int(
            np.linalg.matrix_rank(np.column_stack([np.ones(len(design)), design]))
        ),
        "required_design_rank": int(design.shape[1] + 1),
        "contrasts": contrasts,
        "decision_count": len(contrasts),
    }


def contract_manifest(
    column_names: Sequence[str], design_matrix: Sequence[Sequence[float]]
) -> Dict[str, Any]:
    """Return the complete machine-readable policy frozen before collection."""
    contrasts = primary_contrasts(column_names)
    return {
        "central_interval_mass": CENTRAL_INTERVAL_MASS,
        "interval_quantiles": [0.05, 0.95],
        "log_alpha_rope_half_width": LOG_ALPHA_ROPE,
        "menu_size_rope_half_width": MENU_SIZE_ROPE,
        "bulk_ess_minimum": BULK_ESS_MINIMUM,
        "tail_ess_minimum": TAIL_ESS_MINIMUM,
        "primary_contrasts": [contrast.to_dict() for contrast in contrasts],
        "primary_decisions_per_pool": len(contrasts),
        "rq3": {
            "status": "secondary_descriptive_existing_fit",
            "estimand": "sigma_cell and sigma_cell * z_alpha cell residuals",
            "separate_saturated_fit": False,
            "reason": "The full interaction exactly spans the additive residual cell space and adds no likelihood information.",
            "aliasing": interaction_aliasing_report(design_matrix, column_names),
        },
        "rq4": {"status": "descriptive"},
        "rq5": {
            "status": "confirmatory_preparation_implemented_validation_pending",
            "estimand": "within-model hiring-minus-procurement contrast in a dedicated matched-item fit",
            "representation": "assessment_anchored_no_pca",
            "artifact_directory": "matched_rq5",
            "cross_pool_difference_in_differences": "supporting",
        },
        "rq6": {
            "parameter": "gamma_size",
            "decision_rule": "central 90% interval excludes zero and posterior median exceeds the menu-size ROPE",
        },
        "multiplicity": {
            "adjustment": "none",
            "mitigation": ["hierarchical shrinkage", "ROPE"],
            "report_full_family": True,
            "report_decision_count": True,
            "selection_from_family_prohibited": True,
        },
    }
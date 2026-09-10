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
    "assert_sampler_gates",
    "classify_interval",
    "compare_presentation_reports",
    "compare_utility_reports",
    "contract_manifest",
    "cross_pool_descriptive_report",
    "interaction_aliasing_report",
    "linear_contrast_report",
    "matched_rq5_contract",
    "matched_rq5_design",
    "parameter_report",
    "posterior_fit_report",
    "primary_contrasts",
    "rq3_descriptive_report",
    "summarize_draws",
    "complete_confirmatory_report",
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


def assert_sampler_gates(diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    """Require the frozen sampler gates before confirmatory reporting."""
    required = {
        "max_rhat",
        "min_ess_bulk",
        "min_ess_tail",
        "min_ebfmi",
        "divergences",
        "treedepth_saturated_share",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"Sampler gates missing required fields: {missing}")
    checks = {
        "rhat": float(diagnostics["max_rhat"]) < 1.01,
        "bulk_ess": float(diagnostics["min_ess_bulk"]) >= BULK_ESS_MINIMUM,
        "tail_ess": float(diagnostics["min_ess_tail"]) >= TAIL_ESS_MINIMUM,
        "ebfmi": float(diagnostics["min_ebfmi"]) >= 0.3,
        "divergences": int(diagnostics["divergences"]) == 0,
        "treedepth": float(diagnostics["treedepth_saturated_share"]) == 0.0,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Sampler gates failed: {failed}")
    return {"passed": True, "checks": checks, "diagnostics": dict(diagnostics)}


def summarize_draws(
    draws: Sequence[float], *, rope_half_width: float | None = None
) -> Dict[str, Any]:
    """Summarize one posterior estimand under the frozen central interval."""
    values = np.asarray(draws, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Posterior draws must be nonempty and finite")
    lower, median, upper = np.quantile(values, [0.05, 0.5, 0.95])
    summary: Dict[str, Any] = {
        "mean": float(np.mean(values)),
        "median": float(median),
        "lower_90": float(lower),
        "upper_90": float(upper),
        "median_sign": "positive" if median > 0 else "negative" if median < 0 else "zero",
    }
    if rope_half_width is not None:
        summary["rope_half_width"] = float(rope_half_width)
        summary["decision"] = classify_interval(
            lower=float(lower),
            median=float(median),
            upper=float(upper),
            rope_half_width=rope_half_width,
        )
        summary["substantive_interpretation"] = (
            "positive_practical"
            if median > rope_half_width
            else "negative_practical"
            if median < -rope_half_width
            else "within_rope"
        )
    return summary


def linear_contrast_report(
    gamma_draws: Sequence[Sequence[float]],
    contrasts: Sequence[ContrastSpec],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply named contrasts draw by draw, preserving posterior covariance."""
    gate = assert_sampler_gates(diagnostics)
    gamma = np.asarray(gamma_draws, dtype=float)
    if gamma.ndim != 2 or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma_draws must be a finite two-dimensional array")
    rows = []
    for contrast in contrasts:
        weights = np.asarray(contrast.coefficients, dtype=float)
        if gamma.shape[1] != len(weights):
            raise ValueError(
                f"Contrast {contrast.contrast_id} has {len(weights)} weights for "
                f"gamma draws with shape {gamma.shape}"
            )
        rows.append(
            {
                **contrast.to_dict(),
                **summarize_draws(
                    gamma @ weights, rope_half_width=contrast.rope_half_width
                ),
            }
        )
    return {"sampler_gates": gate, "decision_count": len(rows), "rows": rows}


def parameter_report(
    parameter_id: str,
    draws: Sequence[float],
    diagnostics: Mapping[str, Any],
    *,
    rope_half_width: float | None = None,
) -> Dict[str, Any]:
    """Report one scalar parameter after enforcing sampler diagnostics."""
    return {
        "parameter_id": parameter_id,
        "sampler_gates": assert_sampler_gates(diagnostics),
        **summarize_draws(draws, rope_half_width=rope_half_width),
    }


def rq3_descriptive_report(
    sigma_cell_draws: Sequence[float],
    z_alpha_draws: Sequence[Sequence[float]],
    cell_ids: Sequence[str],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Report residual cell heterogeneity without a point-null decision."""
    gate = assert_sampler_gates(diagnostics)
    sigma = np.asarray(sigma_cell_draws, dtype=float).reshape(-1)
    z_alpha = np.asarray(z_alpha_draws, dtype=float)
    if z_alpha.shape != (len(sigma), len(cell_ids)):
        raise ValueError("z_alpha draws must have one column per cell ID")
    residuals = sigma[:, None] * z_alpha
    return {
        "status": "descriptive",
        "sampler_gates": gate,
        "sigma_cell": summarize_draws(sigma),
        "cell_residuals": [
            {"cell_id": cell_id, **summarize_draws(residuals[:, index])}
            for index, cell_id in enumerate(cell_ids)
        ],
    }


def cross_pool_descriptive_report(
    venture_gamma_draws: Sequence[Sequence[float]],
    hiring_gamma_draws: Sequence[Sequence[float]],
    contrasts: Sequence[ContrastSpec],
    venture_diagnostics: Mapping[str, Any],
    hiring_diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe fixed-domain contrast differences without a variance claim."""
    assert_sampler_gates(venture_diagnostics)
    assert_sampler_gates(hiring_diagnostics)
    venture = np.asarray(venture_gamma_draws, dtype=float)
    hiring = np.asarray(hiring_gamma_draws, dtype=float)
    if venture.ndim != 2 or hiring.ndim != 2 or venture.shape[1] != hiring.shape[1]:
        raise ValueError("Cross-pool gamma draws must have the same parameter columns")
    rows = []
    for contrast in contrasts:
        weights = np.asarray(contrast.coefficients, dtype=float)
        venture_values = venture @ weights
        hiring_values = hiring @ weights
        venture_summary = summarize_draws(venture_values)
        hiring_summary = summarize_draws(hiring_values)
        rows.append(
            {
                "contrast_id": contrast.contrast_id,
                "venture": venture_summary,
                "hiring": hiring_summary,
                "median_difference_hiring_minus_venture": (
                    hiring_summary["median"] - venture_summary["median"]
                ),
                "median_sign_agrees": venture_summary["median_sign"]
                == hiring_summary["median_sign"],
            }
        )
    return {
        "status": "descriptive_two_fixed_domains",
        "population_variance_claim": False,
        "sigma_cell_used_as_cross_pool_variance": False,
        "cross_pool_draw_pairing": False,
        "rows": rows,
    }


def compare_presentation_reports(
    primary: Mapping[str, Any],
    presentation_1: Mapping[str, Any],
    presentation_2: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flag prespecified changes across full and presentation-only fits."""
    return _compare_reports(
        {
        "primary": primary,
        "presentation_1_only": presentation_1,
        "presentation_2_only": presentation_2,
        },
        sensitivity="presentation_dependence",
    )


def compare_utility_reports(
    primary: Mapping[str, Any],
    utility_035: Mapping[str, Any],
    utility_065: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flag changes across the frozen full-data utility-scale grid."""
    return _compare_reports(
        {"primary": primary, "utility_035": utility_035, "utility_065": utility_065},
        sensitivity="utility_scale",
    )


def _compare_reports(
    reports: Mapping[str, Mapping[str, Any]], *, sensitivity: str
) -> Dict[str, Any]:
    indexed = {
        name: {row.get("contrast_id", row.get("parameter_id")): row for row in report["rows"]}
        for name, report in reports.items()
    }
    contrast_ids = set(indexed["primary"])
    if any(set(rows) != contrast_ids for rows in indexed.values()):
        raise ValueError("Presentation reports must contain identical estimands")
    rows = []
    for contrast_id in sorted(contrast_ids):
        values = {name: report[contrast_id] for name, report in indexed.items()}
        signs = {row["median_sign"] for row in values.values()}
        decisions = {row["decision"] for row in values.values()}
        interpretations = {
            row["substantive_interpretation"] for row in values.values()
        }
        rows.append(
            {
                "contrast_id": contrast_id,
                "sign_changed": len(signs) > 1,
                "interval_decision_changed": len(decisions) > 1,
                "substantive_interpretation_changed": len(interpretations) > 1,
                "estimates": values,
            }
        )
    return {
        "sensitivity": sensitivity,
        "selection_conditioned_on_outcomes": False,
        "any_disagreement": any(
            row["sign_changed"]
            or row["interval_decision_changed"]
            or row["substantive_interpretation_changed"]
            for row in rows
        ),
        "rows": rows,
    }


def posterior_fit_report(
    *,
    gamma_draws: Sequence[Sequence[float]],
    gamma_size_draws: Sequence[float],
    sigma_cell_draws: Sequence[float],
    z_alpha_draws: Sequence[Sequence[float]],
    contrasts: Sequence[ContrastSpec],
    cell_ids: Sequence[str],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build all report sections supplied by one anchored posterior fit."""
    contrast_report = linear_contrast_report(gamma_draws, contrasts, diagnostics)
    rq6 = parameter_report(
        "rq6_gamma_size",
        gamma_size_draws,
        diagnostics,
        rope_half_width=MENU_SIZE_ROPE,
    )
    decision_rows = list(contrast_report["rows"]) + [rq6]
    return {
        "sampler_gates": contrast_report["sampler_gates"],
        "contrast_decisions": contrast_report,
        "rq3": rq3_descriptive_report(
            sigma_cell_draws, z_alpha_draws, cell_ids, diagnostics
        ),
        "rq6": rq6,
        "rows": decision_rows,
    }


def complete_confirmatory_report(
    *,
    pool_variants: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pool_contrasts: Mapping[str, Sequence[ContrastSpec]],
    matched_variants: Mapping[str, Mapping[str, Any]],
    matched_contrasts: Sequence[ContrastSpec],
) -> Dict[str, Any]:
    """Assemble the frozen primary, matched, and dependence-sensitivity report."""
    required_variants = {
        "primary",
        "presentation_1_only",
        "presentation_2_only",
        "utility_035",
        "utility_065",
    }
    pool_reports = {}
    for pool_id, variants in pool_variants.items():
        if set(variants) != required_variants:
            raise ValueError(f"Pool {pool_id} must supply all three fit variants")
        reports = {
            name: posterior_fit_report(
                contrasts=pool_contrasts[pool_id], **payload
            )
            for name, payload in variants.items()
        }
        reports["presentation_sensitivity"] = compare_presentation_reports(
            reports["primary"],
            reports["presentation_1_only"],
            reports["presentation_2_only"],
        )
        reports["utility_sensitivity"] = compare_utility_reports(
            reports["primary"], reports["utility_035"], reports["utility_065"]
        )
        pool_reports[pool_id] = reports

    if set(pool_reports) != {"venture", "hiring"}:
        raise ValueError("Confirmatory report requires venture and hiring pools")
    venture_primary = pool_variants["venture"]["primary"]
    hiring_primary = pool_variants["hiring"]["primary"]
    rq4 = cross_pool_descriptive_report(
        venture_primary["gamma_draws"],
        hiring_primary["gamma_draws"],
        pool_contrasts["venture"],
        venture_primary["diagnostics"],
        hiring_primary["diagnostics"],
    )

    if set(matched_variants) != required_variants:
        raise ValueError("Matched RQ5 report must supply all three fit variants")
    matched_reports = {
        name: posterior_fit_report(contrasts=matched_contrasts, **payload)
        for name, payload in matched_variants.items()
    }
    matched_reports["presentation_sensitivity"] = compare_presentation_reports(
        matched_reports["primary"],
        matched_reports["presentation_1_only"],
        matched_reports["presentation_2_only"],
    )
    matched_reports["utility_sensitivity"] = compare_utility_reports(
        matched_reports["primary"],
        matched_reports["utility_035"],
        matched_reports["utility_065"],
    )
    return {
        "schema_version": 1,
        "central_interval_mass": CENTRAL_INTERVAL_MASS,
        "pools": pool_reports,
        "rq4": rq4,
        "matched_rq5": matched_reports,
        "multiplicity": {
            "adjustment": "none",
            "full_family_reported": True,
            "selection_from_family": False,
        },
    }


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
        "status": "confirmatory_fit_validated",
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
            "status": "confirmatory_fit_validated",
            "estimand": "within-model hiring-minus-procurement contrast in a dedicated matched-item fit",
            "representation": "assessment_anchored_no_pca",
            "artifact_directory": "matched_rq5",
            "cross_pool_difference_in_differences": "supporting",
        },
        "rq6": {
            "parameter": "gamma_size",
            "decision_rule": "central 90% interval excludes zero and posterior median exceeds the menu-size ROPE",
        },
        "presentation_dependence_sensitivity": {
            "status": "required_post_collection",
            "primary_analysis": "both frozen presentations",
            "sensitivity_analyses": ["presentation_1_only", "presentation_2_only"],
            "utility_middle": 0.5,
            "outcome_conditioning": False,
            "report_disagreement_in": [
                "sign",
                "interval_decision",
                "substantive_interpretation",
            ],
        },
        "formal_sbc": {
            "decision": "not_run",
            "reason": "Exact production-geometry recovery directly calibrated the anchored estimands across 40 datasets for each primary fit geometry.",
            "scope_limitation": "SBC under the fitted independent-observation model would not test misspecification from repeated-menu dependence.",
        },
        "multiplicity": {
            "adjustment": "none",
            "mitigation": ["hierarchical shrinkage", "ROPE"],
            "report_full_family": True,
            "report_decision_count": True,
            "selection_from_family_prohibited": True,
        },
    }
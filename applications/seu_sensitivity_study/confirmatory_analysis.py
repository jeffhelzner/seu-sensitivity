"""Executable confirmatory contrast and decision contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .config import MODELS, PROMPT_CONDITIONS, REFERENCE_MODEL, REFERENCE_PROMPT
from .config import SEUSensitivityStudyConfig, build_cells, get_model_spec

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
_CROSS_POOL_COMBINATION_CAP = 100_000
_CROSS_POOL_SEED = 20260911
ESTIMAND_VERSION = "amendment5_realized_log_sensitivity_v1"


@dataclass(frozen=True)
class ContrastSpec:
    """Legacy gamma weights and explicit realized-cell weights for one name."""

    contrast_id: str
    research_question: str
    label: str
    coefficients: Tuple[float, ...]
    column_names: Tuple[str, ...]
    rope_half_width: float = LOG_ALPHA_ROPE
    expected_direction: str = "two_sided"
    coefficient_estimand: str = "additive_gamma"
    realized_cell_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

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


def _model_cell_weights(pool_id: str, model_weights: Mapping[str, float]) -> Dict[str, float]:
    return {
        cell.cell_id: model_weights[cell.model_name] / len(PROMPT_CONDITIONS)
        for cell in build_cells([pool_id]) if cell.model_name in model_weights
    }


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
                realized_cell_weights={
                    pool: _model_cell_weights(pool, {model.name: 1.0, REFERENCE_MODEL: -1.0})
                    for pool in ("venture", "hiring")
                },
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
                realized_cell_weights={
                    pool: _model_cell_weights(pool, {flagship.name: 1.0, small.name: -1.0})
                    for pool in ("venture", "hiring")
                },
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
                realized_cell_weights={
                    pool: {
                        cell.cell_id: (1.0 if cell.prompt_condition == prompt else -1.0) / len(MODELS)
                        for cell in build_cells([pool])
                        if cell.prompt_condition in (prompt, REFERENCE_PROMPT)
                    }
                    for pool in ("venture", "hiring")
                },
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
    """Require the frozen sampler gates before confirmatory reporting.

    Complete finite diagnostic evidence is validated by the report loader;
    these threshold checks do not replace that artifact-level validation.
    """
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
    if gamma.ndim != 2 or not gamma.size or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma_draws must be a nonempty finite two-dimensional array")
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
                "estimand": "additive_gamma",
                **summarize_draws(
                    gamma @ weights, rope_half_width=contrast.rope_half_width
                ),
            }
        )
    return {"estimand": "additive_gamma", "sampler_gates": gate, "decision_count": len(rows), "rows": rows}


def _realized_fit_values(gamma_draws, sigma_cell_draws, z_alpha_draws, contrasts, cell_ids):
    gamma = np.asarray(gamma_draws, dtype=float)
    sigma = np.asarray(sigma_cell_draws, dtype=float)
    residual = np.asarray(z_alpha_draws, dtype=float)
    if not contrasts:
        raise ValueError("Realized reporting requires contrast specifications")
    columns = tuple(contrasts[0].column_names)
    if (not columns or len(set(columns)) != len(columns)
            or any(tuple(contrast.column_names) != columns for contrast in contrasts)):
        raise ValueError("Contrasts must share unique design columns")
    if gamma.ndim != 2 or not gamma.size or gamma.shape[1] != len(columns) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma_draws must be nonempty, finite and match design columns")
    if sigma.shape != (len(gamma),) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("sigma_cell_draws must be finite, nonnegative and match draw count")
    if not len(cell_ids) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be nonempty and unique")
    if residual.shape != (len(gamma), len(cell_ids)) or not np.all(np.isfinite(residual)):
        raise ValueError("z_alpha_draws must be finite with one column per cell ID and match draw count")
    canonical_cells = {cell.cell_id: cell for cell in build_cells(["venture", "hiring"])}
    if any(cell_id not in canonical_cells for cell_id in cell_ids):
        raise ValueError("Realized reporting requires canonical cell IDs")
    pools = {canonical_cells[cell_id].pool_id for cell_id in cell_ids}
    if "task_hiring" in columns:
        group = "matched_rq5"
        cells = build_cells(["venture", "hiring"])
        design, canonical_columns = matched_rq5_design(cells)
        canonical_ids = [cell.cell_id for cell in cells]
    else:
        if len(pools) != 1:
            raise ValueError("Primary realized reporting requires exactly one pool")
        group = next(iter(pools))
        design, canonical_columns, canonical_ids = SEUSensitivityStudyConfig().design_matrix_for_pool(group)
    if set(columns) != set(canonical_columns):
        raise ValueError("Realized reporting requires canonical design columns")
    indices = [canonical_ids.index(cell_id) for cell_id in cell_ids]
    design = design[indices][:, [list(canonical_columns).index(column) for column in columns]]
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(cell_ids)), design])) != len(columns) + 1:
        raise ValueError("Retained design rank must be unchanged and full with intercept")
    values = gamma @ design.T + sigma[:, None] * residual
    if not np.all(np.isfinite(values)):
        raise ValueError("Reconstructed realized log sensitivities must be finite")
    return values, group


def _cell_contrast_values(values, cell_ids, weights):
    coefficients = np.asarray(list(weights.values()), dtype=float)
    if (coefficients.ndim != 1 or not coefficients.size or not np.all(np.isfinite(coefficients))
            or not np.any(coefficients > 0) or not np.any(coefficients < 0)
            or not np.isclose(coefficients.sum(), 0.0, atol=1e-12, rtol=0)):
        raise ValueError("Realized cell weights must be finite, nonempty and sum to zero")
    missing = sorted(set(weights) - set(cell_ids))
    if missing:
        return None, missing
    indices = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    result = values[:, [indices[cell_id] for cell_id in weights]] @ coefficients
    if not np.all(np.isfinite(result)):
        raise ValueError("Realized contrast draws must be finite")
    return result, []


def _realized_contrast_report(values, cell_ids, contrasts, group, diagnostics):
    rows = []
    for contrast in contrasts:
        if group not in contrast.realized_cell_weights:
            raise ValueError(f"Contrast {contrast.contrast_id} lacks realized cell weights for {group}; gamma fallback is prohibited")
        weights = contrast.realized_cell_weights[group]
        draws, missing = _cell_contrast_values(values, cell_ids, weights)
        row = {
            **contrast.to_dict(),
            "estimand": "realized_log_sensitivity",
            "estimand_version": ESTIMAND_VERSION,
            "cell_weights": dict(weights),
            "included_in_primary_family": True,
            "status": "unavailable" if missing else "available",
            "missing_cell_ids": missing,
        }
        if draws is None:
            row["reason"] = "Required cells are missing; no renormalization, imputation or gamma fallback"
        else:
            row.update(summarize_draws(draws, rope_half_width=contrast.rope_half_width))
        rows.append(row)
    available = sum(row["status"] == "available" for row in rows)
    return {
        "sampler_gates": assert_sampler_gates(diagnostics),
        "decision_count": len(rows),
        "available_decision_count": available,
        "unavailable_decision_count": len(rows) - available,
        "rows": rows,
    }


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
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("RQ3 cell IDs must be unique")
    residuals = sigma[:, None] * z_alpha
    cell_indices = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    canonical_cells = build_cells()
    pool_ids = sorted({
        cell.pool_id for cell in canonical_cells if cell.cell_id in cell_indices
    })
    canonical_ids = {
        (cell.pool_id, cell.model_name, cell.prompt_condition): cell.cell_id
        for cell in canonical_cells
    }
    did_rows = []
    for pool_id in pool_ids:
        for model in MODELS:
            if model.name == REFERENCE_MODEL:
                continue
            for prompt in PROMPT_CONDITIONS:
                if prompt == REFERENCE_PROMPT:
                    continue
                required_ids = [
                    canonical_ids[pool_id, model.name, prompt],
                    canonical_ids[pool_id, model.name, REFERENCE_PROMPT],
                    canonical_ids[pool_id, REFERENCE_MODEL, prompt],
                    canonical_ids[pool_id, REFERENCE_MODEL, REFERENCE_PROMPT],
                ]
                missing_ids = [cell_id for cell_id in required_ids if cell_id not in cell_indices]
                row = {
                    "contrast_id": f"rq3_{pool_id}_{model.slug}_x_{prompt}",
                    "pool_id": pool_id,
                    "model": model.name,
                    "reference_model": REFERENCE_MODEL,
                    "prompt": prompt,
                    "reference_prompt": REFERENCE_PROMPT,
                    "cell_ids": required_ids,
                    "coefficients": [1.0, -1.0, -1.0, 1.0],
                    "status": "unavailable" if missing_ids else "descriptive",
                    "missing_cell_ids": missing_ids,
                }
                if not missing_ids:
                    indices = [cell_indices[cell_id] for cell_id in required_ids]
                    row.update(summarize_draws(residuals[:, indices] @ np.array([1.0, -1.0, -1.0, 1.0])))
                did_rows.append(row)
    return {
        "status": "descriptive",
        "sampler_gates": gate,
        "sigma_cell": summarize_draws(sigma),
        "residual_definition": "ordinary raw sigma_cell * z_alpha; not orthogonally projected",
        "did_definition": "(model,prompt) - (model,reference_prompt) - (reference_model,prompt) + (reference_model,reference_prompt), within each posterior draw",
        "model_by_prompt_dids": did_rows,
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
    *,
    venture_sigma_cell_draws: Sequence[float] | None = None,
    venture_z_alpha_draws: Sequence[Sequence[float]] | None = None,
    venture_cell_ids: Sequence[str] | None = None,
    hiring_sigma_cell_draws: Sequence[float] | None = None,
    hiring_z_alpha_draws: Sequence[Sequence[float]] | None = None,
    hiring_cell_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Describe independent fixed-domain posteriors without new decisions.

    Use the exact empirical Cartesian posterior up to 100,000 combinations;
    otherwise independently resample whole draw indices with replacement using
    a fixed seed. Reuse indices for every estimand to preserve full-vector joint
    dependence within each pool. Summarize one scalar contrast at a time to
    bound working memory independently of the posterior Cartesian product.
    """
    assert_sampler_gates(venture_diagnostics)
    assert_sampler_gates(hiring_diagnostics)
    if any(value is None for value in (
        venture_sigma_cell_draws, venture_z_alpha_draws, venture_cell_ids,
        hiring_sigma_cell_draws, hiring_z_alpha_draws, hiring_cell_ids,
    )):
        raise ValueError("RQ4 requires realized cell information; gamma fallback is prohibited")
    venture, venture_group = _realized_fit_values(
        venture_gamma_draws, venture_sigma_cell_draws, venture_z_alpha_draws, contrasts, venture_cell_ids
    )
    hiring, hiring_group = _realized_fit_values(
        hiring_gamma_draws, hiring_sigma_cell_draws, hiring_z_alpha_draws, contrasts, hiring_cell_ids
    )
    if (venture_group, hiring_group) != ("venture", "hiring"):
        raise ValueError("RQ4 requires venture and hiring primary pool designs")
    combination_count = len(venture) * len(hiring)
    if combination_count <= _CROSS_POOL_COMBINATION_CAP:
        venture_indices = np.repeat(np.arange(len(venture)), len(hiring))
        hiring_indices = np.tile(np.arange(len(hiring)), len(venture))
        method = "exact_cartesian"
    else:
        generator = np.random.default_rng(_CROSS_POOL_SEED)
        venture_indices = generator.integers(len(venture), size=_CROSS_POOL_COMBINATION_CAP)
        hiring_indices = generator.integers(len(hiring), size=_CROSS_POOL_COMBINATION_CAP)
        method = "independent_resampling_with_replacement"

    def describe(venture_values, hiring_values):
        venture_summary = summarize_draws(venture_values)
        hiring_summary = summarize_draws(hiring_values)
        return {
            "status": "descriptive",
            "venture": venture_summary,
            "hiring": hiring_summary,
            "median_difference_hiring_minus_venture": (
                hiring_summary["median"] - venture_summary["median"]
            ),
            "hiring_minus_venture": summarize_draws(
                hiring_values[hiring_indices] - venture_values[venture_indices]
            ),
            "median_sign_agrees": venture_summary["median_sign"] == hiring_summary["median_sign"],
            "posterior_probability_same_sign": float(
                np.mean(venture_values > 0) * np.mean(hiring_values > 0)
                + np.mean(venture_values < 0) * np.mean(hiring_values < 0)
            ),
        }

    def describe_cells(weights_by_pool, *, ordering=False):
        if set(weights_by_pool) != {"venture", "hiring"}:
            raise ValueError("RQ4 contrasts require explicit realized weights for both pools")
        venture_values, venture_missing = _cell_contrast_values(venture, venture_cell_ids, weights_by_pool["venture"])
        hiring_values, hiring_missing = _cell_contrast_values(hiring, hiring_cell_ids, weights_by_pool["hiring"])
        result = {
            "estimand": "realized_log_sensitivity",
            "cell_weights_by_pool": {pool: dict(weights) for pool, weights in weights_by_pool.items()},
            "missing_cell_ids_by_pool": {"venture": venture_missing, "hiring": hiring_missing},
        }
        if venture_values is None or hiring_values is None:
            result.update(status="unavailable", reason="Required cells are missing; no renormalization, imputation or gamma fallback")
            return result
        result.update(describe(venture_values, hiring_values))
        if ordering:
            for pool_id, values in (("venture", venture_values), ("hiring", hiring_values)):
                result[pool_id].update({
                    "probability_first_model_greater": float(np.mean(values > 0)),
                    "probability_second_model_greater": float(np.mean(values < 0)),
                    "probability_tie": float(np.mean(values == 0)),
                })
        return result

    rows = [
        {"contrast_id": contrast.contrast_id, **describe_cells(contrast.realized_cell_weights)}
        for contrast in contrasts
    ]
    model_orderings = []
    for first_model, second_model in combinations(MODELS, 2):
        weights = {
            pool: _model_cell_weights(pool, {first_model.name: 1.0, second_model.name: -1.0})
            for pool in ("venture", "hiring")
        }
        model_orderings.append({
            "first_model": first_model.name,
            "second_model": second_model.name,
            "contrast_id": f"rq4_{first_model.slug}_minus_{second_model.slug}",
            **describe_cells(weights, ordering=True),
        })
    return {
        "estimand_version": ESTIMAND_VERSION,
        "estimand": "equally_weighted_realized_log_sensitivity_contrasts",
        "decision_count": 0,
        "status": "descriptive_two_fixed_domains",
        "population_variance_claim": False,
        "sigma_cell_used_as_cross_pool_variance": False,
        "cross_pool_draw_pairing": False,
        "independent_posterior_combination": {
            "method": method,
            "combination_count": len(venture_indices),
            "combination_cap": _CROSS_POOL_COMBINATION_CAP,
            "seed": _CROSS_POOL_SEED if method != "exact_cartesian" else None,
            "shared_whole_draw_indices_across_estimands": True,
            "policy": "Exact Cartesian product up to cap; otherwise independent whole-draw resampling with replacement. No aligned-chain or matched-row pairing across pools.",
        },
        "same_sign_policy": "Exact product of independent marginal positive/negative probabilities; zeros excluded.",
        "model_orderings": model_orderings,
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
        unavailable = [name for name, row in values.items() if row.get("status") == "unavailable"]
        if unavailable:
            rows.append({
                "contrast_id": contrast_id,
                "status": "unavailable",
                "reason": "Required contrast unavailable in one or more fit variants",
                "unavailable_variants": unavailable,
                "sign_changed": None,
                "interval_decision_changed": None,
                "substantive_interpretation_changed": None,
                "estimates": values,
            })
            continue
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
        "all_estimands_comparable": all(row.get("status") != "unavailable" for row in rows),
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
    values, group = _realized_fit_values(
        gamma_draws, sigma_cell_draws, z_alpha_draws, contrasts, cell_ids
    )
    if np.asarray(gamma_size_draws).shape != (len(values),):
        raise ValueError("gamma_size_draws must match draw count")
    contrast_report = _realized_contrast_report(values, cell_ids, contrasts, group, diagnostics)
    gamma = np.asarray(gamma_draws, dtype=float)
    companion_rows = []
    for contrast in contrasts:
        coefficients = np.asarray(contrast.coefficients, dtype=float)
        if coefficients.shape != (gamma.shape[1],) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Gamma companion weights must be finite and match design columns")
        companion_rows.append({
            **contrast.to_dict(),
            "estimand": "additive_gamma",
            "status": "descriptive",
            "included_in_primary_family": False,
            **summarize_draws(gamma @ coefficients),
        })
    rq6 = parameter_report(
        "rq6_gamma_size",
        gamma_size_draws,
        diagnostics,
        rope_half_width=MENU_SIZE_ROPE,
    )
    decision_rows = list(contrast_report["rows"]) + [rq6]
    return {
        "estimand_version": ESTIMAND_VERSION,
        "sampler_gates": contrast_report["sampler_gates"],
        "decision_count": len(decision_rows),
        "contrast_decisions": contrast_report,
        "gamma_companion": {
            "status": "descriptive",
            "included_in_primary_family": False,
            "decision_count": 0,
            "rows": companion_rows,
        },
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
            raise ValueError(f"Pool {pool_id} must supply all five fit variants")
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
        venture_sigma_cell_draws=venture_primary["sigma_cell_draws"],
        venture_z_alpha_draws=venture_primary["z_alpha_draws"],
        venture_cell_ids=venture_primary["cell_ids"],
        hiring_sigma_cell_draws=hiring_primary["sigma_cell_draws"],
        hiring_z_alpha_draws=hiring_primary["z_alpha_draws"],
        hiring_cell_ids=hiring_primary["cell_ids"],
    )

    if set(matched_variants) != required_variants:
        raise ValueError("Matched RQ5 report must supply all five fit variants")
    matched_reports = {
        name: posterior_fit_report(contrasts=matched_contrasts, **payload)
        for name, payload in matched_variants.items()
    }
    for report in matched_reports.values():
        report["rq6"]["status"] = "matched_menu_size_sensitivity"
        report["rq6"]["included_in_primary_family"] = False
        report["primary_decision_count"] = report["contrast_decisions"]["decision_count"]
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
    pool_decision_counts = {
        pool_id: reports["primary"]["decision_count"]
        for pool_id, reports in pool_reports.items()
    }
    matched_decision_count = matched_reports["primary"]["primary_decision_count"]
    unavailable_decisions = sum(
        reports["primary"]["contrast_decisions"]["unavailable_decision_count"]
        for reports in pool_reports.values()
    ) + matched_reports["primary"]["contrast_decisions"]["unavailable_decision_count"]
    primary_decision_count = sum(pool_decision_counts.values()) + matched_decision_count
    return {
        "schema_version": 2,
        "estimand_contract": _estimand_contract(),
        "central_interval_mass": CENTRAL_INTERVAL_MASS,
        "pools": pool_reports,
        "rq4": rq4,
        "matched_rq5": matched_reports,
        "multiplicity": {
            "adjustment": "none",
            "full_family_reported": True,
            "selection_from_family": False,
            "primary_decisions_by_pool": pool_decision_counts,
            "matched_rq5_primary_decisions": matched_decision_count,
            "primary_decision_count": primary_decision_count,
            "available_primary_decision_count": primary_decision_count - unavailable_decisions,
            "unavailable_primary_decision_count": unavailable_decisions,
            "decision_count_policy": "Counts retain all named primary decisions, including explicitly unavailable contrasts",
            "expected_primary_decision_count": 26,
            "distinct_primary_decisions_up_to_sign": 24,
            "gamma_companion_primary_decision_count": 0,
            "family_definition": "Nine RQ1/RQ2 contrasts plus one RQ6 slope per pool; six matched RQ5 contrasts. Matched menu-size slope is sensitivity only, not an additional RQ6 primary family. RQ3 and RQ4 are descriptive.",
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
                realized_cell_weights={
                    "matched_rq5": {
                        cell.cell_id: (1.0 if cell.pool_id == "hiring" else -1.0) / len(PROMPT_CONDITIONS)
                        for cell in build_cells(["venture", "hiring"])
                        if cell.model_name == model.name
                    }
                },
            ).to_dict()
        )
    return {
        "estimand_contract": _estimand_contract(),
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


def _estimand_contract() -> Dict[str, Any]:
    return {
        "version": ESTIMAND_VERSION,
        "amendment": 5,
        "primary_estimand": "equally_weighted_realized_log_sensitivity_contrasts",
        "reconstruction": "X @ gamma + sigma_cell * z_alpha; shared gamma0 cancels",
        "size_reference": "common menu size across all cells in each contrast; shared gamma_size term cancels",
        "observation_count_weighting": False,
        "rq1_weights": "mean of three prompt cells per model minus comparator mean",
        "rq2_weights": "mean over six models of prompt minus neutral",
        "rq5_weights": "within-model mean of three hiring prompts minus mean of three procurement prompts in matched fit",
        "rq4_estimand": "same realized model averages as RQ1; descriptive independent fixed-domain comparison",
        "gamma_companion": {
            "estimand": "additive_gamma",
            "coefficient_fields": ["coefficients", "column_names"],
            "status": "descriptive",
            "included_in_primary_family": False,
            "decision_count": 0,
        },
        "missing_cell_policy": "affected contrasts unavailable with explicit missing cell IDs; no renormalization, imputation or gamma fallback",
        "global_rank_check": "retained design must remain full rank with intercept",
        "primary_decision_count": 26,
        "distinct_primary_decisions_up_to_sign": 24,
        "fit_count": 15,
    }


def contract_manifest(
    column_names: Sequence[str], design_matrix: Sequence[Sequence[float]]
) -> Dict[str, Any]:
    """Return the complete machine-readable policy frozen before collection."""
    contrasts = primary_contrasts(column_names)
    return {
        "estimand_contract": _estimand_contract(),
        "central_interval_mass": CENTRAL_INTERVAL_MASS,
        "interval_quantiles": [0.05, 0.95],
        "log_alpha_rope_half_width": LOG_ALPHA_ROPE,
        "menu_size_rope_half_width": MENU_SIZE_ROPE,
        "bulk_ess_minimum": BULK_ESS_MINIMUM,
        "tail_ess_minimum": TAIL_ESS_MINIMUM,
        "primary_contrasts": [contrast.to_dict() for contrast in contrasts],
        "rq1_rq2_decisions_per_pool": len(contrasts),
        "primary_decisions_per_pool": len(contrasts) + 1,
        "matched_rq5_primary_decisions": len(MODELS),
        "primary_decision_count": 2 * (len(contrasts) + 1) + len(MODELS),
        "rq3": {
            "status": "secondary_descriptive_existing_fit",
            "estimand": "sigma_cell, ordinary raw sigma_cell * z_alpha cell residuals, and draw-wise model-by-prompt difference-in-differences",
            "residuals_orthogonally_projected": False,
            "separate_saturated_fit": False,
            "reason": "The full interaction exactly spans the additive residual cell space and adds no likelihood information.",
            "aliasing": interaction_aliasing_report(design_matrix, column_names),
        },
        "rq4": {"status": "descriptive", "estimand": "equally_weighted_realized_model_log_sensitivity_contrasts"},
        "rq5": {
            "status": "confirmatory_fit_validated",
            "estimand": "within-model equally weighted realized log-sensitivity hiring-minus-procurement contrast in a dedicated matched-item fit",
            "representation": "assessment_anchored_no_pca",
            "artifact_directory": "matched_rq5",
            "cross_pool_difference_in_differences": "supporting",
        },
        "rq6": {
            "parameter": "gamma_size",
            "decision_rule": "central 90% interval excludes zero and posterior median exceeds the menu-size ROPE",
            "primary_scope": "one slope per primary pool; matched slope is sensitivity only, not an extra RQ6 primary family",
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
            "reason": "Historical production-geometry recovery evaluated the anchored model and additive estimands across 40 datasets per fit geometry; it does not validate all Amendment 5 realized-cell contrasts.",
            "amended_contrast_validation": "pending saved-draw verification, including realized-cell RQ5; no new coverage or power claim",
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
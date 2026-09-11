"""Tests for the frozen confirmatory analysis contract."""

import math

import numpy as np
import pytest

from applications.seu_sensitivity_study import confirmatory_analysis as ca
from applications.seu_sensitivity_study import config


@pytest.fixture
def design():
    matrix, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool(
        "venture"
    )
    return matrix, columns


@pytest.fixture
def design_columns(design):
    return design[1]


def test_primary_family_has_seven_rq1_and_two_rq2_contrasts(design_columns):
    contrasts = ca.primary_contrasts(design_columns)
    assert sum(c.research_question == "RQ1" for c in contrasts) == 7
    assert sum(c.research_question == "RQ2" for c in contrasts) == 2
    assert len({contrast.contrast_id for contrast in contrasts}) == 9


def test_openai_flagship_minus_small_uses_reference_correctly(design_columns):
    contrast = next(
        c for c in ca.primary_contrasts(design_columns)
        if c.contrast_id == "rq1_openai_flagship_minus_small"
    )
    weights = dict(zip(contrast.column_names, contrast.coefficients))
    assert weights["model_gpt_4o_mini"] == -1.0
    assert sum(abs(value) for value in weights.values()) == 1.0


def test_anthropic_flagship_minus_small_is_derived_difference(design_columns):
    contrast = next(
        c for c in ca.primary_contrasts(design_columns)
        if c.contrast_id == "rq1_anthropic_flagship_minus_small"
    )
    weights = dict(zip(contrast.column_names, contrast.coefficients))
    assert weights["model_claude_sonnet_4_5"] == 1.0
    assert weights["model_claude_haiku_4_5"] == -1.0


def test_decision_requires_interval_exclusion_and_magnitude():
    rope = math.log(1.25)
    assert ca.classify_interval(
        lower=0.10, median=0.30, upper=0.50, rope_half_width=rope
    ) == "detected_positive"
    assert ca.classify_interval(
        lower=0.10, median=0.20, upper=0.30, rope_half_width=rope
    ) == "not_detected"
    assert ca.classify_interval(
        lower=-0.10, median=0.30, upper=0.50, rope_half_width=rope
    ) == "not_detected"


def test_interaction_exactly_spans_the_residual_cell_space(design):
    report = ca.interaction_aliasing_report(*design)
    assert report == {
        "n_cells": 18,
        "additive_rank": 8,
        "interaction_columns": 10,
        "interaction_rank": 10,
        "residual_cell_dimension": 10,
        "residual_interaction_rank": 10,
        "full_rank": 18,
        "exactly_aliased_with_cell_residuals": True,
    }


def test_matched_rq5_design_identifies_six_within_model_task_contrasts():
    cells = config.build_cells(["venture", "hiring"])
    design_matrix, columns = ca.matched_rq5_design(cells)
    contract = ca.matched_rq5_contract(cells)

    assert design_matrix.shape == (36, 13)
    assert np.linalg.matrix_rank(
        np.column_stack([np.ones(len(design_matrix)), design_matrix])
    ) == 14
    assert contract["decision_count"] == 6
    assert contract["representation"] == "assessment_anchored_no_pca"

    for contrast in contract["contrasts"]:
        weights = dict(zip(contrast["column_names"], contrast["coefficients"]))
        model_slug = contrast["contrast_id"].removeprefix("rq5_").removesuffix(
            "_hiring_minus_procurement"
        )
        model = next(model for model in config.MODELS if model.slug == model_slug)
        assert weights["task_hiring"] == 1.0
        interaction = f"task_hiring_x_model_{model.slug}"
        if model.name == config.REFERENCE_MODEL:
            assert interaction not in weights
            assert sum(abs(value) for value in weights.values()) == 1.0
        else:
            assert weights[interaction] == 1.0
            assert sum(abs(value) for value in weights.values()) == 2.0


def test_manifest_freezes_approved_rules_and_pending_estimands(design):
    design_matrix, design_columns = design
    manifest = ca.contract_manifest(design_columns, design_matrix)
    assert manifest["interval_quantiles"] == [0.05, 0.95]
    assert manifest["bulk_ess_minimum"] == 400
    assert manifest["tail_ess_minimum"] == 400
    assert manifest["rq1_rq2_decisions_per_pool"] == 9
    assert manifest["primary_decisions_per_pool"] == 10
    assert manifest["matched_rq5_primary_decisions"] == 6
    assert manifest["primary_decision_count"] == 26
    assert "sensitivity only" in manifest["rq6"]["primary_scope"]
    assert manifest["rq3"]["residuals_orthogonally_projected"] is False
    assert manifest["multiplicity"]["adjustment"] == "none"
    assert manifest["multiplicity"]["report_full_family"] is True
    assert manifest["rq3"]["status"] == "secondary_descriptive_existing_fit"
    assert manifest["rq3"]["separate_saturated_fit"] is False
    assert manifest["rq4"]["status"] == "descriptive"
    assert manifest["rq5"]["status"] == "confirmatory_fit_validated"
    assert manifest["presentation_dependence_sensitivity"] == {
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
    }
    assert manifest["formal_sbc"]["decision"] == "not_run"
    assert "repeated-menu dependence" in manifest["formal_sbc"]["scope_limitation"]
    assert manifest["rq5"]["representation"] == "assessment_anchored_no_pca"
import math
from dataclasses import replace

import numpy as np
import pytest

from applications.seu_sensitivity_study import confirmatory_analysis as ca
from applications.seu_sensitivity_study import config


DIAGNOSTICS = {
    "max_rhat": 1.005,
    "min_ess_bulk": 700,
    "min_ess_tail": 650,
    "min_ebfmi": 0.8,
    "divergences": 0,
    "treedepth_saturated_share": 0.0,
}


def _contrast(contrast_id="effect"):
    return ca.ContrastSpec(
        contrast_id=contrast_id,
        research_question="RQ1",
        label="Synthetic effect",
        coefficients=(1.0, -1.0),
        column_names=("a", "b"),
    )


def _rq4_report(venture, hiring, contrasts, venture_diagnostics, hiring_diagnostics):
    _, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    if tuple(contrasts[0].column_names) == ("a", "b"):
        venture = np.pad(venture, ((0, 0), (0, len(columns) - 2)))
        hiring = np.pad(hiring, ((0, 0), (0, len(columns) - 2)))
        contrasts = [replace(
            contrast,
            column_names=tuple(columns),
            coefficients=contrast.coefficients + (0.0,) * (len(columns) - 2),
            realized_cell_weights={
                pool: {
                    cell.cell_id: weight / 3
                    for cell in config.build_cells([pool])
                    for model, weight in (
                        (config.MODELS[1].name, contrast.coefficients[0]),
                        (config.MODELS[2].name, contrast.coefficients[1]),
                        (config.REFERENCE_MODEL, -sum(contrast.coefficients)),
                    ) if cell.model_name == model and weight != 0
                } for pool in ("venture", "hiring")
            },
        ) for contrast in contrasts]
    return ca.cross_pool_descriptive_report(
        venture.tolist(), hiring.tolist(), contrasts, venture_diagnostics, hiring_diagnostics,
        venture_sigma_cell_draws=np.ones(len(venture)).tolist(),
        venture_z_alpha_draws=np.zeros((len(venture), 18)).tolist(),
        venture_cell_ids=[cell.cell_id for cell in config.build_cells(["venture"])],
        hiring_sigma_cell_draws=np.ones(len(hiring)).tolist(),
        hiring_z_alpha_draws=np.zeros((len(hiring), 18)).tolist(),
        hiring_cell_ids=[cell.cell_id for cell in config.build_cells(["hiring"])],
    )


def test_linear_contrasts_preserve_draw_covariance_and_apply_decision_rule():
    base = np.linspace(0.3, 0.5, 1000)
    gamma = np.column_stack([base, base - 0.3])
    report = ca.linear_contrast_report(gamma, [_contrast()], DIAGNOSTICS)

    row = report["rows"][0]
    assert row["median"] == pytest.approx(0.3)
    assert row["lower_90"] == pytest.approx(0.3)
    assert row["decision"] == "detected_positive"


def test_primary_realized_contrast_includes_residual_model_average():
    _, columns, cell_ids = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    cells = config.build_cells(["venture"])
    residuals = np.array([
        0.6 if cell.model_name == config.MODELS[1].name else 0.0
        for cell in cells
    ])
    report = ca.posterior_fit_report(
        gamma_draws=np.zeros((20, len(columns))),
        gamma_size_draws=np.zeros(20),
        sigma_cell_draws=np.ones(20),
        z_alpha_draws=np.tile(residuals, (20, 1)),
        contrasts=ca.primary_contrasts(columns),
        cell_ids=cell_ids,
        diagnostics=DIAGNOSTICS,
    )
    row = report["contrast_decisions"]["rows"][0]
    assert row["median"] == pytest.approx(0.6)
    assert row["decision"] == "detected_positive"
    companion = report["gamma_companion"]
    assert companion["decision_count"] == 0
    assert companion["rows"][0]["median"] == 0.0
    assert "decision" not in companion["rows"][0]
    assert report["decision_count"] == 10


def test_reporting_fails_closed_on_tail_ess():
    diagnostics = {**DIAGNOSTICS, "min_ess_tail": 399}
    with pytest.raises(ValueError, match="tail_ess"):
        ca.linear_contrast_report(np.ones((10, 2)), [_contrast()], diagnostics)


def test_reporting_fails_closed_on_missing_sampler_evidence():
    diagnostics = dict(DIAGNOSTICS)
    del diagnostics["min_ess_tail"]
    with pytest.raises(ValueError, match="missing required fields"):
        ca.linear_contrast_report(np.ones((10, 2)), [_contrast()], diagnostics)


def test_rq3_reports_sigma_and_residuals_without_a_null_decision():
    sigma = np.full(100, 0.2)
    z_alpha = np.column_stack([np.ones(100), -np.ones(100)])
    report = ca.rq3_descriptive_report(
        sigma, z_alpha, ["cell-a", "cell-b"], DIAGNOSTICS
    )

    assert report["status"] == "descriptive"
    assert "decision" not in report["sigma_cell"]
    assert report["cell_residuals"][0]["median"] == pytest.approx(0.2)
    assert report["model_by_prompt_dids"] == []


@pytest.mark.parametrize("interaction", [0.0, 0.7, -0.7])
def test_rq3_dids_cancel_drawwise_additive_effects_and_preserve_sign(interaction):
    cells = config.build_cells(["venture"])
    target_model = config.MODELS[1].name
    target_prompt = config.PROMPT_CONDITIONS[1]
    shared = np.linspace(-2.0, 3.0, 101)
    residuals = np.column_stack([
        shared + 2 * shared * (cell.model_name == target_model)
        - shared * (cell.prompt_condition == target_prompt)
        + interaction * (cell.model_name == target_model and cell.prompt_condition == target_prompt)
        for cell in cells
    ])
    report = ca.rq3_descriptive_report(
        np.ones(101), residuals, [cell.cell_id for cell in cells], DIAGNOSTICS
    )
    assert len(report["model_by_prompt_dids"]) == 10
    row = next(row for row in report["model_by_prompt_dids"]
               if row["model"] == target_model and row["prompt"] == target_prompt)
    for key in ("lower_90", "median", "upper_90"):
        assert row[key] == pytest.approx(interaction, abs=1e-14)
    assert "decision" not in row
    assert report["cell_residuals"][0]["median"] == pytest.approx(0.5)
    assert "not orthogonally projected" in report["residual_definition"]


def test_rq3_missing_canonical_cell_marks_only_affected_dids_unavailable():
    cells = config.build_cells(["venture"])
    removed = cells.pop(4)
    report = ca.rq3_descriptive_report(
        np.ones(20), np.zeros((20, len(cells))),
        [cell.cell_id for cell in cells], DIAGNOSTICS,
    )
    unavailable = [row for row in report["model_by_prompt_dids"] if row["status"] == "unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0]["missing_cell_ids"] == [removed.cell_id]
    assert "median" not in unavailable[0]
    assert len(report["model_by_prompt_dids"]) == 10


def test_rq4_compares_fixed_domain_effects_without_cross_pool_variance_claim():
    venture = np.column_stack([np.full(100, 0.4), np.zeros(100)])
    hiring = np.column_stack([np.full(80, -0.4), np.zeros(80)])
    report = _rq4_report(
        venture, hiring, [_contrast()], DIAGNOSTICS, DIAGNOSTICS
    )

    assert report["population_variance_claim"] is False
    assert report["sigma_cell_used_as_cross_pool_variance"] is False
    assert report["rows"][0]["median_sign_agrees"] is False
    assert report["cross_pool_draw_pairing"] is False
    assert report["rows"][0][
        "median_difference_hiring_minus_venture"
    ] == pytest.approx(-0.8)


def test_menu_size_report_uses_frozen_rope():
    report = ca.parameter_report(
        "rq6_gamma_size",
        np.full(100, math.log(1.06)),
        DIAGNOSTICS,
        rope_half_width=ca.MENU_SIZE_ROPE,
    )
    assert report["decision"] == "detected_positive"


def test_rq4_uncertainty_is_independent_cartesian_not_aligned_draws():
    venture = np.column_stack([[-2.0, 0.0, 3.0], np.zeros(3)])
    hiring = np.column_stack([[-1.0, 4.0], np.zeros(2)])
    report = _rq4_report(
        venture, hiring, [_contrast()], DIAGNOSTICS, DIAGNOSTICS
    )
    expected = ca.summarize_draws((hiring[:, 0, None] - venture[:, 0]).ravel())
    row = report["rows"][0]
    assert row["hiring_minus_venture"] == expected
    assert row["posterior_probability_same_sign"] == pytest.approx(1 / 3)
    assert report["independent_posterior_combination"]["method"] == "exact_cartesian"
    assert len(report["model_orderings"]) == 15
    assert "decision" not in row["hiring_minus_venture"]


def test_rq4_all_model_orderings_use_joint_gamma_posterior():
    _, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    gamma = np.zeros((4, len(columns)))
    first_column = columns.index(f"model_{config.MODELS[1].slug}")
    second_column = columns.index(f"model_{config.MODELS[2].slug}")
    gamma[:, first_column] = [-2, -1, 1, 2]
    gamma[:, second_column] = gamma[:, first_column] - 0.5
    report = _rq4_report(
        gamma, gamma, ca.primary_contrasts(columns), DIAGNOSTICS, DIAGNOSTICS
    )
    assert len(report["model_orderings"]) == 15
    assert len({row["contrast_id"] for row in report["model_orderings"]}) == 15
    row = next(row for row in report["model_orderings"]
               if row["first_model"] == config.MODELS[1].name
               and row["second_model"] == config.MODELS[2].name)
    assert row["venture"]["lower_90"] == pytest.approx(0.5)
    assert row["venture"]["upper_90"] == pytest.approx(0.5)
    assert row["venture"]["probability_first_model_greater"] == 1.0
    assert row["posterior_probability_same_sign"] == 1.0
    reference_row = report["model_orderings"][0]
    assert reference_row["venture"]["probability_first_model_greater"] == 0.5
    assert reference_row["posterior_probability_same_sign"] == 0.5
    assert report["sigma_cell_used_as_cross_pool_variance"] is False


def test_rq4_large_posterior_resampling_is_bounded_repeatable_and_independent(monkeypatch):
    monkeypatch.setattr(ca, "_CROSS_POOL_COMBINATION_CAP", 2000)
    values = np.linspace(-1, 1, 100)
    gamma = np.column_stack([values, values - 0.25])
    contrasts = [_contrast(), ca.ContrastSpec("first", "RQ1", "First", (1.0, 0.0), ("a", "b"))]
    report = _rq4_report(gamma, gamma, contrasts, DIAGNOSTICS, DIAGNOSTICS)
    repeated = _rq4_report(gamma, gamma, contrasts, DIAGNOSTICS, DIAGNOSTICS)
    assert report == repeated
    policy = report["independent_posterior_combination"]
    assert policy["method"] == "independent_resampling_with_replacement"
    assert policy["combination_count"] == 2000
    assert policy["shared_whole_draw_indices_across_estimands"] is True
    assert report["rows"][0]["hiring_minus_venture"]["upper_90"] == pytest.approx(0.0)
    difference = report["rows"][1]["hiring_minus_venture"]
    assert difference["lower_90"] < -1.0
    assert difference["upper_90"] > 1.0


def test_presentation_report_flags_sign_decision_and_interpretation_changes():
    def report(value):
        row = {
            "contrast_id": "effect",
            **ca.summarize_draws(
                np.full(100, value), rope_half_width=ca.LOG_ALPHA_ROPE
            ),
        }
        return {"rows": [row]}

    comparison = ca.compare_presentation_reports(
        report(0.4), report(0.35), report(-0.4)
    )
    row = comparison["rows"][0]
    assert comparison["any_disagreement"] is True
    assert row["sign_changed"] is True
    assert row["interval_decision_changed"] is True
    assert row["substantive_interpretation_changed"] is True


def test_complete_report_requires_and_emits_all_frozen_sections():
    _, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    contrasts = ca.primary_contrasts(columns)
    matched_cells = config.build_cells(["venture", "hiring"])
    matched = ca.matched_rq5_contract(matched_cells)
    matched_contrasts = [ca.ContrastSpec(**row) for row in matched["contrasts"]]

    def payload(value, selected_cells, column_count):
        gamma = np.full((100, column_count), value)
        return {
            "gamma_draws": gamma,
            "gamma_size_draws": np.full(100, math.log(1.06)),
            "sigma_cell_draws": np.full(100, 0.2),
            "z_alpha_draws": np.zeros((100, len(selected_cells))),
            "cell_ids": [cell.cell_id for cell in selected_cells],
            "diagnostics": DIAGNOSTICS,
        }

    def variants(selected_cells, column_count):
        return {
            name: payload(value, selected_cells, column_count)
            for name, value in (
                ("primary", 0.4), ("presentation_1_only", 0.35),
                ("presentation_2_only", -0.4), ("utility_035", 0.3), ("utility_065", 0.5),
            )
        }

    report = ca.complete_confirmatory_report(
        pool_variants={pool: variants(config.build_cells([pool]), len(columns)) for pool in ("venture", "hiring")},
        pool_contrasts={"venture": contrasts, "hiring": contrasts},
        matched_variants=variants(matched_cells, len(matched["design_columns"])),
        matched_contrasts=matched_contrasts,
    )

    assert set(report["pools"]) == {"venture", "hiring"}
    assert report["rq4"]["status"] == "descriptive_two_fixed_domains"
    assert report["matched_rq5"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 6
    assert report["matched_rq5"]["presentation_sensitivity"][
        "any_disagreement"
    ] is True
    assert report["matched_rq5"]["utility_sensitivity"]["sensitivity"] == (
        "utility_scale"
    )
    assert report["multiplicity"]["full_family_reported"] is True
    assert report["multiplicity"]["primary_decision_count"] == 26
    assert report["matched_rq5"]["primary"]["rq6"]["included_in_primary_family"] is False


def test_complete_canonical_family_counts_primary_decisions_not_sensitivities():
    _, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    contrasts = ca.primary_contrasts(columns)
    cells = config.build_cells(["venture", "hiring"])
    matched_contract = ca.matched_rq5_contract(cells)
    matched_contrasts = [ca.ContrastSpec(**row) for row in matched_contract["contrasts"]]

    def variants(column_count, selected_cells):
        payload = {
            "gamma_draws": np.zeros((10, column_count)),
            "gamma_size_draws": np.zeros(10),
            "sigma_cell_draws": np.ones(10),
            "z_alpha_draws": np.zeros((10, len(selected_cells))),
            "cell_ids": [cell.cell_id for cell in selected_cells],
            "diagnostics": DIAGNOSTICS,
        }
        return {name: payload for name in (
            "primary", "presentation_1_only", "presentation_2_only", "utility_035", "utility_065"
        )}

    report = ca.complete_confirmatory_report(
        pool_variants={pool_id: variants(len(columns), config.build_cells([pool_id]))
                       for pool_id in ("venture", "hiring")},
        pool_contrasts={pool_id: contrasts for pool_id in ("venture", "hiring")},
        matched_variants=variants(len(matched_contract["design_columns"]), cells),
        matched_contrasts=matched_contrasts,
    )
    counts = report["multiplicity"]
    assert counts["primary_decisions_by_pool"] == {"venture": 10, "hiring": 10}
    assert counts["matched_rq5_primary_decisions"] == 6
    assert counts["primary_decision_count"] == counts["expected_primary_decision_count"] == 26
    assert len(report["rq4"]["model_orderings"]) == 15
    assert len(report["matched_rq5"]["primary"]["rq3"]["model_by_prompt_dids"]) == 20


@pytest.mark.parametrize("diagnostic", [
    "max_rhat", "min_ess_bulk", "min_ess_tail", "min_ebfmi", "treedepth_saturated_share",
])
def test_sampler_thresholds_already_reject_nan(diagnostic):
    with pytest.raises(ValueError, match="Sampler gates failed"):
        ca.assert_sampler_gates({**DIAGNOSTICS, diagnostic: float("nan")})
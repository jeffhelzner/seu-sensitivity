import math

import numpy as np
import pytest

from applications.seu_sensitivity_study import confirmatory_analysis as ca


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


def test_linear_contrasts_preserve_draw_covariance_and_apply_decision_rule():
    base = np.linspace(0.3, 0.5, 1000)
    gamma = np.column_stack([base, base - 0.3])
    report = ca.linear_contrast_report(gamma, [_contrast()], DIAGNOSTICS)

    row = report["rows"][0]
    assert row["median"] == pytest.approx(0.3)
    assert row["lower_90"] == pytest.approx(0.3)
    assert row["decision"] == "detected_positive"


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


def test_rq4_compares_fixed_domain_effects_without_using_sigma_cell():
    venture = np.column_stack([np.full(100, 0.4), np.zeros(100)])
    hiring = np.column_stack([np.full(80, -0.4), np.zeros(80)])
    report = ca.cross_pool_descriptive_report(
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
    contrast = _contrast()

    def payload(value, n_cells=2):
        gamma = np.column_stack([np.full(100, value), np.zeros(100)])
        return {
            "gamma_draws": gamma,
            "gamma_size_draws": np.full(100, math.log(1.06)),
            "sigma_cell_draws": np.full(100, 0.2),
            "z_alpha_draws": np.zeros((100, n_cells)),
            "cell_ids": [f"cell-{index}" for index in range(n_cells)],
            "diagnostics": DIAGNOSTICS,
        }

    variants = {
        "primary": payload(0.4),
        "presentation_1_only": payload(0.35),
        "presentation_2_only": payload(-0.4),
        "utility_035": payload(0.3),
        "utility_065": payload(0.5),
    }
    report = ca.complete_confirmatory_report(
        pool_variants={"venture": variants, "hiring": variants},
        pool_contrasts={"venture": [contrast], "hiring": [contrast]},
        matched_variants=variants,
        matched_contrasts=[contrast],
    )

    assert set(report["pools"]) == {"venture", "hiring"}
    assert report["rq4"]["status"] == "descriptive_two_fixed_domains"
    assert report["matched_rq5"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 1
    assert report["matched_rq5"]["presentation_sensitivity"][
        "any_disagreement"
    ] is True
    assert report["matched_rq5"]["utility_sensitivity"]["sensitivity"] == (
        "utility_scale"
    )
    assert report["multiplicity"]["full_family_reported"] is True
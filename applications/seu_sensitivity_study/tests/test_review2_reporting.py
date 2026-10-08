import copy

import pytest

from applications.seu_sensitivity_study.confirmatory_reporting import _attach_contrast_dependence_text


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
@pytest.mark.parametrize("variant", ["primary", "presentation_1_only", "presentation_2_only", "utility_035", "utility_065"])
def test_dependence_text_preserves_results_and_references_existing_diagnostics(group, variant):
    rows = [{"research_question": question, "cell_weights": {"cell_b": -0.5, "cell_a": 0.5},
             "median": 0.3, "lower_90": 0.1, "upper_90": 0.5, "decision": True}
            for question in ("RQ1", "RQ2", "RQ5")]
    rows.append({"research_question": "RQ6", "median": 0.2})
    report = {"contrast_decisions": {"rows": rows}, "rows": rows, "decision_count": 4}
    original = copy.deepcopy(report)
    _attach_contrast_dependence_text(report, group=group, variant=variant)
    for row in rows[:3]:
        qualification = row.pop("dependence_qualification")
        assert qualification["contributing_cell_ids"] == ["cell_a", "cell_b"]
        assert qualification["paired_diagnostic_path"] == ["posterior_predictive_checks", group, variant, "a4"]
        assert qualification["paired_diagnostic_section_when_available"] == "pairs"
        assert qualification["presentation_comparison_path"] == (
            (["matched_rq5"] if group == "matched_rq5" else ["pools", group]) + ["presentation_sensitivity"])
        assert qualification["primary_decision_unchanged"] is True
        assert "Unavailable paired diagnostics" in qualification["text"]
        assert "applies" not in qualification
    assert report == original


def test_missing_contrast_remains_unavailable_with_conditional_text():
    row = {"research_question": "RQ1", "cell_weights": {"missing": 1, "retained": -1},
           "status": "unavailable", "missing_cell_ids": ["missing"]}
    original = copy.deepcopy(row)
    _attach_contrast_dependence_text({"contrast_decisions": {"rows": [row]}}, group="venture", variant="primary")
    assert row.pop("dependence_qualification")["contributing_cell_ids"] == ["missing", "retained"]
    assert row == original
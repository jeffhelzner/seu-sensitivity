import json
import math

import pytest

from analysis import study_results_writeup as writeup


def row(contrast_id, median, lower, upper, decision, question="RQ1", label=None):
    return {"contrast_id": contrast_id, "research_question": question, "label": label or contrast_id,
            "median": median, "lower_90": lower, "upper_90": upper, "decision": decision}


def fake_report():
    rows = [row("rq1_a_minus_b", -1.0, -1.2, -0.8, "detected_negative"),
            {"parameter_id": "rq6_gamma_size", "median": 0.05, "lower_90": 0.02, "upper_90": 0.08,
             "decision": "not_detected"}]
    section = {"primary": {"rows": rows}}
    return {"pools": {"venture": section, "hiring": section}, "matched_rq5": section}


def test_decision_table_reports_log_contrasts_and_ratios():
    text = writeup.decision_table(fake_report()["pools"]["venture"]["primary"])
    assert "| RQ1 | rq1_a_minus_b | −1.000 | [−1.200, −0.800] | 0.37 | [0.30, 0.45] | detected − |" in text
    assert "menu-size slope (per added alternative)" in text and "not detected" in text


def test_render_fills_markers_and_rejects_unknown_or_unresolved():
    report = fake_report()
    text = writeup.render("A {{TABLE:t}} B {{VALUE:v}} C {{RATIO:venture:rq1_a_minus_b}}", {"t": "T"}, {"v": 3}, report)
    assert text == f"A T B 3 C {math.exp(-1):.2f} [{math.exp(-1.2):.2f}, {math.exp(-0.8):.2f}]"
    with pytest.raises(KeyError):
        writeup.render("{{VALUE:missing}}", {}, {}, report)
    with pytest.raises(ValueError, match="Unresolved"):
        writeup.render("{{BOGUS}}", {}, {}, report)


def scale_section(offset=0.0, zero_sd=(), slope_unchanged=True):
    return {"primary": {"assessment_scale": {
        "rows": [{"research_question": "RQ2", "deterministic_offset": offset, "zero_sd_cell_ids": list(zero_sd)}],
        "rq6": {"slope_unchanged": slope_unchanged}}}}


@pytest.mark.parametrize("kwargs", [{"offset": 0.1}, {"zero_sd": ["cell"]}, {"slope_unchanged": False}])
def test_amendment6_statement_requires_every_invariance(kwargs):
    good = {"pools": {"venture": scale_section(), "hiring": scale_section()}, "matched_rq5": scale_section()}
    assert "cancel exactly" in writeup.amendment6_checks(good)
    bad = {"pools": {"venture": scale_section(**kwargs), "hiring": scale_section()}, "matched_rq5": scale_section()}
    with pytest.raises(ValueError):
        writeup.amendment6_checks(bad)


def test_fit_directories_are_rerooted_at_this_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(writeup, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(writeup, "ROOT", tmp_path / "checkout")
    (tmp_path / "fit_manifest.json").write_text(json.dumps({"fits": {"venture": {"primary": {
        "chain_path": "/elsewhere/old-root/applications/seu_sensitivity_study/results/fit_a/chains"}}}}))
    assert writeup.fit_directories() == {
        ("venture", "primary"): tmp_path / "checkout/applications/seu_sensitivity_study/results/fit_a"}

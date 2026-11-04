"""Offline tests for the results-report evidence exporter and renderer (portable supplement only)."""

import copy
import json
import math
import re

import pytest
import yaml

from analysis import study_results_evidence as evidence_module
from analysis import study_results_render as render

REPORT = render.REPORT / "02_results.qmd"
META = re.compile(r"\{\{< meta (rv(?:\.[A-Za-z0-9_]+)+) >\}\}")
INCLUDE = re.compile(r"\{\{< include (_results/[a-z0-9_]+\.qmd) >\}\}")
# Frozen design constants and structural numbers that the prose may state directly. Anything else must come
# from the generated values file.
ALLOWED_NUMBERS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "15", "16", "18", "24", "26", "27", "36", "40",
    "60", "95", "140", "280", "400", "2026", "0.05", "0.10", "0.2", "0.3", "0.35", "0.5", "0.50", "0.65", "1.0",
    "1.01", "1.05", "1.25", "2.5", "4.5", "10%", "30%", "90%",
}


@pytest.fixture(scope="module")
def evidence():
    return json.loads(evidence_module.SUPPLEMENT.read_text())


@pytest.fixture(scope="module")
def values():
    return yaml.safe_load(render.VALUES.read_text())


def lookup(values, key):
    node = values
    for part in key.split("."):
        node = node[part]
    return node


def prose(text):
    """Report text with front matter, shortcodes, math, code, links and cross-references removed."""
    text = text.split("\n---\n", 1)[1]
    text = META.sub(" ", text)
    text = INCLUDE.sub(" ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", " ", text)
    text = re.sub(r"`[^`\n]+`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"\{#[^}]*\}", " ", text)
    text = re.sub(r"@(?:sec|tbl|fig|eq)-[a-z0-9-]+", " ", text)
    text = re.sub(r"@[a-z]+\d{4}", " ", text)
    return text


def test_supplement_renders_exactly(evidence):
    render.check_outputs(evidence)


def test_every_meta_reference_resolves_to_a_scalar(values):
    keys = META.findall(REPORT.read_text())
    assert keys, "The report must quote generated values"
    for key in keys:
        value = lookup(values, key)
        assert isinstance(value, (str, int, float)), key


def test_includes_exist_and_are_all_used():
    used = set(INCLUDE.findall(REPORT.read_text()))
    present = {f"_results/{path.name}" for path in render.INCLUDES.glob("*.qmd")}
    assert used == present


def test_prose_contains_no_hand_typed_result_numbers():
    text = prose(REPORT.read_text())
    numbers = re.findall(r"(?<![A-Za-z_\-.\d])\d[\d,]*(?:\.\d+)?%?", text)
    unexpected = sorted({number.rstrip(",") for number in numbers} - ALLOWED_NUMBERS)
    assert not unexpected, f"Numbers not generated or allowlisted as design constants: {unexpected}"


def test_family_has_26_decisions_and_exact_openai_reversal(evidence):
    rows = {group: {row["id"]: row for row in evidence["decisions"][group]} for group in render.GROUPS}
    assert [len(rows[group]) for group in render.GROUPS] == [10, 10, 7]
    rq5 = [row for row in rows["matched_rq5"].values() if row["research_question"] == "RQ5"]
    assert 10 + 10 + len(rq5) == evidence["multiplicity"]["primary_decision_count"] == 26
    for pool in render.POOLS:
        mini, openai = rows[pool]["rq1_gpt_4o_mini_minus_gpt_4o"], rows[pool]["rq1_openai_flagship_minus_small"]
        assert math.isclose(mini["median"], -openai["median"], abs_tol=1e-12)
        assert math.isclose(mini["lower_90"], -openai["upper_90"], abs_tol=1e-12)


def test_decision_rule_reproduces_every_label(evidence):
    for group in render.GROUPS:
        for row in evidence["decisions"][group]:
            excludes = row["lower_90"] > 0 or row["upper_90"] < 0
            practical = abs(row["median"]) > row["rope_half_width"]
            expected = ("not_detected" if not (excludes and practical)
                        else "detected_positive" if row["median"] > 0 else "detected_negative")
            assert row["decision"] == expected, (group, row["id"])


def test_detection_count_uses_primary_decisions_only(evidence, values):
    counted = sum(row["decision"] != "not_detected" for group in render.GROUPS
                  for row in render.primary_rows(evidence, group))
    assert values["rv"]["n"]["detected"] == counted
    assert all("decision" not in value for pool in render.POOLS for value in evidence["sonnet"][pool].values())


def test_flag_counts_never_exceed_denominators(evidence):
    for key, value in evidence["a4"]["flags"].items():
        for name in ("choice_pattern", "trend", "repetition", "position"):
            assert 0 <= value[name]["flagged"] <= value[name]["evaluated"], (key, name)
        if key.endswith(("presentation_1_only", "presentation_2_only")):
            assert value["repetition"]["evaluated"] == 0 and value["unavailable_sizes_or_pairs"]


def test_flag_counts_follow_the_interpretation_rules():
    def stat(flag, status="available", difference=0.1):
        return {"status": status, "descriptive_review_flag": flag, "difference_from_predictive_median": difference}

    a4 = {"trends": [{"statistics": {"maximizer_fraction": stat(True), "mean_regret": stat(False),
                                     "filler_fraction": stat(None, "unavailable")}}],
          "groups": [{"statistics": {"maximizer_fraction": stat(True), "mean_regret": stat(True),
                                     "filler_fraction": stat(False)}}],
          "items": [{"statistics": {"conditional_rate": stat(False, "unavailable"), "unconditional_share": stat(True)}}],
          "positions": [{"fraction": stat(False)}, {"fraction": stat(True)}],
          "pairs": [{"same_item": stat(True, difference=0.2)}, {"same_item": stat(True, difference=-0.2)}]}
    counts = evidence_module.flag_counts(a4)
    assert counts["trend"] == {"evaluated": 2, "flagged": 1}
    assert counts["choice_pattern"] == {"evaluated": 4, "flagged": 3}
    assert counts["position"] == {"evaluated": 2, "flagged": 1}
    assert counts["repetition"] == {"evaluated": 2, "flagged": 1}


def test_stable_subset_is_annotation_not_decision(evidence):
    stable = evidence["stable"]
    assert "selection" in stable["interpretation"].lower() or "outcome" in stable["interpretation"].lower()
    for pool in render.POOLS:
        value = stable["pools"][pool]
        assert value["menus"] == sum(value["retention"]["retention_after_balance"].values())
        assert len(set(value["retention"]["retention_after_balance"].values())) == 1
        assert {row["id"] for row in value["comparison"]} == {row["id"] for row in evidence["decisions"][pool]}


def test_corrupted_supplement_is_rejected(evidence):
    changed = copy.deepcopy(evidence)
    changed["decisions"]["venture"][0]["median"] += 0.01
    with pytest.raises(ValueError):
        render.check_outputs(changed)


def test_formatting_uses_unicode_minus_and_no_negative_zero():
    assert render.num(-0.0004) == "0.000"
    assert render.num(-1.5, 1) == "−1.5"
    assert render.ratio({"median": 0.0, "lower_90": -0.1, "upper_90": 0.1}) == "1.00 [0.90, 1.11]"


def test_illustration_is_monotone_transform_of_alpha(evidence):
    for pool in render.POOLS:
        for model in render.MODELS:
            cell = f"{model.replace('-', '_')}_neutral_{pool}"
            values = render.illustration(evidence, pool, cell)
            for gap in render.GAPS:
                assert values[gap]["q05"] <= values[gap]["q50"] <= values[gap]["q95"]
            assert values[render.GAPS[0]]["q50"] < values[render.GAPS[1]]["q50"]


def test_supplement_records_sources_with_hashes(evidence):
    assert evidence["schema"] == evidence_module.SCHEMA
    assert evidence["source_count"] == len(evidence["sources"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in evidence["sources"].values())
    assert all(not path.startswith("/") for path in evidence["sources"])

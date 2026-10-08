import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from analysis import study_offset_evidence as offsets
from applications.seu_sensitivity_study.config import build_cells


REVIEWER_VALUES = [
    ("venture", "rq1_gpt_4o_mini_minus_gpt_4o", 0.063),
    ("venture", "rq1_o3_mini_minus_gpt_4o", 0.037),
    ("venture", "rq1_claude_sonnet_4_5_minus_gpt_4o", 0.019),
    ("venture", "rq1_claude_haiku_4_5_minus_gpt_4o", -0.040),
    ("venture", "rq1_claude_sonnet_4_5_thinking_minus_gpt_4o", -0.054),
    ("hiring", "rq1_gpt_4o_mini_minus_gpt_4o", -0.181),
    ("hiring", "rq1_o3_mini_minus_gpt_4o", 0.026),
    ("hiring", "rq1_claude_sonnet_4_5_minus_gpt_4o", 0.059),
    ("hiring", "rq1_claude_haiku_4_5_minus_gpt_4o", -0.133),
    ("hiring", "rq1_claude_sonnet_4_5_thinking_minus_gpt_4o", 0.166),
    ("venture", "context_thinking_minus_sonnet", -0.073),
    ("hiring", "context_thinking_minus_sonnet", 0.107),
    ("matched_rq5", "rq5_gpt_4o_hiring_minus_procurement", -0.197),
    ("matched_rq5", "rq5_gpt_4o_mini_hiring_minus_procurement", -0.409),
    ("matched_rq5", "rq5_o3_mini_hiring_minus_procurement", -0.161),
    ("matched_rq5", "rq5_claude_sonnet_4_5_hiring_minus_procurement", -0.096),
    ("matched_rq5", "rq5_claude_haiku_4_5_hiring_minus_procurement", -0.101),
    ("matched_rq5", "rq5_claude_sonnet_4_5_thinking_hiring_minus_procurement", -0.037),
]


@pytest.fixture(scope="module")
def bundle():
    return yaml.safe_load(offsets.BUNDLE.read_text())


@pytest.fixture(scope="module")
def evidence(bundle):
    return offsets.build_evidence(bundle)


@pytest.mark.parametrize("group,contrast_id,expected", REVIEWER_VALUES)
def test_reviewer_18_rounding_values(evidence, group, contrast_id, expected):
    row = next(row for row in evidence["rows"]
               if (row["group"], row["contrast_id"]) == (group, contrast_id))
    assert round(row["offset"], 3) == expected


def test_all_named_rows_duplicates_and_exact_rq2_cancellation(evidence):
    rows = evidence["rows"]
    assert Counter(row["research_question"] for row in rows) == {
        "RQ1": 14, "RQ2": 4, "RQ5": 6, "context": 2}
    assert len({(row["group"], row["contrast_id"]) for row in rows}) == 26
    for group, openai, anthropic in (("venture", -0.063, 0.059), ("hiring", 0.181, 0.192)):
        by_id = {row["contrast_id"]: row for row in rows if row["group"] == group}
        assert len(by_id) == 10
        assert by_id["rq1_openai_flagship_minus_small"]["offset"] == -by_id["rq1_gpt_4o_mini_minus_gpt_4o"]["offset"]
        assert round(by_id["rq1_openai_flagship_minus_small"]["offset"], 3) == openai
        assert round(by_id["rq1_anthropic_flagship_minus_small"]["offset"], 3) == anthropic
        for prompt in ("seu_maximizing", "deliberative"):
            assert by_id[f"rq2_{prompt}_minus_neutral"]["offset"] == 0.0
    assert all(row["offset"] < 0 for row in rows if row["research_question"] == "RQ5")


def test_full_references_population_sd_and_unrounded_directions(bundle, evidence):
    assert sum(len(reference["arms"]) for reference in evidence["references"].values()) == 24
    cells = {cell.cell_id: cell for cell in build_cells(["venture", "hiring"])}
    for group, reference in evidence["references"].items():
        arm_sds = {}
        for arm in reference["arms"]:
            pool = bundle["pools"][arm["pool_id"]]
            item_ids = sorted(item["id"] for item in pool["items"]
                              if group != "matched_rq5" or item["family"] == offsets.POOL_FAMILIES[arm["pool_id"]])
            assert arm["item_ids"] == item_ids
            assert arm["item_count"] == len(item_ids) == (24 if group == "matched_rq5" else 60)
            assert arm["menu_count"] == (40 if group == "matched_rq5" else 140)
            probabilities = np.asarray([pool["probabilities"][arm["model_name"]][item_id]
                                        for item_id in item_ids])
            eta = (probabilities / probabilities.sum(axis=1, keepdims=True)) @ [0.0, 0.5, 1.0]
            np.testing.assert_allclose(arm["eta"], eta, rtol=0, atol=1e-15)
            assert arm["sd"] == pytest.approx(float(np.std(eta, ddof=0)), abs=1e-15)
            assert arm["sd"] != pytest.approx(float(np.std(eta, ddof=1)), abs=1e-8)
            arm_sds[arm["pool_id"], arm["model_name"]] = arm["sd"]
        for row in (row for row in evidence["rows"] if row["group"] == group):
            expected = math.fsum(weight * math.log(arm_sds[cells[cell_id].pool_id, cells[cell_id].model_name])
                                 for cell_id, weight in row["realized_cell_weights"].items())
            assert row["offset"] == pytest.approx(expected, abs=1e-15)
            assert row["direction"] == row["label"]
            assert row["status"] == "available"
            if row["research_question"] != "RQ2":
                assert row["offset"] != round(row["offset"], 6)


def test_inputs_unchanged_and_probability_normalization(bundle, evidence):
    original = copy.deepcopy(bundle)
    offsets.build_evidence(bundle)
    assert bundle == original
    scaled = copy.deepcopy(bundle)
    for pool in scaled["pools"].values():
        for probabilities in pool["probabilities"].values():
            for item_id, values in probabilities.items():
                probabilities[item_id] = [1.01 * value for value in values]
    rebuilt = offsets.build_evidence(scaled)
    np.testing.assert_allclose([row["offset"] for row in rebuilt["rows"]],
                               [row["offset"] for row in evidence["rows"]], rtol=0, atol=1e-15)
    for group, reference in rebuilt["references"].items():
        for arm, original_arm in zip(reference["arms"], evidence["references"][group]["arms"]):
            np.testing.assert_allclose(arm["eta"], original_arm["eta"], rtol=0, atol=1e-15)
            assert arm["sd"] == pytest.approx(original_arm["sd"], abs=1e-15)


def test_zero_sd_is_unavailable_without_epsilon(bundle):
    modified = copy.deepcopy(bundle)
    probabilities = modified["pools"]["venture"]["probabilities"]["gpt-4o"]
    for item_id in probabilities:
        probabilities[item_id] = [0.0, 1.0, 0.0]
    evidence = offsets.build_evidence(modified)
    cells = {cell.cell_id: cell for cell in build_cells(["venture", "hiring"])}
    for row in evidence["rows"]:
        affected = [cell_id for cell_id in row["realized_cell_weights"]
                    if cells[cell_id].pool_id == "venture" and cells[cell_id].model_name == "gpt-4o"]
        assert row["zero_sd_cell_ids"] == affected
        assert row["status"] == ("unavailable_zero_sd" if affected else "available")
        if affected:
            assert row["offset"] is None
    arm = next(arm for arm in evidence["references"]["venture"]["arms"] if arm["model_name"] == "gpt-4o")
    assert arm["sd"] == 0.0
    assert arm["log_sd"] is None
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("change", ["item", "menu", "probability", "duplicate_menu"])
def test_incomplete_frozen_inputs_fail(bundle, change):
    modified = copy.deepcopy(bundle)
    pool = modified["pools"]["venture"]
    if change == "item":
        pool["items"].pop()
    elif change == "menu":
        pool["menus"].pop()
    elif change == "probability":
        pool["probabilities"]["gpt-4o"].pop(pool["items"][0]["id"])
    else:
        pool["menus"][-1]["id"] = pool["menus"][0]["id"]
    with pytest.raises(ValueError, match="assessment_scale"):
        offsets.build_evidence(modified)


def test_disclosures_and_complete_source_hashes(evidence):
    expected = {"reports/applications/seu_sensitivity_study/data/design_evidence.yml",
                "analysis/study_offset_evidence.py"}
    expected.update(f"applications/seu_sensitivity_study/{name}" for name in (
        "assessment_scale.py", "config.py", "confirmatory_analysis.py", "data_preparation.py",
        "pools.py", "schemas.py", "data/venture.json", "data/hiring.json"))
    assert set(evidence["source_hashes"]) == expected
    for name, digest in evidence["source_hashes"].items():
        assert digest == hashlib.sha256((offsets.ROOT / name).read_bytes()).hexdigest()
    assert evidence["ddof"] == 0
    assert evidence["amendment9_standardized_posterior_extension"] is False
    assert evidence["choice_records_read"] is False
    assert evidence["posterior_fits"] == evidence["provider_calls"] == 0
    assert all(not row["posterior_extension"] and not row["included_in_primary_family"] for row in evidence["rows"])
    rendered = offsets.render_include(evidence)
    assert "not an Amendment 9 standardized posterior extension" in rendered
    assert "not the sign of an unknown production RQ5 effect" in rendered
    assert "not a pooled 48-item SD" in rendered
    assert "no epsilon" in rendered
    assert len([line for line in rendered.splitlines() if line.startswith("| ")]) == 27


def write_temporary_outputs(tmp_path, evidence):
    output = tmp_path / "offset_evidence.json"
    include = tmp_path / "_offset_evidence.qmd"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    include.write_text(offsets.render_include(evidence))
    return output, include


@pytest.mark.parametrize("source", offsets.source_paths(), ids=lambda path: path.name)
def test_consumed_source_tamper_fails_exact_check(bundle, evidence, tmp_path, monkeypatch, source):
    output, include = write_temporary_outputs(tmp_path, evidence)
    original_read = Path.read_bytes

    def tampered_read(path):
        return original_read(path) + (b"\n" if path == source else b"")

    monkeypatch.setattr(Path, "read_bytes", tampered_read)
    rebuilt = offsets.build_evidence(bundle)
    assert rebuilt["rows"] == evidence["rows"]
    with pytest.raises(ValueError, match="source hashes"):
        offsets.check_outputs(rebuilt, output=output, include=include)


@pytest.mark.parametrize("target", ["json", "qmd"])
def test_output_whitespace_tamper_fails_exact_check(evidence, tmp_path, target):
    output, include = write_temporary_outputs(tmp_path, evidence)
    offsets.check_outputs(evidence, output=output, include=include)
    path = output if target == "json" else include
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs"):
        offsets.check_outputs(evidence, output=output, include=include)


def test_published_artifacts_reproduce_exactly(evidence):
    offsets.check_outputs(evidence)
    paths = [*offsets.source_paths(), offsets.OUTPUT, offsets.INCLUDE]
    before = {path: path.read_bytes() for path in paths}
    result = subprocess.run([sys.executable, "-m", "analysis.study_offset_evidence", "--check"],
                            cwd=offsets.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path: path.read_bytes() for path in paths} == before
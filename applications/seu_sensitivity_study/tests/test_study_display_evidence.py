import copy
import math
import shutil
import subprocess
from statistics import NormalDist

import numpy as np
import pytest
import yaml

from analysis.study_display_evidence import BUNDLE, REPORT, build_evidence, render_include, summarize_gaps
from applications.seu_sensitivity_study.assessment_scale import build_reference


@pytest.fixture(scope="module")
def bundle():
    return yaml.safe_load(BUNDLE.read_text())


def test_exact_and_additional_near_ties_are_disjoint():
    assert summarize_gaps([0, 1e-13, 1e-12, np.nextafter(1e-12, math.inf)]) == {
        "menu_count": 4, "exact_ties": 1, "additional_near_ties": 2,
        "ties_with_tolerance": 3, "tie_prevalence": 0.75,
    }


@pytest.mark.parametrize("gaps", [[], [-1e-15], [math.nan], [math.inf], [[0, 0]]])
def test_invalid_gaps_fail(gaps):
    with pytest.raises(ValueError):
        summarize_gaps(gaps)


def test_frozen_geometry_matches_amendment6_without_mutation(bundle):
    original = copy.deepcopy(bundle)
    evidence = build_evidence(bundle)
    assert bundle == original
    assert len(evidence["rows"]) == 24
    for group in ("venture", "hiring", "matched_rq5"):
        pool_ids = ("venture", "hiring") if group == "matched_rq5" else (group,)
        items, problems, probabilities = [], [], {}
        for pool_id in pool_ids:
            pool = bundle["pools"][pool_id]
            family = {"venture": "procurement", "hiring": "matched"}[pool_id]
            items.extend(item for item in pool["items"] if group != "matched_rq5" or item["family"] == family)
            problems.extend(menu for menu in pool["menus"] if group != "matched_rq5" or menu["family"] == family)
            for model, values in pool["probabilities"].items():
                probabilities.setdefault(model, {}).update(values)
        reference = build_reference(group=group, items=items, problems=problems, probabilities=probabilities)
        for arm in reference["arms"]:
            row = next(row for row in evidence["rows"] if row["model"] == arm["model_name"]
                       and row["pool"] == arm["pool_id"]
                       and (row["family"] != "all") == (group == "matched_rq5"))
            assert dict(zip(row["menu_ids"], row["top_two_gaps"])) == dict(zip(arm["menu_ids"], arm["top_two_gaps"]))
            assert row["ties_with_tolerance"] == arm["tie_count"]
            assert row["tie_prevalence"] == arm["tie_prevalence"]
    haiku = next(row for row in evidence["rows"] if row["pool"] == "venture"
                 and row["family"] == "all" and row["model"] == "claude-haiku-4-5")
    assert (haiku["exact_ties"], haiku["additional_near_ties"]) == (19, 2)


@pytest.mark.parametrize("change", ["drop", "duplicate_id"])
def test_incomplete_or_duplicate_menus_fail(bundle, change):
    modified = copy.deepcopy(bundle)
    menus = modified["pools"]["venture"]["menus"]
    if change == "drop":
        menus.pop()
    else:
        menus[-1]["id"] = menus[0]["id"]
    with pytest.raises(ValueError, match="complete fixed menu set"):
        build_evidence(modified)


def test_analytic_interval_and_rendered_disclosures(bundle):
    evidence = build_evidence(bundle)
    lower, upper = evidence["size_ratio_prior_central_90"]
    assert lower * upper == pytest.approx(1)
    assert NormalDist(0, 0.2).cdf(math.log(upper) / 6) == pytest.approx(0.95)
    assert [round(lower, 4), round(upper, 4)] == [0.1389, 7.1982]
    rendered = render_include(evidence)
    assert "19/140 | 2/140 | 21/140 (15.0%)" in rendered
    assert "not independent additional menus" in rendered
    assert "no display rounding" in rendered


def run_filter(source):
    quarto = shutil.which("quarto")
    if not quarto:
        pytest.skip("Quarto is required for the report display filter")
    return subprocess.run([quarto, "pandoc", "--from=markdown", "--to=html",
                           f"--lua-filter={REPORT / 'probability_display.lua'}"],
                          input=source, text=True, capture_output=True)


def test_display_filter_preserves_raw_blocks_and_other_tables():
    source = (
        "| Item | Neutral probabilities |\n|---|---|\n"
        "| H1 | [0.1, 0.3, 0.6000000000000001] |\n\n"
        "| Item | Other values |\n|---|---|\n| H1 | [0.1, 0.3, 0.6000000000000001] |\n\n"
        "```text\nPROBABILITIES: 0.1, 0.3, 0.6000000000000001\n```\n"
    )
    result = run_filter(source)
    assert result.returncode == 0, result.stderr
    assert "[0.100000, 0.300000, 0.600000]" in result.stdout
    assert "[0.1, 0.3, 0.6000000000000001]" in result.stdout
    assert "PROBABILITIES: 0.1, 0.3, 0.6000000000000001" in result.stdout


@pytest.mark.parametrize("values", ["[0.1, 0.9]", "[0.1, -0.1, 1]", '[0.1, "bad", 0.9]'])
def test_display_filter_rejects_malformed_probability_cells(values):
    result = run_filter(f"| Neutral probabilities |\n|---|\n| {values} |\n")
    assert result.returncode != 0
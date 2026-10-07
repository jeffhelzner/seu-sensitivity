import json

import numpy as np
import pytest

from analysis.verify_realized_recovery import cell_weight_vector, realized_log_cells
from analysis.verify_realized_recovery import chain_metadata, read_chain, select_chains
from analysis.verify_realized_recovery import score_draws, summarize_scores, verify_design_rows
from analysis.verify_realized_recovery import canonical_truth
from analysis.verify_realized_recovery import aggregate_iterations, reviewer_comparison
from analysis.verify_realized_recovery import check_output, shared_truth_findings, Inputs


def test_truth_and_draw_reconstruction_permutation():
    design = np.array([[0, 0], [1, 0], [0, 1]])
    gamma = np.array([0.4, -0.2])
    residual_z = np.array([1.0, -1.0, 2.0])
    actual = realized_log_cells(2.5, gamma, 0.3, residual_z, design)
    np.testing.assert_allclose(actual, [2.8, 2.6, 2.9])
    draws = realized_log_cells(np.array([2.5, 7.5]), np.tile(gamma, (2, 1)),
                               np.array([0.3, 0.3]), np.tile(residual_z, (2, 1)), design)
    weights = {"first": -1.0, "second": 1.0}
    ids = ["first", "second", "third"]
    np.testing.assert_allclose(draws @ cell_weight_vector(ids, weights), [-0.2, -0.2])
    permutation = [2, 0, 1]
    np.testing.assert_allclose(actual[permutation] @ cell_weight_vector(
        [ids[index] for index in permutation], weights), actual @ cell_weight_vector(ids, weights))


@pytest.mark.parametrize("gamma0,gamma,sigma,residual_z,design", [
    (0, [1], 1, [1], [[1, 2]]),
    ([0], [[1]], [1], [1], [[1]]),
    (0, [1], -1, [1], [[1]]),
    (0, [np.nan], 1, [1], [[1]]),
])
def test_rejects_malformed_shapes_and_values(gamma0, gamma, sigma, residual_z, design):
    with pytest.raises(ValueError):
        realized_log_cells(gamma0, gamma, sigma, residual_z, design)


@pytest.mark.parametrize("ids,weights", [
    (["first", "first"], {"first": 0}),
    (["first"], {"missing": 0}),
    (["first"], {"first": 1}),
])
def test_rejects_invalid_cell_mapping(ids, weights):
    with pytest.raises(ValueError):
        cell_weight_vector(ids, weights)


def write_chain(path, chain_id=1, seed=54321, warmup_saved=0, rows=None):
    metadata = {"model": "h_m01_size_assessment_anchored_model", "id": chain_id,
                "seed": seed, "num_samples": 2, "num_warmup": 2,
                "save_warmup": warmup_saved, "thin": 1, "max_depth": 12, "delta": 0.95}
    path.write_text("".join(f"# {key} = {value}\n" for key, value in metadata.items())
                    + "lp__,gamma0\n" + (rows or "0,2\n0,3\n"))
    return path


def test_chain_selection_rejects_stale_duplicate_and_missing(tmp_path):
    paths = [write_chain(tmp_path / f"chain{index}.csv", chain_id=index) for index in range(1, 5)]
    references = [str(path) for path in paths]
    write_chain(tmp_path / "stale.csv", seed=1)
    selected, _, extras = select_chains(tmp_path, references[::-1], seed=54321, samples=2, warmup=2)
    assert selected == paths
    assert extras == ["stale.csv"]
    with pytest.raises(ValueError, match="seed/schedule"):
        select_chains(tmp_path, references, seed=123, samples=2, warmup=2)
    write_chain(paths[3], chain_id=1)
    with pytest.raises(ValueError, match="chain IDs"):
        select_chains(tmp_path, references, seed=54321, samples=2, warmup=2)
    paths[3].unlink()
    with pytest.raises(ValueError, match="missing"):
        select_chains(tmp_path, references, seed=54321, samples=2, warmup=2)


def test_saved_warmup_is_removed(tmp_path):
    path = write_chain(tmp_path / "chain.csv", warmup_saved=1, rows="0,900\n0,800\n0,2\n0,3\n")
    frame = read_chain(path, ["gamma0"], chain_metadata(path))
    assert frame.gamma0.tolist() == [2, 3]


def test_cmdstan_boolean_header(tmp_path):
    path = write_chain(tmp_path / "chain.csv")
    path.write_text(path.read_text().replace("save_warmup = 0", "save_warmup = false (Default)"))
    assert chain_metadata(path)["save_warmup"] == 0


@pytest.mark.parametrize("rows", ["0,2\n", "0,2\n0,nan\n", "0,2\n0,3\n0,4\n"])
def test_rejects_bad_chain_rows(tmp_path, rows):
    path = write_chain(tmp_path / "chain.csv", rows=rows)
    with pytest.raises(ValueError):
        read_chain(path, ["gamma0"], chain_metadata(path))


def test_rejects_duplicate_column_and_missing_parameter(tmp_path):
    path = write_chain(tmp_path / "chain.csv")
    with pytest.raises(ValueError, match="required"):
        read_chain(path, ["z_alpha.1"], chain_metadata(path))
    path.write_text(path.read_text().replace("lp__,gamma0", "lp__,lp__"))
    with pytest.raises(ValueError, match="header"):
        chain_metadata(path)


def test_design_permutations_and_invalid_rows():
    design = [[0, 0], [1, 0], [0, 1]]
    assert verify_design_rows([design[2], design[0], design[1]], design, ["a", "b", "c"]) == ["c", "a", "b"]
    with pytest.raises(ValueError):
        verify_design_rows([design[0]] * 3, design, ["a", "b", "c"])
    with pytest.raises(ValueError):
        verify_design_rows([[0, 0], [1, 0], [1, 1]], design, ["a", "b", "c"])


def test_summary_denominators_and_signs():
    rope = np.log(1.25)
    rows = [score_draws(np.array([0.3, 0.4, 0.5]), truth, rope) for truth in (0, 0.1, 0.4, -0.4)]
    rows.append(score_draws(np.array([-0.1, 0, 0.1]), 0.4, rope))
    result = summarize_scores(rows, rope)
    assert result["false_positive_exact_null"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert result["type_S_detected_nonzero_truth"]["count"] == 1
    assert result["type_S_detected_nonzero_truth"]["denominator"] == 3
    assert result["power_outside_ROPE"]["count"] == 2
    assert result["power_outside_ROPE"]["denominator"] == 3
    assert result["correct_sign_power_outside_ROPE"]["count"] == 1
    assert summarize_scores(rows[1:], rope)["false_positive_exact_null"]["rate"] is None


def test_decision_boundary_and_sign_reversal():
    rope = np.log(1.25)
    assert score_draws(np.full(10, rope), rope, rope)["decision"] == 0
    assert score_draws(np.linspace(0.3, 0.6, 100), 0.4, rope)["decision"] == 1
    assert score_draws(-np.linspace(0.3, 0.6, 100), -0.4, rope)["decision"] == -1


def test_empty_optional_contrasts_do_not_change_truth():
    assert canonical_truth({"gamma0": 2.5}) == canonical_truth({"gamma0": 2.5, "contrasts": {}})
    assert canonical_truth({"gamma0": 2.5}) != canonical_truth({"gamma0": 2.6})


def test_named_and_distinct_aggregation_do_not_mix_estimands_or_drop_failures():
    first = {"contrast_id": "rq1_gpt_4o_mini_minus_gpt_4o", "rq": "RQ1", "estimand": "realized",
             **score_draws(np.linspace(0.3, 0.5, 100), 0.4, np.log(1.25))}
    reverse = {"contrast_id": "rq1_openai_flagship_minus_small", "rq": "RQ1", "estimand": "realized",
               **score_draws(-np.linspace(0.3, 0.5, 100), -0.4, np.log(1.25))}
    companion = {**first, "estimand": "additive_companion", "decision": 0}
    result = aggregate_iterations([{"sampler_eligible": False, "scores": [first, reverse, companion]},
                                   {"blocker": "missing chains", "sampler_eligible": False}])
    assert result["realized"]["RQ1"]["named"]["n"] == 2
    assert result["realized"]["RQ1"]["distinct_up_to_sign"]["n"] == 1
    assert result["additive_companion"]["RQ1"]["named"]["detection_all_truths"]["count"] == 0


def test_current_contract_has_26_named_24_distinct_and_15_base_fits(tmp_path):
    from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig, build_cells
    from applications.seu_sensitivity_study.confirmatory_analysis import contract_manifest, matched_rq5_contract
    from applications.seu_sensitivity_study.ceiling_prior import fit_plan, PRIOR_VARIANTS

    design, columns, ids = SEUSensitivityStudyConfig(pool_ids=["venture"]).design_matrix_for_pool("venture")
    manifest = contract_manifest(columns, design)
    signatures = set()
    for spec in manifest["primary_contrasts"]:
        weights = cell_weight_vector(ids, spec["realized_cell_weights"]["venture"])
        np.testing.assert_allclose(weights @ design, spec["coefficients"], atol=1e-12)
        signatures.add(min(tuple(weights), tuple(-weights)))
    assert 2 * (len(manifest["primary_contrasts"]) + 1) + 6 == 26
    assert 2 * (len(signatures) + 1) + 6 == 24
    assert len(matched_rq5_contract(build_cells(["venture", "hiring"]))["contrasts"]) == 6
    assert len([row for row in fit_plan(tmp_path)["fits"] if row["variant"] not in PRIOR_VARIANTS]) == 15


@pytest.mark.parametrize("campaign,estimand,narrative", [
    ("venture", "realized", "A1"),
    ("hiring", "additive_companion", "A1"),
    ("matched_rq5", "additive_companion", "B4"),
])
def test_reviewer_comparison_marks_discrepancy(campaign, estimand, narrative):
    summary = {"detection_bins": {label: {"rate": 0.0} for label in
                                  ("below_1.25", "1.25_to_1.5", "1.5_to_2", "above_2")}}
    result = reviewer_comparison({campaign: {"summaries": {estimand: {"reviewer_pooled_distinct": summary}}}})
    assert result["review_table_detection_comparisons"][0]["agrees_at_reported_precision"] is False
    assert "1 disagreed" in result[narrative]
    assert "reproduced" not in result["A1"] + result["B4"]
    assert "all 40" not in result["B5"]


@pytest.mark.parametrize("summary", [None, {"n": 0}, {"detection_bins": {
    label: {"rate": None} for label in ("below_1.25", "1.25_to_1.5", "1.5_to_2", "above_2")}}])
def test_reviewer_comparison_missing_evidence_is_not_disagreement(summary):
    campaigns = {} if summary is None else {"venture": {"summaries": {
        "realized": {"reviewer_pooled_distinct": summary}}}}
    result = reviewer_comparison(campaigns)
    for field in ("A1", "B4"):
        assert "insufficient evidence" in result[field]
        assert "0 disagreed" in result[field]
        assert "reproduced" not in result[field]


def test_reviewer_comparison_reproduction_requires_all_expected_comparisons():
    campaigns = {}
    for campaign, estimand, rates in (
        ("venture", "additive_companion", [0.15, 0.59, 0.75, 0.90]),
        ("venture", "realized", [0.10, 0.85, 0.99, 1.00]),
        ("hiring", "additive_companion", [0.14, 0.60, 0.79, 0.86]),
        ("hiring", "realized", [0.08, 0.84, 1.00, 1.00]),
        ("matched_rq5", "additive_companion", [0.17, 0.37, 0.82, 0.93]),
    ):
        summary = {"detection_bins": {label: {"rate": value} for label, value in zip(
            ("below_1.25", "1.25_to_1.5", "1.5_to_2", "above_2"), rates)}}
        campaigns.setdefault(campaign, {"summaries": {}})["summaries"][estimand] = {
            "reviewer_pooled_distinct": summary}
    result = reviewer_comparison(campaigns)
    assert "reproduced: 4/4" in result["A1"]
    assert "reproduced: 1/1" in result["B4"]
    del campaigns["hiring"]
    result = reviewer_comparison(campaigns)
    assert "reproduced" not in result["A1"]
    assert "2/4 comparisons agreed" in result["A1"]
    assert "2 with insufficient evidence" in result["A1"]


def test_exact_check_detects_changed_inputs_and_results_without_writing(tmp_path):
    path = tmp_path / "audit.json"
    report = {"input_sha256": {"source": "abc"}, "coverage": 0.9, "weights": (1, -1)}
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    check_output(path, report)
    for changed in ({**report, "coverage": 0.8}, {**report, "input_sha256": {"source": "def"}}):
        with pytest.raises(ValueError, match="differs"):
            check_output(path, changed)
    assert path.read_bytes() == before


def test_missing_truth_blockers_do_not_trigger_unverified_shared_truth_reads(tmp_path):
    campaigns = {name: {"iterations": [{"iteration": 1, "blocker": "Missing truth", "sampler_eligible": False}]}
                 for name in ("venture", "matched_rq5")}
    result = shared_truth_findings(Inputs(tmp_path), campaigns)
    assert result["shifted_residual_stream"]["compared_iterations"] == 0
    assert result["pairwise_equal_counts"]["venture_vs_matched_rq5"]["paired_iterations"] == 0
    assert "insufficient evidence (0 verified pairs)" in result["interpretation"]
    assert "insufficient evidence (0 compared iterations)" in result["interpretation"]
    assert "40 shared draws" not in result["interpretation"]
    assert "truths match" not in result["interpretation"]
    assert "disagreed" not in result["interpretation"]


@pytest.mark.parametrize("equal", [True, False])
def test_shared_truth_narrative_uses_verified_counts(tmp_path, equal):
    fields = ("common_generating_parameters_sha256", "realized_log_truth_sha256", "residual_z_sha256")
    first = {"iteration": 1, "scores": [], **dict.fromkeys(fields, "first")}
    second = {**first, **dict.fromkeys(fields, "first" if equal else "second")}
    campaigns = {"venture": {"iterations": [first]}, "hiring": {"iterations": [second]}}
    result = shared_truth_findings(Inputs(tmp_path), campaigns)
    assert "venture_vs_hiring: 1 verified pairs" in result["interpretation"]
    expected = "1/1 equal, 0/1 disagreed" if equal else "0/1 equal, 1/1 disagreed"
    assert result["interpretation"].count(expected) == 3
    assert "40 shared draws" not in result["interpretation"]
import json
from dataclasses import replace

import numpy as np
import pytest

from applications.seu_sensitivity_study import confirmatory_analysis as ca
from applications.seu_sensitivity_study import config


DIAGNOSTICS = {
    "max_rhat": 1.005, "min_ess_bulk": 700, "min_ess_tail": 650,
    "min_ebfmi": 0.8, "divergences": 0, "treedepth_saturated_share": 0.0,
}


def _payload(group="venture", draw_count=31):
    pools = ["venture", "hiring"] if group == "matched_rq5" else [group]
    cells = config.build_cells(pools)
    if group == "matched_rq5":
        design, columns = ca.matched_rq5_design(cells)
        contrasts = [ca.ContrastSpec(**row) for row in ca.matched_rq5_contract(cells)["contrasts"]]
    else:
        design, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool(group)
        contrasts = ca.primary_contrasts(columns)
    generator = np.random.default_rng(8123)
    payload = {
        "gamma_draws": generator.normal(size=(draw_count, len(columns))),
        "gamma_size_draws": generator.normal(size=draw_count),
        "sigma_cell_draws": np.linspace(0.2, 1.0, draw_count),
        "z_alpha_draws": generator.normal(size=(draw_count, len(cells))),
        "contrasts": contrasts,
        "cell_ids": [cell.cell_id for cell in cells],
        "diagnostics": DIAGNOSTICS,
    }
    realized = payload["gamma_draws"] @ design.T + payload["sigma_cell_draws"][:, None] * payload["z_alpha_draws"]
    return payload, cells, realized


def _assert_summary(row, values):
    for key, expected in ca.summarize_draws(values, rope_half_width=ca.LOG_ALPHA_ROPE).items():
        if isinstance(expected, float):
            assert row[key] == pytest.approx(expected)
        else:
            assert row[key] == expected


@pytest.mark.parametrize("pool", ["venture", "hiring"])
def test_rq1_rq2_are_draw_wise_equal_cell_means(pool):
    payload, cells, realized = _payload(pool)
    report = ca.posterior_fit_report(**payload)
    averages = {
        model.name: realized[:, [index for index, cell in enumerate(cells) if cell.model_name == model.name]].mean(axis=1)
        for model in config.MODELS
    }
    expected = [averages[model.name] - averages[config.REFERENCE_MODEL]
                for model in config.MODELS if model.name != config.REFERENCE_MODEL]
    for vendor in ("openai", "anthropic"):
        flagship = next(model.name for model in config.MODELS if model.vendor == vendor and model.tier == "flagship")
        small = next(model.name for model in config.MODELS if model.vendor == vendor and model.tier == "small")
        expected.append(averages[flagship] - averages[small])
    for prompt in config.PROMPT_CONDITIONS[1:]:
        prompt_indices = [index for index, cell in enumerate(cells) if cell.prompt_condition == prompt]
        neutral_indices = [index for index, cell in enumerate(cells) if cell.prompt_condition == config.REFERENCE_PROMPT]
        expected.append((realized[:, prompt_indices] - realized[:, neutral_indices]).mean(axis=1))
    for row, draws in zip(report["contrast_decisions"]["rows"], expected):
        _assert_summary(row, draws)
        weights = list(row["cell_weights"].values())
        denominator = 6 if row["research_question"] == "RQ2" else 3
        assert len(weights) == 2 * denominator
        assert set(np.abs(weights)) == {1 / denominator}
        assert sum(weights) == pytest.approx(0, abs=1e-15)
    assert not np.allclose(expected[0], payload["gamma_draws"][:, 0])
    for row, contrast in zip(report["gamma_companion"]["rows"], payload["contrasts"]):
        assert row["median"] == pytest.approx(np.median(payload["gamma_draws"] @ contrast.coefficients))
        assert row["included_in_primary_family"] is False
        assert "decision" not in row


@pytest.mark.parametrize("pool", ["venture", "hiring"])
def test_sonnet_descriptive_includes_residuals_with_zero_gamma(pool):
    payload, cells, _ = _payload(pool)
    payload["gamma_draws"][:] = 0
    payload["z_alpha_draws"][:] = 0
    expected = payload["sigma_cell_draws"] * 2
    for index, cell in enumerate(cells):
        if cell.model_name == "claude-sonnet-4-5-thinking":
            payload["z_alpha_draws"][:, index] = (1, 2, 6)[config.PROMPT_CONDITIONS.index(cell.prompt_condition)]
        elif cell.model_name == "claude-sonnet-4-5":
            payload["z_alpha_draws"][:, index] = 1
    report = ca.posterior_fit_report(**payload)
    row = report["sonnet_thinking_descriptive"]
    for key, value in ca.summarize_draws(expected).items():
        assert row[key] == pytest.approx(value) if isinstance(value, float) else row[key] == value
    for key in ("mean", "median", "lower_90", "upper_90"):
        assert row["geometric_mean_sensitivity_ratio"][key] == pytest.approx(ca.summarize_draws(np.exp(expected))[key])
    assert row["probability_positive"] == 1
    assert row["probability_negative"] == row["probability_zero"] == 0
    assert row["included_in_primary_family"] is False
    assert row["decision_count"] == 0
    assert "decision" not in row and "rope_half_width" not in row
    assert report["decision_count"] == 10
    assert all(companion["median"] == 0 for companion in report["gamma_companion"]["rows"])


@pytest.mark.parametrize("pool", ["venture", "hiring"])
def test_sonnet_descriptive_matches_direct_draw_wise_formula(pool):
    payload, cells, realized = _payload(pool)
    thinking = [index for index, cell in enumerate(cells) if cell.model_name == "claude-sonnet-4-5-thinking"]
    base = [index for index, cell in enumerate(cells) if cell.model_name == "claude-sonnet-4-5"]
    expected = realized[:, thinking].mean(axis=1) - realized[:, base].mean(axis=1)
    report = ca.posterior_fit_report(**payload)
    row = report["sonnet_thinking_descriptive"]
    for key in ("mean", "median", "lower_90", "upper_90"):
        assert row[key] == pytest.approx(ca.summarize_draws(expected)[key])
        assert row["geometric_mean_sensitivity_ratio"][key] == pytest.approx(ca.summarize_draws(np.exp(expected))[key])
    assert row["probability_positive"] == np.mean(expected > 0)
    assert row["probability_negative"] == np.mean(expected < 0)
    assert row["probability_zero"] == np.mean(expected == 0)
    assert len(row["cell_weights"]) == 6
    assert set(row["cell_weights"].values()) == {-1 / 3, 1 / 3}
    assert row["policy_version"] == "B3_descriptive_postreview_2026-10-07"
    assert "not a pure causal reasoning effect" in row["interpretation"]
    assert row["contrast_id"] not in {entry.get("contrast_id") for entry in report["rows"]}
    assert report["contrast_decisions"] == ca._realized_contrast_report(
        realized, payload["cell_ids"], payload["contrasts"], pool, DIAGNOSTICS)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pool", ["venture", "hiring"])
@pytest.mark.parametrize("required", [True, False])
def test_sonnet_descriptive_missing_cells_never_reweight_or_fall_back(pool, required):
    payload, cells, _ = _payload(pool)
    expected = ca.posterior_fit_report(**payload)["sonnet_thinking_descriptive"]
    model = "claude-sonnet-4-5-thinking" if required else config.REFERENCE_MODEL
    removed = next(index for index, cell in enumerate(cells) if cell.model_name == model)
    missing_id = payload["cell_ids"].pop(removed)
    payload["z_alpha_draws"] = np.delete(payload["z_alpha_draws"], removed, axis=1)
    row = ca.posterior_fit_report(**payload)["sonnet_thinking_descriptive"]
    assert row["cell_weights"] == expected["cell_weights"]
    if required:
        assert row["status"] == "unavailable"
        assert row["missing_cell_ids"] == [missing_id]
        assert "gamma fallback" in row["reason"]
        assert not {"median", "geometric_mean_sensitivity_ratio", "probability_positive", "decision", "rope_half_width"} & row.keys()
    else:
        assert row == expected


def test_sonnet_descriptive_sign_probabilities_keep_exact_zeros():
    payload, cells, _ = _payload(draw_count=5)
    payload["gamma_draws"][:] = 0
    payload["sigma_cell_draws"][:] = 1
    payload["z_alpha_draws"][:] = 0
    for index, cell in enumerate(cells):
        if cell.model_name == "claude-sonnet-4-5-thinking":
            payload["z_alpha_draws"][:, index] = [-1, 0, 0, 2, 3]
    row = ca.posterior_fit_report(**payload)["sonnet_thinking_descriptive"]
    assert row["probability_positive"] == 0.4
    assert row["probability_negative"] == 0.2
    assert row["probability_zero"] == 0.4


def test_rq5_uses_three_prompt_means_in_the_matched_fit():
    payload, cells, realized = _payload("matched_rq5")
    report = ca.posterior_fit_report(**payload)
    assert "sonnet_thinking_descriptive" not in report
    for model, row in zip(config.MODELS, report["contrast_decisions"]["rows"]):
        hiring = [index for index, cell in enumerate(cells) if cell.model_name == model.name and cell.pool_id == "hiring"]
        procurement = [index for index, cell in enumerate(cells) if cell.model_name == model.name and cell.pool_id == "venture"]
        _assert_summary(row, realized[:, hiring].mean(axis=1) - realized[:, procurement].mean(axis=1))
        assert len(row["cell_weights"]) == 6
        assert set(np.abs(list(row["cell_weights"].values()))) == {1 / 3}


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
def test_realized_contrasts_follow_cell_ids_and_gamma_column_names(group):
    payload, _, _ = _payload(group)
    expected = ca.posterior_fit_report(**payload)
    permutation = np.random.default_rng(17).permutation(len(payload["cell_ids"]))
    reordered = {
        **payload,
        "cell_ids": [payload["cell_ids"][index] for index in permutation],
        "z_alpha_draws": payload["z_alpha_draws"][:, permutation],
        "gamma_draws": payload["gamma_draws"][:, ::-1],
        "contrasts": [replace(contrast, column_names=tuple(reversed(contrast.column_names)),
                              coefficients=tuple(reversed(contrast.coefficients)))
                      for contrast in payload["contrasts"]],
    }
    actual = ca.posterior_fit_report(**reordered)
    if group != "matched_rq5":
        before = expected["sonnet_thinking_descriptive"]
        after = actual["sonnet_thinking_descriptive"]
        assert before["cell_weights"] == after["cell_weights"]
        for key in ("mean", "lower_90", "median", "upper_90", "probability_positive", "probability_negative", "probability_zero"):
            assert before[key] == pytest.approx(after[key])
        for key in ("mean", "lower_90", "median", "upper_90"):
            assert before["geometric_mean_sensitivity_ratio"][key] == pytest.approx(after["geometric_mean_sensitivity_ratio"][key])
    for before, after in zip(expected["rows"], actual["rows"]):
        for key in ("mean", "lower_90", "median", "upper_90"):
            assert before[key] == pytest.approx(after[key])


@pytest.mark.parametrize("group", ["venture", "matched_rq5"])
def test_common_intercept_and_menu_size_terms_cancel(group):
    payload, _, _ = _payload(group)
    expected = ca.posterior_fit_report(**payload)
    shift = np.linspace(-30, 70, len(payload["gamma_draws"]))
    changed = ca.posterior_fit_report(**{
        **payload,
        "z_alpha_draws": payload["z_alpha_draws"] + (shift / payload["sigma_cell_draws"])[:, None],
        "gamma_size_draws": payload["gamma_size_draws"] + 100,
    })
    for before, after in zip(expected["contrast_decisions"]["rows"], changed["contrast_decisions"]["rows"]):
        assert before["median"] == pytest.approx(after["median"])
        assert before["lower_90"] == pytest.approx(after["lower_90"])
        assert before["upper_90"] == pytest.approx(after["upper_90"])


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
def test_only_contrasts_requiring_missing_cells_are_unavailable(group):
    payload, _, _ = _payload(group)
    expected = ca.posterior_fit_report(**payload)
    removed = payload["cell_ids"][4]
    payload["cell_ids"] = payload["cell_ids"][:4] + payload["cell_ids"][5:]
    payload["z_alpha_draws"] = np.delete(payload["z_alpha_draws"], 4, axis=1)
    actual = ca.posterior_fit_report(**payload)
    unavailable = 0
    for before, after in zip(expected["contrast_decisions"]["rows"], actual["contrast_decisions"]["rows"]):
        if removed in before["cell_weights"]:
            unavailable += 1
            assert after["status"] == "unavailable"
            assert after["missing_cell_ids"] == [removed]
            assert "no renormalization" in after["reason"]
            assert "median" not in after and "decision" not in after
        else:
            assert before == after
    assert 0 < unavailable < len(payload["contrasts"])
    assert actual["decision_count"] == expected["decision_count"]
    assert actual["contrast_decisions"]["unavailable_decision_count"] == unavailable
    comparison = ca.compare_presentation_reports(expected, actual, expected)
    assert comparison["all_estimands_comparable"] is False
    for row in comparison["rows"]:
        if row.get("status") == "unavailable":
            assert row["unavailable_variants"] == ["presentation_1_only"]
            assert row["sign_changed"] is None
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("field,value,message", [
    ("gamma_draws", np.zeros((0, 7)), "gamma_draws"),
    ("gamma_draws", np.zeros((31, 6)), "gamma_draws"),
    ("gamma_draws", np.full((31, 7), np.nan), "gamma_draws"),
    ("sigma_cell_draws", np.full(31, -1.0), "sigma_cell_draws"),
    ("sigma_cell_draws", np.full(31, np.inf), "sigma_cell_draws"),
    ("sigma_cell_draws", np.ones((31, 1)), "sigma_cell_draws"),
    ("sigma_cell_draws", np.ones(30), "sigma_cell_draws"),
    ("z_alpha_draws", np.zeros((30, 18)), "z_alpha_draws"),
    ("z_alpha_draws", np.zeros((31, 17)), "z_alpha_draws"),
    ("z_alpha_draws", np.full((31, 18), np.inf), "z_alpha_draws"),
    ("cell_ids", [], "nonempty and unique"),
    ("cell_ids", ["unknown"] * 18, "nonempty and unique"),
    ("cell_ids", [f"unknown-{index}" for index in range(18)], "canonical cell IDs"),
    ("gamma_size_draws", np.zeros(30), "gamma_size_draws"),
    ("gamma_size_draws", np.full(31, np.nan), "finite"),
    ("contrasts", [], "contrast specifications"),
])
def test_realized_input_validation(field, value, message):
    payload, _, _ = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        ca.posterior_fit_report(**payload)


@pytest.mark.parametrize("group", ["venture", "matched_rq5"])
def test_global_rank_check_still_rejects_loss_of_model_cells(group):
    payload, _, _ = _payload(group)
    payload["cell_ids"] = payload["cell_ids"][3:]
    payload["z_alpha_draws"] = payload["z_alpha_draws"][:, 3:]
    with pytest.raises(ValueError, match="design rank"):
        ca.posterior_fit_report(**payload)


@pytest.mark.parametrize("weights", [{}, {"venture": {}}, {"venture": {"unknown": 1.0}},
                                     {"venture": {"first": np.nan, "second": -1.0}}])
def test_no_implicit_gamma_fallback_or_malformed_weights(weights):
    payload, _, _ = _payload()
    payload["contrasts"] = [replace(payload["contrasts"][0], realized_cell_weights=weights)]
    with pytest.raises(ValueError, match="gamma fallback|cell weights"):
        ca.posterior_fit_report(**payload)


def _cross_pool_payload(venture, hiring):
    return {
        "venture_gamma_draws": venture["gamma_draws"],
        "hiring_gamma_draws": hiring["gamma_draws"],
        "contrasts": venture["contrasts"],
        "venture_diagnostics": DIAGNOSTICS, "hiring_diagnostics": DIAGNOSTICS,
        **{f"{pool}_{field}": payload[field]
           for pool, payload in (("venture", venture), ("hiring", hiring))
           for field in ("sigma_cell_draws", "z_alpha_draws", "cell_ids")},
    }


def test_rq4_uses_realized_model_averages_and_independent_draws():
    venture, venture_cells, _ = _payload("venture", 3)
    hiring, hiring_cells, _ = _payload("hiring", 2)
    target = config.MODELS[1].name
    for payload, cells, values in ((venture, venture_cells, [0.3, 0.6, 0.9]),
                                   (hiring, hiring_cells, [-0.6, -0.3])):
        payload["gamma_draws"][:] = 0
        payload["sigma_cell_draws"][:] = 1
        payload["z_alpha_draws"][:] = 0
        for index, cell in enumerate(cells):
            if cell.model_name == target:
                payload["z_alpha_draws"][:, index] = values
    report = ca.cross_pool_descriptive_report(**_cross_pool_payload(venture, hiring))
    row = report["rows"][0]
    assert row["venture"]["median"] == pytest.approx(0.6)
    assert row["hiring"]["median"] == pytest.approx(-0.45)
    assert row["posterior_probability_same_sign"] == 0
    expected_difference = ca.summarize_draws((np.array([-0.6, -0.3])[:, None] - [0.3, 0.6, 0.9]).ravel())
    for key in ("median", "lower_90", "upper_90"):
        assert row["hiring_minus_venture"][key] == pytest.approx(expected_difference[key])
    ordering = report["model_orderings"][0]
    assert ordering["venture"]["probability_second_model_greater"] == 1
    assert ordering["hiring"]["probability_first_model_greater"] == 1
    assert report["decision_count"] == 0
    assert report["independent_posterior_combination"]["combination_count"] == 6


def test_rq4_missing_cells_and_missing_realized_information():
    venture, _, _ = _payload("venture")
    hiring, _, _ = _payload("hiring")
    removed = venture["cell_ids"].pop(4)
    venture["z_alpha_draws"] = np.delete(venture["z_alpha_draws"], 4, axis=1)
    arguments = _cross_pool_payload(venture, hiring)
    report = ca.cross_pool_descriptive_report(**arguments)
    unavailable = [row for row in report["model_orderings"] if row["status"] == "unavailable"]
    assert len(unavailable) == 5
    assert all(row["missing_cell_ids_by_pool"] == {"venture": [removed], "hiring": []} for row in unavailable)
    assert all("hiring_minus_venture" not in row for row in unavailable)
    arguments.pop("venture_z_alpha_draws")
    with pytest.raises(ValueError, match="gamma fallback is prohibited"):
        ca.cross_pool_descriptive_report(**arguments)


def test_current_pool_and_matched_contract_counts_match_fit_plan(tmp_path):
    from applications.seu_sensitivity_study.ceiling_prior import fit_plan

    plan = fit_plan(tmp_path)
    design, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    manifests = [ca.contract_manifest(columns, design),
                 ca.matched_rq5_contract(config.build_cells(["venture", "hiring"]))]
    additional = sum(row["prior_variant"] != "primary" for row in plan["fits"])
    for manifest in manifests:
        contract = manifest["estimand_contract"]
        assert contract["fit_count"] == plan["planned_fit_count"] == len(plan["fits"])
        assert contract["additional_prior_fit_count"] == plan["additional_prior_fits"] == additional
        assert contract["base_fit_count"] + additional == contract["fit_count"]
        assert contract["version"] == ca.ESTIMAND_VERSION


def test_amendment5_contract_retains_names_and_distinct_family_count():
    design, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    manifest = ca.contract_manifest(columns, design)
    contract = manifest["estimand_contract"]
    assert contract["amendment"] == 5
    assert contract["version"] == ca.ESTIMAND_VERSION
    assert contract["fit_count"] == 24
    assert contract["base_fit_count"] == 15
    assert contract["additional_prior_fit_count"] == 9
    assert "planned, not authorized" in contract["fit_count_scope"]
    assert contract["observation_count_weighting"] is False
    assert contract["gamma_companion"]["decision_count"] == 0
    rows = manifest["primary_contrasts"]
    assert [row["contrast_id"] for row in rows] == [
        *(f"rq1_{model.slug}_minus_{config.MODELS[0].slug}" for model in config.MODELS[1:]),
        "rq1_openai_flagship_minus_small", "rq1_anthropic_flagship_minus_small",
        "rq2_seu_maximizing_minus_neutral", "rq2_deliberative_minus_neutral",
    ]
    distinct = set()
    for pool in ("venture", "hiring"):
        for row in rows:
            weights = row["realized_cell_weights"][pool]
            entries = tuple(sorted(weights.items()))
            reversed_entries = tuple((cell_id, -weight) for cell_id, weight in entries)
            distinct.add(min(entries, reversed_entries))
    assert len(distinct) + 2 + len(config.MODELS) == 24
    assert manifest["primary_decision_count"] == 26
    matched = ca.matched_rq5_contract(config.build_cells(["venture", "hiring"]))
    assert matched["estimand_contract"] == contract
    assert all(row["coefficient_estimand"] == "additive_gamma" for row in rows + matched["contrasts"])
    json.dumps(manifest, allow_nan=False)
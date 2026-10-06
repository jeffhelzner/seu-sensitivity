import copy
import json
from collections import Counter

import numpy as np
import pytest
from scipy.special import softmax

from applications.seu_sensitivity_study import assessment_scale as scale
from applications.seu_sensitivity_study import confirmatory_analysis as analysis
from applications.seu_sensitivity_study import config
from applications.seu_sensitivity_study import data_preparation as preparation
from applications.seu_sensitivity_study.pools import load_pool
from applications.seu_sensitivity_study.problem_generation import generate_problem_set
from applications.seu_sensitivity_study.tests.test_realized_estimands import _payload


def fixed_inputs(group, *, varied=False):
    pool_ids = ["venture", "hiring"] if group == "matched_rq5" else [group]
    items, problems = [], []
    probabilities = {model.name: {} for model in config.MODELS}
    for pool_id in pool_ids:
        family = {"venture": "procurement", "hiring": "matched"}[pool_id]
        selected = sorted([item for item in load_pool(pool_id)["items"]
                           if group != "matched_rq5" or item["family"] == family], key=lambda item: item["id"])
        items.extend(selected)
        generated = generate_problem_set(
            load_pool(pool_id), problems_per_family=config.DEFAULT_PROBLEMS_PER_FAMILY[pool_id], seed=42,
        )
        problems.extend(problem for problem in generated["problems"]
                        if group != "matched_rq5" or problem["family"] == family)
        for model_index, model in enumerate(config.MODELS):
            for item_index, item in enumerate(selected):
                vector = ([0.8, 0.0, 0.2], [0.0, 1.0, 0.0], [0.2, 0.0, 0.8])[item_index % 3]
                if varied:
                    spread = (0.1 + 0.04 * model_index) * (1.5 if pool_id == "hiring" else 1)
                    eta = 0.5 + spread * np.sin(item_index + model_index)
                    vector = [1 - eta, 0, eta]
                probabilities[model.name][item["id"]] = vector
    return {"group": group, "items": items, "problems": problems, "probabilities": probabilities}


def test_assessment_scale_contract_preserves_amendment5_estimand():
    design, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    contracts = [analysis.contract_manifest(columns, design),
                 analysis.matched_rq5_contract(config.build_cells(["venture", "hiring"]))]
    for contract in contracts:
        assert contract["estimand_contract"]["amendment"] == 5
        assert contract["assessment_scale"]["policy_version"] == "amendment6_assessment_scale_v1"
        assert contract["assessment_scale"]["utility_middle"] == 0.5
        assert contract["assessment_scale"]["included_in_primary_family"] is False


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
def test_fixed_reference_geometry_and_realized_transforms(group):
    inputs = fixed_inputs(group, varied=True)
    reference = scale.build_reference(**inputs)
    assert scale.validate_reference(reference, group=group) == reference
    payload, _, realized = _payload(group)
    report = analysis.posterior_fit_report(**payload, assessment_scale_reference=reference)
    for arm in reference["arms"]:
        assert arm["item_count"] == (24 if group == "matched_rq5" else 60)
        assert arm["sd"] == pytest.approx(np.std(arm["eta"], ddof=0))
        assert arm["mean"] == pytest.approx(np.mean(arm["eta"]))
        assert arm["menu_count"] == (40 if group == "matched_rq5" else 140)
    arms = {(arm["pool_id"], arm["model_name"]): arm for arm in reference["arms"]}
    cells = {cell.cell_id: cell for cell in config.build_cells(["venture", "hiring"])}
    log_scales = np.array([arms[cells[cell_id].pool_id, cells[cell_id].model_name]["log_sd"] for cell_id in payload["cell_ids"]])
    for row in report["assessment_scale"]["rows"]:
        expected, _ = analysis._cell_contrast_values(realized + log_scales, payload["cell_ids"], row["cell_weights"])
        np.testing.assert_allclose([row["standardized"][key] for key in ("lower_90", "median", "upper_90")], np.quantile(expected, [0.05, 0.5, 0.95]), atol=1e-14)
        assert "decision" not in row["standardized"]
        assert "rope_half_width" not in row["standardized"]
        if row["research_question"] == "RQ2":
            assert row["deterministic_offset"] == 0.0
            assert row["original"] == row["standardized"]
    slope = report["assessment_scale"]["rq6"]
    assert slope["original"] == slope["standardized"]
    assert slope["deterministic_offset"] == 0.0
    assert report["decision_count"] == (7 if group == "matched_rq5" else 10)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("alpha", [1e-6, 1.0, 50.0, 1e6])
def test_softmax_is_preserved_at_every_menu_size(alpha):
    reference = scale.build_reference(**fixed_inputs("venture", varied=True))
    for arm in reference["arms"]:
        eta = np.asarray(arm["eta"])
        for count in (2, 4, 6, 8):
            logits = alpha * (eta[:count] - eta[:count].max())
            standardized = (eta[:count] - arm["mean"]) / arm["sd"]
            transformed = alpha * arm["sd"] * (standardized - standardized.max())
            np.testing.assert_allclose(softmax(logits), softmax(transformed), atol=2e-14, rtol=2e-13)


def test_zero_sd_only_disables_affected_standardized_rows_and_keeps_primary():
    inputs = fixed_inputs("venture", varied=True)
    model = config.MODELS[1].name
    inputs["probabilities"][model] = {item["id"]: [0.0, 1.0, 0.0] for item in inputs["items"]}
    reference = scale.build_reference(**inputs)
    payload, cells, _ = _payload()
    report = analysis.posterior_fit_report(**payload, assessment_scale_reference=reference)
    for row, primary in zip(report["assessment_scale"]["rows"], report["contrast_decisions"]["rows"]):
        affected = any(cell.model_name == model and cell.cell_id in row["cell_weights"] for cell in cells)
        assert (row["status"] == "unavailable") == affected
        assert primary["status"] == "available"
        if affected:
            assert row["original"] is not None and row["standardized"] is None
            assert row["deterministic_offset"] is None
    assert report["assessment_scale"]["rq6"]["slope_unchanged"]
    json.dumps(report, allow_nan=False)


def test_offset_flags_have_no_detection_rule():
    reversed_row = analysis._scale_comparison(np.linspace(0.1, 0.3, 101), -0.5)
    assert reversed_row["sign_reversal"] is True
    assert reversed_row["interval_zero_exclusion_changed"] is False
    crossing_row = analysis._scale_comparison(np.linspace(0.1, 0.3, 101), -0.2)
    assert crossing_row["interval_zero_exclusion_changed"] is True
    assert "decision" not in json.dumps(crossing_row)
    assert "rope" not in json.dumps(crossing_row)


def test_reference_is_invariant_to_input_order_and_matched_outside_items():
    inputs = fixed_inputs("matched_rq5", varied=True)
    reference = scale.build_reference(**inputs)
    inputs["items"].reverse()
    inputs["problems"].reverse()
    for model in config.MODELS:
        inputs["probabilities"][model.name]["unused-padding"] = [1.0, 0.0, 0.0]
    assert scale.build_reference(**inputs) == reference
    assert all(arm["item_count"] == 24 for arm in reference["arms"])


@pytest.mark.parametrize("mutation", [
    lambda reference: reference["policy"].__setitem__("utility_middle", 0.35),
    lambda reference: reference["policy"].__setitem__("policy_version", "stale"),
    lambda reference: reference["arms"][0].__setitem__("sd", 0.1),
    lambda reference: reference["arms"][0]["eta"].__setitem__(0, float("nan")),
    lambda reference: reference["arms"].pop(),
    lambda reference: reference["items"].pop(),
    lambda reference: reference["items"][0].__setitem__("family", "wrong"),
    lambda reference: reference["arms"][0]["probabilities"].pop(),
    lambda reference: reference["menus"].pop(),
])
def test_rejects_missing_stale_malformed_reference(mutation):
    reference = scale.build_reference(**fixed_inputs("venture"))
    mutation(reference)
    with pytest.raises(ValueError, match="assessment_scale"):
        scale.validate_reference(reference, group="venture")
    with pytest.raises(ValueError, match="assessment_scale"):
        scale.validate_reference(None, group="venture")


def test_ties_use_explicit_absolute_tolerance():
    inputs = fixed_inputs("venture")
    second = inputs["problems"][0]["item_ids"][1]
    for probabilities in inputs["probabilities"].values():
        probabilities.update({item["id"]: [0.5, 0, 0.5] for item in inputs["items"]})
        probabilities[second] = [0.5 - 5e-13, 0, 0.5 + 5e-13]
    reference = scale.build_reference(**inputs)
    assert all(arm["tie_prevalence"] == 1.0 for arm in reference["arms"])
    menu_index = next(index for index, menu in enumerate(reference["menus"])
                      if menu["id"] == inputs["problems"][0]["id"])
    assert all(arm["top_two_gaps"][menu_index] > 0 for arm in reference["arms"])
    assert reference["policy"]["tie_rtol"] == 0
    assert reference["policy"]["ppc_tie_policy_changed"] is False


@pytest.mark.parametrize("group", ["venture", "matched_rq5"])
def test_missing_cells_and_reordered_draws_do_not_redefine_reference(group):
    reference = scale.build_reference(**fixed_inputs(group, varied=True))
    payload, _, _ = _payload(group)
    original = analysis.posterior_fit_report(**payload, assessment_scale_reference=reference)
    missing = payload["cell_ids"].pop(4)
    payload["z_alpha_draws"] = np.delete(payload["z_alpha_draws"], 4, axis=1)
    payload["cell_ids"].reverse()
    payload["z_alpha_draws"] = payload["z_alpha_draws"][:, ::-1]
    changed = analysis.posterior_fit_report(**payload, assessment_scale_reference=reference)
    assert changed["assessment_scale"]["reference"] == original["assessment_scale"]["reference"]
    for before, after in zip(original["assessment_scale"]["rows"], changed["assessment_scale"]["rows"]):
        assert before["deterministic_offset"] == after["deterministic_offset"]
        if missing in before["cell_weights"]:
            assert after["status"] == "unavailable"
            assert after["original"] is None and after["standardized"] is None
        else:
            assert before == after


@pytest.mark.parametrize("draw_count", [31, 401])
def test_rq4_transforms_preserve_independence_and_whole_draw_uncertainty(draw_count):
    venture, _, _ = _payload("venture", draw_count)
    hiring, _, _ = _payload("hiring", draw_count + 3)
    references = {pool: scale.build_reference(**fixed_inputs(pool, varied=True)) for pool in ("venture", "hiring")}
    kwargs = {f"{pool}_{name}": payload[name] for pool, payload in (("venture", venture), ("hiring", hiring))
              for name in ("sigma_cell_draws", "z_alpha_draws", "cell_ids")}
    report = analysis.cross_pool_descriptive_report(
        venture["gamma_draws"], hiring["gamma_draws"], venture["contrasts"],
        venture["diagnostics"], hiring["diagnostics"], **kwargs, assessment_scale_references=references,
    )
    assert report["independent_posterior_combination"]["shared_whole_draw_indices_across_estimands"]
    for row in report["rows"] + report["model_orderings"]:
        result = row["assessment_scale"]
        difference = result["hiring_minus_venture"]
        offset = difference["deterministic_offset"]
        for key in ("lower_90", "median", "upper_90"):
            assert difference["standardized"][key] == pytest.approx(row["hiring_minus_venture"][key] + offset)
        assert difference["standardized"]["upper_90"] - difference["standardized"]["lower_90"] == pytest.approx(
            difference["original"]["upper_90"] - difference["original"]["lower_90"])
        venture_values, _ = analysis._realized_fit_values(venture["gamma_draws"], venture["sigma_cell_draws"], venture["z_alpha_draws"], venture["contrasts"], venture["cell_ids"])
        hiring_values, _ = analysis._realized_fit_values(hiring["gamma_draws"], hiring["sigma_cell_draws"], hiring["z_alpha_draws"], venture["contrasts"], hiring["cell_ids"])
        venture_contrast, _ = analysis._cell_contrast_values(venture_values, venture["cell_ids"], row["cell_weights_by_pool"]["venture"])
        hiring_contrast, _ = analysis._cell_contrast_values(hiring_values, hiring["cell_ids"], row["cell_weights_by_pool"]["hiring"])
        venture_contrast += result["by_pool"]["venture"]["deterministic_offset"]
        hiring_contrast += result["by_pool"]["hiring"]["deterministic_offset"]
        expected_same = np.mean(venture_contrast > 0) * np.mean(hiring_contrast > 0) + np.mean(venture_contrast < 0) * np.mean(hiring_contrast < 0)
        assert result["standardized"]["posterior_probability_same_sign"] == expected_same
        if draw_count == 31:
            expected = (hiring_contrast[None, :] - venture_contrast[:, None]).ravel()
            np.testing.assert_allclose([difference["standardized"][key] for key in ("lower_90", "median", "upper_90")],
                                       np.quantile(expected, [0.05, 0.5, 0.95]), atol=1e-14)
    assert report["decision_count"] == 0


@pytest.mark.parametrize("change", ["missing_observed_items", "duplicated_observations", "excluded_cell", "reordered_items"])
def test_preparation_reference_precedes_all_observation_exclusions(change):
    inputs = fixed_inputs("venture", varied=True)
    design, columns, cell_ids = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    cells = config.build_cells(["venture"])
    records = [{"problem_id": problem["id"], "presentation_id": presentation["presentation_id"], "menu_size": len(problem["item_ids"]),
                "difficulty_stratum": problem["difficulty_stratum"], "family": problem["family"],
                "chosen_position": 1, "chosen_item_id": presentation["order"][0], "resolution_path": "answer_token"}
               for problem in inputs["problems"] for presentation in problem["presentations"]]
    choices = {cell_id: {"cell_id": cell_id, "pool_id": "venture", "choices": copy.deepcopy(records)} for cell_id in cell_ids}
    common = {
        "pool": {"items": inputs["items"]}, "problem_set": {"pool_id": "venture", "problems": inputs["problems"]},
        "choice_sets": choices, "reduced_embeddings": {item["id"]: np.zeros(1) for item in inputs["items"]},
        "design_matrix": design, "cell_ids": cell_ids, "K": 3, "include_menu_size": True,
        "assessment_probabilities": inputs["probabilities"], "cell_model_names": [cell.model_name for cell in cells],
        "utility_values": [0, 0.5, 1], "design_column_names": columns, "include_assessment_scale_reference": True,
    }
    before_data, before = preparation.build_stan_data(**common)
    if change == "missing_observed_items":
        for choice_set in choices.values():
            for record in choice_set["choices"][:20]:
                record.update(chosen_item_id=None, chosen_position=None, resolution_path="unresolved")
    elif change == "duplicated_observations":
        choices[cell_ids[0]]["choices"] *= 4
        with pytest.raises(ValueError, match="missing or unexpected collection"):
            preparation.build_stan_data(**common)
        return
    elif change == "excluded_cell":
        for record in choices[cell_ids[-1]]["choices"]:
            record.update(chosen_item_id=None, chosen_position=None, resolution_path="unresolved")
    else:
        common["reduced_embeddings"] = dict(reversed(list(common["reduced_embeddings"].items())))
        for choice_set in choices.values():
            choice_set["choices"].reverse()
    after_data, after = preparation.build_stan_data(**common)
    assert before["assessment_scale_reference"] == after["assessment_scale_reference"]
    assert all(arm["item_count"] == 60 and arm["menu_count"] == 140 for arm in after["assessment_scale_reference"]["arms"])
    if change == "excluded_cell":
        assert after_data["J"] == before_data["J"] - 1
    elif change != "reordered_items":
        assert after_data["M_total"] != before_data["M_total"]


def test_preparation_rejects_missing_assessments_even_for_excluded_cells():
    inputs = fixed_inputs("venture")
    inputs["probabilities"].pop(config.MODELS[-1].name)
    with pytest.raises(ValueError, match="complete finite assessment"):
        scale.build_reference(**inputs)


def test_reference_rejects_inconsistent_unobserved_presentation():
    inputs = fixed_inputs("venture")
    inputs["problems"][0]["presentations"][1]["order"] = ["not-a-fixed-item"]
    with pytest.raises(ValueError, match="fixed presentations"):
        scale.build_reference(**inputs)


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
@pytest.mark.parametrize("mutation", ["remove", "add", "deduplicate", "wrong_family", "wrong_size"])
def test_rebuild_rejects_incomplete_or_misallocated_full_menus(group, mutation):
    inputs = fixed_inputs(group)
    reference = scale.build_reference(**inputs)
    menus = reference["menus"]
    if mutation == "remove":
        menus.pop()
    elif mutation == "add":
        menus.append({**menus[0], "id": "extra-menu"})
    elif mutation == "deduplicate":
        unique = {(menu["pool_id"], tuple(menu["item_ids"])): menu for menu in menus}
        menus[:] = unique.values()
        assert len(menus) < (80 if group == "matched_rq5" else 140)
    elif mutation == "wrong_family":
        replacement = next(menu for menu in menus
                           if menu["pool_id"] != menus[0]["pool_id"] or
                           next(item["family"] for item in reference["items"] if item["id"] == menu["item_ids"][0]) !=
                           next(item["family"] for item in reference["items"] if item["id"] == menus[0]["item_ids"][0]))
        menus[0]["item_ids"] = replacement["item_ids"][:]
    else:
        menus[0]["item_ids"] = next(menu["item_ids"][:] for menu in menus
                                      if menu["pool_id"] == menus[0]["pool_id"] and len(menu["item_ids"]) == 4)
    with pytest.raises(ValueError, match="full fixed menus"):
        scale.build_reference(group=group, items=reference["items"], problems=menus,
                              probabilities=inputs["probabilities"])
    with pytest.raises(ValueError, match="full fixed menus"):
        scale.validate_reference(reference, group=group)


@pytest.mark.parametrize("group", ["venture", "hiring", "matched_rq5"])
def test_seed42_deduplication_rejects_rebuild_with_unchanged_observations(group):
    inputs = fixed_inputs(group, varied=True)
    reference = scale.build_reference(**inputs)
    item_ids = [item["id"] for item in reference["items"]]
    pool_ids = ["venture", "hiring"] if group == "matched_rq5" else [group]
    cells = [config.build_cells([pool_id])[0] for pool_id in pool_ids]
    observations = [(cell_index, menu) for cell_index, cell in enumerate(cells, start=1)
                    for menu in reference["menus"] if menu["pool_id"] == cell.pool_id]
    data = {
        "R": len(item_ids), "utility_values": [0.0, 0.5, 1.0],
        "eta": [preparation.assessment_expected_utilities(
            inputs["probabilities"][cell.model_name], item_ids=item_ids, utilities=[0.0, 0.5, 1.0],
        ) for cell in cells],
        "cell": [cell_index for cell_index, menu in observations],
        "I": [[int(item_id in menu["item_ids"]) for item_id in item_ids] for cell_index, menu in observations],
    }
    report = {"item_ids": item_ids, "cell_ids": [cell.cell_id for cell in cells]}
    scale.validate_retained_data(reference, data, report, group=group)
    original_data = copy.deepcopy(data)
    support = {(menu["pool_id"], tuple(menu["item_ids"])) for menu in reference["menus"]}
    reference["menus"] = list({(menu["pool_id"], tuple(menu["item_ids"])): menu
                               for menu in reference["menus"]}.values())
    assert Counter(menu["pool_id"] for menu in reference["menus"]) == {
        pool_id: 39 if group == "matched_rq5" else 137 for pool_id in pool_ids
    }
    assert {(menu["pool_id"], tuple(menu["item_ids"])) for menu in reference["menus"]} == support
    with pytest.raises(ValueError, match="full fixed menus"):
        scale.build_reference(group=group, items=reference["items"], problems=reference["menus"],
                              probabilities=inputs["probabilities"])
    with pytest.raises(ValueError, match="full fixed menus"):
        scale.validate_retained_data(reference, data, report, group=group)
    assert data == original_data
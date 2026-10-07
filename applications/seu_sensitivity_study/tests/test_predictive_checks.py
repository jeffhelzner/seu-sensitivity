import numpy as np
import pytest
from copy import deepcopy
from types import SimpleNamespace

from applications.seu_sensitivity_study import predictive_checks as checks


def test_scalar_exact_tails_and_zero_width_band():
    summary = checks.scalar_summary(1, [0, 1, 2, 1])
    assert summary["tails"] == {"less": .25, "equal": .5, "greater": .25}
    assert summary["observed_minus_replicated"]["q50"] == 0
    assert not checks.scalar_summary(0, [0, 0])["descriptive_review_flag"]
    assert checks.scalar_summary(1, [0, 0])["descriptive_review_flag"]


def test_scalar_scores_use_within_draw_difference():
    summary = checks.scalar_summary([1, 101], [0, 100])
    assert summary["observed_minus_replicated"] == {"q05": 1., "q50": 1., "q95": 1.}
    assert summary["descriptive_review_flag"]
    assert summary["draw_dependent"]
    assert summary["difference_from_predictive_median"] is None


def test_scalar_tails_keep_one_ulp_differences():
    summary = checks.scalar_summary(1., [np.nextafter(1., 0.), 1., np.nextafter(1., 2.)])
    assert summary["tails"] == {"less": 1 / 3, "equal": 1 / 3, "greater": 1 / 3}
    assert checks.scalar_summary(1., [np.nextafter(1., 0.)])["descriptive_review_flag"]


def test_bound_roles_validate_full_recipe_and_canonical_labels():
    from applications.seu_sensitivity_study import data_preparation, pools, problem_generation
    from applications.seu_sensitivity_study.tests.test_assessment_scale import fixed_inputs
    from applications.seu_sensitivity_study import assessment_scale
    from copy import deepcopy

    pool = pools.load_pool("venture")
    problems = problem_generation.generate_problem_set(pool, problems_per_family={"startup": 100, "procurement": 40}, seed=42)
    inputs = fixed_inputs("venture")
    reference = assessment_scale.build_reference(**{**inputs, "problems": problems["problems"]})
    frozen = data_preparation.build_observation_reference(reference, problems["problems"])
    result = data_preparation.build_predictive_reference(reference, frozen, pool["items"])
    assert all(len(menu["filler_item_ids"]) == menu["menu_size"] - 2 for menu in result["menus"])
    tampered = deepcopy(pool["items"])
    tampered[0]["quality_label"] = "not-canonical"
    with pytest.raises(ValueError, match="canonical items"):
        data_preparation.build_predictive_reference(reference, frozen, tampered)
    tampered = deepcopy(frozen)
    stratum = tampered["menus"][0]["difficulty_stratum"]
    tampered["menus"][0]["difficulty_stratum"] = "weak" if stratum != "weak" else "strong"
    with pytest.raises(ValueError, match="full frozen"):
        data_preparation.build_predictive_reference(reference, tampered, pool["items"])


@pytest.mark.parametrize("observed,replicated", [(0, []), ([0, 1], [1]), (np.nan, [1])])
def test_scalar_rejects_invalid_draws(observed, replicated):
    with pytest.raises(ValueError, match="finite matched"):
        checks.scalar_summary(observed, replicated)


@pytest.fixture
def small_design(monkeypatch):
    from applications.seu_sensitivity_study import config

    monkeypatch.setattr(config, "build_cells", lambda _: [SimpleNamespace(cell_id=name, pool_id="venture")
                                                         for name in ("first", "second")])
    item_ids = [f"item-{index}" for index in range(9)]
    items = [{"id": item_id, "pool_id": "venture", "family": "startup",
              "quality_label": "strong" if index == 0 else "ambiguous" if index == 1 else "weak"}
             for index, item_id in enumerate(item_ids)]
    menus, frozen, observations = [], [], []
    for size in checks.SIZES:
        active = item_ids[:size]
        menus.append({"id": f"menu-{size}", "pool_id": "venture", "family": "startup",
                      "difficulty_stratum": "strong", "menu_size": size, "filler_item_ids": active[2:]})
        frozen.append({"id": f"menu-{size}", "presentations": [{"presentation_id": 1, "order": active[::-1]},
                                                                {"presentation_id": 2, "order": active}]})
        for cell_id in ("first", "second"):
            for presentation in (1, 2):
                order = active[::-1] if presentation == 1 else active
                chosen = active[-1]
                observations.append({"cell_id": cell_id, "problem_id": f"menu-{size}", "menu_size": size,
                                     "presentation_id": presentation, "difficulty_stratum": "strong",
                                     "item_order": order, "chosen_item_id": chosen,
                                     "chosen_position": order.index(chosen) + 1, "resolution_path": "answer_token"})
    data = {"J": 2, "R": 9, "M_total": len(observations), "eta": [[.5, .5, .8, .1, .2, .3, .4, .6, .9]] * 2,
            "I": [[int(index < row["menu_size"]) for index in range(9)] for row in observations],
            "cell": [1 if row["cell_id"] == "first" else 2 for row in observations],
            "y": [row["menu_size"] for row in observations]}
    preparation = {"cell_ids": ["first", "second"], "item_ids": item_ids, "presentation_id": None,
                   "observations": observations, "exclusions": [], "frozen_observation_reference": {"menus": frozen}}
    reference = {"items": items, "menus": menus}
    predicted = np.ones((4, len(observations)), dtype=int)
    predicted[1] = 2
    predicted[2] = data["y"]
    predicted[3] = [row["menu_size"] if row["presentation_id"] == 1 else 1 for row in observations]
    alpha = np.tile(np.array([1., 2., 3., 4.])[:, None], (1, len(observations)))
    return data, preparation, reference, predicted, alpha


def select(rows, **values):
    return next(row for row in rows if all(row.get(key) == value for key, value in values.items()))


def test_hand_computable_groups_exact_ties_fillers_and_scores(small_design):
    data, preparation, reference, predicted, alpha = small_design
    report = checks._calculate(*small_design)
    size2 = select(report["groups"], scope="cell", id="first", menu_size=2, family=None)
    assert size2["statistics"]["maximizer_fraction"]["observed"] == 1
    assert size2["statistics"]["maximizer_fraction"]["tails"]["equal"] == 1
    filler = size2["statistics"]["filler_fraction"]
    assert filler["structural_zero"] and filler["observed"] == filler["replicated"]["q95"] == 0
    size4 = select(report["groups"], scope="cell", id="first", menu_size=4, family=None)
    assert size4["statistics"]["filler_fraction"]["observed"] == 1
    assert size4["observations_containing_fillers"] == 2
    assert size4["statistics"]["mean_regret"]["observed"] == pytest.approx(.7)
    probabilities = np.exp(alpha[:, 4, None] * np.array([-.3, -.3, 0, -.7]))
    probabilities /= probabilities.sum(axis=1)[:, None]
    assert size4["conditional_filler_probability"] == pytest.approx(checks.quantiles(probabilities[:, 2:].sum(axis=1)))
    expected = np.log(probabilities[:, 3])
    assert size4["statistics"]["mean_log_score"]["observed"] == pytest.approx(checks.quantiles(expected))
    predicted[:, 4:8] = 3
    report = checks._calculate(data, preparation, reference, predicted, alpha)
    size4 = select(report["groups"], scope="cell", id="first", menu_size=4, family=None)
    assert size4["statistics"]["maximizer_fraction"]["replicated"]["q05"] == 1
    assert size4["statistics"]["filler_fraction"]["replicated"]["q05"] == 1


def test_item_exposures_unconditional_shares_and_display_mapping(small_design):
    report = checks._calculate(*small_design)
    unused = select(report["items"], scope="cell", id="first", item_id="item-8")
    assert unused["exposure_count"] == 0
    assert unused["conditional_rate"]["status"] == "unavailable"
    assert unused["unconditional_share"]["observed"] == 0
    assert select(report["items"], scope="cell", id="first", item_id="item-7")["exposure_count"] == 2
    assert select(report["items"], scope="cell", id="first", item_id="item-0")["exposure_count"] == 8
    assert sum(row["unconditional_share"]["observed"] for row in report["items"] if row["scope"] == "cell" and row["id"] == "first") == 1
    position = select(report["positions"], scope="cell", id="first", menu_size=4, presentation_id=1, position=1)
    assert position["fraction"]["observed"] == 1
    position = select(report["positions"], scope="cell", id="first", menu_size=4, presentation_id=2, position=4)
    assert position["fraction"]["observed"] == 1


def test_pairs_use_same_saved_draw_and_conditional_products(small_design):
    report = checks._calculate(*small_design)
    pair = select(report["pairs"], scope="cell", id="first", family="startup", menu_size=4)
    assert pair["complete_pair_count"] == 1
    assert pair["same_item"]["observed"] == 1
    assert pair["same_position"]["observed"] == 0
    assert pair["same_item"]["replicated"] == checks.quantiles([1, 1, 1, 0])
    assert pair["same_position"]["replicated"] == checks.quantiles([0, 0, 0, 1])
    from scipy.special import softmax
    probability = softmax(np.arange(1, 5)[:, None] * np.array([.5, .5, .8, .1]), axis=1)
    assert pair["conditional_same_item"] == pytest.approx(checks.quantiles((probability ** 2).sum(axis=1)))
    assert pair["conditional_same_position"] == pytest.approx(checks.quantiles((probability * probability[:, ::-1]).sum(axis=1)))


def remove_rows(design, indices, *, whole=False):
    data, preparation, reference, predicted, alpha = deepcopy(design)
    indices = set(indices)
    keep = [index for index in range(len(preparation["observations"])) if index not in indices]
    for index in indices:
        row = preparation["observations"][index]
        if not whole or index % 2 == 0:
            row.update(chosen_position=None, chosen_item_id=None, resolution_path="unresolved")
        row["reason"] = "cell_na_exclusion" if whole else "unresolved_choice"
        preparation["exclusions"].append(row)
    preparation["observations"] = [preparation["observations"][index] for index in keep]
    for name in ("I", "cell", "y"):
        data[name] = [data[name][index] for index in keep]
    data["M_total"] = len(keep)
    return data, preparation, reference, predicted[:, keep], alpha[:, keep]


def test_incomplete_pairs_whole_cells_and_missing_size_denominators(small_design):
    design = remove_rows(small_design, [0, 4, 5])
    report = checks._calculate(*design)
    pairs = select(report["pairs"], scope="cell", id="first", family=None)
    assert (pairs["complete_pair_count"], pairs["single_retained_presentation_count"], pairs["neither_retained_pair_count"]) == (2, 1, 1)
    trend = select(report["trends"], scope="cell", id="first", difficulty_stratum="strong")
    assert trend["statistics"]["mean_regret"]["status"] == "unavailable"
    missing = select(report["missingness"]["size_unresolved"], scope="cell", id="first")
    assert missing["by_size"]["4"]["eligible"] == 2
    assert missing["by_size"]["4"]["unresolved_fraction"] == 1
    assert missing["range"] == 1
    whole_indices = [index for index, row in enumerate(small_design[1]["observations"]) if row["cell_id"] == "second"]
    report = checks._calculate(*remove_rows(small_design, whole_indices, whole=True))
    missing = select(report["missingness"]["groups"], scope="cell", id="second", menu_size=None, presentation_id=None)
    assert missing["eligible"] == missing["excluded_total"] == missing["whole_cell_excluded_total"] == 8
    assert missing["resolved_whole_cell_removed"] == missing["unresolved"] == 4
    pairs = select(report["pairs"], scope="pool_task", family=None)
    assert pairs["neither_retained_pair_count"] == 4
    empty = select(report["groups"], scope="cell", id="second", menu_size=None)
    assert empty["observation_count"] == 0
    assert empty["statistics"]["mean_regret"]["status"] == "unavailable"


def test_presentation_only_pair_checks_unavailable_by_design(small_design):
    design = list(small_design)
    design[1]["presentation_id"] = 1
    keep = [index for index, row in enumerate(design[1]["observations"]) if row["presentation_id"] == 1]
    design[1]["observations"] = [design[1]["observations"][index] for index in keep]
    for field in ("I", "cell", "y"):
        design[0][field] = [design[0][field][index] for index in keep]
    design[3], design[4] = design[3][:, keep], design[4][:, keep]
    report = checks._calculate(*design)
    pairs = select(report["pairs"], scope="cell", id="first", family=None)
    assert pairs["same_item"]["status"] == "unavailable"
    assert "by design" in pairs["same_item"]["reason"]
    assert pairs["single_retained_presentation_count"] == 4


def test_equal_size_trend_and_mismatch_hidden_by_pooled_mean():
    actual = dict(zip(checks.SIZES, (0, 0, 1, 1)))
    replicas = {size: np.full(4, value) for size, value in zip(checks.SIZES, (1, 1, 0, 0))}
    assert np.mean(list(actual.values())) == np.mean(list(replicas.values()))
    trend = checks.size_trend(actual, replicas)
    assert trend["observed"] == .2
    assert trend["replicated"]["q50"] == -.2
    assert trend["descriptive_review_flag"]
    modeled = checks.size_trend(actual, {size: np.full(4, value) for size, value in actual.items()})
    assert modeled["replicated"]["q50"] != 0
    assert not modeled["descriptive_review_flag"]


def test_extreme_probabilities_finite_scores_and_exact_not_near_ties(small_design):
    data, preparation, reference, predicted, alpha = small_design
    data["eta"][0][0] = .5 - 1e-14
    alpha[:] = 1e300
    report = checks._calculate(data, preparation, reference, predicted, alpha)
    import json
    json.dumps(report, allow_nan=False)
    size2 = select(report["groups"], scope="cell", id="first", menu_size=2, family=None)
    assert size2["statistics"]["maximizer_fraction"]["replicated"]["q05"] == pytest.approx(.075)
    assert size2["statistics"]["mean_selected_probability"]["replicated"]["q05"] == pytest.approx(.075)


def test_item_and_cell_permutations_leave_summaries_unchanged(small_design):
    baseline = checks._calculate(*small_design)
    data, preparation, reference, predicted, alpha = deepcopy(small_design)
    order = list(reversed(range(data["R"])))
    old_indicators = data["I"]
    data["I"] = [[row[index] for index in order] for row in old_indicators]
    data["eta"] = [[row[index] for index in order] for row in data["eta"]][::-1]
    data["cell"] = [3 - cell for cell in data["cell"]]
    preparation["cell_ids"].reverse()
    preparation["item_ids"].reverse()
    for index, row in enumerate(old_indicators):
        size = sum(row)
        data["y"][index] = size + 1 - data["y"][index]
        predicted[:, index] = size + 1 - predicted[:, index]
    changed = checks._calculate(data, preparation, reference, predicted, alpha)
    for section in ("groups", "items", "positions", "pairs", "trends"):
        if section == "items":
            changed[section].sort(key=lambda row: (row["scope"], row["id"], row["item_id"]))
            baseline[section].sort(key=lambda row: (row["scope"], row["id"], row["item_id"]))
        assert_nested_close(changed[section], baseline[section])


def assert_nested_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected):
            assert_nested_close(actual_row, expected_row)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-14)
    else:
        assert actual == expected


@pytest.fixture
def production_design():
    from applications.seu_sensitivity_study import config, pools, problem_generation, data_preparation
    from applications.seu_sensitivity_study.tests.test_assessment_scale import fixed_inputs

    study = config.SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])
    pool = pools.load_pool("venture")
    problems = problem_generation.generate_problem_set(pool, problems_per_family=study.problems_for("venture"), seed=42)
    design, columns, cell_ids = study.design_matrix_for_pool("venture")
    cells = study.cells_for_pool("venture")
    records = [{"problem_id": problem["id"], "presentation_id": presentation["presentation_id"],
                "menu_size": problem["menu_size"], "difficulty_stratum": problem["difficulty_stratum"],
                "family": problem["family"], "chosen_position": 1, "chosen_item_id": presentation["order"][0],
                "resolution_path": "answer_token"}
               for problem in problems["problems"] for presentation in problem["presentations"]]
    choices = {cell.cell_id: {"cell_id": cell.cell_id, "pool_id": "venture", "choices": records} for cell in cells}
    return data_preparation.build_stan_data(
        pool=pool, problem_set=problems, choice_sets=choices,
        reduced_embeddings={item["id"]: np.zeros(1) for item in pool["items"]},
        design_matrix=design, cell_ids=cell_ids, K=3, include_menu_size=True,
        assessment_probabilities=fixed_inputs("venture", varied=True)["probabilities"],
        cell_model_names=[cell.model_name for cell in cells], utility_values=[0., .5, 1.],
        design_column_names=columns, include_assessment_scale_reference=True)


def assert_exact_replica_agreement(value):
    if isinstance(value, dict):
        if "tails" in value:
            assert value["tails"] == {"less": 0., "equal": 1., "greater": 0.}
            assert value["observed_minus_replicated"] == {"q05": 0., "q50": 0., "q95": 0.}
            assert value["difference_from_predictive_median"] in (None, 0.)
            assert value["descriptive_review_flag"] is False
        for child in value.values():
            assert_exact_replica_agreement(child)
    elif isinstance(value, list):
        for child in value:
            assert_exact_replica_agreement(child)


def test_identical_production_replicas_have_exact_equality_mass(production_design):
    data, preparation = production_design
    predicted = np.tile(data["y"], (4, 1))
    alpha = np.exp(np.linspace(-2, 2, predicted.size).reshape(predicted.shape))
    report = checks.build_predictive_report(data, preparation, predicted, alpha)
    assert_exact_replica_agreement(report)
    assert not any(value["applies"] for value in checks.interpretation(report)["qualifications"].values())


@pytest.mark.parametrize("permutation", ["reverse", "shuffle"])
@pytest.mark.parametrize("identical", [True, False])
def test_production_observation_permutation_is_exact(production_design, permutation, identical):
    data, preparation = production_design
    predicted = np.tile(data["y"], (4, 1))
    if not identical:
        sizes = np.asarray(data["I"]).sum(axis=1)
        predicted = (predicted - 1 + np.arange(4)[:, None]) % sizes + 1
    alpha = np.exp(np.linspace(-2, 2, predicted.size).reshape(predicted.shape))
    baseline = checks.build_predictive_report(data, preparation, predicted, alpha)
    order = (np.arange(data["M_total"])[::-1] if permutation == "reverse"
             else np.random.default_rng(42).permutation(data["M_total"]))
    changed_data, changed_preparation = deepcopy((data, preparation))
    for field in ("I", "cell", "y", "s"):
        changed_data[field] = np.asarray(data[field])[order].tolist()
    changed_preparation["observations"] = [preparation["observations"][index] for index in order]
    changed = checks.build_predictive_report(changed_data, changed_preparation, predicted[:, order], alpha[:, order])
    assert changed == baseline
    assert checks.interpretation(changed) == checks.interpretation(baseline)
    if identical:
        assert_exact_replica_agreement(changed)


def test_production_shape_offline_runtime_and_serialization(production_design, monkeypatch, capsys):
    import json
    import time
    from applications.seu_sensitivity_study.tests.test_confirmatory_reporting import FakeFit
    from applications.seu_sensitivity_study import confirmatory_reporting

    data, preparation = production_design
    assert (data["J"], data["R"], data["M_total"]) == (18, 60, 5040)
    fit = FakeFit(data["P"], data)
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("A4 must reuse saved replicas"))
    start = time.perf_counter()
    payload = confirmatory_reporting.fit_payload(fit, preparation["cell_ids"], data["P"])
    legacy = confirmatory_reporting._predictive_checks(fit, data, preparation["cell_ids"], payload)
    report = checks.build_predictive_report(data, preparation, fit.stan_variable("y_pred"), fit.stan_variable("alpha_obs"))
    qualification = checks.interpretation(report)
    elapsed = time.perf_counter() - start
    serialized = json.dumps({"legacy": legacy, "a4": report, "rq6_interpretation": qualification}, allow_nan=False)
    assert report["draw_count"] == 500
    assert len(report["groups"]) == 19 * 29
    assert len(report["items"]) == 19 * 60
    assert len(report["positions"]) == 19 * 40
    assert len(serialized) < 20_000_000
    assert all(row["same_item"]["status"] == "available" for row in report["pairs"])
    with capsys.disabled():
        print(f"\nA4_OFFLINE_BENCHMARK observations=5040 cells=18 items=60 draws=500 seconds={elapsed:.3f} report_bytes={len(serialized.encode())}")


@pytest.mark.parametrize("mutation", [
    lambda report: report.pop("predictive_reference"),
    lambda report: report["predictive_reference"]["items"][0].__setitem__("quality_label", "wrong"),
    lambda report: report["predictive_reference"]["menus"][0].__setitem__("family", "wrong"),
    lambda report: report["predictive_reference"]["recipes"]["strong"].__setitem__("contenders", ["strong", "strong"]),
    lambda report: report["frozen_observation_reference"]["menus"][0]["presentations"][0]["order"].reverse(),
    lambda report: report["observations"].pop(),
    lambda report: report["na_logs"].pop(next(iter(report["na_logs"]))),
])
def test_bound_metadata_rejects_tampering_and_missing_denominators(production_design, mutation):
    data, preparation = production_design
    mutation(preparation)
    with pytest.raises(ValueError, match="A4|A3|Observation"):
        checks.validate_evidence(data, preparation)


def test_sibling_evidence_preserves_utility_observations(production_design):
    _, preparation = production_design
    checks.validate_sibling_evidence(preparation, deepcopy(preparation), "utility_035")
    sibling = deepcopy(preparation)
    sibling["observations"][0]["chosen_item_id"] = "changed"
    with pytest.raises(ValueError, match="exactly the primary"):
        checks.validate_sibling_evidence(preparation, sibling, "utility_065")


def test_public_entry_rejects_eta_not_matching_utility_variant(production_design):
    data, preparation = production_design
    data["eta"][0][0] += .01
    with pytest.raises(ValueError, match="retained eta disagrees"):
        checks.validate_evidence(data, preparation)


def test_pool_weights_retained_observations_not_cells(small_design):
    design = remove_rows(small_design, [4, 5, 8, 9, 12, 13])
    report = checks._calculate(*design)
    pool = select(report["groups"], scope="pool_task", family=None, menu_size=None)
    first = select(report["groups"], scope="cell", id="first", family=None, menu_size=None)
    second = select(report["groups"], scope="cell", id="second", family=None, menu_size=None)
    assert pool["observation_count"] == 10
    assert pool["statistics"]["filler_fraction"]["observed"] == .6
    assert (first["statistics"]["filler_fraction"]["observed"] + second["statistics"]["filler_fraction"]["observed"]) / 2 == .375


def test_pair_products_allow_distinct_presentation_probabilities(small_design):
    from scipy.special import softmax

    data, preparation, reference, predicted, alpha = small_design
    alpha[:, 5] *= 3
    report = checks._calculate(data, preparation, reference, predicted, alpha)
    pair = select(report["pairs"], scope="cell", id="first", family="startup", menu_size=4)
    first = softmax(alpha[:, 4, None] * np.array([.5, .5, .8, .1]), axis=1)
    second = softmax(alpha[:, 5, None] * np.array([.5, .5, .8, .1]), axis=1)
    expected_item, expected_position = [], []
    for draw in range(4):
        expected_item.append(sum(first[draw, left] * second[draw, right]
                                 for left in range(4) for right in range(4) if left == right))
        expected_position.append(sum(first[draw, left] * second[draw, right]
                                     for left in range(4) for right in range(4) if left == 3 - right))
    assert pair["conditional_same_item"] == pytest.approx(checks.quantiles(expected_item))
    assert pair["conditional_same_position"] == pytest.approx(checks.quantiles(expected_position))


def test_structured_rq6_qualifications_include_missingness_and_unavailable_strata(small_design):
    report = checks._calculate(*remove_rows(small_design, [0, 4, 5]))
    interpretation = checks.interpretation(report)
    assert interpretation["primary_decision_unchanged"]
    assert interpretation["qualifications"]["conditional_on_retention"]["applies"]
    assert interpretation["qualifications"]["conditional_on_retention"]["differential_missingness"]
    assert interpretation["qualifications"]["unavailable_sizes_or_pairs"]["applies"]
    for section in ("groups", "items", "positions", "pairs", "trends"):
        for row in report[section]:
            assert "decision" not in row
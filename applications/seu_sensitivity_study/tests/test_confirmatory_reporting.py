import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from applications.seu_sensitivity_study import confirmatory_reporting as reporting
from applications.seu_sensitivity_study import assessment_scale
from applications.seu_sensitivity_study.data_preparation import assessment_expected_utilities
from applications.seu_sensitivity_study.tests.test_assessment_scale import fixed_inputs


def test_legacy_manifest_requires_explicit_input_evidence():
    fits = {
        group: {variant: "legacy-chains" for variant in reporting.REQUIRED_VARIANTS}
        for group in ("venture", "hiring", "matched_rq5")
    }
    with pytest.raises(ValueError, match="schema_version"):
        reporting.build_report_from_manifest({"fits": fits}, fit_loader=lambda _: None)


class FakeFit:
    def __init__(self, gamma_columns, data=None, data_path=None, residuals=None):
        self.gamma_columns = gamma_columns
        self.chains = 4
        self.metadata = SimpleNamespace(cmdstan_config={"max_depth": 12, "model": f"{reporting.MODEL_NAME}_model"})
        if data_path is not None:
            self.metadata.cmdstan_config["data_file"] = str(data_path)
        cell_count = data["J"] if data else 18
        self.variables = {
            "gamma0": np.full(500, 0.5),
            "gamma": np.full((500, gamma_columns), 0.3),
            "gamma_size": np.full(500, 0.06),
            "sigma_cell": np.full(500, 0.2),
            "z_alpha": np.zeros((500, cell_count)),
        }
        if residuals is not None:
            self.variables["z_alpha"] = np.tile(np.asarray(residuals) / 0.2, (500, 1))
        if data:
            cell = np.asarray(data["cell"]) - 1
            log_alpha = (
                self.variables["gamma0"][:, None]
                + self.variables["gamma"] @ np.asarray(data["X"]).T
                + self.variables["sigma_cell"][:, None] * self.variables["z_alpha"]
            )
            alpha_obs = np.exp(log_alpha[:, cell] + self.variables["gamma_size"][:, None] * data["s"])
            self.variables.update(
                alpha_cell=np.exp(log_alpha), alpha_obs=alpha_obs,
                upsilon=np.tile(data["utility_values"], (500, 1)),
                y_pred=np.tile(data["y"], (500, 1)),
            )
            log_lik = np.empty((500, data["M_total"]))
            for observation, cell_index in enumerate(cell):
                active = np.asarray(data["I"][observation], dtype=bool)
                eta = np.asarray(data["eta"][cell_index])[active]
                logits = alpha_obs[:, observation, None] * (eta - eta.max())
                logs = logits - np.log(np.exp(logits).sum(axis=1))[:, None]
                log_lik[:, observation] = logs[:, data["y"][observation] - 1]
            self.variables["log_lik"] = log_lik
        names = ["gamma0", "gamma_size", "sigma_cell"]
        names += [f"gamma[{index}]" for index in range(1, gamma_columns + 1)]
        names += [f"z_alpha[{index}]" for index in range(1, cell_count + 1)]
        self.summary_frame = pd.DataFrame(
            {"ESS_bulk": 600.0, "ESS_tail": 550.0, "R_hat": 1.001}, index=names
        )

    def stan_variable(self, name):
        return self.variables[name]

    def summary(self):
        return self.summary_frame

    def method_variables(self):
        return {
            "treedepth__": np.full((125, 4), 8),
            "divergent__": np.zeros((125, 4)),
            "energy__": np.tile(np.arange(125)[:, None] % 3, (1, 4)),
        }


def bind_json(path, payload):
    path.write_text(json.dumps(payload))
    return {"path": str(path), "sha256": reporting._sha256_file(path)}


def replace_json(entry, name, transform):
    path = Path(entry[name]["path"])
    payload = json.loads(path.read_text())
    transform(payload)
    entry[name] = bind_json(path, payload)


@pytest.fixture
def artifacts(tmp_path):
    config = reporting.SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])
    fits = {}
    for group in ("venture", "hiring", "matched_rq5"):
        if group == "matched_rq5":
            cells = reporting.build_cells(["venture", "hiring"])
            design, columns = reporting.confirmatory_analysis.matched_rq5_design(cells)
            cell_ids = [cell.cell_id for cell in cells]
            contract = reporting.confirmatory_analysis.matched_rq5_contract(cells)
        else:
            design, columns, cell_ids = config.design_matrix_for_pool(group)
            contract = reporting.confirmatory_analysis.contract_manifest(columns, design)
        fits[group] = {}
        inputs = fixed_inputs(group)
        reference = assessment_scale.build_reference(**inputs)
        item_ids = sorted(item["id"] for item in inputs["items"])
        cells_by_id = {cell.cell_id: cell for cell in reporting.build_cells(["venture", "hiring"])}
        indicators = []
        for cell_id in cell_ids:
            pool_id = cells_by_id[cell_id].pool_id
            for size in (2, 4):
                menu = next(menu for menu in reference["menus"]
                            if menu["pool_id"] == pool_id and len(menu["item_ids"]) == size)
                indicators.append([int(item_id in menu["item_ids"]) for item_id in item_ids])
        for variant in reporting.REQUIRED_VARIANTS:
            directory = tmp_path / group / variant
            chain_directory = directory / "chains"
            chain_directory.mkdir(parents=True)
            for chain in range(1, 5):
                (chain_directory / f"chain-{chain}.csv").write_text(f"synthetic {group} {variant} {chain}")
            middle = {"utility_035": 0.35, "utility_065": 0.65}.get(variant, 0.5)
            cell_count = len(cell_ids)
            data = {
                "J": cell_count, "P": len(columns), "K": 3, "R": len(item_ids),
                "M_total": cell_count * 2,
                "cell": np.repeat(np.arange(1, cell_count + 1), 2).tolist(),
                "I": indicators,
                "y": [1, 3] * cell_count, "s": [-1.0, 1.0] * cell_count,
                "eta": [assessment_expected_utilities(inputs["probabilities"][cells_by_id[cell_id].model_name],
                            item_ids=item_ids, utilities=[0.0, middle, 1.0]) for cell_id in cell_ids],
                "utility_values": [0.0, middle, 1.0],
                "X": design.tolist(), "M_per_cell": [2] * cell_count,
            }
            preparation = {
                "pool_id": group, "cell_ids": cell_ids, "design_columns": list(columns),
                "presentation_id": {"presentation_1_only": 1, "presentation_2_only": 2}.get(variant),
                "confirmatory_design_rank": len(columns) + 1,
                "confirmatory_design_required_rank": len(columns) + 1,
                "item_ids": item_ids,
                "assessment_scale_reference": reference,
            }
            fits[group][variant] = {
                "chain_path": str(chain_directory),
                "chain_sha256": {path.name: reporting._sha256_file(path) for path in chain_directory.glob("*.csv")},
                "stan_data": bind_json(directory / "stan_data_size.json", data),
                "preparation_report": bind_json(directory / "preparation_report.json", preparation),
                "analysis_contract": bind_json(directory / "analysis_contract.json", contract),
            }
    return {"schema_version": 1, "max_treedepth": 12, "fits": fits}


def load_fit(paths):
    data_path = Path(paths[0]).parent.parent / "stan_data_size.json"
    data = json.loads(data_path.read_text())
    return FakeFit(data["P"], data, data_path)


def test_build_report_from_saved_fit_manifest(artifacts, tmp_path, monkeypatch):
    from copy import deepcopy

    core_report = reporting.confirmatory_analysis.complete_confirmatory_report
    original = {}

    def capture_core(**kwargs):
        result = core_report(**kwargs)
        original.update(deepcopy(result))
        return result

    monkeypatch.setattr(reporting.confirmatory_analysis, "complete_confirmatory_report", capture_core)
    report = reporting.build_report_from_manifest(
        artifacts, fit_loader=load_fit
    )
    for name in ("pools", "matched_rq5", "rq4", "multiplicity"):
        actual = deepcopy(report[name])
        sections = actual.values() if name == "pools" else [actual] if name == "matched_rq5" else []
        for section in sections:
            for variant in reporting.REQUIRED_VARIANTS:
                section[variant]["rq6"].pop("predictive_interpretation")
        assert actual == original[name]

    assert report["pools"]["venture"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 9
    assert report["matched_rq5"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 6
    assert len(report["fit_artifact_hashes"]) == 15
    assert report["schema_version"] == 3
    assert report["estimand_contract"]["amendment"] == 5
    assert report["multiplicity"]["primary_decision_count"] == 26
    assert report["multiplicity"]["available_primary_decision_count"] == 26
    assert report["multiplicity"]["unavailable_primary_decision_count"] == 0
    assert report["multiplicity"]["distinct_primary_decisions_up_to_sign"] == 24
    assert report["multiplicity"]["gamma_companion_primary_decision_count"] == 0
    companion = report["pools"]["venture"]["primary"]["gamma_companion"]
    assert companion["decision_count"] == 0
    assert all("decision" not in row for row in companion["rows"])
    provenance = report["fit_provenance"]["venture"]["primary"]
    assert provenance["binding"] == "declared_input_binding"
    assert provenance["cryptographic_execution_proof"] is False
    assert provenance["cmdstan_data_path_check"] == "matched"
    assert provenance["model"] == f"{reporting.MODEL_NAME}_model"
    assert provenance["chains"] == 4
    assert provenance["max_treedepth"] == 12
    assert report["posterior_predictive_checks"]["venture"]["primary"]["a4"]["status"] == "unavailable"
    assert report["pools"]["venture"]["primary"]["rq6"]["predictive_interpretation"]["status"] == "unavailable"
    comparison = report["pools"]["venture"]["primary"]["rq6"]["predictive_interpretation"]["presentation_comparison"]
    assert comparison["presentation_1_only"]["change_from_primary"] == {"median": 0., "lower_90": 0., "upper_90": 0.}
    reporting.write_report(tmp_path / "report.json", report)
    json.dumps(report, allow_nan=False)


def test_normal_runner_preparation_to_all_fifteen_bound_reports(tmp_path, monkeypatch):
    from applications.seu_sensitivity_study import problem_generation, pools
    from applications.seu_sensitivity_study.config import MODELS
    from applications.seu_sensitivity_study.study_runner import SEUSensitivityStudyRunner

    config = reporting.SEUSensitivityStudyConfig(
        pool_ids=["venture", "hiring"], results_dir=str(tmp_path),
        stan_model="h_m01_size_assessment_anchored",
    )
    runner = SEUSensitivityStudyRunner(config)
    fixed_pools = {pool: pools.load_pool(pool) for pool in config.pool_ids}
    problems = {pool: problem_generation.generate_problem_set(
        fixed_pools[pool], problems_per_family=config.problems_for(pool), seed=42,
    ) for pool in config.pool_ids}
    probabilities = {pool: fixed_inputs(pool, varied=True)["probabilities"] for pool in config.pool_ids}
    choices = {}
    for pool in config.pool_ids:
        records = [{"problem_id": problem["id"], "presentation_id": presentation["presentation_id"],
                    "menu_size": problem["menu_size"], "difficulty_stratum": problem["difficulty_stratum"],
                    "family": problem["family"], "chosen_position": 1,
                    "chosen_item_id": presentation["order"][0], "resolution_path": "answer_token"}
                   for problem in problems[pool]["problems"] for presentation in problem["presentations"]]
        choices[pool] = {cell.cell_id: {"cell_id": cell.cell_id, "pool_id": pool, "choices": records,
                          "model_name": cell.model_name, "prompt_condition": cell.prompt_condition}
                         for cell in config.cells_for_pool(pool)}
    monkeypatch.setattr(runner, "_load_pool_artifact", lambda pool: fixed_pools[pool])
    monkeypatch.setattr(runner, "_load_problem_set", lambda pool: problems[pool])
    monkeypatch.setattr(runner, "_load_reduced_embeddings", lambda pool: {item["id"]: np.zeros(1) for item in fixed_pools[pool]["items"]})
    monkeypatch.setattr(runner, "_load_all_choice_sets", lambda pool: choices[pool])
    monkeypatch.setattr(runner, "_load_assessment_probabilities", lambda pool, model: probabilities[pool][model])
    for pool in config.pool_ids:
        runner._phase_stan_data(pool)
    runner._phase_matched_rq5_stan_data()
    suffixes = {"primary": "", "presentation_1_only": "_presentation_1",
                "presentation_2_only": "_presentation_2", "utility_035": "_u035", "utility_065": "_u065"}
    fits = {}
    for group in ("venture", "hiring", "matched_rq5"):
        directory = tmp_path / "matched_rq5" if group == "matched_rq5" else tmp_path / "pools" / group
        fits[group] = {}
        for variant, suffix in suffixes.items():
            chains = directory / variant / "chains"
            chains.mkdir(parents=True)
            for chain in range(4):
                (chains / f"chain-{chain}.csv").write_text(f"offline {group} {variant} {chain}")
            paths = {"stan_data": directory / f"stan_data_size{suffix}.json",
                     "preparation_report": directory / f"stan_data_size{suffix}_assembly_report.json",
                     "analysis_contract": directory / "analysis_contract.json"}
            fits[group][variant] = {
                "chain_path": str(chains),
                "chain_sha256": {path.name: reporting._sha256_file(path) for path in chains.glob("*.csv")},
                **{name: {"path": str(path), "sha256": reporting._sha256_file(path)} for name, path in paths.items()},
            }

    def loader(paths):
        directory = Path(paths[0]).parent.parent.parent
        variant = Path(paths[0]).parent.parent.name
        data_path = directory / f"stan_data_size{suffixes[variant]}.json"
        data = json.loads(data_path.read_text())
        return FakeFit(data["P"], data, data_path)

    report = reporting.build_report_from_manifest({"schema_version": 1, "max_treedepth": 12, "fits": fits}, fit_loader=loader)
    assert len(report["fit_artifact_hashes"]) == 15
    assert report["multiplicity"]["primary_decision_count"] == 26
    for group in ("venture", "hiring", "matched_rq5"):
        section = report["matched_rq5"] if group == "matched_rq5" else report["pools"][group]
        reference = section["primary"]["assessment_scale"]["reference"]
        assert len(reference["arms"]) == len(MODELS) * (2 if group == "matched_rq5" else 1)
        assert all(arm["item_count"] == (24 if group == "matched_rq5" else 60) for arm in reference["arms"])
        assert all(arm["menu_count"] == (40 if group == "matched_rq5" else 140) for arm in reference["arms"])
        for variant in suffixes:
            assert ("assessment_scale" in section[variant]) == (variant == "primary")
    assert len(report["rq4"]["model_orderings"]) == 15
    assert all("assessment_scale" in row for row in report["rq4"]["model_orderings"])
    json.dumps(report, allow_nan=False)

    from applications.seu_sensitivity_study import ceiling_prior

    for group in ("venture", "hiring", "matched_rq5"):
        directory = tmp_path / "matched_rq5" if group == "matched_rq5" else tmp_path / "pools" / group
        for variant in ceiling_prior.PRIOR_VARIANTS:
            chains = directory / variant / "chains"
            chains.mkdir(parents=True)
            for chain in range(4):
                (chains / f"chain-{chain}.csv").write_text(f"offline A3 {group} {variant} {chain}")
            suffixes[variant] = f"_{variant}"

    def a3_loader(paths):
        fit = loader(paths)
        variant = Path(paths[0]).parent.parent.name
        if variant in ceiling_prior.PRIOR_VARIANTS:
            fit.metadata.cmdstan_config["model"] = ceiling_prior.SENSITIVITY_MODEL
            fit.variables["prior_settings"] = np.tile(list(ceiling_prior.prior_fields(variant).values()), (500, 1))
        return fit

    a3_report = reporting.build_report_from_manifest(ceiling_prior.fit_manifest(tmp_path), fit_loader=a3_loader)
    assert len(a3_report["fit_artifact_hashes"]) == 24
    assert a3_report["multiplicity"] == report["multiplicity"]
    assert a3_report["pools"] == report["pools"]
    assert a3_report["ceiling_prior_sensitivity"]["complete_fit_count"] == 9
    assert len(a3_report["ceiling_diagnostics"]["matched_rq5"]["cells"]) == 36
    assert a3_report["predictive_check_policy"]["complete_fit_count"] == 24
    assert a3_report["predictive_check_policy"]["planned_fit_count"] == 24
    assert a3_report["matched_rq5"] == report["matched_rq5"]
    assert a3_report["rq4"] == report["rq4"]
    assert a3_report["multiplicity"]["primary_decision_count"] == 26
    assert a3_report["multiplicity"]["distinct_primary_decisions_up_to_sign"] == 24
    for group in ("venture", "hiring", "matched_rq5"):
        for variant in (*reporting.REQUIRED_VARIANTS, *ceiling_prior.PRIOR_VARIANTS):
            if variant in ceiling_prior.PRIOR_VARIANTS:
                fit_report = a3_report["ceiling_prior_sensitivity"]["groups"][group]["variants"][variant]["report"]
            else:
                section = a3_report["matched_rq5"] if group == "matched_rq5" else a3_report["pools"][group]
                fit_report = section[variant]
            assert ("sonnet_thinking_descriptive" in fit_report) == (group != "matched_rq5")
            if group != "matched_rq5":
                sonnet = fit_report["sonnet_thinking_descriptive"]
                assert sonnet["status"] == "descriptive"
                assert sonnet["pool_id"] == group
                assert sonnet["median"] == pytest.approx(0)
                assert sonnet["geometric_mean_sensitivity_ratio"]["median"] == pytest.approx(1)
                assert sonnet["decision_count"] == 0
                assert sonnet["included_in_primary_family"] is False
                assert not {"decision", "rope_half_width", "substantive_interpretation", "assessment_scale"} & sonnet.keys()
            a4 = a3_report["posterior_predictive_checks"][group][variant]["a4"]
            assert a4["status"] == "descriptive"
            assert a4["decision_count"] == 0
            assert a4["source_bindings"]["cryptographic_execution_proof"] is False
            assert len(a4["source_bindings"]["observation_set_sha256"]) == 64
            expected_pools = {"venture", "hiring"} if group == "matched_rq5" else {group}
            assert {row["pool_id"] for row in a4["groups"]} == expected_pools
            if variant.startswith("presentation_"):
                assert all(row["same_item"]["status"] == "unavailable" for row in a4["pairs"])
        section = a3_report["matched_rq5"] if group == "matched_rq5" else a3_report["pools"][group]
        assert section["primary"]["rq6"]["predictive_interpretation"]["primary_decision_unchanged"]
        assert set(section["primary"]["rq6"]["predictive_interpretation"]["presentation_comparison"]) == {
            "presentation_1_only", "presentation_2_only"}


@pytest.mark.parametrize("mutation", [
    lambda preparation: preparation.pop("assessment_scale_reference"),
    lambda preparation: preparation.pop("item_ids"),
    lambda preparation: preparation["item_ids"].reverse(),
    lambda preparation: preparation["assessment_scale_reference"]["arms"][0].__setitem__("sd", 0.2),
    lambda preparation: preparation["assessment_scale_reference"]["policy"].__setitem__("utility_middle", 0.35),
])
def test_bound_reference_is_required_and_checked_before_loading_fit(artifacts, mutation):
    replace_json(artifacts["fits"]["venture"]["primary"], "preparation_report", mutation)
    with pytest.raises(ValueError, match="assessment_scale"):
        reporting.build_report_from_manifest(artifacts, fit_loader=lambda _: pytest.fail("Invalid reference reached fit loader"))


def test_rejects_eta_that_disagrees_with_sibling_arm_reference(artifacts):
    replace_json(artifacts["fits"]["venture"]["primary"], "stan_data", lambda data: data["eta"][0].__setitem__(0, 0.4))
    with pytest.raises(ValueError, match="retained eta disagrees"):
        reporting.build_report_from_manifest(artifacts, fit_loader=lambda _: pytest.fail("Bad eta reached fit loader"))


def test_sibling_references_cannot_silently_use_different_fixed_menu_sets(artifacts):
    entry = artifacts["fits"]["venture"]["presentation_1_only"]
    inputs = fixed_inputs("venture")
    inputs["problems"][0]["id"] = "different-fixed-menu"
    reference = assessment_scale.build_reference(**inputs)
    replace_json(entry, "preparation_report", lambda preparation: preparation.__setitem__("assessment_scale_reference", reference))
    with pytest.raises(ValueError, match="differs across sibling"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


def test_matched_off_support_padding_does_not_define_scale(artifacts):
    baseline = reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)
    entry = artifacts["fits"]["matched_rq5"]["primary"]
    preparation = json.loads(Path(entry["preparation_report"]["path"]).read_text())
    other_items = {item["id"] for item in preparation["assessment_scale_reference"]["items"] if item["pool_id"] == "hiring"}
    other_indices = [index for index, item_id in enumerate(preparation["item_ids"]) if item_id in other_items]

    def change_padding(data):
        for row in data["eta"][:18]:
            for index in other_indices:
                row[index] = 0.0

    replace_json(entry, "stan_data", change_padding)
    changed = reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)
    assert baseline["matched_rq5"]["primary"]["assessment_scale"] == changed["matched_rq5"]["primary"]["assessment_scale"]

    def wrong_menu(data):
        active = np.flatnonzero(data["I"][0])[0]
        data["I"][0][active] = 0
        data["I"][0][other_indices[0]] = 1

    replace_json(entry, "stan_data", wrong_menu)
    with pytest.raises(ValueError, match="matched family support"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


def test_saved_fit_residuals_use_cell_weights_not_observation_counts(artifacts):
    entry = artifacts["fits"]["venture"]["primary"]
    cells = reporting.build_cells(["venture"])
    thinking = [index for index, cell in enumerate(cells) if cell.model_name == "claude-sonnet-4-5-thinking"]
    base = [index for index, cell in enumerate(cells) if cell.model_name == "claude-sonnet-4-5"]

    def unbalance_observations(data):
        data["M_total"] += 1
        data["M_per_cell"][0] += 1
        for field in ("cell", "I", "y"):
            data[field].append(data[field][0])
        observation = data["cell"].index(thinking[0] + 1)
        data["M_total"] += 20
        data["M_per_cell"][thinking[0]] += 20
        for field in ("cell", "I", "y"):
            data[field].extend([data[field][observation]] * 20)
        sizes = np.asarray(data["I"]).sum(axis=1)
        data["s"] = (sizes - sizes.mean()).tolist()

    replace_json(entry, "stan_data", unbalance_observations)

    def load_residual_fit(paths):
        data_path = Path(paths[0]).parent.parent / "stan_data_size.json"
        data = json.loads(data_path.read_text())
        residuals = np.arange(data["J"]) / 10
        return FakeFit(data["P"], data, data_path, residuals=residuals)

    report = reporting.build_report_from_manifest(artifacts, fit_loader=load_residual_fit)
    primary = report["pools"]["venture"]["primary"]
    residuals = np.arange(len(cells)) / 10
    expected = residuals[thinking].mean() - residuals[base].mean()
    row = primary["sonnet_thinking_descriptive"]
    assert row["median"] == pytest.approx(expected)
    assert row["geometric_mean_sensitivity_ratio"]["median"] == pytest.approx(np.exp(expected))
    data = json.loads(Path(entry["stan_data"]["path"]).read_text())
    counts = np.asarray(data["M_per_cell"])
    observation_weighted = np.average(residuals[thinking], weights=counts[thinking]) - np.average(residuals[base], weights=counts[base])
    assert row["median"] != pytest.approx(observation_weighted)
    assert primary["contrast_decisions"]["rows"][0]["median"] == pytest.approx(0.6)
    assert primary["gamma_companion"]["rows"][0]["median"] == pytest.approx(0.3)
    assert primary["contrast_decisions"]["rows"][7]["median"] == pytest.approx(0.4)
    assert report["rq4"]["rows"][0]["venture"]["median"] == pytest.approx(0.6)
    matched = report["matched_rq5"]["primary"]
    assert matched["contrast_decisions"]["rows"][0]["median"] == pytest.approx(2.1)
    assert matched["gamma_companion"]["rows"][0]["median"] == pytest.approx(0.3)
    assert len(report["fit_artifact_hashes"]) == 15


@pytest.mark.parametrize("group", ["venture", "matched_rq5"])
def test_historical_gamma_contract_is_not_amendment5(artifacts, group):
    entry = artifacts["fits"][group]["primary"]

    def historical_contract(contract):
        contract.pop("estimand_contract")
        for row in contract["contrasts" if group == "matched_rq5" else "primary_contrasts"]:
            row.pop("coefficient_estimand")
            row.pop("realized_cell_weights")

    replace_json(entry, "analysis_contract", historical_contract)

    def forbidden_loader(paths):
        if Path(paths[0]).parent.parent.parent.name == group:
            pytest.fail("Historical contract must be rejected before its fit is loaded")
        return load_fit(paths)

    with pytest.raises(ValueError, match="analysis_contract differs"):
        reporting.build_report_from_manifest(artifacts, fit_loader=forbidden_loader)


def test_fit_payload_rejects_wrong_parameter_dimensions():
    with pytest.raises(ValueError, match="gamma draws have shape"):
        reporting.fit_payload(
            FakeFit(gamma_columns=6),
            [f"cell-{index}" for index in range(18)],
            gamma_columns=7,
        )


@pytest.mark.parametrize("chains,depth", [(3, 12), (4, 10), (4, None)])
def test_fit_payload_requires_actual_sampler_metadata(chains, depth):
    fit = FakeFit(7)
    fit.chains = chains
    fit.metadata.cmdstan_config["max_depth"] = depth
    with pytest.raises(ValueError, match="four actual chains|TD12"):
        reporting.fit_payload(fit, [f"cell-{index}" for index in range(18)], 7)


def test_retained_seventeen_cell_design(artifacts):
    entry = artifacts["fits"]["venture"]["primary"]

    def drop_last_cell(data):
        data["J"] -= 1
        data["M_total"] -= 2
        for name in ("eta", "X", "M_per_cell"):
            data[name] = data[name][:-1]
        for name in ("cell", "I", "y", "s"):
            data[name] = data[name][:-2]

    replace_json(entry, "stan_data", drop_last_cell)
    replace_json(entry, "preparation_report", lambda report: report["cell_ids"].pop())
    report = reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)
    cells = report["pools"]["venture"]["primary"]["rq3"]["cell_residuals"]
    retained = json.loads(Path(entry["preparation_report"]["path"]).read_text())["cell_ids"]
    assert [cell["cell_id"] for cell in cells] == retained
    assert len(cells) == 17
    assert len(report["posterior_predictive_checks"]["venture"]["primary"]["cells"]) == 17
    primary = report["pools"]["venture"]["primary"]
    missing = reporting.build_cells(["venture"])[-1].cell_id
    for row in primary["contrast_decisions"]["rows"]:
        if missing in row["cell_weights"]:
            assert row["status"] == "unavailable"
            assert row["missing_cell_ids"] == [missing]
            assert "decision" not in row
        else:
            assert row["status"] == "available"
    assert report["multiplicity"]["primary_decision_count"] == 26
    assert report["multiplicity"]["unavailable_primary_decision_count"] == 2
    assert report["multiplicity"]["available_primary_decision_count"] == 24
    assert report["pools"]["venture"]["presentation_sensitivity"]["all_estimands_comparable"] is False


@pytest.mark.parametrize("name,transform,message", [
    ("stan_data", lambda data: data["X"][0].__setitem__(0, 0.125), "X must equal"),
    ("stan_data", lambda data: data.__setitem__("J", 17), "J/P/K"),
    ("stan_data", lambda data: data.__setitem__("P", 6), "J/P/K"),
    ("stan_data", lambda data: data.__setitem__("M_total", 35), "shape"),
    ("stan_data", lambda data: data["cell"].__setitem__(0, 0), "cell/M_per_cell"),
    ("stan_data", lambda data: data["M_per_cell"].__setitem__(0, 1), "actual cell"),
    ("stan_data", lambda data: data["I"][0].__setitem__(0, 2), "I must be binary"),
    ("stan_data", lambda data: data["y"].__setitem__(0, 3), "y must index"),
    ("stan_data", lambda data: data["s"].__setitem__(0, 1.0), "s must equal"),
    ("stan_data", lambda data: data["eta"].pop(), "eta must be finite with shape"),
    ("stan_data", lambda data: data["eta"][0].__setitem__(0, float("nan")), "eta must be finite"),
    ("preparation_report", lambda report: report.pop("design_columns"), "design_columns"),
    ("preparation_report", lambda report: report["design_columns"].reverse(), "design_columns"),
    ("preparation_report", lambda report: report.pop("presentation_id"), "presentation_id"),
    ("preparation_report", lambda report: report.__setitem__("presentation_id", 1), "presentation_id"),
    ("preparation_report", lambda report: report.__setitem__("pool_id", "hiring"), "pool_id"),
    ("preparation_report", lambda report: report["cell_ids"].reverse(), "canonical order"),
    ("preparation_report", lambda report: report["cell_ids"].__setitem__(1, report["cell_ids"][0]), "unique canonical"),
    ("preparation_report", lambda report: report["cell_ids"].__setitem__(0, "unknown"), "canonical subset"),
    ("preparation_report", lambda report: report.__setitem__("confirmatory_design_rank", 7), "design rank"),
    ("analysis_contract", lambda contract: contract["primary_contrasts"][0]["column_names"].reverse(), "analysis_contract"),
])
def test_rejects_wrong_input_evidence(artifacts, name, transform, message):
    replace_json(artifacts["fits"]["venture"]["primary"], name, transform)
    with pytest.raises(ValueError, match=message):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("variant", reporting.REQUIRED_VARIANTS)
def test_checks_expected_utility_for_every_variant(artifacts, variant):
    entry = artifacts["fits"]["venture"][variant]
    replace_json(entry, "stan_data", lambda data: data["utility_values"].__setitem__(1, 0.4))
    with pytest.raises(ValueError, match="utility variant"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("name", ["stan_data", "preparation_report", "analysis_contract"])
def test_rejects_wrong_digest(artifacts, name):
    artifacts["fits"]["venture"]["primary"][name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("name", ["chain_path", "chain_sha256", "stan_data", "preparation_report", "analysis_contract"])
def test_rejects_missing_bindings(artifacts, name):
    artifacts["fits"]["venture"]["primary"].pop(name)
    with pytest.raises(ValueError, match="explicit artifact bindings"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("group,variant,copied", [
    ("venture", "presentation_1_only", False),
    ("hiring", "primary", False),
    ("matched_rq5", "primary", True),
])
def test_rejects_reused_chains(artifacts, group, variant, copied):
    source = artifacts["fits"]["venture"]["primary"]
    target = artifacts["fits"][group][variant]
    if copied:
        for source_path in Path(source["chain_path"]).glob("*.csv"):
            (Path(target["chain_path"]) / source_path.name).write_bytes(source_path.read_bytes())
    else:
        target["chain_path"] = source["chain_path"]
    target["chain_sha256"] = source["chain_sha256"]
    with pytest.raises(ValueError, match="Chain reuse"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("name", ["gamma0", "gamma", "gamma_size", "sigma_cell", "z_alpha", "alpha_cell", "alpha_obs", "upsilon", "log_lik", "y_pred"])
def test_rejects_nonfinite_draws(artifacts, name):
    def loader(paths):
        fit = load_fit(paths)
        fit.variables[name] = fit.variables[name].astype(float)
        fit.variables[name].flat[0] = np.nan
        return fit

    with pytest.raises(ValueError, match="finite"):
        reporting.build_report_from_manifest(artifacts, fit_loader=loader)


@pytest.mark.parametrize("name", ["gamma0", "gamma_size", "sigma_cell", "gamma[7]", "z_alpha[18]"])
@pytest.mark.parametrize("diagnostic", [None, "ESS_bulk", "ESS_tail", "R_hat"])
def test_requires_complete_finite_structural_diagnostics(name, diagnostic):
    fit = FakeFit(7)
    if diagnostic is None:
        fit.summary_frame = fit.summary_frame.drop(index=name)
    else:
        fit.summary_frame.loc[name, diagnostic] = np.inf
    with pytest.raises(ValueError, match="structural|Structural"):
        reporting.fit_payload(fit, [f"cell-{index}" for index in range(18)], 7)


@pytest.mark.parametrize("field,value", [("model", "h_m01_size_pinned"), ("model", None), ("data_file", "/wrong/data.json")])
def test_rejects_wrong_fit_metadata(artifacts, field, value):
    def loader(paths):
        fit = load_fit(paths)
        fit.metadata.cmdstan_config[field] = value
        return fit

    with pytest.raises(ValueError, match="metadata"):
        reporting.build_report_from_manifest(artifacts, fit_loader=loader)


def test_missing_data_path_is_explicitly_declared_not_proven(artifacts):
    def loader(paths):
        fit = load_fit(paths)
        fit.metadata.cmdstan_config.pop("data_file")
        return fit

    report = reporting.build_report_from_manifest(artifacts, fit_loader=loader)
    provenance = report["fit_provenance"]["venture"]["primary"]
    assert provenance["cmdstan_data_path_check"] == "unavailable"
    assert provenance["cryptographic_execution_proof"] is False


@pytest.mark.parametrize("name", ["alpha_cell", "alpha_obs", "upsilon", "log_lik"])
def test_rejects_draws_inconsistent_with_inputs(artifacts, name):
    def loader(paths):
        fit = load_fit(paths)
        fit.variables[name].flat[0] += 0.01
        return fit

    with pytest.raises(ValueError, match="disagree"):
        reporting.build_report_from_manifest(artifacts, fit_loader=loader)


def test_ppc_uses_actual_y_and_predictions_per_cell_and_pool(artifacts):
    entry = artifacts["fits"]["venture"]["primary"]
    data = json.loads(Path(entry["stan_data"]["path"]).read_text())
    data["eta"][0] = [0.5] * data["R"]
    preparation = json.loads(Path(entry["preparation_report"]["path"]).read_text())
    fit = FakeFit(data["P"], data)
    fit.variables["y_pred"][:] = 1
    payload = reporting.fit_payload(fit, preparation["cell_ids"], data["P"])
    checks = reporting._predictive_checks(fit, data, preparation["cell_ids"], payload)
    cells = list(checks["cells"].values())
    assert cells[0]["tied_eta_fraction"] == 1.0
    assert cells[0]["observed_modal_fraction"] == 1.0
    modal_flags = []
    replicated_flags = []
    for observation, cell_index in enumerate(np.asarray(data["cell"]) - 1):
        eta = np.asarray(data["eta"][cell_index])[np.asarray(data["I"][observation], dtype=bool)]
        modal_flags.append(eta[data["y"][observation] - 1] == eta.max())
        replicated_flags.append(eta[0] == eta.max())
    assert cells[1]["observed_modal_fraction"] == np.mean(modal_flags[2:4])
    assert cells[1]["replicated_modal_fraction"]["q50"] == np.mean(replicated_flags[2:4])
    pool = checks["pools"]["venture"]
    assert pool["observation_count"] == 36
    assert pool["tied_eta_fraction"] >= 1 / 18
    assert pool["observed_modal_fraction"] == pytest.approx(np.mean(modal_flags))
    assert pool["choice_position_distribution"][0]["observed_fraction"] == 0.5
    assert pool["choice_position_distribution"][0]["replicated_fraction"]["q50"] == 1.0
    assert "alpha_obs_quantiles" in pool
    assert "observed_mean_log_score" in pool
    assert checks["status"] == "descriptive"


def test_rejects_rank_losing_canonical_subset(artifacts):
    entry = artifacts["fits"]["venture"]["primary"]

    def retain_seven(data):
        data["J"] = 7
        data["M_total"] = 14
        for name in ("X", "eta", "M_per_cell"):
            data[name] = data[name][:7]
        for name in ("cell", "I", "y", "s"):
            data[name] = data[name][:14]

    replace_json(entry, "stan_data", retain_seven)
    replace_json(entry, "preparation_report", lambda report: report.__setitem__("cell_ids", report["cell_ids"][:7]))
    with pytest.raises(ValueError, match="design rank must be unchanged"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("mode", ["missing_chain", "wrong_digest", "missing_group", "missing_variant", "legacy_entry"])
def test_rejects_incomplete_fit_manifest(artifacts, mode):
    entry = artifacts["fits"]["venture"]["primary"]
    if mode == "missing_chain":
        (Path(entry["chain_path"]) / "chain-4.csv").unlink()
    elif mode == "wrong_digest":
        entry["chain_sha256"]["chain-1.csv"] = "0" * 64
    elif mode == "missing_group":
        artifacts["fits"].pop("hiring")
    elif mode == "missing_variant":
        artifacts["fits"]["venture"].pop("utility_065")
    else:
        artifacts["fits"]["venture"]["primary"] = entry["chain_path"]
    with pytest.raises(ValueError, match="four actual chain|chain_sha256|exactly|explicit artifact bindings"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_fit)


@pytest.mark.parametrize("mode", ["empty", "wrong_count", "fractional", "out_of_bounds"])
def test_rejects_invalid_prediction_arrays(artifacts, mode):
    def loader(paths):
        fit = load_fit(paths)
        predicted = fit.variables["y_pred"].astype(float)
        if mode == "empty":
            predicted = predicted[:0]
        elif mode == "wrong_count":
            predicted = predicted[:-1]
        elif mode == "fractional":
            predicted[0, 0] = 1.5
        else:
            predicted[0, 0] = 3
        fit.variables["y_pred"] = predicted
        return fit

    with pytest.raises(ValueError, match="nonempty|shape|integer|sorted active"):
        reporting.build_report_from_manifest(artifacts, fit_loader=loader)


@pytest.mark.parametrize("name", ["treedepth__", "divergent__", "energy__"])
def test_rejects_nonfinite_sampler_arrays(name):
    fit = FakeFit(7)
    methods = fit.method_variables()
    methods[name] = methods[name].astype(float)
    methods[name][0, 0] = np.nan
    fit.method_variables = lambda: methods
    with pytest.raises(ValueError, match="finite diagnostics"):
        reporting.fit_payload(fit, [f"cell-{index}" for index in range(18)], 7)


def test_rejects_one_chain_with_undefined_ebfmi():
    fit = FakeFit(7)
    methods = fit.method_variables()
    methods["energy__"][:, 0] = 0
    fit.method_variables = lambda: methods
    with pytest.raises(ValueError, match="finite.*diagnostics|diagnostics.*finite"):
        reporting.fit_payload(fit, [f"cell-{index}" for index in range(18)], 7)


def test_rejects_low_ess_even_with_complete_summary():
    fit = FakeFit(7)
    fit.summary_frame.loc["z_alpha[18]", "ESS_tail"] = 399
    payload = reporting.fit_payload(fit, [f"cell-{index}" for index in range(18)], 7)
    with pytest.raises(ValueError, match="tail_ess"):
        reporting.confirmatory_analysis.assert_sampler_gates(payload["diagnostics"])


def test_v2_rejects_stale_a4_reference_before_fit_loader(artifacts, monkeypatch):
    artifacts["schema_version"] = 2
    replace_json(artifacts["fits"]["venture"]["primary"], "preparation_report",
                 lambda preparation: preparation.__setitem__("observation_metadata_version", 2))
    monkeypatch.setattr(assessment_scale, "validate_retained_data", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="regenerate preparation reports.*refresh manifest"):
        reporting.build_report_from_manifest(artifacts, fit_loader=lambda _: pytest.fail("Stale A4 reached loader"))


@pytest.mark.parametrize("mode", ["missing", "failed", "sampler_failed", "missing_files", "no_fit"])
def test_a4_prior_incompleteness_explicit(artifacts, monkeypatch, tmp_path, mode):
    from applications.seu_sensitivity_study import ceiling_prior

    artifacts["schema_version"] = 2
    for variants in artifacts["fits"].values():
        for entry in variants.values():
            replace_json(entry, "preparation_report", lambda preparation: preparation.__setitem__("observation_metadata_version", 2))
    monkeypatch.setattr(reporting.a4_checks, "validate_evidence", lambda *args: {})
    monkeypatch.setattr(reporting.ceiling_diagnostics, "validate_observations", lambda *args: [])
    monkeypatch.setattr(reporting.ceiling_diagnostics, "retained_ceiling_report", lambda *args: {})
    if mode in ("missing", "failed"):
        artifacts["fits"]["venture"]["prior_L"] = {"status": mode, "reason": "offline declared incomplete"}
    else:
        directory = tmp_path / "venture" / "prior_L"
        chains = directory / "chains"
        chains.mkdir(parents=True)
        for index in range(4):
            (chains / f"chain-{index}.csv").write_text(f"unique prior chain {index}")
        primary = artifacts["fits"]["venture"]["primary"]
        data = json.loads(Path(primary["stan_data"]["path"]).read_text())
        model = Path(reporting.__file__).resolve().parents[2] / "models" / f"{ceiling_prior.SENSITIVITY_MODEL}.stan"
        artifacts["fits"]["venture"]["prior_L"] = {
            **primary, "chain_path": str(chains),
            "chain_sha256": {path.name: reporting._sha256_file(path) for path in chains.glob("*.csv")},
            "stan_data": bind_json(directory / "stan_data_size.json", ceiling_prior.sensitivity_data(data, "prior_L")),
            "prior_contract": bind_json(directory / "prior_contract.json", ceiling_prior.prior_contract("prior_L")),
            "model_source": {"path": str(model), "sha256": reporting._sha256_file(model)}}
        if mode == "missing_files":
            for path in chains.glob("*.csv"):
                path.unlink()

    def loader(paths):
        if Path(paths[0]).parent.parent.name != "prior_L":
            return load_fit(paths)
        if mode == "no_fit":
            return None
        fit = load_fit(paths)
        fit.metadata.cmdstan_config["model"] = ceiling_prior.SENSITIVITY_MODEL
        fit.variables["prior_settings"] = np.tile(list(ceiling_prior.prior_fields("prior_L").values()), (500, 1))
        fit.summary_frame.loc["gamma0", "ESS_tail"] = 10
        return fit

    report = reporting.build_report_from_manifest(artifacts, fit_loader=loader)
    a4 = report["posterior_predictive_checks"]["venture"]["prior_L"]["a4"]
    assert a4["status"] == "unavailable"
    assert a4["fit_status"] == {"missing_files": "missing", "no_fit": "failed"}.get(mode, mode)
    assert a4["reason"]
    assert report["multiplicity"]["primary_decision_count"] == 26
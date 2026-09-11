import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from applications.seu_sensitivity_study import confirmatory_reporting as reporting


def test_legacy_manifest_requires_explicit_input_evidence():
    fits = {
        group: {variant: "legacy-chains" for variant in reporting.REQUIRED_VARIANTS}
        for group in ("venture", "hiring", "matched_rq5")
    }
    with pytest.raises(ValueError, match="schema_version"):
        reporting.build_report_from_manifest({"fits": fits}, fit_loader=lambda _: None)


class FakeFit:
    def __init__(self, gamma_columns, data=None, data_path=None):
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
        if data:
            cell = np.asarray(data["cell"]) - 1
            log_alpha = self.variables["gamma0"][:, None] + self.variables["gamma"] @ np.asarray(data["X"]).T
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
        for variant in reporting.REQUIRED_VARIANTS:
            directory = tmp_path / group / variant
            chain_directory = directory / "chains"
            chain_directory.mkdir(parents=True)
            for chain in range(1, 5):
                (chain_directory / f"chain-{chain}.csv").write_text(f"synthetic {group} {variant} {chain}")
            middle = {"utility_035": 0.35, "utility_065": 0.65}.get(variant, 0.5)
            cell_count = len(cell_ids)
            data = {
                "J": cell_count, "P": len(columns), "K": 3, "R": 3,
                "M_total": cell_count * 2,
                "cell": np.repeat(np.arange(1, cell_count + 1), 2).tolist(),
                "I": [[1, 0, 1], [1, 1, 1]] * cell_count,
                "y": [1, 3] * cell_count, "s": [-0.5, 0.5] * cell_count,
                "eta": [[0.2, middle, 0.8]] * cell_count,
                "utility_values": [0.0, middle, 1.0],
                "X": design.tolist(), "M_per_cell": [2] * cell_count,
            }
            preparation = {
                "pool_id": group, "cell_ids": cell_ids, "design_columns": list(columns),
                "presentation_id": {"presentation_1_only": 1, "presentation_2_only": 2}.get(variant),
                "confirmatory_design_rank": len(columns) + 1,
                "confirmatory_design_required_rank": len(columns) + 1,
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


def test_build_report_from_saved_fit_manifest(artifacts, tmp_path):
    report = reporting.build_report_from_manifest(
        artifacts, fit_loader=load_fit
    )

    assert report["pools"]["venture"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 9
    assert report["matched_rq5"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 6
    assert len(report["fit_artifact_hashes"]) == 15
    provenance = report["fit_provenance"]["venture"]["primary"]
    assert provenance["binding"] == "declared_input_binding"
    assert provenance["cryptographic_execution_proof"] is False
    assert provenance["cmdstan_data_path_check"] == "matched"
    assert provenance["model"] == f"{reporting.MODEL_NAME}_model"
    assert provenance["chains"] == 4
    assert provenance["max_treedepth"] == 12
    reporting.write_report(tmp_path / "report.json", report)
    json.dumps(report, allow_nan=False)


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
    replace_json(entry, "stan_data", lambda data: data["eta"].__setitem__(0, [0.5, 0.5, 0.5]))

    def loader(paths):
        fit = load_fit(paths)
        fit.variables["y_pred"][:] = 1
        return fit

    report = reporting.build_report_from_manifest(artifacts, fit_loader=loader)
    checks = report["posterior_predictive_checks"]["venture"]["primary"]
    cells = list(checks["cells"].values())
    assert cells[0]["tied_eta_fraction"] == 1.0
    assert cells[0]["observed_modal_fraction"] == 1.0
    assert cells[1]["observed_modal_fraction"] == 0.5
    assert cells[1]["replicated_modal_fraction"]["q50"] == 0.0
    pool = checks["pools"]["venture"]
    assert pool["observation_count"] == 36
    assert pool["tied_eta_fraction"] == pytest.approx(1 / 18)
    assert pool["observed_modal_fraction"] == pytest.approx(19 / 36)
    assert pool["choice_position_distribution"][0]["observed_fraction"] == 0.5
    assert pool["choice_position_distribution"][0]["replicated_fraction"]["q50"] == 1.0
    assert "alpha_obs_quantiles" in pool
    assert "observed_mean_log_score" in pool
    assert checks["status"] == "descriptive"
    assert set(report["posterior_predictive_checks"]["matched_rq5"]["primary"]["pools"]) == {"venture", "hiring"}


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
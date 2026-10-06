import copy
import json
from pathlib import Path

import numpy as np
import pytest

from applications.seu_sensitivity_study import ceiling_prior as prior
from applications.seu_sensitivity_study import confirmatory_reporting as reporting
from applications.seu_sensitivity_study.data_preparation import build_stan_data
from applications.seu_sensitivity_study.tests.test_a3_observation_audit import assembly_inputs
from applications.seu_sensitivity_study.tests.test_confirmatory_reporting import (
    artifacts, bind_json, load_fit, replace_json,
)


@pytest.fixture
def a3_artifacts(artifacts):
    artifacts["schema_version"] = 2
    model = Path(reporting.__file__).resolve().parents[2] / "models" / f"{prior.SENSITIVITY_MODEL}.stan"
    for group, variants in artifacts["fits"].items():
        for variant, entry in variants.items():
            kwargs = assembly_inputs(group, {"presentation_1_only": 1, "presentation_2_only": 2}.get(variant))
            kwargs["utility_values"] = [0., {"utility_035": .35, "utility_065": .65}.get(variant, .5), 1.]
            data, preparation = build_stan_data(**kwargs)
            entry["stan_data"] = bind_json(Path(entry["stan_data"]["path"]), data)
            entry["preparation_report"] = bind_json(Path(entry["preparation_report"]["path"]), preparation)
        entry = variants["primary"]
        data = json.loads(Path(entry["stan_data"]["path"]).read_text())
        preparation = json.loads(Path(entry["preparation_report"]["path"]).read_text())
        for variant in prior.PRIOR_VARIANTS:
            directory = Path(entry["stan_data"]["path"]).parent.parent / variant
            chains = directory / "chains"
            chains.mkdir(parents=True)
            for chain in range(4):
                (chains / f"chain-{chain}.csv").write_text(f"A3 {group} {variant} {chain}")
            variants[variant] = {
                "chain_path": str(chains), "chain_sha256": {path.name: reporting._sha256_file(path) for path in chains.glob("*.csv")},
                "stan_data": bind_json(directory / "stan_data_size.json", prior.sensitivity_data(data, variant)),
                "preparation_report": bind_json(directory / "preparation.json", preparation),
                "analysis_contract": entry["analysis_contract"],
                "prior_contract": bind_json(directory / "prior.json", prior.prior_contract(variant)),
                "model_source": {"path": str(model), "sha256": reporting._sha256_file(model)},
            }
    return artifacts


def load_a3(paths):
    fit = load_fit(paths)
    variant = Path(paths[0]).parent.parent.name
    if variant in prior.PRIOR_VARIANTS:
        fit.metadata.cmdstan_config["model"] = prior.SENSITIVITY_MODEL + "_model"
        fit.variables["prior_settings"] = np.tile(list(prior.prior_fields(variant).values()), (500, 1))
    return fit


def test_all_24_fits_preserve_primary_family_and_separate_annotations(a3_artifacts):
    baseline = copy.deepcopy(a3_artifacts)
    baseline["schema_version"] = 1
    for variants in baseline["fits"].values():
        for variant in prior.PRIOR_VARIANTS:
            del variants[variant]
    old = reporting.build_report_from_manifest(baseline, fit_loader=load_a3)
    result = reporting.build_report_from_manifest(a3_artifacts, fit_loader=load_a3)
    assert result["multiplicity"] == old["multiplicity"]
    assert result["pools"] == old["pools"]
    assert result["matched_rq5"] == old["matched_rq5"]
    assert result["rq4"] == old["rq4"]
    assert len(result["fit_artifact_hashes"]) == 24
    sensitivity = result["ceiling_prior_sensitivity"]
    assert sensitivity["status"] == "complete"
    assert sensitivity["complete_fit_count"] == 9
    assert sensitivity["decision_count"] == 0
    for group in sensitivity["groups"].values():
        for entry in group["variants"].values():
            assert "assessment_scale" not in entry["report"]
            assert entry["report"]["decision_count"] == 0
            for comparison in entry["comparisons"]:
                assert comparison["median_shift"] == 0
                assert not comparison["decision_rule_changed"]
                assert comparison["alternative"]["probability_positive"] in (0, 1)
            assert set(next(iter(entry["cell_quantiles"].values()))["t"]) == {"q05", "q50", "q95", "q99"}
    assert all(len(report["model_orderings"]) == 15 for report in sensitivity["rq4"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", ["missing", "failed", "rhat", "nonfinite", "missing_chains", "empty_chains", "nonfinite_draw"])
def test_missing_and_sampler_failed_fits_are_incomplete(a3_artifacts, mode):
    variants = a3_artifacts["fits"]["venture"]
    if mode == "missing":
        del variants["prior_L"]
    elif mode == "failed":
        variants["prior_L"] = {"status": "failed", "reason": "Authorized run failed"}
    elif mode == "missing_chains":
        variants["prior_L"]["chain_path"] += "-not-created"
    elif mode == "empty_chains":
        directory = Path(variants["prior_L"]["chain_path"]).with_name("empty-chains")
        directory.mkdir()
        variants["prior_L"]["chain_path"] = str(directory)

    def loader(paths):
        fit = load_a3(paths)
        if "/venture/prior_L/" in paths[0]:
            if mode == "rhat":
                fit.summary_frame.loc["gamma0", "R_hat"] = 1.2
            elif mode == "nonfinite":
                fit.summary_frame.loc["gamma0", "ESS_bulk"] = np.nan
            elif mode == "nonfinite_draw":
                fit.variables["gamma"][0, 0] = np.nan
        return fit

    result = reporting.build_report_from_manifest(a3_artifacts, fit_loader=loader)
    sensitivity = result["ceiling_prior_sensitivity"]
    assert sensitivity["status"] == "incomplete"
    assert sensitivity["complete_fit_count"] == 8
    assert result["multiplicity"]["primary_decision_count"] == 26
    assert sensitivity["rq4"]["prior_L"]["status"] == "incomplete"
    assert all(not row["assessment_complete"] for row in sensitivity["groups"]["venture"]["annotations"])


@pytest.mark.parametrize("mode", ["prior", "data", "preparation", "contract", "model_hash", "wrong_model", "wrong_echo", "no_echo", "chain_hash", "primary_prior", "wrong_model_and_nan"])
def test_tampering_fails_closed(a3_artifacts, mode):
    entry = a3_artifacts["fits"]["venture"]["prior_L"]
    if mode == "prior":
        replace_json(entry, "stan_data", lambda data: data.update(prior_gamma_sd=99))
    elif mode == "data":
        replace_json(entry, "stan_data", lambda data: data["y"].__setitem__(0, 2))
    elif mode == "preparation":
        replace_json(entry, "preparation_report", lambda data: data["observations"][0].update(problem_id="changed"))
    elif mode == "contract":
        replace_json(entry, "prior_contract", lambda data: data.update(variant="prior_H"))
    elif mode == "model_hash":
        entry["model_source"]["sha256"] = "0" * 64
    elif mode == "chain_hash":
        entry["chain_sha256"] = {}
    elif mode == "primary_prior":
        replace_json(a3_artifacts["fits"]["venture"]["primary"], "stan_data", lambda data: data.update(prior_gamma_sd=.5))

    def loader(paths):
        fit = load_a3(paths)
        if "/venture/prior_L/" in paths[0]:
            if mode in {"wrong_model", "wrong_model_and_nan"}:
                fit.metadata.cmdstan_config["model"] = reporting.MODEL_NAME
                if mode == "wrong_model_and_nan":
                    fit.variables["gamma"][0, 0] = np.nan
            elif mode == "wrong_echo":
                fit.variables["prior_settings"][:, 1] = .5
            elif mode == "no_echo":
                del fit.variables["prior_settings"]
        return fit

    with pytest.raises(ValueError):
        reporting.build_report_from_manifest(a3_artifacts, fit_loader=loader)


def test_changed_decision_is_annotation_not_primary_replacement(a3_artifacts):
    def loader(paths):
        fit = load_a3(paths)
        if "/venture/prior_L/" in paths[0]:
            from applications.seu_sensitivity_study.tests.test_confirmatory_reporting import FakeFit

            data_path = Path(paths[0]).parent.parent / "stan_data_size.json"
            data = json.loads(data_path.read_text())
            residuals = np.zeros(data["J"])
            residuals[3:6] = -2.0
            fit = FakeFit(data["P"], data, data_path, residuals=residuals)
            fit.metadata.cmdstan_config["model"] = prior.SENSITIVITY_MODEL
            fit.variables["prior_settings"] = np.tile(list(prior.prior_fields("prior_L").values()), (500, 1))
        return fit

    report = reporting.build_report_from_manifest(a3_artifacts, fit_loader=loader)
    sensitivity = report["ceiling_prior_sensitivity"]["groups"]["venture"]
    comparisons = sensitivity["variants"]["prior_L"]["comparisons"]
    assert any(row["decision_rule_changed"] and row["median_sign_changed"] for row in comparisons)
    assert any(row["interpretation"] == "prior-sensitive under the specified checks" for row in sensitivity["annotations"])
    assert report["multiplicity"]["primary_decision_count"] == 26
    assert all(row["median"] >= 0 for row in report["pools"]["venture"]["primary"]["rows"] if row.get("research_question") == "RQ2")


def test_a3_manifest_rejects_historical_observation_evidence(artifacts):
    artifacts["schema_version"] = 2
    with pytest.raises(ValueError, match="version-2"):
        reporting.build_report_from_manifest(artifacts, fit_loader=load_a3)
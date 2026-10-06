import pytest

from applications.seu_sensitivity_study.ceiling_prior import (
    PRIOR_VARIANTS, prior_fields, sensitivity_data, validate_prior_data, fit_plan, write_prior_inputs,
)


def test_only_declared_prior_components_change():
    primary = {"eta": [[0.1, 0.5]], "y": [2], "s": [0.0]}
    expected = ({"prior_gamma0_sd"}, {"prior_gamma_sd", "prior_sigma_cell_sd"},
                {"prior_gamma_size_sd"})
    for variant, changed in zip(PRIOR_VARIANTS, expected):
        fields = prior_fields(variant)
        assert {key for key in fields if fields[key] != prior_fields("primary")[key]} == changed
        data = sensitivity_data(primary, variant)
        validate_prior_data(data, primary, variant)
        with pytest.raises(ValueError):
            validate_prior_data({**data, "y": [1]}, primary, variant)
        with pytest.raises(ValueError):
            validate_prior_data({**data, "prior_z_sd": 2.0}, primary, variant)
        with pytest.raises(ValueError):
            validate_prior_data(primary, primary, variant)


def test_plan_and_input_artifacts(tmp_path):
    import json

    primary = {"utility_values": [0.0, .5, 1.0], "eta": [[.2, .8]]}
    preparation = {"presentation_id": None}
    files = write_prior_inputs(tmp_path, primary, preparation)
    assert len(files) == 3
    for variant, filename in zip(PRIOR_VARIANTS, files):
        validate_prior_data(json.loads((tmp_path / filename).read_text()), primary, variant)
    plan = fit_plan(tmp_path)
    assert len(plan["fits"]) == 24
    assert sum(row["variant"] in PRIOR_VARIANTS for row in plan["fits"]) == 9
    assert all(row["status"] == "planned_not_authorized" for row in plan["fits"])
    assert all(row["utility_middle"] == .5 and row["presentation_id"] is None
               for row in plan["fits"] if row["variant"] in PRIOR_VARIANTS)
    with pytest.raises(ValueError):
        write_prior_inputs(tmp_path, primary, {"presentation_id": 1})


def test_primary_stan_priors_equal_sibling_primary_configuration():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "models"
    original = (root / "h_m01_size_assessment_anchored.stan").read_text()
    sibling = (root / "h_m01_size_assessment_anchored_prior.stan").read_text()
    for name, value in prior_fields("primary").items():
        sibling = sibling.replace(name, str(value))
    for parameter in ("gamma0", "gamma", "gamma_size", "sigma_cell", "z_alpha"):
        pattern = rf"\b{parameter}\s*~\s*([^;]+);"
        assert re.search(pattern, original).group(1) == re.search(pattern, sibling).group(1)
    assert "sigma_cell * z_alpha" in sibling


def test_comparison_shifts_sign_and_zero_changes_are_distinct():
    from applications.seu_sensitivity_study.confirmatory_analysis import summarize_draws
    from applications.seu_sensitivity_study.prior_sensitivity import compare_rows

    primary = {"rows": [{"contrast_id": "contrast", **summarize_draws([-1, 0, 1, 2, 3], rope_half_width=.2)}]}
    alternative = {"rows": [{"contrast_id": "contrast", **summarize_draws([1, 2, 3, 4, 5], rope_half_width=.2)}]}
    row = compare_rows(primary, alternative)[0]
    assert row["median_shift"] == 2
    assert row["lower_endpoint_shift"] == pytest.approx(2)
    assert row["upper_endpoint_shift"] == pytest.approx(2)
    assert row["interval_width_ratio"] == pytest.approx(1)
    assert row["zero_exclusion_changed"]
    assert row["decision_rule_changed"]
    assert not row["median_sign_changed"]


def test_fit_plan_cli_is_offline(tmp_path, monkeypatch):
    import json
    import sys
    from applications.seu_sensitivity_study.cli import main

    output = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["seu_sensitivity_study", "fit-plan", "--results-dir", str(tmp_path), "--output", str(output)])
    main()
    assert json.loads(output.read_text())["planned_fit_count"] == 24


@pytest.mark.parametrize("variant", ["prior_L", "primary"])
def test_fit_manifest_empty_directory(tmp_path, monkeypatch, variant):
    from applications.seu_sensitivity_study import ceiling_prior

    chains = tmp_path / variant / "chains"
    chains.mkdir(parents=True)
    monkeypatch.setattr(ceiling_prior, "fit_plan", lambda root: {"fits": [
        {"group": "venture", "variant": variant, "stan_data": str(tmp_path / "data.json")}
    ]})
    if variant == "primary":
        with pytest.raises(FileNotFoundError, match="no CmdStan CSV"):
            ceiling_prior.fit_manifest(tmp_path)
    else:
        entry = ceiling_prior.fit_manifest(tmp_path)["fits"]["venture"][variant]
        assert entry["status"] == "missing"
        assert "no CmdStan CSV" in entry["reason"]


def test_fit_manifest_does_not_suppress_malformed_prior_chains(tmp_path, monkeypatch):
    from applications.seu_sensitivity_study import ceiling_prior, confirmatory_reporting

    monkeypatch.setattr(ceiling_prior, "fit_plan", lambda root: {"fits": [
        {"group": "venture", "variant": "prior_L", "stan_data": str(tmp_path / "data.json")}
    ]})

    def malformed(path):
        raise ValueError("Malformed supplied chains")

    monkeypatch.setattr(confirmatory_reporting, "_fit_files", malformed)
    with pytest.raises(ValueError, match="Malformed supplied chains"):
        ceiling_prior.fit_manifest(tmp_path)
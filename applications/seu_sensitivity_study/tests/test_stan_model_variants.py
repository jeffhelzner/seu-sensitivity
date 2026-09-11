import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.seu_sensitivity_study import config as study_config
from analysis.assessment_anchored_prior_predictive import (
    _menu_metrics,
    _prior_predictive,
)
from analysis.hierarchical_parameter_recovery import (
    _generate_fixed_eta,
    _linear_contrast_summary,
    _load_completed_iteration,
    _rejected_proposal_counts,
    _summarize_sampler_diagnostics,
)
from analysis.hierarchical_power import fit_diagnostics
from scripts.run_hierarchical_parameter_recovery import (
    _build_study_design,
    _load_config,
)
from scripts.build_matched_rq5_validation_template import _placeholder_choice_sets
from utils.cmdstan_artifacts import gzip_csv_files


ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "h_m01_size.stan"
PINNED_MODEL = ROOT / "models" / "h_m01_size_pinned.stan"
ANCHORED_MODEL = ROOT / "models" / "h_m01_size_assessment_anchored.stan"
ANCHORED_SIM_MODEL = ROOT / "models" / "h_m01_size_assessment_anchored_sim.stan"


def test_rq5_validation_template_uses_every_menu_presentation():
    cells = [
        cell
        for cell in study_config.build_cells(["venture"])
        if cell.model_name == "gpt-4o"
    ]
    problem_set = {
        "problems": [
            {
                "id": "P1",
                "menu_size": 2,
                "difficulty_stratum": "strong",
                "family": "procurement",
                "presentations": [
                    {"presentation_id": "a", "order": ["v1", "v2"]},
                    {"presentation_id": "b", "order": ["v2", "v1"]},
                ],
            }
        ]
    }

    choices = _placeholder_choice_sets("venture", problem_set, cells)

    assert len(choices) == 3
    for payload in choices.values():
        assert [row["presentation_id"] for row in payload["choices"]] == ["a", "b"]
        assert [row["chosen_item_id"] for row in payload["choices"]] == ["v1", "v2"]


def test_linear_contrast_summary_uses_joint_draws():
    gamma_draws = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    summary = _linear_contrast_summary(gamma_draws, (1.0, 1.0))
    assert summary == {"Mean": 0.0, "5%": 0.0, "95%": 0.0}
    with pytest.raises(ValueError, match="2 weights"):
        _linear_contrast_summary(gamma_draws[:, :1], (1.0, 1.0))


def test_recovery_design_and_eta_load_from_stan_template(tmp_path):
    template = {
        "J": 2,
        "K": 3,
        "R": 3,
        "P": 1,
        "M_total": 4,
        "M_per_cell": [2, 2],
        "X": [[0.0], [1.0]],
        "cell": [1, 1, 2, 2],
        "I": [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]],
        "s": [-0.5, 0.5, -0.5, 0.5],
        "eta": [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]],
        "utility_values": [0.0, 0.5, 1.0],
    }
    path = tmp_path / "stan_data_size.json"
    path.write_text(json.dumps(template))

    design = _build_study_design({"stan_data_template_path": str(path)})
    data = design.get_data_dict()
    fixed_eta = _generate_fixed_eta(
        {"stan_data_template_path": str(path)}, J=2, K=3, R=3
    )

    assert data["X"] == template["X"]
    assert data["I"] == template["I"]
    assert data["cell"] == template["cell"]
    assert data["s"] == template["s"]
    assert fixed_eta == {
        "eta": template["eta"],
        "utility_values": template["utility_values"],
    }


def _block(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_pinned_model_fixes_delta_outside_parameters():
    source = PINNED_MODEL.read_text()
    parameters = _block(source, "parameters {", "transformed parameters {")
    model = _block(source, "model {", "generated quantities {")

    assert "simplex[K-1] delta" not in parameters
    assert "simplex[K-1] delta = rep_vector(1.0 / (K - 1), K - 1);" in source
    assert "delta ~" not in model
    assert "ordered[K] upsilon = cumulative_sum(append_row(0, delta));" in source


def test_pinned_model_preserves_data_and_generated_quantities_contracts():
    base = BASE_MODEL.read_text()
    pinned = PINNED_MODEL.read_text()

    assert _block(pinned, "data {", "transformed data {") == _block(
        base, "data {", "transformed data {"
    )
    assert pinned.split("generated quantities {", 1)[1] == base.split(
        "generated quantities {", 1
    )[1]


def test_anchored_simulator_uses_fixed_eta_and_no_latent_belief_map():
    inference = ANCHORED_MODEL.read_text()
    simulator = ANCHORED_SIM_MODEL.read_text()

    for source in (inference, simulator):
        data = _block(source, "data {", "transformed data {")
        assert "matrix<lower=0,upper=1>[J, R] eta" in data
        assert "utility_values" in data
        assert "array[R] vector[D] w" not in data
        assert "beta" not in source


def test_fixed_eta_generation_is_seeded_and_shares_declared_rows():
    config = {
        "seed": 42,
        "distribution": "beta",
        "row_groups": ["a", "a", "b", "b"],
        "utility_values": [0.0, 0.5, 1.0],
    }
    first = _generate_fixed_eta(config, J=4, K=3, R=5)
    second = _generate_fixed_eta(config, J=4, K=3, R=5)

    assert first == second
    assert first["eta"][0] == first["eta"][1]
    assert first["eta"][2] == first["eta"][3]
    assert first["eta"][0] != first["eta"][2]
    assert all(0 <= value <= 1 for row in first["eta"] for value in row)


def test_fixed_eta_generation_rejects_wrong_group_count():
    with pytest.raises(ValueError, match="length J=3"):
        _generate_fixed_eta({"row_groups": [0, 1]}, J=3, K=3, R=4)


def test_fixed_eta_generation_loads_assessments_in_pool_order(tmp_path):
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        '{"items": [{"id": "second"}, {"id": "first"}]}'
    )
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(
        '{"assessments": ['
        '{"item_id": "first", "parse_ok": true, "probabilities": [0.1, 0.2, 0.7]},'
        '{"item_id": "second", "parse_ok": true, "probabilities": [0.6, 0.3, 0.1]}'
        ']}'
    )

    result = _generate_fixed_eta(
        {
            "pool_path": str(pool_path),
            "assessment_files": [str(assessment_path), str(assessment_path)],
            "utility_values": [0.0, 0.5, 1.0],
        },
        J=2,
        K=3,
        R=2,
    )

    assert np.allclose(result["eta"], [[0.25, 0.8], [0.25, 0.8]])


def test_exact_venture_problem_set_reconstructs_production_geometry():
    design = _build_study_design(
        {
            "factors": [6, 3],
            "reference_indices": [0, 0],
            "K": 3,
            "D": 1,
            "R": 60,
            "M_per_cell": 280,
            "menu_sizes": [2, 4, 6, 8],
            "problem_set_path": str(
                ROOT
                / "applications/seu_sensitivity_study/results/pools/venture/problems.json"
            ),
            "pool_path": str(
                ROOT / "applications/seu_sensitivity_study/results/pools/venture/pool.json"
            ),
        }
    )

    assert design.M_total == 5040
    assert design.get_data_dict()["M_per_cell"] == [280] * 18
    assert design.I.shape == (5040, 60)
    assert design.cell.tolist() == np.repeat(np.arange(1, 19), 280).tolist()
    assert np.array_equal(design.I[:280], design.I[280:560])
    assert sorted(np.unique(design.I[:280].sum(axis=1)).tolist()) == [2, 4, 6, 8]


def test_recovery_config_can_override_top_level_base_values(tmp_path):
    base_path = tmp_path / "base.json"
    base_path.write_text('{"chains": 4, "nested": {"kept": true}}')
    child_path = tmp_path / "child.json"
    child_path.write_text(
        '{"base_config_path": "' + str(base_path) + '", "chains": 2}'
    )

    assert _load_config(str(child_path)) == {
        "chains": 2,
        "nested": {"kept": True},
    }


def test_recovery_config_resolves_nested_inheritance(tmp_path):
    base_path = tmp_path / "base.json"
    base_path.write_text('{"chains": 4, "samples": 500}')
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        '{"base_config_path": "' + str(base_path) + '", "samples": 1000}'
    )
    child_path = tmp_path / "child.json"
    child_path.write_text(
        '{"base_config_path": "' + str(middle_path) + '", "warmup": 1000}'
    )

    assert _load_config(str(child_path)) == {
        "chains": 4,
        "samples": 1000,
        "warmup": 1000,
    }


def test_anchored_prior_predictive_reflects_gamma_size_prior_scale():
    problems = [
        {"item_ids": ["low", "high"], "menu_size": 2},
        {"item_ids": ["low", "mid", "high", "other"], "menu_size": 4},
    ]
    eta = {"low": 0.1, "mid": 0.4, "high": 0.9, "other": 0.2}
    common = {
        "eta_by_cell": [eta],
        "problems": problems,
        "design_matrix": np.zeros((1, 1)),
        "draws": 2000,
        "seed": 42,
    }

    narrow = _prior_predictive(**common, gamma_size_sd=0.1)
    wide = _prior_predictive(**common, gamma_size_sd=0.5)

    assert narrow["alpha_ratio_largest_to_smallest_menu"]["q95"] < wide[
        "alpha_ratio_largest_to_smallest_menu"
    ]["q95"]
    gaps, by_size = _menu_metrics(eta, problems)
    assert gaps.tolist() == pytest.approx([0.8, 0.5])
    assert sorted(by_size) == [2, 4]


def test_gzip_csv_files_preserves_content_and_retains_source(tmp_path):
    source = tmp_path / "chain.csv"
    source.write_bytes(b"lp__,gamma0\n-1.0,2.5\n")

    preserved = gzip_csv_files([source])

    assert preserved == [tmp_path / "chain.csv.gz"]
    assert source.read_bytes() == b"lp__,gamma0\n-1.0,2.5\n"
    with gzip.open(preserved[0], "rb") as preserved_file:
        assert preserved_file.read() == b"lp__,gamma0\n-1.0,2.5\n"


def test_rejected_proposal_counts_reads_each_chain_log(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    marker = "categorical_logit_lpmf: log odds parameter[1] is inf"
    first.write_text(f"before\n{marker}\nafter\n{marker}\n")
    second.write_text("clean\n")

    assert _rejected_proposal_counts([first, second]) == [2, 0]


def test_completed_recovery_iteration_requires_all_durable_outputs(tmp_path):
    iteration = tmp_path / "iteration_1"
    iteration.mkdir()
    (iteration / "true_parameters.json").write_text('{"gamma0": 2.5}')
    (iteration / "posterior_summary.csv").write_text(
        ",Mean,5%,95%\ngamma0,2.4,2.0,2.8\n"
    )
    assert _load_completed_iteration(iteration) is None

    (iteration / "diagnostics.json").write_text("{}")
    true_params, summary = _load_completed_iteration(iteration)
    assert true_params == {"gamma0": 2.5}
    assert summary.loc["gamma0", "Mean"] == 2.4


def test_sampler_summary_applies_campaign_thresholds(tmp_path):
    for iteration, max_rhat in ((1, 1.009), (2, 1.011)):
        directory = tmp_path / f"iteration_{iteration}"
        directory.mkdir()
        (directory / "diagnostics.json").write_text(
            '{"seconds": 10, "max_rhat": '
            + str(max_rhat)
            + ', "min_ess_bulk": 500, "min_ebfmi": 0.8, '
            + '"min_ess_tail": 500, '
            '"divergences": 0, "treedepth_saturated_share": 0, '
            '"nonfinite_proposals_total": 3}'
        )

    summary = _summarize_sampler_diagnostics(tmp_path)

    assert summary["completed"] == 2
    assert summary["passed"] == 1
    assert not summary["all_passed"]
    assert summary["total_fit_seconds"] == 20


def test_sampler_summary_recovers_tail_evidence_and_rejects_missing(tmp_path):
    directory = tmp_path / "iteration_1"
    directory.mkdir()
    (directory / "diagnostics.json").write_text(json.dumps({
        "seconds": 1, "max_rhat": 1.005, "min_ess_bulk": 425,
        "min_ebfmi": 0.8, "divergences": 0,
        "treedepth_saturated_share": 0, "nonfinite_proposals_total": 0,
    }))
    assert not _summarize_sampler_diagnostics(tmp_path)["all_passed"]
    (directory / "posterior_summary.csv").write_text(
        ",ESS_tail\nsigma_cell,310.525\ngamma0,800\n"
    )
    report = _summarize_sampler_diagnostics(tmp_path)
    assert report["iterations"][0]["min_ess_tail"] == 310.525
    assert not report["all_passed"]


def test_fit_diagnostics_covers_parameters_and_excludes_observation_arrays():
    class FakeFit:
        metadata = type("Metadata", (), {"cmdstan_config": {"model": "h_m01_size_pinned"}})()

        def summary(self):
            return pd.DataFrame(
                {
                    "ESS_bulk": [40.0, 120.0, 80.0, 500.0, float("nan")],
                        "ESS_tail": [30.0, 480.0, 70.0, 490.0, float("nan")],
                    "R_hat": [1.02, 1.001, 1.009, 1.000, float("nan")],
                },
                index=["lp__", "gamma0", "beta[1,1,1]", "eta[1]", "delta[1]"],
            )

        def method_variables(self):
            return {
                "treedepth__": [[9, 10], [10, 12]],
                "divergent__": [[0, 0], [1, 0]],
                "energy__": [[1.0, 2.0], [2.0, 4.0], [1.0, 2.0]],
            }

    diagnostics = fit_diagnostics(FakeFit(), seconds=200.0, max_treedepth=12)

    assert diagnostics["min_ess_bulk"] == 80.0
    assert diagnostics["max_rhat"] == 1.009
    assert diagnostics["min_ess_tail"] == 70.0
    assert diagnostics["mean_treedepth"] == 10.25
    assert diagnostics["treedepth_saturated_share"] == 0.25
    assert diagnostics["divergences"] == 1
    assert diagnostics["ebfmi_by_chain"] == pytest.approx([4.5, 4.5])
    assert diagnostics["min_ebfmi"] == pytest.approx(4.5)
    assert diagnostics["ess_bulk_per_1000_seconds"] == 400.0
    assert "lp__" not in diagnostics["ess_bulk"]
    assert "beta[1,1,1]" in diagnostics["ess_bulk"]
    assert "eta[1]" not in diagnostics["ess_bulk"]
    assert "delta[1]" not in diagnostics["ess_bulk"]

    FakeFit.metadata.cmdstan_config["model"] = "h_m01_size"
    invalid = fit_diagnostics(FakeFit(), seconds=200.0, max_treedepth=12)
    assert invalid["invalid_diagnostic_parameters"] == ["delta[1]"]
    assert invalid["min_ess_bulk"] is None
    assert invalid["min_ess_tail"] is None
    assert invalid["max_rhat"] is None
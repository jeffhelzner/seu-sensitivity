import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.assessment_anchored_prior_predictive import (
    _menu_metrics,
    _prior_predictive,
)
from analysis.hierarchical_parameter_recovery import _generate_fixed_eta
from analysis.hierarchical_power import fit_diagnostics
from scripts.run_hierarchical_parameter_recovery import _build_study_design
from utils.cmdstan_artifacts import gzip_csv_files


ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "h_m01_size.stan"
PINNED_MODEL = ROOT / "models" / "h_m01_size_pinned.stan"
ANCHORED_MODEL = ROOT / "models" / "h_m01_size_assessment_anchored.stan"
ANCHORED_SIM_MODEL = ROOT / "models" / "h_m01_size_assessment_anchored_sim.stan"


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


def test_fit_diagnostics_covers_parameters_and_excludes_observation_arrays():
    class FakeFit:
        def summary(self):
            return pd.DataFrame(
                {
                    "ESS_bulk": [120.0, 80.0, 500.0, float("nan")],
                    "R_hat": [1.001, 1.009, 1.000, float("nan")],
                },
                index=["gamma0", "beta[1,1,1]", "eta[1]", "delta[1]"],
            )

        def method_variables(self):
            return {
                "treedepth__": [[9, 10], [10, 12]],
                "divergent__": [[0, 0], [1, 0]],
            }

    diagnostics = fit_diagnostics(FakeFit(), seconds=200.0, max_treedepth=12)

    assert diagnostics["min_ess_bulk"] == 80.0
    assert diagnostics["max_rhat"] == 1.009
    assert diagnostics["mean_treedepth"] == 10.25
    assert diagnostics["treedepth_saturated_share"] == 0.25
    assert diagnostics["divergences"] == 1
    assert diagnostics["ess_bulk_per_1000_seconds"] == 400.0
    assert "beta[1,1,1]" in diagnostics["ess_bulk"]
    assert "eta[1]" not in diagnostics["ess_bulk"]
    assert "delta[1]" not in diagnostics["ess_bulk"]
import gzip
from pathlib import Path

import pandas as pd

from analysis.hierarchical_power import fit_diagnostics
from utils.cmdstan_artifacts import gzip_csv_files


ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "h_m01_size.stan"
PINNED_MODEL = ROOT / "models" / "h_m01_size_pinned.stan"


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
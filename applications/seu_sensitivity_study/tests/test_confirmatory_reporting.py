from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.seu_sensitivity_study import confirmatory_reporting as reporting


class FakeFit:
    def __init__(self, gamma_columns):
        self.gamma_columns = gamma_columns

    def stan_variable(self, name):
        if name == "gamma":
            return np.full((500, self.gamma_columns), 0.3)
        if name == "gamma_size":
            return np.full(500, 0.06)
        if name == "sigma_cell":
            return np.full(500, 0.2)
        if name == "z_alpha":
            cell_count = 36 if self.gamma_columns == 13 else 18
            return np.zeros((500, cell_count))
        raise KeyError(name)

    def summary(self):
        return pd.DataFrame(
            {
                "ESS_bulk": [600.0, 650.0],
                "ESS_tail": [550.0, 580.0],
                "R_hat": [1.001, 1.002],
            },
            index=["gamma0", "gamma_size"],
        )

    def method_variables(self):
        return {
            "treedepth__": np.full((500, 4), 8),
            "divergent__": np.zeros((500, 4)),
            "energy__": np.tile(np.arange(500)[:, None] % 3, (1, 4)),
        }


def test_build_report_from_saved_fit_manifest(tmp_path):
    fits = {}
    for group in ("venture", "hiring", "matched_rq5"):
        fits[group] = {}
        for variant in reporting.REQUIRED_VARIANTS:
            directory = tmp_path / group / variant
            directory.mkdir(parents=True)
            (directory / "chain-1.csv").write_text("synthetic")
            fits[group][variant] = str(directory)

    def load_fit(paths: list[str]):
        return FakeFit(13 if "matched_rq5" in paths[0] else 7)

    report = reporting.build_report_from_manifest(
        {"max_treedepth": 12, "fits": fits}, fit_loader=load_fit
    )

    assert report["pools"]["venture"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 9
    assert report["matched_rq5"]["primary"]["contrast_decisions"][
        "decision_count"
    ] == 6
    assert len(report["fit_artifact_hashes"]) == 15


def test_fit_payload_rejects_wrong_parameter_dimensions():
    with pytest.raises(ValueError, match="gamma draws have shape"):
        reporting.fit_payload(
            FakeFit(gamma_columns=6),
            [f"cell-{index}" for index in range(18)],
            gamma_columns=7,
        )
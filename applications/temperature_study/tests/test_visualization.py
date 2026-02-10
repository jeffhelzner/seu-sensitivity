"""Tests for visualization.py — DESIGN.md §6.1 primary analysis helpers."""
import pytest
import numpy as np
from ..visualization import (
    posterior_monotonicity_prob,
    alpha_slope,
    alpha_summary_table,
    forest_plot,
    position_bias_bars,
    consistency_line_plot,
    na_rate_bar_chart,
)


@pytest.fixture
def alpha_posteriors():
    """Synthetic α posteriors: higher temp → lower α (on average)."""
    rng = np.random.default_rng(42)
    return {
        0.0: rng.lognormal(1.0, 0.3, size=500),
        0.7: rng.lognormal(0.7, 0.3, size=500),
        1.5: rng.lognormal(0.4, 0.3, size=500),
    }


class TestPosteriorMonotonicity:
    def test_strongly_decreasing(self, alpha_posteriors):
        prob = posterior_monotonicity_prob(alpha_posteriors)
        # With well-separated means the probability should be high
        assert prob > 0.5

    def test_single_temp(self):
        draws = {0.0: np.ones(100)}
        assert posterior_monotonicity_prob(draws) == 1.0

    def test_equal_draws_zero_prob(self):
        """Identical draws across temps → strict decrease never holds."""
        rng = np.random.default_rng(99)
        same = rng.standard_normal(200)
        draws = {0.0: same.copy(), 1.0: same.copy()}
        prob = posterior_monotonicity_prob(draws)
        # Same draws → diff == 0, which is NOT strictly < 0
        assert prob == 0.0


class TestAlphaSlope:
    def test_negative_slope(self, alpha_posteriors):
        result = alpha_slope(alpha_posteriors)
        assert result["slope"] < 0  # decreasing with temperature
        assert result["ci_low"] < result["ci_high"]

    def test_two_temps(self):
        draws = {0.0: np.full(100, 2.0), 1.0: np.full(100, 1.0)}
        result = alpha_slope(draws)
        assert result["slope"] == pytest.approx(-1.0, abs=0.2)

    def test_single_temp(self):
        result = alpha_slope({0.0: np.ones(50)})
        assert result["slope"] == 0.0


class TestAlphaSummaryTable:
    def test_rows(self, alpha_posteriors):
        table = alpha_summary_table(alpha_posteriors, ci=0.90)
        assert len(table) == 3
        for row in table:
            assert "temperature" in row
            assert "median" in row
            assert "ci_low" in row
            assert "ci_high" in row
            assert row["ci_low"] < row["median"] < row["ci_high"]


class TestPlotFunctions:
    """Smoke tests — just verify they run without error and return a Figure."""

    def test_forest_plot(self, alpha_posteriors, tmp_path):
        fig = forest_plot(alpha_posteriors, save_path=str(tmp_path / "forest.png"))
        if fig is not None:
            assert (tmp_path / "forest.png").exists()
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_position_bias_bars(self, tmp_path):
        rates = {
            "0.0": {1: 0.5, 2: 0.3, 3: 0.2},
            "1.0": {1: 0.4, 2: 0.35, 3: 0.25},
        }
        fig = position_bias_bars(rates, save_path=str(tmp_path / "pos.png"))
        if fig is not None:
            assert (tmp_path / "pos.png").exists()
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_consistency_line_plot(self, tmp_path):
        summary = {
            "per_temperature": {
                "0.0": {
                    "unanimity": {"rate": 0.8, "n_unanimous": 8, "n_problems": 10},
                    "modal_agreement": {"mean_rate": 0.9, "per_problem": {}, "n_problems": 10},
                },
                "1.0": {
                    "unanimity": {"rate": 0.5, "n_unanimous": 5, "n_problems": 10},
                    "modal_agreement": {"mean_rate": 0.7, "per_problem": {}, "n_problems": 10},
                },
            }
        }
        fig = consistency_line_plot(summary, save_path=str(tmp_path / "cons.png"))
        if fig is not None:
            assert (tmp_path / "cons.png").exists()
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_na_rate_bar_chart(self, tmp_path):
        na_summary = {
            "per_temperature": {
                "0.0": {"na_rate": 0.01, "valid": 99, "na": 1, "total": 100},
                "1.0": {"na_rate": 0.05, "valid": 95, "na": 5, "total": 100},
            }
        }
        fig = na_rate_bar_chart(na_summary, save_path=str(tmp_path / "na.png"))
        if fig is not None:
            assert (tmp_path / "na.png").exists()
            import matplotlib.pyplot as plt
            plt.close(fig)

import math

import numpy as np
import pytest
from scipy.special import softmax

from applications.seu_sensitivity_study.ceiling_diagnostics import (
    conditional_log_likelihood, likelihood_slices, retained_ceiling_report,
)


@pytest.mark.parametrize("etas,choices,classification,limit", [
    ([[.2, .2]], [1], "constant", -math.log(2)),
    ([[.1, .5, .5]], [2], "finite_supremum_at_infinity", -math.log(2)),
    ([[.1, .5]], [1], "eventual_upper_tail_decay", None),
    ([[.5 - 1e-14, .5]], [1], "eventual_upper_tail_decay", None),
])
def test_analytic_limits(etas, choices, classification, limit):
    menus = [np.asarray(eta) for eta in etas]
    result = likelihood_slices(menus, choices, [0.0], np.arange(100) / 20, np.zeros(100))
    assert result["classification"] == classification
    assert result["analytic_limit"] == limit
    values = conditional_log_likelihood(menus, choices, [0.0], [0, 5, 40], .2)
    if classification == "constant":
        np.testing.assert_allclose(values, limit)
    elif limit is not None:
        assert np.all(np.diff(values) >= 0)
        assert values[-1] == pytest.approx(limit)
    else:
        assert values[-1] < values[0]


def test_direct_softmax_all_alternatives_and_slope():
    menus = [np.array([.1, .5, .3]), np.array([.2, .4])]
    actual = conditional_log_likelihood(menus, [2, 1], [-1, 1], [.7], .3)[0]
    expected = math.log(softmax(np.exp(.4) * menus[0])[1]) + math.log(softmax(np.exp(1.0) * menus[1])[0])
    assert actual == pytest.approx(expected)
    assert actual != conditional_log_likelihood(menus, [2, 1], [-1, 1], [.7], 0)[0]
    assert conditional_log_likelihood([np.array([.1, .5, .5])], [2], [0], [1000], 0)[0] == pytest.approx(-math.log(2))
    result = likelihood_slices(menus, [2, 1], [-1, 1], [1000, 1001], [0, .1])
    assert result["status"] == "unresolved"


def example():
    data = dict(J=1, R=3, M_total=2, cell=[1, 1], I=[[1, 0, 1]] * 2,
                y=[1, 2], eta=[[.5 - 1e-14, .1, .5]], s=[0, 0])
    preparation = dict(cell_ids=["cell"], item_ids=["a", "unused", "b"],
                       observation_metadata_version=1, exclusions=[dict(cell_id="cell", problem_id="excluded",
                       presentation_id=2, menu_size=4, reason="unresolved_choice")],
                       observations=[dict(cell_id="cell", problem_id=menu, presentation_id=1,
                       menu_size=2, item_order=["b", "a"], chosen_item_id=chosen, chosen_position=position)
                       for menu, chosen, position in [("first", "a", 2), ("same-composition", "b", 1)]])
    return data, preparation


@pytest.fixture
def synthetic_rows_only(monkeypatch):
    from applications.seu_sensitivity_study import ceiling_diagnostics

    monkeypatch.setattr(ceiling_diagnostics, "_validate_observation_universe", lambda *args: None)


def test_mapping_counts_near_ties_empty_sizes(synthetic_rows_only):
    data, preparation = example()
    cell = retained_ceiling_report(data, preparation, np.ones((10, 1)), np.zeros(10))["cells"]["cell"]
    assert cell["distinct_menu_ids"] == 2
    assert cell["exact"]["maximizer_choices"] == 1
    assert cell["near"]["maximizer_choices"] == 2
    assert cell["by_size"]["4"]["regret"] is None
    assert cell["by_size_and_presentation"]["4/2"]["exclusions"] == 1
    assert cell["individual"][0]["regret"] > 0
    data["y"][0] = 2
    with pytest.raises(ValueError, match="mapping"):
        retained_ceiling_report(data, preparation, np.ones((10, 1)), np.zeros(10))


def test_grid_refines_for_added_posterior_maximum():
    result = likelihood_slices([np.array([0., 1.])] * 3, [2, 2, 1], [0] * 3,
                               np.full(10, math.log(math.log(2))), np.zeros(10))
    row = result["slices"][0]
    assert row["refined"]
    assert row["initial_grid_count"] == 401
    assert row["endpoints"] == [-5, 10]
    assert row["evaluations"]["q50"]["difference_from_grid_maximum"] == 0


def test_cell_reordering_and_repeated_presentations(synthetic_rows_only):
    data, preparation = example()
    data.update(J=2, cell=[1, 2], eta=[[.5 - 1e-14, .1, .5], [.2, .1, .8]])
    preparation["cell_ids"] = ["first", "second"]
    preparation["exclusions"] = []
    for row, cell in zip(preparation["observations"], preparation["cell_ids"]):
        row["cell_id"] = cell
    levels = np.tile([1., 2.], (10, 1))
    baseline = retained_ceiling_report(data, preparation, levels, np.zeros(10))
    data["eta"].reverse()
    data["cell"] = [2, 1]
    preparation["cell_ids"].reverse()
    reordered = retained_ceiling_report(data, preparation, levels[:, ::-1], np.zeros(10))
    assert baseline["cells"] == reordered["cells"]
    data, preparation = example()
    preparation["observations"][1].update(problem_id="first", presentation_id=2)
    report = retained_ceiling_report(data, preparation, np.ones((10, 1)), np.zeros(10))
    assert report["cells"]["cell"]["distinct_menu_ids"] == 1
    assert report["cells"]["cell"]["observations"] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_inputs_unresolved_not_analytic_ceiling(bad):
    result = likelihood_slices([np.array([bad, .5])], [2], [0], [1, 2], [0, 1])
    assert result["status"] == "unresolved"
    assert "classification" not in result
    assert np.isnan(conditional_log_likelihood([np.array([bad, .5])], [2], [0], [1], 0)[0])
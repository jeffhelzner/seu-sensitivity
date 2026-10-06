import numpy as np
import pytest

from applications.seu_sensitivity_study.a3_prior_predictive import prior_draws
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig, build_cells
from applications.seu_sensitivity_study.confirmatory_analysis import matched_rq5_design


def test_realized_prior_contrasts_include_residuals_and_matched_coding():
    config = SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])
    for design in (config.design_matrix_for_pool("venture")[0], matched_rq5_design(build_cells(["venture", "hiring"]))[0]):
        levels, slope, gamma = prior_draws(design, "primary", 10000, np.random.default_rng(123))
        repeated, _, _ = prior_draws(design, "primary", 10000, np.random.default_rng(123))
        np.testing.assert_array_equal(levels, repeated)
        realized = levels[:, 0] - levels[:, -1]
        additive = gamma @ (design[0] - design[-1])
        assert np.std(realized - additive) > .4
        assert .19 < slope.std() < .21
        wider, _, _ = prior_draws(design, "prior_H", 10000, np.random.default_rng(456))
        assert 1.9 < np.std(wider[:, 0] - wider[:, -1]) / realized.std() < 2.1


def test_frozen_inputs_neutral_hash_bound_without_choice_reads(tmp_path):
    import hashlib
    import json
    from applications.seu_sensitivity_study.a3_prior_predictive import load_frozen
    from applications.seu_sensitivity_study.config import MODELS

    hashes = {}

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    for pool in ("venture", "hiring"):
        prefix = f"pools/{pool}/"
        write(prefix + "pool.json", {"items": [{"id": "item"}]})
        write(prefix + "problems.json", {"problems": []})
        for model in MODELS:
            write(prefix + f"assessments/{model.slug}.json", {
                "instruction": "neutral", "model_name": model.name,
                "assessments": [{"item_id": "item", "probabilities": [.2, .3, .5], "parse_ok": True}]})
    write("preflight_manifest.json", {"source_hashes": dict(hashes)})
    pools, sources = load_frozen(tmp_path)
    assert set(pools) == {"venture", "hiring"}
    assert len(sources) == 17
    assert not any("choices" in source for source in sources)
    write("pools/venture/problems.json", {"problems": [{"id": "tampered"}]})
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen(tmp_path)


@pytest.mark.parametrize("model,menu_id,expected", [
    ("claude-haiku-4-5", "VEN0044", .483134),
    ("claude-sonnet-4-5-thinking", "VEN0138", .519009),
])
def test_frozen_exact_maximizers_use_canonical_arithmetic(model, menu_id, expected):
    from scipy.special import softmax
    from applications.seu_sensitivity_study.a3_prior_predictive import DEFAULT_STAGE, load_frozen, menu_utilities
    from applications.seu_sensitivity_study.data_preparation import assessment_expected_utilities

    pools, _ = load_frozen(DEFAULT_STAGE)
    pool = pools["venture"]
    menu = next(menu for menu in pool["menus"] if menu["id"] == menu_id)
    probabilities = pool["probabilities"][model]
    actual = menu_utilities(probabilities, [menu])[0]
    canonical = np.asarray(assessment_expected_utilities(probabilities, item_ids=menu["item_ids"], utilities=[0., .5, 1.]))
    np.testing.assert_array_equal(actual, canonical)
    exact = actual == actual.max()
    choice = softmax(12.2 * (actual - actual.max()))
    assert choice[exact].sum() == pytest.approx(expected, abs=5e-7)
    old = np.array([.5 * probabilities[item][1] + probabilities[item][2] for item in menu["item_ids"]])
    assert not np.array_equal(exact, old == old.max())


def test_menu_utilities_normalize_before_dot_product():
    from applications.seu_sensitivity_study.a3_prior_predictive import menu_utilities
    from applications.seu_sensitivity_study.data_preparation import assessment_expected_utilities

    probabilities = {"first": [.1, .2, .7000000000000001], "second": [.2, .2, .6]}
    expected = assessment_expected_utilities(probabilities, item_ids=list(probabilities), utilities=[0., .5, 1.])
    np.testing.assert_array_equal(menu_utilities(probabilities, [{"item_ids": list(probabilities)}])[0], expected)
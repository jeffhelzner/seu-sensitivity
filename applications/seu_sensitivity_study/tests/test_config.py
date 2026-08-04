"""
Tests for the study configuration and the per-pool design matrix (§3, §8.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.seu_sensitivity_study import config as cfg
from applications.seu_sensitivity_study import schemas


class TestModels:
    def test_six_models_spanning_three_tiers(self):
        assert len(cfg.MODELS) == 6
        assert {m.tier for m in cfg.MODELS} == {"small", "flagship", "reasoning"}
        assert {m.vendor for m in cfg.MODELS} == {"openai", "anthropic"}

    def test_reasoning_models_declare_no_temperature(self):
        """o3-mini accepts no temperature parameter; thinking mode constrains it."""
        for model in cfg.MODELS:
            if model.tier == "reasoning":
                assert model.temperature is None
                assert model.request_params

    def test_each_vendor_has_a_flagship_and_a_small(self):
        """H1 is a within-vendor contrast, so both arms must exist per vendor."""
        for vendor in {m.vendor for m in cfg.MODELS}:
            tiers = {m.tier for m in cfg.MODELS if m.vendor == vendor}
            assert {"flagship", "small"} <= tiers

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            cfg.get_model_spec("gpt-5")


class TestCells:
    def test_full_factorial_is_54_cells(self):
        cells = cfg.build_cells()
        assert len(cells) == 54
        assert len({c.cell_id for c in cells}) == 54

    def test_eighteen_cells_per_pool(self):
        cells = cfg.build_cells(["venture"])
        assert len(cells) == 18
        assert {c.pool_id for c in cells} == {"venture"}

    def test_assessment_key_omits_prompt_condition(self):
        """Assessments are shared across the three prompt cells (B2)."""
        cells = cfg.build_cells(["venture"])
        keys = {c.assessment_key for c in cells}
        assert len(keys) == 6  # one per model, not per cell

    def test_assessment_jobs_force_the_neutral_instruction(self):
        jobs = cfg.assessment_keys(cfg.build_cells())
        assert len(jobs) == 18  # 6 models x 3 pools
        assert all(
            job.prompt_condition == schemas.ASSESSMENT_INSTRUCTION
            for job in jobs.values()
        )


class TestStudyConfig:
    def test_defaults_build_the_full_design(self):
        config = cfg.SEUSensitivityStudyConfig()
        assert len(config.cells) == 54
        assert config.menu_sizes == list(schemas.MENU_SIZES)
        assert config.num_presentations == schemas.NUM_PRESENTATIONS

    def test_presentation_count_cannot_be_overridden(self):
        with pytest.raises(ValueError, match="frozen"):
            cfg.SEUSensitivityStudyConfig(num_presentations=3)

    def test_expected_choice_calls_matches_the_plan(self):
        """54 cells x (100 + matched strata) menus x 2 presentations (§12)."""
        config = cfg.SEUSensitivityStudyConfig()
        # insurance 100, venture 140, hiring 140 menus; 18 cells each; 2 presentations
        expected = (100 + 140 + 140) * 18 * 2
        assert config.expected_choice_calls() == expected

    def test_expected_assessment_calls_is_per_model_pool(self):
        config = cfg.SEUSensitivityStudyConfig()
        counts = {"insurance": 30, "venture": 40, "hiring": 40}
        # 6 models x each pool's item count
        assert config.expected_assessment_calls(counts) == 6 * (30 + 40 + 40)

    def test_roundtrip_yaml(self, tmp_path):
        config = cfg.SEUSensitivityStudyConfig(pool_ids=["venture"])
        path = tmp_path / "study.yaml"
        config.save_yaml(str(path))
        restored = cfg.SEUSensitivityStudyConfig.from_yaml(str(path))
        assert restored.pool_ids == ["venture"]
        assert restored.menu_sizes == config.menu_sizes


class TestDesignMatrix:
    def test_shape_is_18_by_7(self):
        """Model-coded: 5 model dummies + 2 prompt dummies (B1)."""
        config = cfg.SEUSensitivityStudyConfig()
        X, names, cell_ids = config.design_matrix_for_pool("venture")
        assert X.shape == (18, 7)
        assert len(names) == 7
        assert len(cell_ids) == 18

    def test_reference_cell_row_is_all_zeros(self):
        config = cfg.SEUSensitivityStudyConfig()
        X, _, cell_ids = config.design_matrix_for_pool("venture")
        reference = cfg.get_model_spec(cfg.REFERENCE_MODEL).slug
        index = cell_ids.index(f"{reference}_{cfg.REFERENCE_PROMPT}_venture")
        assert not X[index].any()

    def test_each_row_has_at_most_one_model_and_one_prompt_dummy(self):
        config = cfg.SEUSensitivityStudyConfig()
        X, _, _ = config.design_matrix_for_pool("venture")
        assert set(np.unique(X)) <= {0.0, 1.0}
        assert X[:, :5].sum(axis=1).max() == 1
        assert X[:, 5:].sum(axis=1).max() == 1

    def test_matrix_is_full_rank(self):
        """A rank-deficient X would make the contrasts inestimable."""
        config = cfg.SEUSensitivityStudyConfig()
        X, _, _ = config.design_matrix_for_pool("venture")
        with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        assert np.linalg.matrix_rank(with_intercept) == with_intercept.shape[1]

    def test_main_effects_leave_room_but_interaction_would_saturate(self):
        """8 parameters for 18 cells; a full interaction would be exactly 18."""
        config = cfg.SEUSensitivityStudyConfig()
        X, _, _ = config.design_matrix_for_pool("venture")
        n_cells, p_main = X.shape
        assert 1 + p_main == 8
        assert 1 + p_main + (5 * 2) == n_cells

    def test_row_order_matches_cells_for_pool(self):
        config = cfg.SEUSensitivityStudyConfig()
        _, _, cell_ids = config.design_matrix_for_pool("venture")
        assert cell_ids == [c.cell_id for c in config.cells_for_pool("venture")]

    def test_unknown_pool_raises(self):
        config = cfg.SEUSensitivityStudyConfig(pool_ids=["venture"])
        with pytest.raises(KeyError, match="No cells configured"):
            config.design_matrix_for_pool("hiring")

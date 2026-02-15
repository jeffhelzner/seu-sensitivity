"""
Tests for the data_preparation module.

Covers:
- EmbeddingReducer (pooled PCA fit/transform)
- filter_valid_choices (NA filtering)
- StanDataBuilder (assembly + validation)
"""
import json
import pytest
import numpy as np
from pathlib import Path

from applications.temperature_study.data_preparation import (
    EmbeddingReducer,
    StanDataBuilder,
    filter_valid_choices,
    save_stan_data,
    save_na_log,
    save_reduced_embeddings,
)
from applications.temperature_study.config import StudyConfig


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def raw_embeddings_per_temp(rng):
    """Simulate raw embeddings for 2 temperatures, keyed by claim ID."""
    dims = 64  # small for test speed
    embs: dict[float, dict[str, np.ndarray]] = {}
    for temp in [0.0, 0.7]:
        d = {}
        for cid in ["C001", "C002", "C003", "C004", "C005"]:
            d[cid] = rng.standard_normal(dims)
        embs[temp] = d
    return embs


@pytest.fixture
def mock_problems():
    """Three minimal problems."""
    return [
        {
            "id": "P0001",
            "claim_ids": ["C001", "C002", "C003"],
            "num_alternatives": 3,
            "presentations": [
                {"presentation_id": 1, "order": ["C001", "C002", "C003"]},
                {"presentation_id": 2, "order": ["C003", "C001", "C002"]},
            ],
        },
        {
            "id": "P0002",
            "claim_ids": ["C001", "C002"],
            "num_alternatives": 2,
            "presentations": [
                {"presentation_id": 1, "order": ["C001", "C002"]},
                {"presentation_id": 2, "order": ["C002", "C001"]},
            ],
        },
        {
            "id": "P0003",
            "claim_ids": ["C002", "C003"],
            "num_alternatives": 2,
            "presentations": [
                {"presentation_id": 1, "order": ["C002", "C003"]},
                {"presentation_id": 2, "order": ["C003", "C002"]},
            ],
        },
    ]


@pytest.fixture
def all_valid_choices():
    """Choices dict where all entries are valid."""
    return {
        "temperature": 0.7,
        "total_choices": 6,
        "valid_choices": 6,
        "na_choices": 0,
        "na_rate": 0.0,
        "choices": [
            {
                "problem_id": "P0001",
                "presentation_id": 1,
                "claim_order": ["C001", "C002", "C003"],
                "position_chosen": 2,
                "claim_chosen": "C002",
                "valid": True,
                "raw_response": "2",
            },
            {
                "problem_id": "P0001",
                "presentation_id": 2,
                "claim_order": ["C003", "C001", "C002"],
                "position_chosen": 3,
                "claim_chosen": "C002",
                "valid": True,
                "raw_response": "3",
            },
            {
                "problem_id": "P0002",
                "presentation_id": 1,
                "claim_order": ["C001", "C002"],
                "position_chosen": 1,
                "claim_chosen": "C001",
                "valid": True,
                "raw_response": "1",
            },
            {
                "problem_id": "P0002",
                "presentation_id": 2,
                "claim_order": ["C002", "C001"],
                "position_chosen": 2,
                "claim_chosen": "C001",
                "valid": True,
                "raw_response": "2",
            },
            {
                "problem_id": "P0003",
                "presentation_id": 1,
                "claim_order": ["C002", "C003"],
                "position_chosen": 1,
                "claim_chosen": "C002",
                "valid": True,
                "raw_response": "1",
            },
            {
                "problem_id": "P0003",
                "presentation_id": 2,
                "claim_order": ["C003", "C002"],
                "position_chosen": 1,
                "claim_chosen": "C003",
                "valid": True,
                "raw_response": "1",
            },
        ],
    }


@pytest.fixture
def mixed_choices():
    """Choices dict with some NAs."""
    return {
        "temperature": 1.0,
        "total_choices": 4,
        "valid_choices": 3,
        "na_choices": 1,
        "na_rate": 0.25,
        "choices": [
            {
                "problem_id": "P0001",
                "presentation_id": 1,
                "claim_order": ["C001", "C002", "C003"],
                "position_chosen": 2,
                "claim_chosen": "C002",
                "valid": True,
                "raw_response": "2",
            },
            {
                "problem_id": "P0001",
                "presentation_id": 2,
                "claim_order": ["C003", "C001", "C002"],
                "position_chosen": None,
                "claim_chosen": None,
                "valid": False,
                "raw_response": "I think Claim 2 but also Claim 1...",
            },
            {
                "problem_id": "P0002",
                "presentation_id": 1,
                "claim_order": ["C001", "C002"],
                "position_chosen": 1,
                "claim_chosen": "C001",
                "valid": True,
                "raw_response": "1",
            },
            {
                "problem_id": "P0003",
                "presentation_id": 1,
                "claim_order": ["C002", "C003"],
                "position_chosen": 2,
                "claim_chosen": "C003",
                "valid": True,
                "raw_response": "Claim 2",
            },
        ],
    }


# ── EmbeddingReducer Tests ──────────────────────────────────────────


class TestEmbeddingReducer:

    def test_fit_and_transform(self, rng):
        embs = {f"key_{i}": rng.standard_normal(128) for i in range(50)}
        reducer = EmbeddingReducer(target_dim=8)
        reducer.fit(embs)
        reduced = reducer.transform(embs)

        assert len(reduced) == 50
        for v in reduced.values():
            assert v.shape == (8,)

    def test_explained_variance_between_0_and_1(self, rng):
        embs = {f"k{i}": rng.standard_normal(64) for i in range(100)}
        reducer = EmbeddingReducer(target_dim=16)
        reducer.fit(embs)
        summary = reducer.get_summary()
        assert 0 < summary["total_explained_variance"] <= 1.0

    def test_clamp_target_dim_to_n_samples(self, rng):
        # Only 5 samples — target_dim=32 should be clamped
        embs = {f"k{i}": rng.standard_normal(128) for i in range(5)}
        reducer = EmbeddingReducer(target_dim=32)
        reducer.fit(embs)
        assert reducer.pca.n_components_ <= 5

    def test_fit_transform_pooled(self, raw_embeddings_per_temp):
        reducer = EmbeddingReducer(target_dim=8)
        per_temp_reduced = reducer.fit_transform_pooled(raw_embeddings_per_temp)

        assert set(per_temp_reduced.keys()) == {0.0, 0.7}
        for temp, embs in per_temp_reduced.items():
            for v in embs.values():
                assert v.shape == (8,)

    def test_same_keys_after_transform(self, raw_embeddings_per_temp):
        reducer = EmbeddingReducer(target_dim=8)
        per_temp_reduced = reducer.fit_transform_pooled(raw_embeddings_per_temp)

        for temp in [0.0, 0.7]:
            assert set(per_temp_reduced[temp].keys()) == set(
                raw_embeddings_per_temp[temp].keys()
            )

    def test_error_on_empty(self):
        reducer = EmbeddingReducer(target_dim=8)
        with pytest.raises(ValueError, match="empty"):
            reducer.fit({})

    def test_error_on_transform_before_fit(self, rng):
        reducer = EmbeddingReducer(target_dim=8)
        with pytest.raises(RuntimeError, match="fit"):
            reducer.transform({"k": rng.standard_normal(64)})


# ── filter_valid_choices Tests ───────────────────────────────────────


class TestFilterValidChoices:

    def test_all_valid(self, all_valid_choices):
        valid, na_log = filter_valid_choices(all_valid_choices)
        assert len(valid) == 6
        assert na_log["removed_observations"] == 0
        assert na_log["na_rate"] == 0.0

    def test_mixed(self, mixed_choices):
        valid, na_log = filter_valid_choices(mixed_choices)
        assert len(valid) == 3
        assert na_log["removed_observations"] == 1
        assert len(na_log["removed_entries"]) == 1
        assert na_log["removed_entries"][0]["problem_id"] == "P0001"
        assert na_log["removed_entries"][0]["presentation_id"] == 2

    def test_na_log_has_required_fields(self, mixed_choices):
        _, na_log = filter_valid_choices(mixed_choices)
        assert "temperature" in na_log
        assert "total_observations" in na_log
        assert "valid_observations" in na_log
        assert "removed_observations" in na_log
        assert "na_rate" in na_log
        assert "removed_entries" in na_log
        assert "filtered_at" in na_log


# ── StanDataBuilder Tests ────────────────────────────────────────────


class TestStanDataBuilder:

    def _make_reduced_embs(self, problems, dim=8):
        """Generate fake reduced embeddings keyed by claim ID."""
        rng = np.random.default_rng(42)
        embs = {}
        for p in problems:
            for cid in p["claim_ids"]:
                if cid not in embs:
                    embs[cid] = rng.standard_normal(dim)
        return embs

    def test_build_produces_valid_stan_data(
        self, mock_problems, all_valid_choices
    ):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )

        assert stan_data["M"] == 6
        assert stan_data["K"] == 3
        assert stan_data["D"] == 8
        assert stan_data["R"] == 3  # C001, C002, C003

    def test_validation_passes(self, mock_problems, all_valid_choices):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )
        issues = StanDataBuilder.validate_stan_data(stan_data)
        assert issues == []

    def test_y_values_within_bounds(self, mock_problems, all_valid_choices):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )

        for m in range(stan_data["M"]):
            N_m = sum(stan_data["I"][m])
            assert 1 <= stan_data["y"][m] <= N_m

    def test_indicator_row_sums(self, mock_problems, all_valid_choices):
        """Each observation's I row should sum to the problem's num_alternatives."""
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )

        prob_lookup = {p["id"]: p for p in mock_problems}
        for m, entry in enumerate(all_valid_choices["choices"]):
            expected_n = prob_lookup[entry["problem_id"]]["num_alternatives"]
            assert sum(stan_data["I"][m]) == expected_n

    def test_w_dimensions(self, mock_problems, all_valid_choices):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )

        assert len(stan_data["w"]) == stan_data["R"]
        for vec in stan_data["w"]:
            assert len(vec) == stan_data["D"]

    def test_error_on_empty_choices(self, mock_problems):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        with pytest.raises(ValueError, match="No valid choices"):
            builder.build([], {}, mock_problems)

    def test_error_on_missing_embedding(
        self, mock_problems, all_valid_choices
    ):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        # Empty embeddings → should fail
        with pytest.raises(ValueError, match="No reduced embedding"):
            builder.build(all_valid_choices["choices"], {}, mock_problems)

    def test_with_na_filtered_input(self, mock_problems, mixed_choices):
        """Test end-to-end: filter NAs then build Stan data."""
        valid_entries, _ = filter_valid_choices(mixed_choices)
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        embs = self._make_reduced_embs(mock_problems)
        stan_data = builder.build(valid_entries, embs, mock_problems)

        assert stan_data["M"] == 3  # One NA removed
        issues = StanDataBuilder.validate_stan_data(stan_data)
        assert issues == []


# ── Serialization Tests ──────────────────────────────────────────────


class TestSerialization:

    def test_save_stan_data(self, tmp_path, mock_problems, all_valid_choices):
        config = StudyConfig(num_problems=3, K=3, target_dim=8)
        builder = StanDataBuilder(config)
        rng = np.random.default_rng(42)
        embs = {}
        for p in mock_problems:
            for cid in p["claim_ids"]:
                if cid not in embs:
                    embs[cid] = rng.standard_normal(8)

        stan_data = builder.build(
            all_valid_choices["choices"], embs, mock_problems
        )
        path = save_stan_data(stan_data, tmp_path / "stan_data.json")
        assert path.exists()

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["M"] == 6

    def test_save_na_log(self, tmp_path, mixed_choices):
        _, na_log = filter_valid_choices(mixed_choices)
        path = save_na_log(na_log, tmp_path / "na_log.json")
        assert path.exists()

    def test_save_reduced_embeddings(self, tmp_path, rng):
        embs = {f"k{i}": rng.standard_normal(8) for i in range(10)}
        path = save_reduced_embeddings(embs, tmp_path / "reduced.npz")
        assert path.exists()
        loaded = np.load(path)
        assert len(loaded.files) == 10

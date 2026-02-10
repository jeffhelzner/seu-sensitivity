"""
Tests for the DeliberationCollector module.

Uses mock LLM / embedding clients to test logic without API calls.
"""
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from applications.temperature_study.config import StudyConfig
from applications.temperature_study.problem_generator import ProblemGenerator
from applications.temperature_study.deliberation_collector import (
    DeliberationCollector,
)


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLMClient:
    """Mock LLM that returns a canned deliberation response."""

    model = "mock-model"
    total_input_tokens = 0
    total_output_tokens = 0

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=256):
        return f"This claim shows moderate fraud indicators. Temperature was {temperature}."

    def get_estimated_cost(self):
        return 0.01

    def get_usage_summary(self):
        return {
            "model": self.model,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "estimated_cost_usd": 0.01,
        }


class MockEmbeddingClient:
    """Mock embedding client that returns deterministic vectors."""

    model = "mock-embedding"
    total_tokens = 0
    _dim = 1536

    def embed_single(self, text):
        # Deterministic: hash-based so different texts get different vectors
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.standard_normal(self._dim).tolist()

    def embed(self, texts):
        return [self.embed_single(t) for t in texts]

    def get_estimated_cost(self):
        return 0.001

    def get_usage_summary(self):
        return {
            "model": self.model,
            "total_tokens": 100,
            "estimated_cost_usd": 0.001,
        }


@pytest.fixture
def mock_config(tmp_path, sample_claims):
    """Config pointing to tmp dirs with a valid claims + prompts file."""
    claims_path = tmp_path / "data" / "claims.json"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claims_path, "w") as f:
        json.dump(
            {"claims": sample_claims, "consequences": ["c1", "c2", "c3"]}, f
        )

    prompts_path = tmp_path / "configs" / "prompts.yaml"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    prompts = {
        "deliberation": {
            "system": "You are a claims analyst.",
            "user": "Claims:\n{claims_list}\n\nAnalyze Claim {target_letter}.",
        },
        "choice": {
            "system": "You are a claims analyst.",
            "user": "Claims:\n{claims_list}\n\nChoose ({num_range}).",
        },
    }
    with open(prompts_path, "w") as f:
        yaml.dump(prompts, f)

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    return StudyConfig(
        temperatures=[0.0, 0.7],
        num_problems=3,
        min_alternatives=2,
        max_alternatives=3,
        num_presentations=2,
        seed=42,
        claims_file=str(claims_path),
        prompts_file=str(prompts_path),
        results_dir=str(results_dir),
    )


@pytest.fixture
def mock_generator(mock_config):
    return ProblemGenerator.from_config(mock_config)


@pytest.fixture
def mock_problems(mock_generator, mock_config):
    return mock_generator.generate_problems_from_config(mock_config)


# ── Tests ────────────────────────────────────────────────────────────


class TestDeliberationCollector:
    """Tests for deliberation collection logic."""

    def test_collect_temperature_returns_correct_structure(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        delib_dict, raw_embs = collector.collect_temperature(
            mock_problems, 0.7
        )

        assert delib_dict["temperature"] == 0.7
        assert "total_deliberations" in delib_dict
        assert "deliberations" in delib_dict
        assert delib_dict["total_deliberations"] == len(
            delib_dict["deliberations"]
        )

    def test_one_deliberation_per_claim_per_problem(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        delib_dict, _ = collector.collect_temperature(mock_problems, 0.0)

        expected_count = sum(p["num_alternatives"] for p in mock_problems)
        assert delib_dict["total_deliberations"] == expected_count

    def test_embedding_keys_match_entries(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        delib_dict, raw_embs = collector.collect_temperature(
            mock_problems, 0.7
        )

        for entry in delib_dict["deliberations"]:
            key = entry["embedding_key"]
            assert key in raw_embs
            assert raw_embs[key].shape == (1536,)

    def test_deliberation_entries_have_required_fields(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        delib_dict, _ = collector.collect_temperature(mock_problems, 0.3)

        for entry in delib_dict["deliberations"]:
            assert "problem_id" in entry
            assert "claim_id" in entry
            assert "target_letter" in entry
            assert "temperature" in entry
            assert "response" in entry
            assert "embedding_key" in entry
            assert entry["temperature"] == 0.3

    def test_collect_all_temperatures(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        per_temp, pooled_raw = collector.collect_all_temperatures(mock_problems)

        assert set(per_temp.keys()) == {0.0, 0.7}
        # Pooled embeddings have temperature-tagged keys
        for key in pooled_raw:
            assert "_T" in key

    def test_save_deliberations(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        delib_dict, raw_embs = collector.collect_temperature(
            mock_problems, 0.7
        )
        json_path, npz_path = collector.save_deliberations(
            0.7, delib_dict, raw_embs
        )

        assert json_path.exists()
        assert npz_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["temperature"] == 0.7

    def test_usage_summary(self, mock_config, mock_generator):
        collector = DeliberationCollector(
            mock_config,
            mock_generator,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        summary = collector.get_usage_summary()
        assert "llm" in summary
        assert "embedding" in summary
        assert "total_estimated_cost_usd" in summary

"""
Tests for the AssessmentCollector module.

Uses mock LLM / embedding clients to test logic without API calls.
"""
import json
import pytest
import numpy as np
from pathlib import Path

from applications.temperature_study.config import StudyConfig
from applications.temperature_study.assessment_collector import (
    AssessmentCollector,
)


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLMClient:
    """Mock LLM that returns a canned assessment response."""

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
        "assessment": {
            "system": "You are evaluating insurance claims.",
            "user": "Claim description:\n{claim_description}\n\nProvide a brief assessment.",
        },
        "choice": {
            "system": "You are selecting a claim for investigation.",
            "user": "Assessments:\n{assessments_list}\n\nChoose ({num_range}).",
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


# ── Tests ────────────────────────────────────────────────────────────


class TestAssessmentCollector:
    """Tests for assessment collection logic."""

    def test_collect_temperature_returns_correct_structure(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, raw_embs = collector.collect_temperature(0.7)

        assert assess_dict["temperature"] == 0.7
        assert "total_assessments" in assess_dict
        assert "assessments" in assess_dict
        assert assess_dict["total_assessments"] == len(
            assess_dict["assessments"]
        )

    def test_one_assessment_per_claim(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, _ = collector.collect_temperature(0.0)

        # Exactly one assessment per claim
        assert assess_dict["total_assessments"] == len(sample_claims)

    def test_embedding_keys_are_claim_ids(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, raw_embs = collector.collect_temperature(0.7)

        expected_ids = {c["id"] for c in sample_claims}
        assert set(raw_embs.keys()) == expected_ids
        for key, vec in raw_embs.items():
            assert vec.shape == (1536,)

    def test_embedding_keys_match_entries(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, raw_embs = collector.collect_temperature(0.7)

        for entry in assess_dict["assessments"]:
            key = entry["embedding_key"]
            assert key in raw_embs
            assert raw_embs[key].shape == (1536,)

    def test_assessment_entries_have_required_fields(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, _ = collector.collect_temperature(0.3)

        for entry in assess_dict["assessments"]:
            assert "claim_id" in entry
            assert "temperature" in entry
            assert "response" in entry
            assert "embedding_key" in entry
            assert entry["temperature"] == 0.3

    def test_no_problem_id_in_entries(
        self, mock_config, sample_claims
    ):
        """Assessment entries should NOT have a problem_id field —
        assessments are per-claim, not per-problem."""
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, _ = collector.collect_temperature(0.0)

        for entry in assess_dict["assessments"]:
            assert "problem_id" not in entry

    def test_collect_all_temperatures(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        per_temp, per_temp_raw = collector.collect_all_temperatures()

        assert set(per_temp.keys()) == {0.0, 0.7}
        assert set(per_temp_raw.keys()) == {0.0, 0.7}
        # Each temperature should have claim-keyed embeddings
        for temp, raw in per_temp_raw.items():
            assert set(raw.keys()) == {c["id"] for c in sample_claims}

    def test_get_assessment_texts(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, _ = collector.collect_temperature(0.7)
        texts = AssessmentCollector.get_assessment_texts(assess_dict)

        assert isinstance(texts, dict)
        assert set(texts.keys()) == {c["id"] for c in sample_claims}
        for text in texts.values():
            assert isinstance(text, str)
            assert len(text) > 0

    def test_save_assessments(
        self, mock_config, sample_claims
    ):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        assess_dict, raw_embs = collector.collect_temperature(0.7)
        json_path, npz_path = collector.save_assessments(
            0.7, assess_dict, raw_embs
        )

        assert json_path.exists()
        assert npz_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["temperature"] == 0.7

    def test_usage_summary(self, mock_config, sample_claims):
        collector = AssessmentCollector(
            mock_config,
            sample_claims,
            llm_client=MockLLMClient(),
            embedding_client=MockEmbeddingClient(),
        )
        summary = collector.get_usage_summary()
        assert "llm" in summary
        assert "embedding" in summary
        assert "total_estimated_cost_usd" in summary

"""
Integration test for the TemperatureStudyRunner.

Uses mock LLM / embedding clients to exercise the full pipeline
(phases 1–3) without making API calls.
"""
import json
import pytest
import yaml
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from applications.temperature_study.config import StudyConfig
from applications.temperature_study.study_runner import TemperatureStudyRunner
from applications.temperature_study.llm_client import LLMClient, EmbeddingClient


# ── Mock clients ─────────────────────────────────────────────────────


class MockLLM(LLMClient):
    """Mock LLM: deliberation returns text, choice returns "1"."""

    def __init__(self, **kwargs):
        super().__init__(model="mock", **kwargs)

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=256):
        # Distinguish deliberation vs. choice by max_tokens or content
        if max_tokens <= 64:
            return "1"
        return "This claim shows moderate fraud indicators worth investigating."

    def get_estimated_cost(self):
        return 0.0


class MockEmbedder(EmbeddingClient):
    """Mock embedding client returning deterministic 64-dim vectors."""

    def __init__(self, **kwargs):
        # Skip real OpenAI init
        self.model = "mock-embed"
        self.max_retries = 1
        self.retry_delay = 0
        self.total_tokens = 0
        self._dim = 64

    def embed_single(self, text):
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.standard_normal(self._dim).tolist()

    def embed(self, texts):
        return [self.embed_single(t) for t in texts]

    def get_estimated_cost(self):
        return 0.0

    def get_usage_summary(self):
        return {"model": self.model, "total_tokens": 0, "estimated_cost_usd": 0.0}


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def study_config(tmp_path, sample_claims):
    """A minimal study config pointing at tmp dirs."""
    claims_path = tmp_path / "data" / "claims.json"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claims_path, "w") as f:
        json.dump({"claims": sample_claims, "consequences": ["c1", "c2", "c3"]}, f)

    prompts_path = tmp_path / "configs" / "prompts.yaml"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
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

    return StudyConfig(
        temperatures=[0.0, 0.7],
        num_problems=3,
        min_alternatives=2,
        max_alternatives=3,
        num_presentations=2,
        K=3,
        target_dim=8,
        seed=42,
        claims_file=str(claims_path),
        prompts_file=str(prompts_path),
        results_dir=str(results_dir),
        fit_models=False,
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestStudyRunnerIntegration:
    """End-to-end pipeline test (phases 1–3) with mocked API clients."""

    def _patch_clients(self, runner):
        """Replace lazy-init clients with mocks."""
        mock_llm = MockLLM()
        mock_emb = MockEmbedder()

        # Inject into the collectors that will be lazily created
        gen = runner._get_generator()

        from applications.temperature_study.deliberation_collector import (
            DeliberationCollector,
        )
        from applications.temperature_study.choice_collector import ChoiceCollector

        runner._delib_collector = DeliberationCollector(
            runner.config, gen, llm_client=mock_llm, embedding_client=mock_emb
        )
        runner._choice_collector = ChoiceCollector(
            runner.config, gen, llm_client=mock_llm
        )

    def test_full_pipeline_runs(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)

        summary = runner.run(skip_model_fitting=True)

        assert "phases" in summary
        assert "phase1" in summary["phases"]
        assert "phase2a_deliberation" in summary["phases"]
        assert "phase2b_choices" in summary["phases"]
        assert "phase3_data_prep" in summary["phases"]

    def test_stan_data_files_created(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        results = Path(study_config.results_dir)
        assert (results / "stan_data_T0_0.json").exists()
        assert (results / "stan_data_T0_7.json").exists()

    def test_na_logs_created(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        results = Path(study_config.results_dir)
        assert (results / "na_removal_log_T0_0.json").exists()
        assert (results / "na_removal_log_T0_7.json").exists()

    def test_problems_saved(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        results = Path(study_config.results_dir)
        assert (results / "problems.json").exists()
        with open(results / "problems.json") as f:
            data = json.load(f)
        assert len(data["problems"]) == 3

    def test_stan_data_valid(self, study_config):
        """Check that produced Stan data passes validation."""
        from applications.temperature_study.data_preparation import StanDataBuilder

        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        results = Path(study_config.results_dir)
        for temp_str in ["T0_0", "T0_7"]:
            with open(results / f"stan_data_{temp_str}.json") as f:
                stan_data = json.load(f)
            issues = StanDataBuilder.validate_stan_data(stan_data)
            assert issues == [], f"Validation issues at {temp_str}: {issues}"

    def test_run_summary_saved(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        summary_path = Path(study_config.results_dir) / "run_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert "duration_seconds" in summary

    def test_reduced_embeddings_saved(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        runner.run(skip_model_fitting=True)

        results = Path(study_config.results_dir)
        assert (results / "embeddings_reduced_T0_0.npz").exists()
        assert (results / "embeddings_reduced_T0_7.npz").exists()

    def test_pca_summary_in_output(self, study_config):
        runner = TemperatureStudyRunner(study_config)
        self._patch_clients(runner)
        summary = runner.run(skip_model_fitting=True)

        pca = summary["phases"]["phase3_data_prep"]["pca_summary"]
        assert pca["fitted"] is True
        assert 0 < pca["total_explained_variance"] <= 1.0

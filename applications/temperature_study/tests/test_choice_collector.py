"""
Tests for the ChoiceCollector module.

Uses mock LLM clients to test parsing, NA handling, and position
metadata without API calls.
"""
import json
import pytest
import yaml
from pathlib import Path

from applications.temperature_study.config import StudyConfig
from applications.temperature_study.problem_generator import ProblemGenerator
from applications.temperature_study.choice_collector import ChoiceCollector


# ── Mock clients ─────────────────────────────────────────────────────


class MockLLMClientValid:
    """Mock LLM that always returns a valid choice (position 1)."""

    model = "mock-valid"
    total_input_tokens = 0
    total_output_tokens = 0

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=64):
        return "1"

    def get_estimated_cost(self):
        return 0.005

    def get_usage_summary(self):
        return {
            "model": self.model,
            "total_input_tokens": 50,
            "total_output_tokens": 10,
            "estimated_cost_usd": 0.005,
        }


class MockLLMClientMixed:
    """Mock LLM that returns valid choices sometimes, NA others."""

    model = "mock-mixed"
    total_input_tokens = 0
    total_output_tokens = 0
    _call_count = 0

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=64):
        self._call_count += 1
        if self._call_count % 4 == 0:
            # Ambiguous → should parse as NA
            return "I would recommend Claim 2, but Claim 1 also has merit..."
        return "2"

    def get_estimated_cost(self):
        return 0.005

    def get_usage_summary(self):
        return {
            "model": self.model,
            "total_input_tokens": 50,
            "total_output_tokens": 10,
            "estimated_cost_usd": 0.005,
        }


class MockLLMClientAllNA:
    """Mock LLM that always returns unparseable responses."""

    model = "mock-na"
    total_input_tokens = 0
    total_output_tokens = 0

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=64):
        return "I cannot determine which claim to select given the information."

    def get_estimated_cost(self):
        return 0.005

    def get_usage_summary(self):
        return {
            "model": self.model,
            "total_input_tokens": 50,
            "total_output_tokens": 10,
            "estimated_cost_usd": 0.005,
        }


@pytest.fixture
def mock_config(tmp_path, sample_claims):
    """Config with tmp dirs and valid claims/prompts files."""
    claims_path = tmp_path / "data" / "claims.json"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claims_path, "w") as f:
        json.dump(
            {"claims": sample_claims, "consequences": ["c1", "c2", "c3"]}, f
        )

    prompts_path = tmp_path / "configs" / "prompts.yaml"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture
def mock_generator(mock_config):
    return ProblemGenerator.from_config(mock_config)


@pytest.fixture
def mock_problems(mock_generator, mock_config):
    return mock_generator.generate_problems_from_config(mock_config)


# ── Tests ────────────────────────────────────────────────────────────


class TestChoiceCollectorValid:
    """Tests with a mock that always returns valid choices."""

    def test_collect_returns_correct_structure(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.7)

        assert result["temperature"] == 0.7
        assert "total_choices" in result
        assert "valid_choices" in result
        assert "na_choices" in result
        assert "na_rate" in result
        assert "choices" in result

    def test_all_valid_when_parseable(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.0)

        assert result["na_choices"] == 0
        assert result["valid_choices"] == result["total_choices"]
        assert result["na_rate"] == 0.0

    def test_total_choices_matches_presentations(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.0)

        expected = sum(len(p["presentations"]) for p in mock_problems)
        assert result["total_choices"] == expected

    def test_choice_entries_have_required_fields(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.7)

        for c in result["choices"]:
            assert "problem_id" in c
            assert "presentation_id" in c
            assert "claim_order" in c
            assert "position_chosen" in c
            assert "claim_chosen" in c
            assert "valid" in c
            assert "raw_response" in c

    def test_valid_entries_have_correct_claim_mapping(
        self, mock_config, mock_generator, mock_problems
    ):
        """Position → claim_id mapping must be consistent with claim_order."""
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.0)

        for c in result["choices"]:
            if c["valid"]:
                pos = c["position_chosen"]
                assert c["claim_chosen"] == c["claim_order"][pos - 1]


class TestChoiceCollectorNA:
    """Tests for NA handling."""

    def test_mixed_responses_have_nas(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientMixed()
        )
        result = collector.collect_temperature(mock_problems, 0.7)

        assert result["na_choices"] > 0
        assert result["valid_choices"] > 0
        assert result["na_rate"] > 0

    def test_na_entries_have_null_fields(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientMixed()
        )
        result = collector.collect_temperature(mock_problems, 1.0)

        na_entries = [c for c in result["choices"] if not c["valid"]]
        assert len(na_entries) > 0
        for c in na_entries:
            assert c["position_chosen"] is None
            assert c["claim_chosen"] is None
            assert c["raw_response"] is not None  # raw response always recorded

    def test_all_na_when_unparseable(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientAllNA()
        )
        result = collector.collect_temperature(mock_problems, 1.5)

        assert result["valid_choices"] == 0
        assert result["na_choices"] == result["total_choices"]
        assert result["na_rate"] == 1.0

    def test_get_na_entries(self, mock_config, mock_generator, mock_problems):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientMixed()
        )
        result = collector.collect_temperature(mock_problems, 0.7)
        na_entries = ChoiceCollector.get_na_entries(result)
        assert len(na_entries) == result["na_choices"]


class TestChoiceCollectorAcrossTemps:
    """Tests for multi-temperature collection."""

    def test_collect_all_temperatures(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        per_temp = collector.collect_all_temperatures(mock_problems)

        assert set(per_temp.keys()) == {0.0, 0.7}
        for temp, result in per_temp.items():
            assert result["temperature"] == temp

    def test_summarize_na(self, mock_config, mock_generator, mock_problems):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientMixed()
        )
        per_temp = collector.collect_all_temperatures(mock_problems)
        summary = ChoiceCollector.summarize_na(per_temp)

        assert "per_temperature" in summary
        assert "overall" in summary
        assert summary["overall"]["total"] > 0


class TestPositionBias:
    """Tests for position bias analysis helpers."""

    def test_position_choice_rates(
        self, mock_config, mock_generator, mock_problems
    ):
        # MockLLMClientValid always picks position 1
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.0)
        rates = ChoiceCollector.position_choice_rates(result)

        # All choices go to position 1
        assert rates[1] == 1.0

    def test_position_choice_rates_empty_when_all_na(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientAllNA()
        )
        result = collector.collect_temperature(mock_problems, 0.0)
        rates = ChoiceCollector.position_choice_rates(result)
        assert rates == {}


class TestSerialization:
    """Tests for save/load."""

    def test_save_load_roundtrip(
        self, mock_config, mock_generator, mock_problems
    ):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        result = collector.collect_temperature(mock_problems, 0.7)
        path = collector.save_choices(0.7, result)

        assert path.exists()
        loaded = ChoiceCollector.load_choices(path)
        assert loaded["temperature"] == 0.7
        assert loaded["total_choices"] == result["total_choices"]

    def test_save_all(self, mock_config, mock_generator, mock_problems):
        collector = ChoiceCollector(
            mock_config, mock_generator, llm_client=MockLLMClientValid()
        )
        per_temp = collector.collect_all_temperatures(mock_problems)
        paths = collector.save_all(per_temp)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

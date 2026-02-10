"""Tests for consistency_analysis.py — DESIGN.md §6.3."""
import pytest
from ..consistency_analysis import (
    unanimity_rate,
    modal_agreement_rate,
    consistency_by_temperature,
    entropy_of_choices,
)


class TestUnanimityRate:
    def test_all_unanimous(self):
        """If every presentation picks the same claim → rate = 1.0."""
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["B", "A"]},
        ]
        result = unanimity_rate(entries)
        assert result["rate"] == 1.0
        assert result["n_unanimous"] == 1
        assert result["n_problems"] == 1

    def test_no_unanimity(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "B",
             "claim_order": ["B", "A"]},
        ]
        result = unanimity_rate(entries)
        assert result["rate"] == 0.0

    def test_mixed(self, sample_choice_entries):
        result = unanimity_rate(sample_choice_entries)
        # P0002 is unanimous (C002 chosen all 3 times), P0001 and P0003 not
        assert result["n_unanimous"] == 1
        assert result["n_problems"] == 3

    def test_skips_single_presentation_problems(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
        ]
        result = unanimity_rate(entries)
        assert result["n_problems"] == 0

    def test_excludes_invalid(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": False, "problem_id": "P1", "claim_chosen": None,
             "claim_order": ["B", "A"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["B", "A"]},
        ]
        # Only 2 valid for P1, both chose A → unanimous
        result = unanimity_rate(entries)
        assert result["rate"] == 1.0


class TestModalAgreementRate:
    def test_perfect_agreement(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["B", "A"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
        ]
        result = modal_agreement_rate(entries)
        assert result["mean_rate"] == 1.0

    def test_two_out_of_three(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["B", "A"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "B",
             "claim_order": ["A", "B"]},
        ]
        result = modal_agreement_rate(entries)
        assert result["mean_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_from_fixture(self, sample_choice_entries):
        result = modal_agreement_rate(sample_choice_entries)
        assert 0 < result["mean_rate"] <= 1.0
        assert result["n_problems"] == 3


class TestConsistencyByTemperature:
    def test_returns_per_temp(self, per_temp_choices):
        result = consistency_by_temperature(per_temp_choices)
        assert "per_temperature" in result
        for t_key in ["0.0", "0.7"]:
            assert t_key in result["per_temperature"]
            assert "unanimity" in result["per_temperature"][t_key]
            assert "modal_agreement" in result["per_temperature"][t_key]


class TestEntropyOfChoices:
    def test_unanimous_zero_entropy(self):
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["B", "A"]},
        ]
        result = entropy_of_choices(entries)
        assert result["mean_entropy"] == pytest.approx(0.0, abs=0.01)

    def test_max_entropy(self):
        """Equal split between 2 claims → entropy = 1 bit."""
        entries = [
            {"valid": True, "problem_id": "P1", "claim_chosen": "A",
             "claim_order": ["A", "B"]},
            {"valid": True, "problem_id": "P1", "claim_chosen": "B",
             "claim_order": ["B", "A"]},
        ]
        result = entropy_of_choices(entries)
        assert result["mean_entropy"] == pytest.approx(1.0, abs=0.01)

    def test_from_fixture(self, sample_choice_entries):
        result = entropy_of_choices(sample_choice_entries)
        assert result["n_problems"] == 3
        assert result["mean_entropy"] >= 0

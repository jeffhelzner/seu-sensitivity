"""Tests for position_analysis.py — DESIGN.md §6.2."""
import pytest
from ..position_analysis import (
    position_choice_rates,
    position_choice_counts,
    chi_squared_uniformity_test,
    cramers_v,
    position_bias_by_temperature,
    per_claim_position_analysis,
)


class TestPositionChoiceRates:
    def test_all_valid(self, sample_choice_entries):
        rates = position_choice_rates(sample_choice_entries)
        # 9 valid entries, 6 chose position 1, 1 chose position 2, 1 chose position 2, 1 chose position 3
        assert sum(rates.values()) == pytest.approx(1.0, abs=0.01)
        assert rates[1] > 0  # position 1 was chosen at least once

    def test_empty_list(self):
        assert position_choice_rates([]) == {}

    def test_no_valid(self):
        entries = [{"valid": False, "position_chosen": None}]
        assert position_choice_rates(entries) == {}

    def test_counts(self, sample_choice_entries):
        counts = position_choice_counts(sample_choice_entries)
        assert sum(counts.values()) == 9  # all 9 entries valid


class TestChiSquaredUniformity:
    def test_perfectly_uniform(self):
        """If every position is equally chosen the test should not reject."""
        entries = []
        for pos in [1, 2, 3]:
            for _ in range(30):
                entries.append({
                    "valid": True,
                    "position_chosen": pos,
                    "claim_order": ["A", "B", "C"],
                })
        result = chi_squared_uniformity_test(entries)
        assert result["chi2"] == pytest.approx(0.0, abs=0.01)
        assert result["p_value"] > 0.5

    def test_strong_bias(self):
        """Heavy position-1 bias should yield small p-value."""
        entries = []
        for _ in range(90):
            entries.append({
                "valid": True,
                "position_chosen": 1,
                "claim_order": ["A", "B", "C"],
            })
        for _ in range(5):
            entries.append({
                "valid": True,
                "position_chosen": 2,
                "claim_order": ["A", "B", "C"],
            })
        for _ in range(5):
            entries.append({
                "valid": True,
                "position_chosen": 3,
                "claim_order": ["A", "B", "C"],
            })
        result = chi_squared_uniformity_test(entries)
        assert result["significant"]

    def test_excludes_invalid(self):
        entries = [
            {"valid": False, "position_chosen": None, "claim_order": ["A", "B"]},
            {"valid": True, "position_chosen": 1, "claim_order": ["A", "B"]},
            {"valid": True, "position_chosen": 2, "claim_order": ["A", "B"]},
        ]
        result = chi_squared_uniformity_test(entries)
        assert result["chi2"] == pytest.approx(0.0, abs=0.01)


class TestCramersV:
    def test_uniform_gives_zero(self):
        entries = []
        for pos in [1, 2]:
            for _ in range(50):
                entries.append({
                    "valid": True,
                    "position_chosen": pos,
                    "claim_order": ["A", "B"],
                })
        v = cramers_v(entries)
        assert v == pytest.approx(0.0, abs=0.01)

    def test_strong_bias_large_v(self):
        entries = [
            {"valid": True, "position_chosen": 1, "claim_order": ["A", "B"]}
        ] * 95 + [
            {"valid": True, "position_chosen": 2, "claim_order": ["A", "B"]}
        ] * 5
        v = cramers_v(entries)
        assert v > 0.5


class TestPositionBiasByTemperature:
    def test_returns_per_temp(self, per_temp_choices):
        result = position_bias_by_temperature(per_temp_choices)
        assert "per_temperature" in result
        assert "0.0" in result["per_temperature"]
        assert "0.7" in result["per_temperature"]
        for info in result["per_temperature"].values():
            assert "rates" in info
            assert "chi_squared" in info
            assert "cramers_v" in info


class TestPerClaimPositionAnalysis:
    def test_basic(self, sample_choice_entries):
        result = per_claim_position_analysis(sample_choice_entries)
        # Should have entries for all claim IDs that appear
        assert len(result) > 0
        for cid, info in result.items():
            assert "pos1_rate" in info
            assert "other_rate" in info
            assert "diff" in info

    def test_specific_claims(self, sample_choice_entries):
        result = per_claim_position_analysis(
            sample_choice_entries, claim_ids=["C001"]
        )
        assert list(result.keys()) == ["C001"]

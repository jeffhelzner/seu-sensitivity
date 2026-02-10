"""Tests for na_analysis.py — DESIGN.md §6.4."""
import pytest
from ..na_analysis import (
    na_rates_by_temperature,
    na_concentration,
    effective_sample_summary,
    worst_case_imputation,
    na_quality_report,
)


class TestNaRatesByTemperature:
    def test_zero_na_temp(self, per_temp_choices):
        result = na_rates_by_temperature(per_temp_choices)
        assert result["per_temperature"]["0.0"]["na_rate"] == 0.0
        assert result["per_temperature"]["0.0"]["effective_M"] == 9

    def test_nonzero_na_temp(self, per_temp_choices):
        result = na_rates_by_temperature(per_temp_choices)
        assert result["per_temperature"]["0.7"]["na_rate"] > 0

    def test_overall_totals(self, per_temp_choices):
        result = na_rates_by_temperature(per_temp_choices)
        total = sum(
            info["total"]
            for info in result["per_temperature"].values()
        )
        assert result["overall"]["total"] == total


class TestNaConcentration:
    def test_no_nas(self, sample_choice_entries):
        result = na_concentration(sample_choice_entries)
        assert result["uniform"]
        assert result["total_na"] == 0

    def test_with_nas(self, sample_choice_entries_with_na):
        result = na_concentration(sample_choice_entries_with_na)
        assert result["total_na"] == 2
        assert result["n_problems"] == 3

    def test_concentrated_nas(self):
        """All NAs in one problem → concentration detected."""
        entries = []
        for pid in ["P1", "P2", "P3"]:
            entries.append({
                "valid": True, "problem_id": pid,
                "claim_order": ["A", "B"], "position_chosen": 1,
            })
        # Add 10 NAs all in P1
        for _ in range(10):
            entries.append({
                "valid": False, "problem_id": "P1",
                "claim_order": ["A", "B"], "position_chosen": None,
            })
        result = na_concentration(entries)
        assert "P1" in result["concentrated_problems"]


class TestEffectiveSampleSummary:
    def test_retention(self, per_temp_choices):
        result = effective_sample_summary(per_temp_choices, nominal_M=10)
        assert result["0.0"]["effective_M"] == 9
        assert result["0.0"]["retention_rate"] == 0.9
        assert result["0.7"]["effective_M"] < 10


class TestWorstCaseImputation:
    def test_imputes_all_nas(self, sample_choice_entries_with_na):
        augmented = worst_case_imputation(sample_choice_entries_with_na)
        assert all(c["valid"] for c in augmented)
        assert len(augmented) == len(sample_choice_entries_with_na)

    def test_imputed_flag(self, sample_choice_entries_with_na):
        augmented = worst_case_imputation(sample_choice_entries_with_na)
        imputed = [c for c in augmented if c.get("imputed")]
        assert len(imputed) == 2
        for c in imputed:
            assert c["position_chosen"] == 1

    def test_claim_chosen_set(self, sample_choice_entries_with_na):
        augmented = worst_case_imputation(sample_choice_entries_with_na)
        for c in augmented:
            if c.get("imputed") and c["claim_order"]:
                assert c["claim_chosen"] == c["claim_order"][0]

    def test_original_entries_unchanged(self, sample_choice_entries_with_na):
        original_na_count = sum(
            1 for c in sample_choice_entries_with_na if not c["valid"]
        )
        _ = worst_case_imputation(sample_choice_entries_with_na)
        after_na_count = sum(
            1 for c in sample_choice_entries_with_na if not c["valid"]
        )
        assert original_na_count == after_na_count


class TestNaQualityReport:
    def test_full_report(self, per_temp_choices):
        report = na_quality_report(per_temp_choices, nominal_M=10)
        assert "na_rates" in report
        assert "concentration" in report
        assert "effective_sample_size" in report
        assert isinstance(report["sensitivity_imputation_recommended"], bool)
        assert "max_na_rate" in report

    def test_sensitivity_flag(self, per_temp_choices):
        """If max NA rate > 5%, sensitivity imputation is recommended."""
        report = na_quality_report(per_temp_choices, nominal_M=10)
        max_rate = report["max_na_rate"]
        if max_rate > 0.05:
            assert report["sensitivity_imputation_recommended"]
        else:
            assert not report["sensitivity_imputation_recommended"]

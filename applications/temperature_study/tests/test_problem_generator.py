"""
Tests for the ProblemGenerator module.
"""
import json
import pytest
from pathlib import Path

from applications.temperature_study.problem_generator import ProblemGenerator
from applications.temperature_study.config import StudyConfig


class TestProblemGeneration:
    """Tests for basic problem generation."""

    def test_generate_correct_count(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=10, seed=42)
        assert len(problems) == 10

    def test_problem_structure(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=5, seed=42)
        for p in problems:
            assert "id" in p
            assert "claim_ids" in p
            assert "num_alternatives" in p
            assert "presentations" in p
            assert p["id"].startswith("P")

    def test_alternatives_within_bounds(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=50, min_alternatives=2, max_alternatives=4, seed=42
        )
        for p in problems:
            n = p["num_alternatives"]
            assert 2 <= n <= 4
            assert len(p["claim_ids"]) == n

    def test_claim_ids_from_pool(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        valid_ids = {c["id"] for c in sample_claims}
        problems = gen.generate_problems(num_problems=20, seed=42)
        for p in problems:
            for cid in p["claim_ids"]:
                assert cid in valid_ids

    def test_no_duplicate_claims_in_problem(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=30, seed=42)
        for p in problems:
            assert len(set(p["claim_ids"])) == len(p["claim_ids"])

    def test_seed_reproducibility(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        p1 = gen.generate_problems(num_problems=10, seed=123)
        p2 = gen.generate_problems(num_problems=10, seed=123)
        assert p1 == p2

    def test_different_seeds_differ(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        p1 = gen.generate_problems(num_problems=10, seed=1)
        p2 = gen.generate_problems(num_problems=10, seed=2)
        # Very unlikely to be identical
        ids1 = [p["claim_ids"] for p in p1]
        ids2 = [p["claim_ids"] for p in p2]
        assert ids1 != ids2


class TestPresentations:
    """Tests for presentation shuffling."""

    def test_correct_number_of_presentations(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=5, num_presentations=3, seed=42
        )
        for p in problems:
            assert len(p["presentations"]) == 3

    def test_presentation_structure(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=5, num_presentations=3, seed=42
        )
        for p in problems:
            for pres in p["presentations"]:
                assert "presentation_id" in pres
                assert "order" in pres
                assert set(pres["order"]) == set(p["claim_ids"])

    def test_first_presentation_canonical(self, sample_claims):
        """First presentation preserves the canonical ordering."""
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=20, num_presentations=3, seed=42
        )
        for p in problems:
            assert p["presentations"][0]["order"] == p["claim_ids"]

    def test_presentations_cover_same_claims(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=10, num_presentations=5, seed=42
        )
        for p in problems:
            base_set = set(p["claim_ids"])
            for pres in p["presentations"]:
                assert set(pres["order"]) == base_set

    def test_presentation_ids_sequential(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(
            num_problems=5, num_presentations=4, seed=42
        )
        for p in problems:
            ids = [pres["presentation_id"] for pres in p["presentations"]]
            assert ids == [1, 2, 3, 4]


class TestPromptFormatting:
    """Tests for prompt formatting helpers."""

    def test_deliberation_claims_list(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        result = gen.format_deliberation_claims_list(["C001", "C002"])
        assert "Claim A:" in result
        assert "Claim B:" in result
        assert "C001" not in result  # IDs should not appear, only descriptions

    def test_choice_claims_list(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        result = gen.format_choice_claims_list(["C003", "C001"])
        assert "Claim 1:" in result
        assert "Claim 2:" in result

    def test_num_range_str_2(self):
        assert ProblemGenerator.num_range_str(2) == "1 or 2"

    def test_num_range_str_3(self):
        assert ProblemGenerator.num_range_str(3) == "1, 2, or 3"

    def test_num_range_str_4(self):
        assert ProblemGenerator.num_range_str(4) == "1, 2, 3, or 4"

    def test_claim_index_to_letter(self):
        assert ProblemGenerator.claim_index_to_letter(0) == "A"
        assert ProblemGenerator.claim_index_to_letter(2) == "C"


class TestCoverage:
    """Tests for coverage statistics."""

    def test_coverage_stats_structure(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=50, seed=42)
        stats = gen.get_coverage_stats(problems)
        assert "num_problems" in stats
        assert "num_claims" in stats
        assert "coverage_rate" in stats
        assert "min_appearances" in stats
        assert "max_appearances" in stats
        assert "mean_appearances" in stats

    def test_high_coverage_with_many_problems(self, sample_claims):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=100, seed=42)
        stats = gen.get_coverage_stats(problems)
        # With 100 problems and 5 claims, coverage should be 100%
        assert stats["coverage_rate"] == 1.0


class TestSerialization:
    """Tests for save/load."""

    def test_save_load_roundtrip(self, sample_claims, tmp_path):
        gen = ProblemGenerator(claims=sample_claims)
        problems = gen.generate_problems(num_problems=10, seed=42)
        filepath = tmp_path / "problems.json"
        gen.save_problems(problems, filepath)

        loaded, K = ProblemGenerator.load_problems(filepath)
        assert len(loaded) == 10
        assert loaded[0]["id"] == problems[0]["id"]
        assert loaded[0]["claim_ids"] == problems[0]["claim_ids"]

    def test_from_claims_file(self, tmp_path, sample_claims):
        claims_path = tmp_path / "claims.json"
        with open(claims_path, "w") as f:
            json.dump(
                {"claims": sample_claims, "consequences": ["c1", "c2", "c3"]},
                f,
            )
        gen = ProblemGenerator(claims_file=str(claims_path))
        assert len(gen.claims) == len(sample_claims)

    def test_from_config(self, tmp_path, sample_claims):
        claims_path = tmp_path / "claims.json"
        with open(claims_path, "w") as f:
            json.dump({"claims": sample_claims, "consequences": ["c1", "c2", "c3"]}, f)
        config = StudyConfig(
            claims_file=str(claims_path),
            num_problems=5,
            seed=42,
        )
        gen = ProblemGenerator.from_config(config)
        problems = gen.generate_problems_from_config(config)
        assert len(problems) == 5

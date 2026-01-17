"""
Tests for the prompt framing study pipeline.
"""
import pytest
import numpy as np
from pathlib import Path


class TestPromptManager:
    """Tests for PromptManager and PromptVariant."""
    
    def test_load_default_variants(self):
        """Test that default variants load correctly."""
        from prompt_framing_study.prompt_manager import PromptManager
        
        manager = PromptManager()
        variants = manager.get_all_variants()
        
        assert len(variants) >= 4
        assert "minimal" in manager.list_variant_names()
        assert "baseline" in manager.list_variant_names()
        assert "enhanced" in manager.list_variant_names()
        assert "maximal" in manager.list_variant_names()
    
    def test_variant_levels_ordered(self):
        """Test that variants are ordered by level."""
        from prompt_framing_study.prompt_manager import PromptManager
        
        manager = PromptManager()
        variants = manager.get_all_variants()
        
        levels = [v.level for v in variants]
        assert levels == sorted(levels)
    
    def test_format_choice_prompt(self):
        """Test prompt formatting with claims."""
        from prompt_framing_study.prompt_manager import PromptManager
        
        manager = PromptManager()
        variant = manager.get_variant("baseline")
        
        claims = ["Claim about water damage", "Claim about auto accident"]
        prompt = variant.format_choice_prompt(claims)
        
        assert "Claim 1:" in prompt
        assert "Claim 2:" in prompt
        assert "water damage" in prompt
        assert "auto accident" in prompt
    
    def test_format_embedding_prompt(self):
        """Test embedding prompt formatting."""
        from prompt_framing_study.prompt_manager import PromptManager
        
        manager = PromptManager()
        variant = manager.get_variant("enhanced")
        
        claim = "Test claim description"
        prompt = variant.format_embedding_prompt(claim)
        
        assert "Test claim description" in prompt
        assert "{claim}" not in prompt  # Should be replaced


class TestProblemGenerator:
    """Tests for ProblemGenerator."""
    
    def test_generate_problems(self, sample_claims):
        """Test basic problem generation."""
        from prompt_framing_study.problem_generator import ProblemGenerator
        
        generator = ProblemGenerator(claims=sample_claims)
        problems = generator.generate_problems(num_problems=10, seed=42)
        
        assert len(problems) == 10
        for problem in problems:
            assert "id" in problem
            assert "claim_ids" in problem
            assert "claims" in problem
            assert len(problem["claim_ids"]) >= 2
    
    def test_problem_reproducibility(self, sample_claims):
        """Test that seed produces reproducible problems."""
        from prompt_framing_study.problem_generator import ProblemGenerator
        
        gen1 = ProblemGenerator(claims=sample_claims)
        gen2 = ProblemGenerator(claims=sample_claims)
        
        problems1 = gen1.generate_problems(num_problems=5, seed=123)
        problems2 = gen2.generate_problems(num_problems=5, seed=123)
        
        for p1, p2 in zip(problems1, problems2):
            assert p1["claim_ids"] == p2["claim_ids"]
    
    def test_save_and_load_problems(self, sample_claims, temp_dir):
        """Test saving and loading problems."""
        from prompt_framing_study.problem_generator import ProblemGenerator
        
        generator = ProblemGenerator(claims=sample_claims)
        problems = generator.generate_problems(num_problems=5, seed=42)
        
        filepath = temp_dir / "test_problems.json"
        generator.save_problems(problems, str(filepath))
        
        loaded_problems, K = ProblemGenerator.load_problems(str(filepath))
        
        assert len(loaded_problems) == len(problems)
        assert K == 3


class TestContextualizedEmbedding:
    """Tests for ContextualizedEmbeddingManager."""
    
    def test_dimension_reduction_pca(self, mock_embeddings):
        """Test PCA dimension reduction."""
        from prompt_framing_study.contextualized_embedding import ContextualizedEmbeddingManager
        
        manager = ContextualizedEmbeddingManager()
        reduced, reducer = manager.reduce_dimensions(
            mock_embeddings, 
            target_dim=32, 
            method="pca"
        )
        
        assert len(reduced) == len(mock_embeddings)
        for cid, emb in reduced.items():
            assert len(emb) == 32
    
    def test_dimension_reduction_random(self, mock_embeddings):
        """Test random projection dimension reduction."""
        from prompt_framing_study.contextualized_embedding import ContextualizedEmbeddingManager
        
        manager = ContextualizedEmbeddingManager()
        reduced, reducer = manager.reduce_dimensions(
            mock_embeddings,
            target_dim=64,
            method="random"
        )
        
        assert len(reduced) == len(mock_embeddings)
        for cid, emb in reduced.items():
            assert len(emb) == 64
    
    def test_save_and_load_embeddings(self, mock_embeddings, temp_dir):
        """Test saving and loading embeddings."""
        from prompt_framing_study.contextualized_embedding import ContextualizedEmbeddingManager
        
        all_embeddings = {
            "minimal": mock_embeddings,
            "baseline": mock_embeddings
        }
        
        filepath = temp_dir / "test_embeddings.npz"
        manager = ContextualizedEmbeddingManager()
        manager.save_embeddings(all_embeddings, str(filepath))
        
        loaded = ContextualizedEmbeddingManager.load_embeddings(str(filepath))
        
        assert "minimal" in loaded
        assert "baseline" in loaded
        assert len(loaded["minimal"]) == len(mock_embeddings)


class TestCostEstimator:
    """Tests for CostEstimator."""
    
    def test_estimate_embedding_cost(self):
        """Test embedding cost estimation."""
        from prompt_framing_study.cost_estimator import CostEstimator
        
        estimator = CostEstimator()
        estimate = estimator.estimate_embedding_cost(
            num_claims=20,
            num_variants=4,
            embedding_model="text-embedding-3-small"
        )
        
        assert "num_embeddings" in estimate
        assert estimate["num_embeddings"] == 80  # 20 * 4
        assert "estimated_cost_usd" in estimate
        assert estimate["estimated_cost_usd"] > 0
    
    def test_estimate_choice_collection_cost(self):
        """Test choice collection cost estimation."""
        from prompt_framing_study.cost_estimator import CostEstimator
        
        estimator = CostEstimator()
        estimate = estimator.estimate_choice_collection_cost(
            num_problems=100,
            num_variants=4,
            num_repetitions=1,
            avg_alternatives=3.0,
            llm_model="gpt-4"
        )
        
        assert "num_api_calls" in estimate
        assert estimate["num_api_calls"] == 400  # 100 * 4 * 1
        assert "estimated_cost_usd" in estimate
    
    def test_total_study_cost(self):
        """Test total study cost estimation."""
        from prompt_framing_study.cost_estimator import CostEstimator
        
        estimator = CostEstimator()
        estimate = estimator.estimate_total_study_cost(
            num_claims=20,
            num_problems=100,
            num_variants=4
        )
        
        assert "embedding_costs" in estimate
        assert "choice_collection_costs" in estimate
        assert "total_estimated_cost_usd" in estimate


class TestChoiceCollector:
    """Tests for choice extraction utilities."""
    
    def test_extract_choice_vector(self, sample_problems):
        """Test extracting choice vector from results."""
        from prompt_framing_study.choice_collector import extract_choice_vector
        
        # Mock choices result
        choices = {
            "variant": "baseline",
            "choices": [
                {"problem_id": "P0001", "choice": 1, "choice_1indexed": 2},
                {"problem_id": "P0002", "choice": 0, "choice_1indexed": 1},
                {"problem_id": "P0003", "choice": 2, "choice_1indexed": 3}
            ]
        }
        
        y = extract_choice_vector(choices, sample_problems)
        
        assert len(y) == len(sample_problems)
        assert y == [2, 1, 3]


class TestValidation:
    """Tests for validation utilities."""
    
    def test_validate_config_valid(self, sample_config):
        """Test validation of valid config."""
        from prompt_framing_study.validation import validate_config
        
        warnings = validate_config(sample_config)
        # Should not raise, may have warnings
        assert isinstance(warnings, list)
    
    def test_validate_config_missing_required(self):
        """Test validation catches missing required fields."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({"temperature": 0.7})  # Missing num_problems, K
    
    def test_validate_claims_file(self, claims_file):
        """Test claims file validation."""
        from prompt_framing_study.validation import validate_claims_file
        
        data = validate_claims_file(claims_file)
        assert "claims" in data
        assert len(data["claims"]) > 0
    
    def test_validate_stan_data_m0(self, sample_problems, mock_reduced_embeddings):
        """Test Stan data validation for m_0."""
        from prompt_framing_study.validation import validate_stan_data
        
        # Create minimal valid Stan data
        M = 3
        K = 3
        D = 32
        R = 5
        
        stan_data = {
            "M": M,
            "K": K,
            "D": D,
            "R": R,
            "w": [[0.1] * D for _ in range(R)],
            "I": [[1, 1, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
            "y": [1, 2, 1]
        }
        
        # Should not raise
        validate_stan_data(stan_data, model="m_0")
    
    def test_validate_stan_data_invalid(self):
        """Test Stan data validation catches errors."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        # Missing fields
        with pytest.raises(ValidationError):
            validate_stan_data({"M": 10}, model="m_0")


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_problem_generation_to_stan_data(self, sample_claims, mock_reduced_embeddings):
        """Test generating problems and creating Stan data."""
        from prompt_framing_study.problem_generator import ProblemGenerator
        from prompt_framing_study.validation import validate_stan_data
        
        # Generate problems
        generator = ProblemGenerator(claims=sample_claims)
        problems = generator.generate_problems(num_problems=5, seed=42)
        
        # Create claim_id to index mapping
        claim_id_to_idx = {c["id"]: i for i, c in enumerate(sample_claims)}
        
        # Create Stan data
        M = len(problems)
        R = len(sample_claims)
        D = 32
        K = 3
        
        # Build I matrix
        I = [[0] * R for _ in range(M)]
        for m, problem in enumerate(problems):
            for cid in problem["claim_ids"]:
                I[m][claim_id_to_idx[cid]] = 1
        
        # Build w matrix
        w = [list(mock_reduced_embeddings[c["id"]]) for c in sample_claims]
        
        # Mock choices
        y = [1] * M
        
        stan_data = {
            "M": M,
            "K": K,
            "D": D,
            "R": R,
            "w": w,
            "I": I,
            "y": y
        }
        
        # Should validate
        validate_stan_data(stan_data, model="m_0")

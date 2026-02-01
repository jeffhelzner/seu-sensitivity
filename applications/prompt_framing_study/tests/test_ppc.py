"""
Tests for Posterior Predictive Checks

Tests the PPC statistics in Stan models and the Python analysis module.
"""
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path


class TestStanPPCVariables:
    """Tests that Stan models produce expected PPC variables."""
    
    @pytest.fixture
    def minimal_stan_data(self):
        """Create minimal valid data for m_0 model."""
        # 5 problems, 3 consequences, 4 dimensions, 3 distinct alternatives
        M, K, D, R = 5, 3, 4, 3
        
        # Random embeddings for alternatives
        np.random.seed(42)
        w = np.random.randn(R, D).tolist()
        
        # Each problem uses 2-3 alternatives
        I = [
            [1, 1, 0],  # Problem 1: alternatives 1, 2
            [1, 0, 1],  # Problem 2: alternatives 1, 3
            [0, 1, 1],  # Problem 3: alternatives 2, 3
            [1, 1, 1],  # Problem 4: all three
            [1, 1, 0],  # Problem 5: alternatives 1, 2
        ]
        
        # Choices (1-indexed within each problem)
        y = [1, 2, 1, 3, 2]
        
        return {
            "M": M,
            "K": K,
            "D": D,
            "R": R,
            "w": w,
            "I": I,
            "y": y
        }
    
    @pytest.mark.slow
    def test_m0_ppc_variables_exist(self, minimal_stan_data):
        """Test that m_0.stan produces all expected PPC variables."""
        pytest.importorskip("cmdstanpy")
        from cmdstanpy import CmdStanModel
        
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0.stan"
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")
        
        model = CmdStanModel(stan_file=str(model_path))
        
        # Quick fit with minimal samples
        fit = model.sample(
            data=minimal_stan_data,
            chains=1,
            iter_warmup=100,
            iter_sampling=100,
            seed=42
        )
        
        # Check all PPC variables exist
        expected_vars = [
            "log_lik", "y_pred",
            "T_obs_ll", "T_rep_ll", "ppc_ll",
            "T_obs_modal", "T_rep_modal", "ppc_modal",
            "T_obs_prob", "T_rep_prob", "ppc_prob"
        ]
        
        for var in expected_vars:
            try:
                values = fit.stan_variable(var)
                assert values is not None, f"Variable {var} is None"
                assert len(values) > 0, f"Variable {var} is empty"
            except Exception as e:
                pytest.fail(f"Failed to extract variable {var}: {e}")
    
    @pytest.mark.slow
    def test_ppc_indicators_are_binary(self, minimal_stan_data):
        """Test that PPC indicator variables are 0 or 1."""
        pytest.importorskip("cmdstanpy")
        from cmdstanpy import CmdStanModel
        
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0.stan"
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")
        
        model = CmdStanModel(stan_file=str(model_path))
        
        fit = model.sample(
            data=minimal_stan_data,
            chains=1,
            iter_warmup=100,
            iter_sampling=100,
            seed=42
        )
        
        for var in ["ppc_ll", "ppc_modal", "ppc_prob"]:
            values = fit.stan_variable(var)
            assert np.all((values == 0) | (values == 1)), \
                f"{var} contains values other than 0 or 1: {np.unique(values)}"


class TestPosteriorPredictiveChecker:
    """Tests for the PosteriorPredictiveChecker class."""
    
    @pytest.fixture
    def mock_fit(self):
        """Create a mock fit object with PPC variables."""
        class MockFit:
            def __init__(self):
                np.random.seed(42)
                n_samples = 1000
                
                # Simulate p-values near 0.5 (well-calibrated)
                self._data = {
                    "ppc_ll": np.random.binomial(1, 0.5, n_samples),
                    "ppc_modal": np.random.binomial(1, 0.45, n_samples),
                    "ppc_prob": np.random.binomial(1, 0.55, n_samples),
                    "T_obs_ll": np.full(n_samples, -50.0),
                    "T_rep_ll": np.random.normal(-50.0, 5.0, n_samples),
                    "T_obs_modal": np.full(n_samples, 3),
                    "T_rep_modal": np.random.poisson(3, n_samples),
                    "T_obs_prob": np.full(n_samples, 2.5),
                    "T_rep_prob": np.random.normal(2.5, 0.3, n_samples),
                }
            
            def stan_variable(self, name):
                return self._data.get(name)
        
        return MockFit()
    
    @pytest.fixture
    def mock_data(self):
        """Mock observed data."""
        return {"M": 5, "K": 3, "y": [1, 2, 1, 3, 2]}
    
    def test_compute_p_values(self, mock_fit, mock_data):
        """Test that p-values are computed correctly."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        p_values = checker.compute_p_values()
        
        assert "ll" in p_values
        assert "modal" in p_values
        assert "prob" in p_values
        
        # All p-values should be in [0, 1]
        for name, p in p_values.items():
            assert 0 <= p <= 1, f"p-value {name}={p} out of range"
    
    def test_p_values_cached(self, mock_fit, mock_data):
        """Test that p-values are cached after first computation."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        p1 = checker.compute_p_values()
        p2 = checker.compute_p_values()
        
        assert p1 is p2  # Same object (cached)
    
    def test_interpret_p_value_good(self, mock_fit, mock_data):
        """Test interpretation of well-calibrated p-value."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        symbol, text = checker.interpret_p_value(0.5)
        
        assert symbol == "✓"
        assert "Good fit" in text
    
    def test_interpret_p_value_extreme_low(self, mock_fit, mock_data):
        """Test interpretation of extreme low p-value."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        symbol, text = checker.interpret_p_value(0.01)
        
        assert symbol == "⚠️"
        assert "BETTER" in text
    
    def test_interpret_p_value_extreme_high(self, mock_fit, mock_data):
        """Test interpretation of extreme high p-value."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        symbol, text = checker.interpret_p_value(0.99)
        
        assert symbol == "⚠️"
        assert "WORSE" in text
    
    def test_summary_string(self, mock_fit, mock_data):
        """Test that summary produces readable output."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        summary = checker.summary()
        
        assert "Posterior Predictive Check Summary" in summary
        assert "ll" in summary
        assert "modal" in summary
        assert "prob" in summary
    
    def test_to_dict(self, mock_fit, mock_data):
        """Test JSON-serializable output."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        result = checker.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0
        
        # Check structure
        assert "model_type" in result
        assert "p_values" in result
        assert "interpretation" in result
        assert "has_extreme_p_values" in result
    
    def test_model_type_detection_m0(self, mock_fit, mock_data):
        """Test m_0 model type detection."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(mock_fit, mock_data)
        assert checker.model_type == "m_0"
    
    def test_model_type_detection_m1(self, mock_fit):
        """Test m_1 model type detection."""
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        m1_data = {"M": 5, "N": 3, "y": [1, 2, 1, 3, 2], "z": [1, 2, 1]}
        checker = PosteriorPredictiveChecker(mock_fit, m1_data)
        assert checker.model_type == "m_1"


class TestPPCIntegration:
    """Integration tests for PPC with study pipeline."""
    
    def test_run_posterior_predictive_checks_function(self):
        """Test the convenience function."""
        from analysis.posterior_predictive_checks import run_posterior_predictive_checks
        
        # Create mock fit
        class MockFit:
            def stan_variable(self, name):
                np.random.seed(42)
                n = 100
                if name.startswith("ppc_"):
                    return np.random.binomial(1, 0.5, n)
                elif name.startswith("T_obs"):
                    return np.full(n, 1.0)
                elif name.startswith("T_rep"):
                    return np.random.normal(1.0, 0.1, n)
                return None
        
        mock_data = {"M": 5, "y": [1, 2, 1, 2, 1]}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_posterior_predictive_checks(
                fit=MockFit(),
                observed_data=mock_data,
                output_dir=tmpdir,
                verbose=False
            )
            
            # Check results structure
            assert "p_values" in results
            assert "model_type" in results
            
            # Check files created
            assert (Path(tmpdir) / "ppc_summary.json").exists()
            assert (Path(tmpdir) / "ppc_summary.txt").exists()


class TestPPCWellSpecified:
    """Test that well-specified model produces reasonable p-values."""
    
    @pytest.mark.slow
    def test_simulated_data_reasonable_p_values(self):
        """
        Generate data from the model, fit it back, and verify p-values aren't extreme.
        
        This is a key validation: if data comes from the model, p-values should be
        roughly uniform, so most should be in [0.1, 0.9].
        """
        pytest.importorskip("cmdstanpy")
        from cmdstanpy import CmdStanModel
        
        # Use the SBC model to generate "true" data
        sbc_model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0_sbc.stan"
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0.stan"
        
        if not sbc_model_path.exists() or not model_path.exists():
            pytest.skip("Models not found")
        
        # Generate data
        np.random.seed(42)
        M, K, D, R = 20, 3, 4, 5
        w = np.random.randn(R, D).tolist()
        
        # Random indicator matrix
        I = []
        for _ in range(M):
            row = [0] * R
            # Include 2-3 alternatives
            indices = np.random.choice(R, size=np.random.randint(2, min(4, R+1)), replace=False)
            for idx in indices:
                row[idx] = 1
            I.append(row)
        
        sbc_data = {"M": M, "K": K, "D": D, "R": R, "w": w, "I": I}
        
        sbc_model = CmdStanModel(stan_file=str(sbc_model_path))
        
        # Generate one dataset from prior
        sbc_fit = sbc_model.sample(
            data=sbc_data,
            chains=1,
            iter_warmup=0,
            iter_sampling=1,
            fixed_param=True,
            seed=42
        )
        
        # Extract generated y
        y = sbc_fit.stan_variable("y")[0].astype(int).tolist()
        
        # Now fit with the main model
        fit_data = {**sbc_data, "y": y}
        
        model = CmdStanModel(stan_file=str(model_path))
        fit = model.sample(
            data=fit_data,
            chains=2,
            iter_warmup=500,
            iter_sampling=500,
            seed=42
        )
        
        # Check PPC p-values
        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker
        
        checker = PosteriorPredictiveChecker(fit, fit_data)
        p_values = checker.compute_p_values()
        
        # For well-specified data, we expect p-values not to be extreme
        # Allow some tolerance since this is stochastic
        extreme_count = sum(1 for p in p_values.values() if p < 0.05 or p > 0.95)
        
        # At most 1 out of 3 should be extreme by chance
        assert extreme_count <= 1, \
            f"Too many extreme p-values for well-specified data: {p_values}"


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    @pytest.mark.slow  
    def test_log_lik_still_works(self):
        """Verify log_lik variable still produced correctly."""
        pytest.importorskip("cmdstanpy")
        from cmdstanpy import CmdStanModel
        
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0.stan"
        if not model_path.exists():
            pytest.skip("Model not found")
        
        np.random.seed(42)
        data = {
            "M": 3, "K": 3, "D": 2, "R": 2,
            "w": np.random.randn(2, 2).tolist(),
            "I": [[1, 1], [1, 1], [1, 1]],
            "y": [1, 2, 1]
        }
        
        model = CmdStanModel(stan_file=str(model_path))
        fit = model.sample(data=data, chains=1, iter_warmup=50, iter_sampling=50, seed=42)
        
        log_lik = fit.stan_variable("log_lik")
        assert log_lik.shape[1] == 3  # M problems
        assert np.all(log_lik <= 0)  # Log probabilities are negative
    
    @pytest.mark.slow
    def test_y_pred_still_works(self):
        """Verify y_pred variable still produced correctly."""
        pytest.importorskip("cmdstanpy")
        from cmdstanpy import CmdStanModel
        
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "m_0.stan"
        if not model_path.exists():
            pytest.skip("Model not found")
        
        np.random.seed(42)
        data = {
            "M": 3, "K": 3, "D": 2, "R": 2,
            "w": np.random.randn(2, 2).tolist(),
            "I": [[1, 1], [1, 1], [1, 1]],
            "y": [1, 2, 1]
        }
        
        model = CmdStanModel(stan_file=str(model_path))
        fit = model.sample(data=data, chains=1, iter_warmup=50, iter_sampling=50, seed=42)
        
        y_pred = fit.stan_variable("y_pred")
        assert y_pred.shape[1] == 3  # M problems
        assert np.all(y_pred >= 1)  # 1-indexed
        assert np.all(y_pred <= 2)  # At most 2 alternatives per problem

"""
Tests for validation module.
"""
import pytest
import json
from pathlib import Path


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_minimal_config(self):
        """Test minimal valid configuration."""
        from prompt_framing_study.validation import validate_config
        
        config = {
            "num_problems": 100,
            "K": 3
        }
        warnings = validate_config(config)
        assert isinstance(warnings, list)
    
    def test_valid_full_config(self):
        """Test full configuration."""
        from prompt_framing_study.validation import validate_config
        
        config = {
            "num_problems": 100,
            "K": 3,
            "min_alternatives": 2,
            "max_alternatives": 4,
            "temperature": 0.7,
            "num_repetitions": 1,
            "target_dim": 32
        }
        warnings = validate_config(config)
        assert isinstance(warnings, list)
    
    def test_missing_num_problems(self):
        """Test error on missing num_problems."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            validate_config({"K": 3})
        assert "num_problems" in str(excinfo.value)
    
    def test_missing_k(self):
        """Test error on missing K."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            validate_config({"num_problems": 100})
        assert "K" in str(excinfo.value)
    
    def test_invalid_num_problems_type(self):
        """Test error on invalid num_problems type."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({"num_problems": "one hundred", "K": 3})
    
    def test_invalid_num_problems_value(self):
        """Test error on num_problems < 1."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({"num_problems": 0, "K": 3})
    
    def test_invalid_k_value(self):
        """Test error on K < 2."""
        from prompt_framing_study.validation import validate_config, ValidationError
        
        with pytest.raises(ValidationError):
            validate_config({"num_problems": 100, "K": 1})
    
    def test_warning_few_problems(self):
        """Test warning when num_problems < 50."""
        from prompt_framing_study.validation import validate_config
        
        warnings = validate_config({"num_problems": 20, "K": 3})
        assert any("too few" in w.lower() for w in warnings)
    
    def test_warning_zero_temperature(self):
        """Test warning when temperature = 0."""
        from prompt_framing_study.validation import validate_config
        
        warnings = validate_config({"num_problems": 100, "K": 3, "temperature": 0})
        assert any("temperature" in w.lower() for w in warnings)
    
    def test_warning_repetitions_with_zero_temp(self):
        """Test warning for multiple repetitions with temperature=0."""
        from prompt_framing_study.validation import validate_config
        
        warnings = validate_config({
            "num_problems": 100,
            "K": 3,
            "temperature": 0,
            "num_repetitions": 3
        })
        assert any("repetitions" in w.lower() for w in warnings)


class TestClaimsValidation:
    """Tests for claims file validation."""
    
    def test_valid_claims_file(self, temp_dir):
        """Test validation of valid claims file."""
        from prompt_framing_study.validation import validate_claims_file
        
        filepath = temp_dir / "valid_claims.json"
        data = {
            "claims": [
                {"id": "C001", "description": "First claim"},
                {"id": "C002", "description": "Second claim"}
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        result = validate_claims_file(str(filepath))
        assert "claims" in result
        assert len(result["claims"]) == 2
    
    def test_missing_claims_array(self, temp_dir):
        """Test error on missing claims array."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        filepath = temp_dir / "no_claims.json"
        with open(filepath, 'w') as f:
            json.dump({"metadata": {}}, f)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_claims_file(str(filepath))
        assert "claims" in str(excinfo.value).lower()
    
    def test_too_few_claims(self, temp_dir):
        """Test error on fewer than 2 claims."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        filepath = temp_dir / "one_claim.json"
        data = {
            "claims": [{"id": "C001", "description": "Only claim"}]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValidationError):
            validate_claims_file(str(filepath))
    
    def test_missing_claim_id(self, temp_dir):
        """Test error on claim missing ID."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        filepath = temp_dir / "missing_id.json"
        data = {
            "claims": [
                {"description": "Missing ID"},
                {"id": "C002", "description": "Has ID"}
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_claims_file(str(filepath))
        assert "id" in str(excinfo.value).lower()
    
    def test_missing_claim_description(self, temp_dir):
        """Test error on claim missing description."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        filepath = temp_dir / "missing_desc.json"
        data = {
            "claims": [
                {"id": "C001"},
                {"id": "C002", "description": "Has description"}
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_claims_file(str(filepath))
        assert "description" in str(excinfo.value).lower()
    
    def test_duplicate_claim_ids(self, temp_dir):
        """Test error on duplicate claim IDs."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        filepath = temp_dir / "duplicate_ids.json"
        data = {
            "claims": [
                {"id": "C001", "description": "First"},
                {"id": "C001", "description": "Duplicate ID"}
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_claims_file(str(filepath))
        assert "duplicate" in str(excinfo.value).lower()
    
    def test_file_not_found(self):
        """Test error on non-existent file."""
        from prompt_framing_study.validation import validate_claims_file, ValidationError
        
        with pytest.raises(ValidationError) as excinfo:
            validate_claims_file("/nonexistent/path/claims.json")
        assert "not found" in str(excinfo.value).lower()


class TestStanDataValidation:
    """Tests for Stan data validation."""
    
    @pytest.fixture
    def valid_m0_data(self):
        """Valid m_0 Stan data."""
        return {
            "M": 3,
            "K": 3,
            "D": 4,
            "R": 5,
            "w": [[0.1, 0.2, 0.3, 0.4] for _ in range(5)],
            "I": [
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 1]
            ],
            "y": [1, 2, 1]
        }
    
    def test_valid_m0_data(self, valid_m0_data):
        """Test valid m_0 data passes validation."""
        from prompt_framing_study.validation import validate_stan_data
        
        validate_stan_data(valid_m0_data, model="m_0")
    
    def test_missing_field_m0(self, valid_m0_data):
        """Test error on missing required field."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        del valid_m0_data["y"]
        
        with pytest.raises(ValidationError) as excinfo:
            validate_stan_data(valid_m0_data, model="m_0")
        assert "y" in str(excinfo.value)
    
    def test_invalid_m_value(self, valid_m0_data):
        """Test error on invalid M value."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["M"] = 0
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_invalid_k_value(self, valid_m0_data):
        """Test error on K < 2."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["K"] = 1
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_wrong_w_rows(self, valid_m0_data):
        """Test error on wrong number of rows in w."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["w"] = [[0.1, 0.2, 0.3, 0.4] for _ in range(3)]  # Should be 5
        
        with pytest.raises(ValidationError) as excinfo:
            validate_stan_data(valid_m0_data, model="m_0")
        assert "w" in str(excinfo.value).lower()
    
    def test_wrong_w_cols(self, valid_m0_data):
        """Test error on wrong number of columns in w."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["w"] = [[0.1, 0.2] for _ in range(5)]  # Should be 4 columns
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_wrong_i_rows(self, valid_m0_data):
        """Test error on wrong number of rows in I."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["I"] = [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]  # Should be 3
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_invalid_i_values(self, valid_m0_data):
        """Test error on I containing values other than 0 and 1."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["I"][0][0] = 2
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_too_few_alternatives(self, valid_m0_data):
        """Test error on problem with fewer than 2 alternatives."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["I"][0] = [1, 0, 0, 0, 0]  # Only 1 alternative
        
        with pytest.raises(ValidationError) as excinfo:
            validate_stan_data(valid_m0_data, model="m_0")
        assert "fewer than 2" in str(excinfo.value).lower()
    
    def test_invalid_y_length(self, valid_m0_data):
        """Test error on wrong y length."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["y"] = [1, 2]  # Should be 3
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_invalid_y_value(self, valid_m0_data):
        """Test error on invalid choice value in y."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        valid_m0_data["y"][0] = 0  # Should be >= 1
        
        with pytest.raises(ValidationError):
            validate_stan_data(valid_m0_data, model="m_0")
    
    def test_y_exceeds_alternatives(self, valid_m0_data):
        """Test error on y value exceeding number of alternatives."""
        from prompt_framing_study.validation import validate_stan_data, ValidationError
        
        # Problem 0 has 2 alternatives (indices 0 and 1 are 1)
        valid_m0_data["y"][0] = 5  # Too high
        
        with pytest.raises(ValidationError) as excinfo:
            validate_stan_data(valid_m0_data, model="m_0")
        assert "exceeds" in str(excinfo.value).lower()


class TestEmbeddingsValidation:
    """Tests for embeddings validation."""
    
    def test_valid_embeddings(self):
        """Test valid embeddings pass validation."""
        import numpy as np
        from prompt_framing_study.validation import validate_embeddings
        
        embeddings = {
            "C001": np.array([0.1, 0.2, 0.3]),
            "C002": np.array([0.4, 0.5, 0.6])
        }
        
        warnings = validate_embeddings(embeddings, expected_claims=["C001", "C002"])
        assert isinstance(warnings, list)
    
    def test_missing_embedding(self):
        """Test error on missing embedding for expected claim."""
        import numpy as np
        from prompt_framing_study.validation import validate_embeddings, ValidationError
        
        embeddings = {
            "C001": np.array([0.1, 0.2, 0.3])
        }
        
        with pytest.raises(ValidationError) as excinfo:
            validate_embeddings(embeddings, expected_claims=["C001", "C002"])
        assert "missing" in str(excinfo.value).lower()
    
    def test_inconsistent_dimensions(self):
        """Test error on inconsistent embedding dimensions."""
        import numpy as np
        from prompt_framing_study.validation import validate_embeddings, ValidationError
        
        embeddings = {
            "C001": np.array([0.1, 0.2, 0.3]),
            "C002": np.array([0.4, 0.5])  # Different dimension
        }
        
        with pytest.raises(ValidationError) as excinfo:
            validate_embeddings(embeddings, expected_claims=["C001", "C002"])
        assert "inconsistent" in str(excinfo.value).lower()
    
    def test_nan_values(self):
        """Test error on NaN values in embeddings."""
        import numpy as np
        from prompt_framing_study.validation import validate_embeddings, ValidationError
        
        embeddings = {
            "C001": np.array([0.1, np.nan, 0.3]),
            "C002": np.array([0.4, 0.5, 0.6])
        }
        
        with pytest.raises(ValidationError) as excinfo:
            validate_embeddings(embeddings, expected_claims=["C001", "C002"])
        assert "nan" in str(excinfo.value).lower()
    
    def test_dimension_warning(self):
        """Test warning on unexpected dimension."""
        import numpy as np
        from prompt_framing_study.validation import validate_embeddings
        
        embeddings = {
            "C001": np.array([0.1, 0.2, 0.3]),
            "C002": np.array([0.4, 0.5, 0.6])
        }
        
        warnings = validate_embeddings(
            embeddings,
            expected_claims=["C001", "C002"],
            expected_dim=5
        )
        assert len(warnings) > 0
        assert any("dimension" in w.lower() for w in warnings)

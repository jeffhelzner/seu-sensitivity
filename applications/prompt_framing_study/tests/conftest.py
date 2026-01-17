"""
Test fixtures for prompt framing study tests.
"""
import pytest
from pathlib import Path
import json
import tempfile
import shutil


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        {
            "id": "C001",
            "description": "A homeowner filed a claim for water damage. The claim includes $15,000 for repairs."
        },
        {
            "id": "C002",
            "description": "An auto insurance claim for a rear-end collision. Repair estimate is $4,200."
        },
        {
            "id": "C003",
            "description": "A business interruption claim following a kitchen fire. Claim requests $50,000."
        },
        {
            "id": "C004",
            "description": "A health insurance claim for an emergency room visit. Total claim is $2,300."
        },
        {
            "id": "C005",
            "description": "A theft claim for stolen items from a home. Claimed items total $12,000."
        }
    ]


@pytest.fixture
def sample_problems(sample_claims):
    """Sample decision problems for testing."""
    return [
        {
            "id": "P0001",
            "claim_ids": ["C001", "C002", "C003"],
            "claims": [c["description"] for c in sample_claims[:3]],
            "num_alternatives": 3
        },
        {
            "id": "P0002",
            "claim_ids": ["C002", "C004"],
            "claims": [sample_claims[1]["description"], sample_claims[3]["description"]],
            "num_alternatives": 2
        },
        {
            "id": "P0003",
            "claim_ids": ["C001", "C003", "C005"],
            "claims": [sample_claims[0]["description"], sample_claims[2]["description"], sample_claims[4]["description"]],
            "num_alternatives": 3
        }
    ]


@pytest.fixture
def sample_config():
    """Sample study configuration."""
    return {
        "num_problems": 10,
        "K": 3,
        "min_alternatives": 2,
        "max_alternatives": 4,
        "temperature": 0.7,
        "num_repetitions": 1,
        "target_dim": 16,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4",
        "provider": "openai",
        "seed": 42
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def claims_file(temp_dir, sample_claims):
    """Create a temporary claims file."""
    filepath = temp_dir / "claims.json"
    data = {
        "claims": sample_claims,
        "consequences": ["bad", "neutral", "good"],
        "metadata": {"K": 3}
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)
    return str(filepath)


@pytest.fixture
def mock_embeddings(sample_claims):
    """Generate mock embeddings for testing."""
    import numpy as np
    np.random.seed(42)
    
    embeddings = {}
    for claim in sample_claims:
        embeddings[claim["id"]] = np.random.randn(1536)  # text-embedding-3-small dimension
    
    return embeddings


@pytest.fixture
def mock_reduced_embeddings(sample_claims):
    """Generate mock reduced embeddings for testing."""
    import numpy as np
    np.random.seed(42)
    
    embeddings = {}
    for claim in sample_claims:
        embeddings[claim["id"]] = np.random.randn(32)  # Reduced dimension
    
    return embeddings

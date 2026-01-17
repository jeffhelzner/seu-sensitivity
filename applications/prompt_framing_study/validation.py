"""
Validation utilities for configuration and data formats.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate study configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of warning messages (empty if all good)
        
    Raises:
        ValidationError: If critical validation fails
    """
    warnings = []
    
    # Required fields
    required = ["num_problems", "K"]
    for field in required:
        if field not in config:
            raise ValidationError(f"Missing required config field: {field}")
    
    # Type checks
    if not isinstance(config.get("num_problems"), int) or config["num_problems"] < 1:
        raise ValidationError("num_problems must be a positive integer")
    
    if not isinstance(config.get("K"), int) or config["K"] < 2:
        raise ValidationError("K must be an integer >= 2")
    
    # Warnings for potentially problematic values
    if config.get("num_problems", 0) < 50:
        warnings.append(f"num_problems={config['num_problems']} may be too few for reliable estimation")
    
    if config.get("temperature", 0) == 0:
        warnings.append("temperature=0 means deterministic responses; consider T>0 for variability")
    
    if config.get("num_repetitions", 1) > 1 and config.get("temperature", 0) == 0:
        warnings.append("Multiple repetitions with temperature=0 will yield identical results")
    
    # Check for reasonable embedding dimension
    target_dim = config.get("target_dim", 32)
    if target_dim < 8:
        warnings.append(f"target_dim={target_dim} is very low; consider at least 16")
    elif target_dim > 256:
        warnings.append(f"target_dim={target_dim} is high; may lead to overfitting")
    
    return warnings


def validate_stan_data(data: Dict[str, Any], model: str = "m_0") -> None:
    """
    Validate Stan data package matches expected schema.
    
    Args:
        data: Stan data dictionary
        model: Model name ("m_0" or "m_1")
        
    Raises:
        ValidationError: If validation fails
    """
    if model == "m_0":
        _validate_m0_data(data)
    elif model == "m_1":
        _validate_m1_data(data)
    else:
        raise ValidationError(f"Unknown model: {model}")


def _validate_m0_data(data: Dict[str, Any]) -> None:
    """Validate data for m_0 model."""
    required_fields = {"M", "K", "D", "R", "w", "I", "y"}
    
    # Check all required fields present
    missing = required_fields - set(data.keys())
    if missing:
        raise ValidationError(f"Missing required fields for m_0: {missing}")
    
    M, K, D, R = data["M"], data["K"], data["D"], data["R"]
    
    # Validate dimensions
    if not isinstance(M, int) or M < 1:
        raise ValidationError(f"M must be positive integer, got {M}")
    if not isinstance(K, int) or K < 2:
        raise ValidationError(f"K must be integer >= 2, got {K}")
    if not isinstance(D, int) or D < 1:
        raise ValidationError(f"D must be positive integer, got {D}")
    if not isinstance(R, int) or R < 1:
        raise ValidationError(f"R must be positive integer, got {R}")
    
    # Validate w array shape (R x D)
    w = data["w"]
    if len(w) != R:
        raise ValidationError(f"w should have {R} rows, got {len(w)}")
    if any(len(row) != D for row in w):
        raise ValidationError(f"w rows should have {D} columns")
    
    # Validate I array shape (M x R)
    I = data["I"]
    if len(I) != M:
        raise ValidationError(f"I should have {M} rows, got {len(I)}")
    if any(len(row) != R for row in I):
        raise ValidationError(f"I rows should have {R} columns")
    
    # Validate I contains only 0s and 1s
    for m, row in enumerate(I):
        if not all(v in (0, 1) for v in row):
            raise ValidationError(f"I[{m}] contains values other than 0 or 1")
        if sum(row) < 2:
            raise ValidationError(f"Problem {m} has fewer than 2 alternatives (sum(I[{m}])={sum(row)})")
    
    # Validate y
    y = data["y"]
    if len(y) != M:
        raise ValidationError(f"y should have {M} elements, got {len(y)}")
    
    # y values should be valid alternative indices
    for m, choice in enumerate(y):
        if not isinstance(choice, int) or choice < 1:
            raise ValidationError(f"y[{m}]={choice} is not a positive integer")
        # Check choice is among available alternatives for this problem
        num_alts = sum(I[m])
        if choice > num_alts:
            raise ValidationError(f"y[{m}]={choice} exceeds number of alternatives ({num_alts})")
    
    logger.info(f"Stan data validated: M={M}, K={K}, D={D}, R={R}")


def _validate_m1_data(data: Dict[str, Any]) -> None:
    """Validate data for m_1 model (extends m_0 with risky choice)."""
    # First validate m_0 fields
    _validate_m0_data(data)
    
    # Additional m_1 fields
    m1_fields = {"N", "p", "y_risky"}
    missing = m1_fields - set(data.keys())
    if missing:
        raise ValidationError(f"Missing required fields for m_1: {missing}")
    
    N, K = data["N"], data["K"]
    
    # Validate N
    if not isinstance(N, int) or N < 1:
        raise ValidationError(f"N must be positive integer, got {N}")
    
    # Validate p dimensions
    p = data["p"]
    if len(p) != N:
        raise ValidationError(f"p should have {N} elements, got {len(p)}")
    
    # Validate y_risky
    y_risky = data["y_risky"]
    if len(y_risky) != N:
        raise ValidationError(f"y_risky should have {N} elements, got {len(y_risky)}")
    
    logger.info(f"Stan data validated for m_1: N={N} risky problems")


def validate_claims_file(filepath: str) -> Dict[str, Any]:
    """
    Validate and load claims file.
    
    Returns:
        Validated claims data
        
    Raises:
        ValidationError: If validation fails
    """
    path = Path(filepath)
    if not path.exists():
        raise ValidationError(f"Claims file not found: {filepath}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Required fields
    if "claims" not in data:
        raise ValidationError("Claims file must contain 'claims' array")
    
    claims = data["claims"]
    if not isinstance(claims, list) or len(claims) < 2:
        raise ValidationError("Must have at least 2 claims")
    
    # Validate each claim
    claim_ids = set()
    for i, claim in enumerate(claims):
        if "id" not in claim:
            raise ValidationError(f"Claim {i} missing 'id' field")
        if "description" not in claim:
            raise ValidationError(f"Claim {i} missing 'description' field")
        
        if claim["id"] in claim_ids:
            raise ValidationError(f"Duplicate claim ID: {claim['id']}")
        claim_ids.add(claim["id"])
    
    logger.info(f"Validated {len(claims)} claims from {filepath}")
    return data


def validate_problems(problems: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> List[str]:
    """
    Validate generated decision problems against claims.
    
    Args:
        problems: List of decision problem dicts
        claims: List of claim dicts with 'id' field
        
    Returns:
        List of warning messages
        
    Raises:
        ValidationError: If critical validation fails
    """
    warnings = []
    claim_ids = {c["id"] for c in claims}
    
    for i, problem in enumerate(problems):
        if "id" not in problem:
            raise ValidationError(f"Problem {i} missing 'id' field")
        if "claim_ids" not in problem:
            raise ValidationError(f"Problem {i} missing 'claim_ids' field")
        
        # Check all claim IDs are valid
        for cid in problem["claim_ids"]:
            if cid not in claim_ids:
                raise ValidationError(f"Problem {problem['id']} references unknown claim: {cid}")
        
        # Check minimum alternatives
        if len(problem["claim_ids"]) < 2:
            raise ValidationError(f"Problem {problem['id']} has fewer than 2 alternatives")
    
    # Check for duplicate problem IDs
    problem_ids = [p["id"] for p in problems]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValidationError("Duplicate problem IDs detected")
    
    return warnings


def validate_embeddings(
    embeddings: Dict[str, Any],
    expected_claims: List[str],
    expected_dim: Optional[int] = None
) -> List[str]:
    """
    Validate embedding dictionary.
    
    Args:
        embeddings: Dict mapping claim_id -> embedding array
        expected_claims: List of expected claim IDs
        expected_dim: Expected embedding dimension (optional)
        
    Returns:
        List of warning messages
        
    Raises:
        ValidationError: If validation fails
    """
    import numpy as np
    
    warnings = []
    
    # Check all expected claims have embeddings
    missing = set(expected_claims) - set(embeddings.keys())
    if missing:
        raise ValidationError(f"Missing embeddings for claims: {missing}")
    
    # Check embedding dimensions are consistent
    dims = set()
    for claim_id, emb in embeddings.items():
        arr = np.asarray(emb)
        if arr.ndim != 1:
            raise ValidationError(f"Embedding for {claim_id} is not 1D: shape={arr.shape}")
        dims.add(len(arr))
    
    if len(dims) > 1:
        raise ValidationError(f"Inconsistent embedding dimensions: {dims}")
    
    actual_dim = dims.pop() if dims else 0
    if expected_dim is not None and actual_dim != expected_dim:
        warnings.append(f"Expected dimension {expected_dim}, got {actual_dim}")
    
    # Check for NaN/Inf values
    for claim_id, emb in embeddings.items():
        arr = np.asarray(emb)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValidationError(f"Embedding for {claim_id} contains NaN or Inf values")
    
    return warnings

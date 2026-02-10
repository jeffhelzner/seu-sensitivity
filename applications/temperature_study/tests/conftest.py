"""
Shared test fixtures for the temperature study test suite.
"""
import pytest
import json
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_claims():
    """A minimal set of 5 claims for testing."""
    return [
        {
            "id": "C001",
            "description": (
                "A homeowner filed a claim for water damage to their basement, "
                "stating that a pipe burst during a cold snap. The claim includes "
                "$15,000 for flooring replacement and $8,000 for damaged furniture."
            ),
        },
        {
            "id": "C002",
            "description": (
                "An auto insurance claim for a rear-end collision in a shopping "
                "center parking lot. The claimant has filed three similar claims "
                "in the past two years."
            ),
        },
        {
            "id": "C003",
            "description": (
                "A business interruption claim from a restaurant owner following "
                "a kitchen fire. Fire department report confirms grease fire but "
                "notes fire suppression system was not properly maintained."
            ),
        },
        {
            "id": "C004",
            "description": (
                "A health insurance claim for an emergency room visit following "
                "a sports injury. Medical records show treatment for a sprained "
                "ankle with X-rays confirming no fracture. Total claim is $2,300."
            ),
        },
        {
            "id": "C005",
            "description": (
                "A theft claim for a stolen laptop and jewelry from a home. "
                "Police report filed but no signs of forced entry noted. "
                "Claimed items total $12,000 including a $7,000 watch."
            ),
        },
    ]


@pytest.fixture
def sample_problems(sample_claims):
    """Three sample problems with presentations."""
    return [
        {
            "id": "P0001",
            "claim_ids": ["C001", "C002", "C003"],
            "num_alternatives": 3,
            "presentations": [
                {"presentation_id": 1, "order": ["C001", "C002", "C003"]},
                {"presentation_id": 2, "order": ["C003", "C001", "C002"]},
                {"presentation_id": 3, "order": ["C002", "C003", "C001"]},
            ],
        },
        {
            "id": "P0002",
            "claim_ids": ["C002", "C004"],
            "num_alternatives": 2,
            "presentations": [
                {"presentation_id": 1, "order": ["C002", "C004"]},
                {"presentation_id": 2, "order": ["C004", "C002"]},
                {"presentation_id": 3, "order": ["C002", "C004"]},
            ],
        },
        {
            "id": "P0003",
            "claim_ids": ["C001", "C003", "C004", "C005"],
            "num_alternatives": 4,
            "presentations": [
                {"presentation_id": 1, "order": ["C001", "C003", "C004", "C005"]},
                {"presentation_id": 2, "order": ["C005", "C001", "C003", "C004"]},
                {"presentation_id": 3, "order": ["C004", "C005", "C001", "C003"]},
            ],
        },
    ]


@pytest.fixture
def sample_config():
    """A minimal study config dict for testing."""
    return {
        "temperatures": [0.0, 0.7, 1.5],
        "num_problems": 5,
        "min_alternatives": 2,
        "max_alternatives": 3,
        "num_presentations": 2,
        "K": 3,
        "target_dim": 8,
        "llm_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "seed": 42,
    }


@pytest.fixture
def sample_raw_embeddings():
    """Simulated raw embeddings (1536-dim) for 5 claims."""
    rng = np.random.default_rng(42)
    return {f"C00{i}": rng.standard_normal(1536) for i in range(1, 6)}


@pytest.fixture
def sample_reduced_embeddings():
    """Simulated reduced embeddings (8-dim) for 5 claims."""
    rng = np.random.default_rng(42)
    return {f"C00{i}": rng.standard_normal(8) for i in range(1, 6)}


@pytest.fixture
def tmp_results_dir(tmp_path):
    """A temporary directory for test outputs."""
    results = tmp_path / "results"
    results.mkdir()
    return results


def _make_choice(
    problem_id, presentation_id, claim_order, position_chosen,
    valid=True, claim_chosen=None, temperature=0.0,
):
    """Helper to build a single choice entry dict."""
    if valid and claim_chosen is None:
        claim_chosen = claim_order[position_chosen - 1]
    return {
        "problem_id": problem_id,
        "presentation_id": presentation_id,
        "claim_order": claim_order,
        "position_chosen": position_chosen if valid else None,
        "claim_chosen": claim_chosen if valid else None,
        "valid": valid,
        "temperature": temperature,
        "raw_response": "Claim 1" if valid else "I don't know",
    }


@pytest.fixture
def sample_choice_entries():
    """
    Choice entries for 3 problems × 3 presentations.  All valid,
    with a mild position-1 bias to make tests interesting.
    """
    entries = []
    orders = {
        "P0001": [
            ["C001", "C002", "C003"],
            ["C003", "C001", "C002"],
            ["C002", "C003", "C001"],
        ],
        "P0002": [
            ["C002", "C004"],
            ["C004", "C002"],
            ["C002", "C004"],
        ],
        "P0003": [
            ["C001", "C003", "C004", "C005"],
            ["C005", "C001", "C003", "C004"],
            ["C004", "C005", "C001", "C003"],
        ],
    }
    # Deliberate pattern: usually pick position 1 (mild bias)
    positions = {
        "P0001": [1, 1, 2],   # C001, C003, C003
        "P0002": [1, 2, 1],   # C002, C002, C002 — unanimous
        "P0003": [1, 1, 3],   # C001, C005, C001
    }
    for pid in ["P0001", "P0002", "P0003"]:
        for i in range(3):
            entries.append(_make_choice(
                problem_id=pid,
                presentation_id=i + 1,
                claim_order=orders[pid][i],
                position_chosen=positions[pid][i],
            ))
    return entries


@pytest.fixture
def sample_choice_entries_with_na(sample_choice_entries):
    """Same as sample_choice_entries but with 2 NAs injected."""
    entries = list(sample_choice_entries)
    # Replace last entry with an NA
    entries[-1] = _make_choice(
        problem_id="P0003",
        presentation_id=3,
        claim_order=["C004", "C005", "C001", "C003"],
        position_chosen=0,
        valid=False,
    )
    # Add another NA for P0001
    entries.append(_make_choice(
        problem_id="P0001",
        presentation_id=4,
        claim_order=["C001", "C002", "C003"],
        position_chosen=0,
        valid=False,
    ))
    return entries


@pytest.fixture
def per_temp_choices(sample_choice_entries, sample_choice_entries_with_na):
    """
    Multi-temperature choice data: T=0.0 (clean), T=0.7 (with NAs).
    """
    valid_entries = sample_choice_entries
    na_entries = sample_choice_entries_with_na

    n_valid_0 = len(valid_entries)
    n_valid_07 = sum(1 for c in na_entries if c["valid"])
    n_na_07 = sum(1 for c in na_entries if not c["valid"])

    return {
        0.0: {
            "choices": valid_entries,
            "valid_choices": n_valid_0,
            "na_choices": 0,
            "total_choices": n_valid_0,
            "na_rate": 0.0,
        },
        0.7: {
            "choices": na_entries,
            "valid_choices": n_valid_07,
            "na_choices": n_na_07,
            "total_choices": len(na_entries),
            "na_rate": round(n_na_07 / len(na_entries), 4),
        },
    }

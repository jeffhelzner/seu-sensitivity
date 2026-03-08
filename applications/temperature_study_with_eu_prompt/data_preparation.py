"""
Data Preparation for the Temperature Study with EU Prompt.

Loads pre-computed embeddings and PCA-reduced features from the base
temperature study, then combines them with newly collected EU-prompt
choices to produce Stan data packages.

No new PCA fitting is needed — the feature space is identical.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import StudyConfig

# Reuse Stan data building and saving infrastructure from the base study
from applications.temperature_study.data_preparation import (
    StanDataBuilder,
    filter_valid_choices,
    save_stan_data,
    save_na_log,
)

logger = logging.getLogger(__name__)


def load_base_study_data(
    config: StudyConfig,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[float, Dict[str, Any]],
    Dict[float, Dict[str, np.ndarray]],
]:
    """
    Load problems, assessments, and reduced embeddings from the base study.

    Args:
        config: Study configuration with ``base_study_results_dir``.

    Returns:
        ``(problems, per_temp_assess, per_temp_reduced)``
    """
    base_dir = Path(config.base_study_results_dir)

    # Load problems
    problems_path = base_dir / "problems.json"
    with open(problems_path) as f:
        problems_data = json.load(f)
    problems = problems_data["problems"]
    logger.info("Loaded %d problems from %s", len(problems), problems_path)

    # Load assessments and reduced embeddings per temperature
    per_temp_assess: Dict[float, Dict[str, Any]] = {}
    per_temp_reduced: Dict[float, Dict[str, np.ndarray]] = {}

    for temp in config.temperatures:
        ts = f"{temp:.1f}".replace(".", "_")

        # Assessments
        assess_path = base_dir / f"assessments_T{ts}.json"
        with open(assess_path) as f:
            per_temp_assess[temp] = json.load(f)
        logger.info("Loaded assessments for T=%.1f from %s", temp, assess_path)

        # Reduced embeddings
        emb_path = base_dir / f"embeddings_reduced_T{ts}.npz"
        data = np.load(emb_path)
        per_temp_reduced[temp] = {k: data[k] for k in data.files}
        logger.info(
            "Loaded %d reduced embeddings for T=%.1f from %s",
            len(per_temp_reduced[temp]),
            temp,
            emb_path,
        )

    return problems, per_temp_assess, per_temp_reduced


def prepare_stan_data(
    config: StudyConfig,
    problems: List[Dict[str, Any]],
    reduced_embeddings: Dict[str, np.ndarray],
    choices_dict: Dict[str, Any],
    temperature: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter NAs and assemble Stan data for one temperature.

    Args:
        config: Study configuration.
        problems: Full problem list.
        reduced_embeddings: {claim_id → D-dim vector} for this temperature.
        choices_dict: Raw choices output for this temperature.
        temperature: The temperature level.

    Returns:
        ``(stan_data, na_log)``
    """
    # Filter NAs
    valid_entries, na_log = filter_valid_choices(choices_dict)

    # Build Stan data
    builder = StanDataBuilder(config)
    stan_data = builder.build(valid_entries, reduced_embeddings, problems)

    # Validate
    issues = StanDataBuilder.validate_stan_data(stan_data)
    if issues:
        for issue in issues:
            logger.error("Stan data issue (T=%.1f): %s", temperature, issue)
        raise ValueError(
            f"Stan data validation failed for T={temperature}: {issues}"
        )

    return stan_data, na_log

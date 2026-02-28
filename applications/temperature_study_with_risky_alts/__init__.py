"""
Temperature Study with Risky Alternatives Module

Extends the temperature study by collecting risky choice data from the LLM
at each temperature level, then merges with existing uncertain data to produce
augmented Stan data packages for models m_1, m_2, and m_3.

This module reuses the LLM client infrastructure from temperature_study
but implements its own problem generation, choice collection, and data
preparation pipeline for the risky alternatives.
"""
import logging

# Configure module-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
from .config import StudyConfig, ConfigError
from .risky_problem_generator import RiskyProblemGenerator
from .risky_choice_collector import RiskyChoiceCollector
from .data_preparation import (
    RiskyStanDataBuilder,
    build_augmented_stan_data,
    filter_valid_risky_choices,
    validate_augmented_stan_data,
)
from .study_runner import RiskyStudyRunner

__all__ = [
    "StudyConfig",
    "ConfigError",
    "RiskyProblemGenerator",
    "RiskyChoiceCollector",
    "RiskyStanDataBuilder",
    "build_augmented_stan_data",
    "filter_valid_risky_choices",
    "validate_augmented_stan_data",
    "RiskyStudyRunner",
]

__version__ = "0.1.0"

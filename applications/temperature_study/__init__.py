"""
Temperature Study Module

Investigates how LLM temperature affects estimated sensitivity (α)
to subjective expected utility maximization.

This module is standalone and does not depend on the legacy llm_rationality module.
"""
import logging

# Configure module-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API — expanded as modules are implemented
from .config import StudyConfig, ConfigError
from .llm_client import (
    LLMClient,
    OpenAIClient,
    AnthropicClient,
    EmbeddingClient,
    create_llm_client,
    parse_choice,
)
from .problem_generator import ProblemGenerator
from .assessment_collector import AssessmentCollector
from .choice_collector import ChoiceCollector
from .data_preparation import EmbeddingReducer, StanDataBuilder, filter_valid_choices
from .study_runner import TemperatureStudyRunner

# Analysis modules
from . import position_analysis
from . import consistency_analysis
from . import na_analysis
from . import visualization

__all__ = [
    "StudyConfig",
    "ConfigError",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "EmbeddingClient",
    "create_llm_client",
    "parse_choice",
    "ProblemGenerator",
    "AssessmentCollector",
    "ChoiceCollector",
    "EmbeddingReducer",
    "StanDataBuilder",
    "filter_valid_choices",
    "TemperatureStudyRunner",
    # Analysis sub-modules
    "position_analysis",
    "consistency_analysis",
    "na_analysis",
    "visualization",
]

__version__ = "0.1.0"

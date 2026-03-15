"""
Ellsberg Study Module

Investigates how LLM temperature affects estimated sensitivity (α)
to subjective expected utility maximization, using Ellsberg-style urn
gambles as alternatives and Claude 3.5 Sonnet (Anthropic).
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import StudyConfig, ConfigError
from .problem_generator import ProblemGenerator
from .assessment_collector import AssessmentCollector
from .choice_collector import ChoiceCollector
from .data_preparation import EllsbergStanDataBuilder, EmbeddingReducer
from .study_runner import EllsbergStudyRunner

__all__ = [
    "StudyConfig",
    "ConfigError",
    "ProblemGenerator",
    "AssessmentCollector",
    "ChoiceCollector",
    "EllsbergStanDataBuilder",
    "EmbeddingReducer",
    "EllsbergStudyRunner",
]

__version__ = "0.1.0"

"""
Temperature Study with EU Prompt Module

Investigates whether explicitly instructing an LLM to maximize expected
utility increases estimated sensitivity (α) compared to the base
temperature study, and whether the temperature–sensitivity relationship
is preserved.

Reuses problems, assessments, and embeddings from the base temperature
study. Only the choice prompt is modified.
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import StudyConfig, ConfigError
from .choice_collector import ChoiceCollector
from .study_runner import EUPromptStudyRunner

__all__ = [
    "StudyConfig",
    "ConfigError",
    "ChoiceCollector",
    "EUPromptStudyRunner",
]

__version__ = "0.1.0"

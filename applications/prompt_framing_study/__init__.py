"""
Prompt Framing Study Module

Investigates how prompt framing affects LLM sensitivity to 
subjective expected utility maximization.

This module is standalone and does not depend on the legacy llm_rationality module.
"""
import logging

# Configure module-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
from .prompt_manager import PromptManager, PromptVariant
from .contextualized_embedding import ContextualizedEmbeddingManager
from .choice_collector import ChoiceCollector
from .problem_generator import ProblemGenerator
from .study_runner import StudyRunner
from .robustness_analysis import RobustnessAnalyzer
from .visualization import StudyVisualizer
from .cost_estimator import CostEstimator
from .validation import validate_stan_data, validate_config

__all__ = [
    "PromptManager",
    "PromptVariant",
    "ContextualizedEmbeddingManager",
    "ChoiceCollector",
    "ProblemGenerator",
    "StudyRunner",
    "RobustnessAnalyzer",
    "StudyVisualizer",
    "CostEstimator",
    "validate_stan_data",
    "validate_config",
]

__version__ = "0.1.0"

"""
Claude × Insurance Study Module (Cell 2,1)

2×2 Factorial experiment cell: Claude 3.5 Sonnet on the insurance
claims triage task.  Reuses the temperature study's task infrastructure
with Anthropic as the LLM provider.
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import StudyConfig, ConfigError
from .study_runner import ClaudeInsuranceStudyRunner

__all__ = [
    "StudyConfig",
    "ConfigError",
    "ClaudeInsuranceStudyRunner",
]

__version__ = "0.1.0"

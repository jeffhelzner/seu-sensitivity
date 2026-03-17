"""
GPT-4o × Ellsberg Study Module (Cell 1,2)

2×2 Factorial experiment cell: GPT-4o on Ellsberg-style urn gambles.
Reuses the Ellsberg study's task infrastructure with OpenAI as the
LLM provider.
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import StudyConfig, ConfigError
from .study_runner import GPT4oEllsbergStudyRunner

__all__ = [
    "StudyConfig",
    "ConfigError",
    "GPT4oEllsbergStudyRunner",
]

__version__ = "0.1.0"

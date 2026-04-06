"""
Alignment Study Module

Investigates how LLM model identity and prompt framing affect
estimated sensitivity (α) to subjective expected utility maximization,
using a hierarchical Bayesian model (h_m01) across a 6-model × 3-prompt
factorial design.
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import AlignmentStudyConfig, CellSpec, build_cells, MODELS, PROMPT_CONDITIONS

__all__ = [
    "AlignmentStudyConfig",
    "CellSpec",
    "build_cells",
    "MODELS",
    "PROMPT_CONDITIONS",
]

__version__ = "0.1.0"

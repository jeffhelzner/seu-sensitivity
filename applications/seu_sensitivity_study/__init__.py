"""
SEU Sensitivity Study Module

Measures how LLM model identity, prompt framing, decision domain, and menu size
affect estimated sensitivity (alpha) to subjective-expected-utility
maximization, using the hierarchical model ``h_m01`` over a 6-model x 3-prompt
x 3-pool factorial design.  See ``local/seu_sensitivity_study_plan.md`` (v0.5).
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import pools, schemas
from .config import (
    MODELS,
    PROMPT_CONDITIONS,
    REFERENCE_MODEL,
    REFERENCE_PROMPT,
    CellSpec,
    ModelSpec,
    SEUSensitivityStudyConfig,
    build_cells,
)

__all__ = [
    "SEUSensitivityStudyConfig",
    "CellSpec",
    "ModelSpec",
    "build_cells",
    "MODELS",
    "PROMPT_CONDITIONS",
    "REFERENCE_MODEL",
    "REFERENCE_PROMPT",
    "pools",
    "schemas",
]

__version__ = "0.2.0"

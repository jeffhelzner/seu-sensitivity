"""
Configuration for the 6-model × 3-prompt alignment study.

All cells use temperature=0.0, insurance claims triage task (K=3),
and the same alternative pool (30 claims from temperature_study).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# --- Cell specification ---

@dataclass
class CellSpec:
    """Specification for one factorial cell."""
    cell_id: str                # e.g. "gpt4o_neutral"
    model_name: str             # e.g. "gpt-4o"
    provider: str               # "openai" or "anthropic"
    prompt_condition: str       # "neutral", "eu_maximizing", "deliberative"
    temperature: float = 0.0
    # Provider-specific kwargs (e.g. reasoning budget for o3)
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)


# --- Factory for the 6×3 design ---

MODELS = [
    {"name": "gpt-4o",                       "provider": "openai"},
    {"name": "gpt-4o-mini",                  "provider": "openai"},
    {"name": "o3-mini",                       "provider": "openai",
     "provider_kwargs": {"reasoning_effort": "medium"}},
    {"name": "claude-sonnet-4-20250514",      "provider": "anthropic"},
    {"name": "claude-3-5-haiku-20241022",     "provider": "anthropic"},
    {"name": "claude-3-7-sonnet-20250219",    "provider": "anthropic",
     "provider_kwargs": {"extended_thinking": True, "budget_tokens": 4096}},
]

PROMPT_CONDITIONS = ["neutral", "eu_maximizing", "deliberative"]


def build_cells() -> List[CellSpec]:
    """Generate all 18 CellSpec instances for the 6×3 factorial."""
    cells = []
    for model_info in MODELS:
        for prompt in PROMPT_CONDITIONS:
            cell_id = f"{model_info['name'].replace('-', '_').replace('.', '_')}_{prompt}"
            cells.append(CellSpec(
                cell_id=cell_id,
                model_name=model_info["name"],
                provider=model_info["provider"],
                prompt_condition=prompt,
                provider_kwargs=model_info.get("provider_kwargs", {}),
            ))
    return cells


# --- Study-level config ---

@dataclass
class AlignmentStudyConfig:
    """Top-level config for the alignment study."""
    cells: List[CellSpec] = field(default_factory=build_cells)

    # Problem parameters (shared across cells)
    num_problems: int = 100
    min_alternatives: int = 2
    max_alternatives: int = 4
    num_presentations: int = 3
    K: int = 3
    target_dim: int = 32

    # Embedding (always OpenAI, regardless of choice LLM)
    embedding_model: str = "text-embedding-3-small"

    # Reproducibility
    seed: int = 42

    # File paths
    claims_file: Optional[str] = None
    prompts_file: Optional[str] = None

    # API robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Storage
    save_raw_embeddings: bool = True
    results_dir: Optional[str] = None

    # Stan model for hierarchical fitting
    stan_model: str = "h_m01"

    def __post_init__(self):
        if self.claims_file is None:
            self.claims_file = str(
                Path(__file__).parent.parent / "temperature_study" / "data" / "claims.json"
            )
        if self.prompts_file is None:
            self.prompts_file = str(
                Path(__file__).parent / "configs" / "prompts.yaml"
            )
        if self.results_dir is None:
            self.results_dir = str(Path(__file__).parent / "results")

    @classmethod
    def from_yaml(cls, path: str) -> AlignmentStudyConfig:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)

        # Build cells from YAML if specified, otherwise use defaults
        cells_data = raw.pop("cells", None)
        if cells_data is not None:
            cells = []
            for cd in cells_data:
                cells.append(CellSpec(**cd))
        else:
            cells = build_cells()

        config = cls(cells=cells, **{k: v for k, v in raw.items() if k != "cells"})
        return config

    def save_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        data = {
            "num_problems": self.num_problems,
            "min_alternatives": self.min_alternatives,
            "max_alternatives": self.max_alternatives,
            "num_presentations": self.num_presentations,
            "K": self.K,
            "target_dim": self.target_dim,
            "embedding_model": self.embedding_model,
            "seed": self.seed,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "save_raw_embeddings": self.save_raw_embeddings,
            "stan_model": self.stan_model,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def get_design_matrix(self) -> tuple:
        """
        Build the J × P design matrix for the regression.

        Encoding: treatment coding with GPT-4o + neutral as reference.
        Returns (X, column_names) where X is np.ndarray of shape (J, P).

        Columns (P=7):
          model_gpt4o_mini, model_o3_mini,
          model_claude_sonnet, model_claude_haiku, model_claude_thinking,
          prompt_eu, prompt_deliberative
        """
        model_names = [m["name"] for m in MODELS]
        ref_model = model_names[0]  # gpt-4o as reference
        ref_prompt = PROMPT_CONDITIONS[0]  # neutral as reference

        # Model dummies (5 columns: all except reference)
        model_columns = []
        model_col_names = []
        for m in model_names[1:]:
            safe_name = m.replace("-", "_").replace(".", "_")
            model_col_names.append(f"model_{safe_name}")
            model_columns.append(m)

        # Prompt dummies (2 columns: all except reference)
        prompt_columns = []
        prompt_col_names = []
        for p in PROMPT_CONDITIONS[1:]:
            prompt_col_names.append(f"prompt_{p}")
            prompt_columns.append(p)

        column_names = model_col_names + prompt_col_names
        P = len(column_names)
        J = len(self.cells)
        X = np.zeros((J, P), dtype=float)

        for j, cell in enumerate(self.cells):
            # Model dummies
            for col_idx, m_name in enumerate(model_columns):
                if cell.model_name == m_name:
                    X[j, col_idx] = 1.0

            # Prompt dummies
            for col_idx, p_name in enumerate(prompt_columns):
                if cell.prompt_condition == p_name:
                    X[j, len(model_columns) + col_idx] = 1.0

        return X, column_names

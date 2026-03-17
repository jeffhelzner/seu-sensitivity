"""
Configuration for the Claude × Insurance Study (Cell 2,1).

Overrides temperature study defaults for Anthropic provider and
Claude temperature grid.
"""
from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)

# Default paths: reuse temperature study's claims and prompts
_MODULE_DIR = Path(__file__).resolve().parent
_TEMP_STUDY_DIR = _MODULE_DIR.parent / "temperature_study"
DEFAULT_CONFIG_PATH = _MODULE_DIR / "configs" / "study_config.yaml"
DEFAULT_PROMPTS_PATH = _TEMP_STUDY_DIR / "configs" / "prompts.yaml"
DEFAULT_CLAIMS_PATH = _TEMP_STUDY_DIR / "data" / "claims.json"


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class StudyConfig:
    """
    Configuration for the Claude × Insurance Study.

    Same task as temperature_study (insurance claims, K=3) but uses
    Anthropic Claude with its temperature grid [0.0, 0.2, 0.5, 0.8, 1.0].
    """

    # Temperature conditions (Anthropic range)
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0]
    )

    # Problem parameters
    num_problems: int = 100
    min_alternatives: int = 2
    max_alternatives: int = 4
    num_presentations: int = 3

    # Model parameters
    K: int = 3
    target_dim: int = 32

    # API settings
    llm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"
    provider: str = "anthropic"

    # Reproducibility
    seed: int = 42

    # File paths
    claims_file: Optional[str] = None
    prompts_file: Optional[str] = None

    # Pipeline control
    fit_models: bool = False
    stan_model: str = "m_01"

    # API robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Storage
    save_raw_embeddings: bool = True
    results_dir: Optional[str] = None

    def __post_init__(self):
        if self.claims_file is None:
            self.claims_file = str(DEFAULT_CLAIMS_PATH)
        if self.prompts_file is None:
            self.prompts_file = str(DEFAULT_PROMPTS_PATH)
        if self.results_dir is None:
            self.results_dir = str(_MODULE_DIR / "results")

        self.validate()

    def validate(self) -> List[str]:
        warnings: List[str] = []

        if not self.temperatures:
            raise ConfigError("temperatures must be a non-empty list")
        if any(t < 0 for t in self.temperatures):
            raise ConfigError("All temperature values must be >= 0")
        if self.num_problems < 1:
            raise ConfigError("num_problems must be >= 1")
        if self.min_alternatives < 2:
            raise ConfigError("min_alternatives must be >= 2")
        if self.max_alternatives < self.min_alternatives:
            raise ConfigError("max_alternatives must be >= min_alternatives")
        if self.num_presentations < 1:
            raise ConfigError("num_presentations must be >= 1")
        if self.K < 2:
            raise ConfigError("K (consequences) must be >= 2")
        if self.target_dim < 1:
            raise ConfigError("target_dim must be >= 1")
        if self.provider not in ("openai", "anthropic"):
            raise ConfigError(f"Unknown provider: {self.provider}")

        # Anthropic temperature limit
        if self.provider == "anthropic":
            if any(t > 1.0 for t in self.temperatures):
                raise ConfigError(
                    "Anthropic API supports temperature in [0.0, 1.0]. "
                    f"Got temperatures: {self.temperatures}"
                )

        if self.num_problems < 20:
            warnings.append(
                f"num_problems={self.num_problems} is low; "
                "consider >= 50 for reliable model fitting"
            )
        if 0.0 in self.temperatures and len(self.temperatures) < 2:
            warnings.append(
                "Only T=0.0 specified; need multiple temperatures for comparison"
            )
        if self.target_dim > 64:
            warnings.append(
                f"target_dim={self.target_dim} is high; "
                "32 is usually sufficient"
            )
        if self.num_presentations < 2:
            warnings.append(
                "num_presentations=1 provides no position counterbalancing"
            )

        for w in warnings:
            logger.warning(w)
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "StudyConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning("Config file %s not found; using defaults", path)
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        logger.info("Loaded config from %s", path)
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)

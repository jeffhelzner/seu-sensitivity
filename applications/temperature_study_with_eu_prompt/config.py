"""
Configuration module for the Temperature Study with EU Prompt.

Defines StudyConfig dataclass with validation and YAML loading.
This study reuses problems, assessments, and embeddings from the
base temperature study, collecting only new choices under a modified
prompt that explicitly instructs EU maximization.
"""
from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)

# Default paths relative to this module
_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _MODULE_DIR / "configs" / "study_config.yaml"
DEFAULT_PROMPTS_PATH = _MODULE_DIR / "configs" / "prompts.yaml"
DEFAULT_BASE_STUDY_RESULTS = (
    _MODULE_DIR.parent / "temperature_study" / "results"
)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class StudyConfig:
    """
    Configuration for the Temperature Study with EU Prompt.

    Attributes:
        temperatures: LLM temperature levels (must match original study).
        K: Number of consequences in the SEU model.
        target_dim: PCA target dimensionality (must match base study).
        stan_model: Which Stan model to fit (default "m_01").
        llm_model: Model name for choice collection.
        provider: LLM API provider ("openai" or "anthropic").
        seed: Random seed for reproducibility.
        prompts_file: Path to the prompts YAML file.
        fit_models: Whether to fit Stan models as part of the pipeline.
        max_retries: Maximum API retry attempts.
        retry_delay: Base delay (seconds) between retries.
        results_dir: Directory for outputs.
        base_study_results_dir: Path to original temperature study results.
    """

    # Temperature conditions
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.7, 1.0, 1.5]
    )

    # Model parameters
    K: int = 3
    target_dim: int = 32
    stan_model: str = "m_01"

    # API settings
    llm_model: str = "gpt-4o"
    provider: str = "openai"

    # Reproducibility
    seed: int = 42

    # File paths
    prompts_file: Optional[str] = None

    # Pipeline control
    fit_models: bool = False

    # API robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Storage
    results_dir: Optional[str] = None
    base_study_results_dir: Optional[str] = None

    def __post_init__(self):
        """Set defaults that depend on module location and validate."""
        if self.prompts_file is None:
            self.prompts_file = str(DEFAULT_PROMPTS_PATH)
        if self.results_dir is None:
            self.results_dir = str(_MODULE_DIR / "results")
        if self.base_study_results_dir is None:
            self.base_study_results_dir = str(DEFAULT_BASE_STUDY_RESULTS)

        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of warning strings (empty if no warnings).

        Raises:
            ConfigError: On hard validation failures.
        """
        warnings: List[str] = []

        # Hard constraints
        if not self.temperatures:
            raise ConfigError("temperatures must be a non-empty list")
        if any(t < 0 for t in self.temperatures):
            raise ConfigError("All temperature values must be ≥ 0")
        if self.K < 2:
            raise ConfigError("K (consequences) must be ≥ 2")
        if self.target_dim < 1:
            raise ConfigError("target_dim must be ≥ 1")
        if self.provider not in ("openai", "anthropic"):
            raise ConfigError(f"Unknown provider: {self.provider}")

        # Check that the base temperature study results exist
        base_dir = Path(self.base_study_results_dir)
        if not base_dir.exists():
            warnings.append(
                f"Base study results dir not found: {base_dir}. "
                "The pipeline requires problems, assessments, and embeddings "
                "from the base temperature study."
            )

        for w in warnings:
            logger.warning(w)

        return warnings

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (for JSON serialization)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyConfig":
        """Create from a dict, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "StudyConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML file.  Falls back to the default
                  ``configs/study_config.yaml`` shipped with the module.

        Returns:
            Validated StudyConfig instance.
        """
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning(
                "Config file %s not found; using defaults", path
            )
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        logger.info("Loaded config from %s", path)
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)

"""
Configuration module for the Temperature Study with Risky Alternatives.

Defines StudyConfig dataclass with validation and YAML loading.
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
DEFAULT_RISKY_ALTS_PATH = _MODULE_DIR / "configs" / "risky_alternatives.json"
DEFAULT_TEMP_STUDY_RESULTS = (
    _MODULE_DIR.parent / "temperature_study" / "results"
)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class StudyConfig:
    """
    Configuration for the Temperature Study with Risky Alternatives.

    Attributes:
        temperatures: LLM temperature levels (must match original study).
        num_problems: Number of risky decision problems to generate.
        min_alternatives: Minimum risky alternatives per problem (≥2).
        max_alternatives: Maximum risky alternatives per problem.
        num_presentations: Shuffled orderings per problem (P).
        K: Number of consequences in the SEU model.
        llm_model: Model name for choice collection.
        provider: LLM API provider ("openai" or "anthropic").
        seed: Random seed for reproducibility.
        risky_alternatives_file: Path to the risky alternatives JSON.
        prompts_file: Path to the prompts YAML file.
        fit_models: Whether to fit Stan models as part of the pipeline.
        max_retries: Maximum API retry attempts.
        retry_delay: Base delay (seconds) between retries.
        results_dir: Directory for outputs.
        temperature_study_results_dir: Path to original temperature study results.
    """

    # Temperature conditions
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.7, 1.0, 1.5]
    )

    # Problem parameters
    num_problems: int = 100
    min_alternatives: int = 2
    max_alternatives: int = 4
    num_presentations: int = 3

    # Model parameters
    K: int = 3

    # API settings
    llm_model: str = "gpt-4o"
    provider: str = "openai"

    # Reproducibility
    seed: int = 42

    # File paths
    risky_alternatives_file: Optional[str] = None
    prompts_file: Optional[str] = None

    # Pipeline control
    fit_models: bool = False

    # API robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Storage
    results_dir: Optional[str] = None
    temperature_study_results_dir: Optional[str] = None

    def __post_init__(self):
        """Set defaults that depend on module location and validate."""
        if self.risky_alternatives_file is None:
            self.risky_alternatives_file = str(DEFAULT_RISKY_ALTS_PATH)
        if self.prompts_file is None:
            self.prompts_file = str(DEFAULT_PROMPTS_PATH)
        if self.results_dir is None:
            self.results_dir = str(_MODULE_DIR / "results")
        if self.temperature_study_results_dir is None:
            self.temperature_study_results_dir = str(DEFAULT_TEMP_STUDY_RESULTS)

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
        if self.num_problems < 1:
            raise ConfigError("num_problems must be ≥ 1")
        if self.min_alternatives < 2:
            raise ConfigError("min_alternatives must be ≥ 2")
        if self.max_alternatives < self.min_alternatives:
            raise ConfigError("max_alternatives must be ≥ min_alternatives")
        if self.num_presentations < 1:
            raise ConfigError("num_presentations must be ≥ 1")
        if self.K < 2:
            raise ConfigError("K (consequences) must be ≥ 2")
        if self.provider not in ("openai", "anthropic"):
            raise ConfigError(f"Unknown provider: {self.provider}")

        # Soft warnings
        if self.num_problems < 20:
            warnings.append(
                f"num_problems={self.num_problems} is low; "
                "consider ≥ 50 for reliable model fitting"
            )
        if self.num_presentations < 2:
            warnings.append(
                "num_presentations=1 provides no position counterbalancing"
            )

        # Check that the original temperature study results exist
        ts_dir = Path(self.temperature_study_results_dir)
        if not ts_dir.exists():
            warnings.append(
                f"temperature_study results dir not found: {ts_dir}. "
                "Augmented Stan data assembly will fail without it."
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

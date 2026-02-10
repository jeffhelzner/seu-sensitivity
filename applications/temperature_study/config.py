"""
Configuration module for the Temperature Study.

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
DEFAULT_CLAIMS_PATH = _MODULE_DIR / "data" / "claims.json"


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class StudyConfig:
    """
    Configuration for the Temperature Study.

    Attributes:
        temperatures: LLM temperature levels to test.
        num_problems: Number of base decision problems to generate.
        min_alternatives: Minimum claims per problem (≥2).
        max_alternatives: Maximum claims per problem.
        num_presentations: Number of shuffled presentations per problem (P).
        K: Number of consequences in the SEU model.
        target_dim: PCA target dimensionality for embeddings.
        llm_model: Model name for deliberation and choice collection.
        embedding_model: Model name for embedding deliberation responses.
        provider: LLM API provider ("openai" or "anthropic").
        seed: Random seed for reproducibility.
        claims_file: Path to the claim pool JSON file.
        prompts_file: Path to the prompts YAML file.
        fit_models: Whether to fit Stan models as part of the pipeline.
        stan_model: Which Stan model to fit (default "m_0").
        max_retries: Maximum API retry attempts.
        retry_delay: Base delay (seconds) between retries.
        save_raw_embeddings: Whether to persist full-dimension embeddings.
        results_dir: Directory for outputs (default: results/ under module).
    """

    # Temperature conditions
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.7, 1.0, 1.5]
    )

    # Problem parameters
    num_problems: int = 100
    min_alternatives: int = 2
    max_alternatives: int = 4
    num_presentations: int = 3  # P in the design doc

    # Model parameters
    K: int = 3  # number of consequences
    target_dim: int = 32  # embedding dimension after PCA

    # API settings
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    provider: str = "openai"

    # Reproducibility
    seed: int = 42

    # File paths (resolved at load time; defaults set in __post_init__)
    claims_file: Optional[str] = None
    prompts_file: Optional[str] = None

    # Pipeline control
    fit_models: bool = False
    stan_model: str = "m_0"

    # API robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Storage
    save_raw_embeddings: bool = True
    results_dir: Optional[str] = None

    def __post_init__(self):
        """Set defaults that depend on module location and validate."""
        if self.claims_file is None:
            self.claims_file = str(DEFAULT_CLAIMS_PATH)
        if self.prompts_file is None:
            self.prompts_file = str(DEFAULT_PROMPTS_PATH)
        if self.results_dir is None:
            self.results_dir = str(_MODULE_DIR / "results")

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
        if self.target_dim < 1:
            raise ConfigError("target_dim must be ≥ 1")
        if self.provider not in ("openai", "anthropic"):
            raise ConfigError(f"Unknown provider: {self.provider}")

        # Soft warnings
        if self.num_problems < 20:
            warnings.append(
                f"num_problems={self.num_problems} is low; "
                "consider ≥ 50 for reliable model fitting"
            )
        if 0.0 in self.temperatures and len(self.temperatures) < 2:
            warnings.append(
                "Only T=0.0 specified; need multiple temperatures for comparison"
            )
        if self.target_dim > 64:
            warnings.append(
                f"target_dim={self.target_dim} is high; "
                "32 is usually sufficient for claim diversity"
            )
        if self.num_presentations < 2:
            warnings.append(
                "num_presentations=1 provides no position counterbalancing"
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

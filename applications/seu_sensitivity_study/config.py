"""
Configuration for the 6-model x 3-prompt x 3-pool SEU sensitivity study.

The design is a 54-cell factorial (study plan §3), fitted **per pool** under
Option B (§8.1), so the design matrix this module builds is the 18-cell,
model-coded matrix of one pool -- five model dummies plus two prompt dummies
against a reference cell (B1, §2, §4).  Tier and vendor claims are derived
linear combinations of the fitted model coefficients, never separate columns:
a tier-coded matrix would pool vendors within a tier and make the within-vendor
H1 contrasts inestimable.

Decoding is deliberately **not uniform** (§3.1).  o3-mini accepts no
temperature parameter at all, and extended-thinking Claude constrains it, so
each model carries its own request parameters -- and for the reasoning tier
those parameters are part of the treatment, which is why they are pinned here
and copied into the run manifest (§6.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pools as pools_module
from . import schemas
from .schemas import MENU_SIZES, NUM_PRESENTATIONS, PROMPT_CONDITIONS

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSpec",
    "CellSpec",
    "MODELS",
    "PROMPT_CONDITIONS",
    "REFERENCE_MODEL",
    "REFERENCE_PROMPT",
    "get_model_spec",
    "build_cells",
    "assessment_keys",
    "SEUSensitivityStudyConfig",
]


# --- Models (§3.1) ---


@dataclass(frozen=True)
class ModelSpec:
    """One model arm, with its decoding settings pinned."""

    #: Unique ARM label.  This is the study's identity for the arm -- it drives
    #: :attr:`slug`, and through it the cell ids, the assessment artefact
    #: filenames, the cache paths and the design-matrix dummy names.  It is NOT
    #: necessarily the string sent to the provider: two arms may share one
    #: endpoint and differ only in ``request_params`` (see the reasoning tier).
    name: str
    provider: str
    #: "small" | "flagship" | "reasoning" -- a *reporting* grouping (§2), not a
    #: fitted factor.
    tier: str
    vendor: str
    #: Provider-specific request parameters.  For the reasoning tier these are
    #: part of the treatment, not incidental plumbing (§3.1).
    request_params: Dict[str, Any] = field(default_factory=dict)
    #: None where the provider accepts no temperature parameter.
    temperature: Optional[float] = 0.0
    #: The dated endpoint actually called, when it differs from :attr:`name`.
    #: Recorded as ``endpoint_id`` in the run manifest (§6.5).
    endpoint_id: Optional[str] = None
    #: Completion-token headroom reserved for tokens the model emits but does
    #: not show.  OpenAI reasoning tokens are billed as output AND consume the
    #: completion budget, so without a reserve the visible answer is truncated
    #: to nothing (measured: o3-mini spends 384-640 reasoning tokens on one
    #: assessment, and returns '' at a 400-token budget).  Anthropic thinking
    #: is handled separately, by adding ``budget_tokens`` inside the client.
    reasoning_token_reserve: int = 0
    #: Set when this arm replaces a deprecated model under the §3.1 rule.
    #: Surfaced in the run manifest so the substitution is a recorded fact
    #: rather than a code comment.
    substituted_for: Optional[str] = None
    substitution_reason: str = ""
    notes: str = ""

    @property
    def slug(self) -> str:
        return self.name.replace("-", "_").replace(".", "_")

    @property
    def endpoint(self) -> str:
        """The provider-facing model id for this arm."""
        return self.endpoint_id or self.name


#: Reason recorded in the run manifest for every §3.1 deprecation substitution.
_SUBSTITUTION_REASON = (
    "Original endpoint retired (HTTP 404 on 2026-08-06). Replaced under the "
    "§3.1 pre-declared rule: nearest available same-tier, same-vendor successor."
)


MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="gpt-4o",
        provider="openai",
        tier="flagship",
        vendor="openai",
        temperature=0.0,
    ),
    ModelSpec(
        name="gpt-4o-mini",
        provider="openai",
        tier="small",
        vendor="openai",
        temperature=0.0,
    ),
    ModelSpec(
        name="o3-mini",
        provider="openai",
        tier="reasoning",
        vendor="openai",
        temperature=None,
        request_params={"reasoning_effort": "medium"},
        reasoning_token_reserve=2048,
        notes="Accepts no temperature parameter; reasoning_effort is the treatment.",
    ),
    ModelSpec(
        name="claude-sonnet-4-5",
        provider="anthropic",
        tier="flagship",
        vendor="anthropic",
        temperature=0.0,
        endpoint_id="claude-sonnet-4-5-20250929",
        substituted_for="claude-sonnet-4-20250514",
        substitution_reason=_SUBSTITUTION_REASON,
        notes=(
            "Substituted for the retired claude-sonnet-4-20250514 under the §3.1 "
            "pre-declared rule (nearest available same-tier, same-vendor "
            "successor)."
        ),
    ),
    ModelSpec(
        name="claude-haiku-4-5",
        provider="anthropic",
        tier="small",
        vendor="anthropic",
        temperature=0.0,
        endpoint_id="claude-haiku-4-5-20251001",
        substituted_for="claude-3-5-haiku-20241022",
        substitution_reason=_SUBSTITUTION_REASON,
        notes=(
            "Substituted for the retired claude-3-5-haiku-20241022 under the "
            "§3.1 pre-declared rule."
        ),
    ),
    ModelSpec(
        name="claude-sonnet-4-5-thinking",
        provider="anthropic",
        tier="reasoning",
        vendor="anthropic",
        temperature=None,
        endpoint_id="claude-sonnet-4-5-20250929",
        request_params={"extended_thinking": True, "budget_tokens": 4096},
        substituted_for="claude-3-7-sonnet-20250219",
        substitution_reason=(
            _SUBSTITUTION_REASON
            + " NOTE: the successor shares a base with the flagship arm, which "
            "REMOVES the base-generation confound §3.1 warned about for this "
            "vendor; the Anthropic reasoning contrast is now exactly "
            "'flagship + extended thinking'."
        ),
        notes=(
            "Substituted for the retired claude-3-7-sonnet-20250219 under the "
            "§3.1 pre-declared rule. Extended thinking forces temperature 1.0. "
            "*** THIS CHANGES THE READING OF THE ANTHROPIC REASONING CONTRAST. "
            "The retired pair had an OLDER base for reasoning than for flagship, "
            "so §3.1 warned the contrast confounded base generation with "
            "inference-time compute. The nearest-successor rule maps BOTH arms "
            "onto claude-sonnet-4-5-20250929, so the confound is REMOVED and the "
            "contrast now is exactly 'flagship + thinking'. §3.1's warning no "
            "longer applies to this vendor and must be restated at E3. Note the "
            "arm NAME differs from the flagship's so that cells, caches and "
            "design-matrix dummies stay distinct. ***"
        ),
    ),
)

#: Treatment-coding reference levels.  Changing either re-bases every contrast,
#: so both are pinned here and recorded in the run manifest.
REFERENCE_MODEL = MODELS[0].name
REFERENCE_PROMPT = PROMPT_CONDITIONS[0]


def get_model_spec(name: str) -> ModelSpec:
    for spec in MODELS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model {name!r}; registered: {[m.name for m in MODELS]}")


# --- Cells (§3) ---


@dataclass(frozen=True)
class CellSpec:
    """One model x prompt x pool cell."""

    cell_id: str
    model_name: str
    provider: str
    prompt_condition: str
    pool_id: str
    request_params: Dict[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = 0.0
    #: Provider-facing model id; falls back to :attr:`model_name`.
    endpoint_id: Optional[str] = None
    #: See :attr:`ModelSpec.reasoning_token_reserve`.
    reasoning_token_reserve: int = 0

    @property
    def endpoint(self) -> str:
        return self.endpoint_id or self.model_name

    @property
    def assessment_key(self) -> str:
        """
        Identifies the shared assessment artefact this cell reads.

        Assessments are collected once per model x pool under the neutral
        instruction and shared across the three prompt cells (B2, §3.2), so the
        key deliberately omits the prompt condition.
        """
        return f"{get_model_spec(self.model_name).slug}__{self.pool_id}"


def build_cells(pool_ids: Optional[Sequence[str]] = None) -> List[CellSpec]:
    """Build the full factorial: 6 models x 3 prompts x len(pool_ids) pools."""
    selected = list(pool_ids) if pool_ids is not None else pools_module.available_pools()
    cells: List[CellSpec] = []
    for pool_id in selected:
        for model in MODELS:
            for prompt in PROMPT_CONDITIONS:
                cells.append(
                    CellSpec(
                        cell_id=f"{model.slug}_{prompt}_{pool_id}",
                        model_name=model.name,
                        provider=model.provider,
                        prompt_condition=prompt,
                        pool_id=pool_id,
                        request_params=dict(model.request_params),
                        temperature=model.temperature,
                        endpoint_id=model.endpoint_id,
                        reasoning_token_reserve=model.reasoning_token_reserve,
                    )
                )
    return cells


def assessment_keys(cells: Sequence[CellSpec]) -> Dict[str, CellSpec]:
    """
    Distinct model x pool assessment jobs implied by *cells*.

    Returns one representative cell per key with its prompt condition forced to
    the neutral reference, so a caller cannot accidentally collect assessments
    under a treatment prompt (B2).
    """
    jobs: Dict[str, CellSpec] = {}
    for cell in cells:
        if cell.assessment_key not in jobs:
            jobs[cell.assessment_key] = replace(
                cell,
                prompt_condition=schemas.ASSESSMENT_INSTRUCTION,
                cell_id=cell.assessment_key,
            )
    return jobs


# --- Study-level config ---

#: Menus per family, per pool (§6.2, §3.3).  The primary family carries the
#: pool's estimand; matched families are smaller strata sized for the paired
#: contrast rather than for the main contrasts.
DEFAULT_PROBLEMS_PER_FAMILY: Dict[str, Dict[str, int]] = {
    "insurance": {"claims": 100},
    "venture": {"startup": 100, "procurement": 40},
    "hiring": {"candidates": 100, "matched": 40},
}


@dataclass
class SEUSensitivityStudyConfig:
    """Top-level configuration for the collection pipeline."""

    pool_ids: List[str] = field(
        default_factory=lambda: list(pools_module.available_pools())
    )
    cells: List[CellSpec] = field(default_factory=list)

    # Design (§3.4, §6.2)
    problems_per_family: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            pool: dict(counts) for pool, counts in DEFAULT_PROBLEMS_PER_FAMILY.items()
        }
    )
    menu_sizes: List[int] = field(default_factory=lambda: list(MENU_SIZES))
    num_presentations: int = NUM_PRESENTATIONS
    presentation_mode: str = "reverse"
    K: int = 3

    # Embedding (§6.1 step 4)
    embedding_model: str = "text-embedding-3-small"
    target_dim: int = 32

    # Pre-choice gate (§5 R3, §6.3 R4).  Overrides for
    # ``configs/gate_thresholds.json``; empty means "use the packaged values",
    # which are PROVISIONAL until the §13 pre-registration freeze at build
    # phase E3.  Kept as a plain dict so the config stays JSON/YAML round-trippable
    # and every gate report can echo exactly what was applied.
    gate_thresholds: Dict[str, Any] = field(default_factory=dict)

    # Reproducibility (§6.5)
    seed: int = 42

    # API robustness
    max_retries: int = 5
    retry_delay: float = 2.0
    max_choice_tokens: int = 64
    max_assessment_tokens: int = 400
    cache_dir: Optional[str] = None

    # Storage
    results_dir: Optional[str] = None

    # Fitting
    stan_model: str = "h_m01"

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = build_cells(self.pool_ids)
        if self.results_dir is None:
            self.results_dir = str(Path(__file__).parent / "results")
        if self.cache_dir is None:
            self.cache_dir = str(Path(self.results_dir) / "_cache")
        if self.num_presentations != NUM_PRESENTATIONS:
            raise ValueError(
                f"num_presentations is frozen at {NUM_PRESENTATIONS} (§6.2); got "
                f"{self.num_presentations}"
            )

    # -- Views --

    def cells_for_pool(self, pool_id: str) -> List[CellSpec]:
        return [cell for cell in self.cells if cell.pool_id == pool_id]

    def assessment_jobs(self) -> Dict[str, CellSpec]:
        return assessment_keys(self.cells)

    def problems_for(self, pool_id: str) -> Dict[str, int]:
        return dict(self.problems_per_family.get(pool_id, {}))

    def expected_choice_calls(self) -> int:
        """Total choice API calls implied by the design (§12)."""
        total = 0
        for pool_id in self.pool_ids:
            menus = sum(self.problems_for(pool_id).values())
            total += menus * self.num_presentations * len(self.cells_for_pool(pool_id))
        return total

    def expected_assessment_calls(self, pool_item_counts: Dict[str, int]) -> int:
        """
        Total assessment API calls, given each pool's item count.

        Collected once per model x pool (B2), which is the ~3x saving that makes
        the §5 predictive-validity gate affordable before choice collection.
        """
        return sum(
            pool_item_counts.get(job.pool_id, 0) for job in self.assessment_jobs().values()
        )

    # -- Design matrix (B1, §4, §8.2) --

    def design_matrix_for_pool(
        self, pool_id: str
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build one pool's 18 x 7 model-coded design matrix.

        Returns ``(X, column_names, cell_ids)``.  Rows follow the order of
        :meth:`cells_for_pool`, which is also what the Stan ``cell`` index
        refers to, so a caller must never re-sort one without the other.

        The matrix carries **no intercept column** -- ``h_m01`` fits ``gamma0``
        separately -- and no interaction columns: with J = 18 cells the full
        model x prompt interaction is *exactly* saturated at 18 parameters, so
        it is a secondary shrinkage-dependent fit, not the primary design
        (§8.1).
        """
        cells = self.cells_for_pool(pool_id)
        if not cells:
            raise KeyError(f"No cells configured for pool {pool_id!r}")

        model_levels = [m.name for m in MODELS if m.name != REFERENCE_MODEL]
        prompt_levels = [p for p in PROMPT_CONDITIONS if p != REFERENCE_PROMPT]

        column_names = [f"model_{get_model_spec(m).slug}" for m in model_levels]
        column_names += [f"prompt_{p}" for p in prompt_levels]

        X = np.zeros((len(cells), len(column_names)), dtype=float)
        for row, cell in enumerate(cells):
            if cell.model_name in model_levels:
                X[row, model_levels.index(cell.model_name)] = 1.0
            if cell.prompt_condition in prompt_levels:
                offset = len(model_levels) + prompt_levels.index(cell.prompt_condition)
                X[row, offset] = 1.0

        return X, column_names, [cell.cell_id for cell in cells]

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_ids": list(self.pool_ids),
            "problems_per_family": {
                pool: dict(counts) for pool, counts in self.problems_per_family.items()
            },
            "menu_sizes": list(self.menu_sizes),
            "num_presentations": self.num_presentations,
            "presentation_mode": self.presentation_mode,
            "K": self.K,
            "embedding_model": self.embedding_model,
            "target_dim": self.target_dim,
            "gate_thresholds": dict(self.gate_thresholds),
            "seed": self.seed,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "max_choice_tokens": self.max_choice_tokens,
            "max_assessment_tokens": self.max_assessment_tokens,
            "stan_model": self.stan_model,
            "reference_model": REFERENCE_MODEL,
            "reference_prompt": REFERENCE_PROMPT,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "SEUSensitivityStudyConfig":
        """Load config from YAML. Reference levels are pinned, not configurable."""
        import yaml

        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        raw.pop("reference_model", None)
        raw.pop("reference_prompt", None)
        return cls(**raw)

    def save_yaml(self, path: str) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(
                self.to_dict(), handle, default_flow_style=False, sort_keys=False
            )


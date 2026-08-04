"""
Prompt loading and rendering (study plan §3.2, §6.1, §6.5).

The YAML files in ``configs/prompts_<pool>.yaml`` hold one neutral assessment
prompt per pool and three choice instructions that differ *only* in their
decision guidance.  This module resolves per-family overrides, renders the
templates, and hashes the resolved text for the run manifest.

Per-family overrides exist for one reason (§3.3): the venture pool hosts the
matched ``procurement`` family, which shares that pool's item file and PCA
space but must be *presented* under its own task label.  Without the override
the matched pair would vary item content and task label together, defeating the
paired contrast.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from . import pools as pools_module
from .schemas import ASSESSMENT_INSTRUCTION, PROMPT_CONDITIONS

logger = logging.getLogger(__name__)

__all__ = [
    "PromptSet",
    "probability_format",
    "consequence_lines",
    "load_prompt_sets",
    "prompt_hashes",
]


@dataclass(frozen=True)
class PromptSet:
    """Fully resolved prompts for one (pool, family)."""

    pool_id: str
    family: str
    assessment_system: str
    assessment_user: str
    choice_system: str
    choice_instructions: Dict[str, str]
    choice_user: str

    # -- Rendering --

    def render_assessment(self, item_text: str, consequences: Sequence[str]) -> str:
        return self.assessment_user.format(
            item_text=item_text.strip(),
            consequence_lines=consequence_lines(consequences),
            probability_format=probability_format(len(consequences)),
        )

    def render_choice(self, condition: str, assessments_in_order: Sequence[str]) -> str:
        """
        Render the choice prompt for one presentation.

        *assessments_in_order* must already be ordered to match the
        presentation order (§3.2, §6.1 step 3) -- this function numbers them
        1..n exactly as given and does not reorder, so the caller owns the
        correspondence between position and item.
        """
        if condition not in self.choice_instructions:
            raise KeyError(
                f"prompt condition {condition!r} not defined for "
                f"{self.pool_id}/{self.family}; have "
                f"{sorted(self.choice_instructions)}"
            )
        block = "\n\n".join(
            f"{index}. {text.strip()}"
            for index, text in enumerate(assessments_in_order, start=1)
        )
        return self.choice_user.format(
            instruction=self.choice_instructions[condition].strip(),
            assessments_list=block,
            n_max=len(assessments_in_order),
        )

    def fingerprint(self) -> Dict[str, str]:
        """Stable per-artefact hashes, for the run manifest (§6.5)."""
        parts = {
            "assessment_system": self.assessment_system,
            "assessment_user": self.assessment_user,
            "choice_system": self.choice_system,
            "choice_user": self.choice_user,
            **{f"instruction_{k}": v for k, v in self.choice_instructions.items()},
        }
        return {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            for name, text in parts.items()
        }


def probability_format(k: int) -> str:
    """The structured line the assessment prompt asks for (§5)."""
    slots = ", ".join(f"<p{index}>" for index in range(1, k + 1))
    return f"PROBABILITIES: {slots}"


def consequence_lines(consequences: Sequence[str]) -> str:
    return "\n".join(
        f"  {index}. {label}" for index, label in enumerate(consequences, start=1)
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_prompt_sets(
    pool_id: str, *, path: Optional[Path] = None
) -> Dict[str, PromptSet]:
    """
    Load and resolve every family's prompts for one pool.

    Returns ``{family: PromptSet}`` covering all families declared for the pool
    in :data:`pools.POOL_SPECS`, with per-family overrides merged over the
    pool-level defaults.
    """
    spec = pools_module.get_pool_spec(pool_id)
    prompt_path = Path(path) if path is not None else spec.prompts_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file for pool {pool_id!r} not found: {prompt_path}")

    with open(prompt_path) as handle:
        raw = yaml.safe_load(handle) or {}

    if raw.get("pool_id") != pool_id:
        raise ValueError(
            f"{prompt_path}: declares pool_id {raw.get('pool_id')!r}, expected "
            f"{pool_id!r}"
        )

    base = {"assessment": raw.get("assessment", {}), "choice": raw.get("choice", {})}
    overrides = raw.get("families", {}) or {}

    unknown = sorted(set(overrides) - set(spec.families))
    if unknown:
        raise ValueError(
            f"{prompt_path}: family override(s) {unknown} not declared for pool "
            f"{pool_id!r} (families: {list(spec.families)})"
        )

    resolved: Dict[str, PromptSet] = {}
    for family in spec.families:
        merged = _deep_merge(base, overrides.get(family, {}))
        resolved[family] = _build_prompt_set(pool_id, family, merged, prompt_path)
    return resolved


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_prompt_set(
    pool_id: str, family: str, merged: Mapping[str, Any], source: Path
) -> PromptSet:
    context = f"{source} [{pool_id}/{family}]"
    assessment = merged.get("assessment") or {}
    choice = merged.get("choice") or {}

    for section, keys in (("assessment", ("system_prompt", "user_prompt")),
                          ("choice", ("system_prompt", "user_prompt", "instructions"))):
        block = merged.get(section) or {}
        missing = [key for key in keys if not block.get(key)]
        if missing:
            raise ValueError(f"{context}: missing {section}.{missing}")

    instructions = dict(choice["instructions"])
    missing_conditions = [c for c in PROMPT_CONDITIONS if c not in instructions]
    if missing_conditions:
        raise ValueError(
            f"{context}: choice.instructions missing condition(s) {missing_conditions}"
        )
    extra = sorted(set(instructions) - set(PROMPT_CONDITIONS))
    if extra:
        raise ValueError(f"{context}: unknown prompt condition(s) {extra}")

    prompt_set = PromptSet(
        pool_id=pool_id,
        family=family,
        assessment_system=assessment["system_prompt"],
        assessment_user=assessment["user_prompt"],
        choice_system=choice["system_prompt"],
        choice_instructions=instructions,
        choice_user=choice["user_prompt"],
    )
    _check_prompt_scope(prompt_set, context)
    return prompt_set


#: Vocabulary the `deliberative` arm must not contain (§3.2, resolved).  Its
#: whole point is to invoke reasoning effort *without* smuggling in SEU
#: content; the earlier wording ("think about the consequences and their
#: likelihoods") partially restated the seu_maximizing arm and contaminated the
#: RQ2 contrast.
_SEU_VOCABULARY = (
    "likelihood",
    "likelihoods",
    "likely",
    "probability",
    "probabilities",
    "probable",
    "consequence",
    "consequences",
    "outcome",
    "outcomes",
    "expected",
    "expectation",
    "utility",
    "value",
    "payoff",
    "maximiz",
    "maximis",
)


def _check_prompt_scope(prompt_set: PromptSet, context: str) -> None:
    """
    Guard the two prompt-scope invariants that the RQ2 contrast rests on.

    Enforcing them here means a future prompt edit fails loudly at load time
    rather than silently changing what RQ2 measures.
    """
    deliberative = prompt_set.choice_instructions.get("deliberative", "").lower()
    offenders = sorted({word for word in _SEU_VOCABULARY if word in deliberative})
    if offenders:
        raise ValueError(
            f"{context}: the 'deliberative' instruction contains SEU vocabulary "
            f"{offenders}. That arm must be a pure effort invocation; any "
            f"likelihood/outcome/value wording partially restates 'seu_maximizing' "
            f"and contaminates the RQ2 contrast (§3.2)."
        )

    if ASSESSMENT_INSTRUCTION not in prompt_set.choice_instructions:
        raise ValueError(
            f"{context}: missing the {ASSESSMENT_INSTRUCTION!r} reference condition"
        )


def prompt_hashes(prompt_sets: Mapping[str, Mapping[str, PromptSet]]) -> Dict[str, str]:
    """
    Flatten ``{pool: {family: PromptSet}}`` into manifest-ready hashes (§6.5).

    Keys look like ``"venture/procurement/choice_user"``.
    """
    flat: Dict[str, str] = {}
    for pool_id, families in prompt_sets.items():
        for family, prompt_set in families.items():
            for name, digest in prompt_set.fingerprint().items():
                flat[f"{pool_id}/{family}/{name}"] = digest
    return flat


def dump_resolved_prompts(prompt_sets: Mapping[str, PromptSet]) -> str:
    """Serialize resolved prompts for archival alongside the raw responses."""
    payload = {
        family: {
            "assessment_system": ps.assessment_system,
            "assessment_user": ps.assessment_user,
            "choice_system": ps.choice_system,
            "choice_instructions": ps.choice_instructions,
            "choice_user": ps.choice_user,
            "fingerprint": ps.fingerprint(),
        }
        for family, ps in prompt_sets.items()
    }
    return json.dumps(payload, indent=2, sort_keys=True)

"""
Item-pool registry and loaders for the SEU sensitivity study (study plan §3.3, §6.3).

A *pool* is a decision domain: a set of richly described items, an ordered list
of K consequences, and a framing valence.  This module is the only place that
knows where pool files live and how to bring them into the canonical schema of
:mod:`applications.seu_sensitivity_study.schemas`.

Three pools are registered (§3.3):

``insurance``
    The replication anchor.  Item text is reused **unchanged** from the
    methodology-paper pipeline; the loader adapts the legacy
    ``{"claims": [...]}`` layout rather than rewriting the file.
``venture``
    The RQ5 comparator backbone.  Carries **two families**: ``startup`` (which
    alone supplies the ordinary venture menus) and ``procurement`` (the matched
    family of §3.3, which forms its own menu stratum and never contaminates the
    comparator).
``hiring``
    The override probe.  Its ``matched`` family is rendered from the same merit
    vectors as the venture pool's ``procurement`` family, linked by
    ``matched_key``, and likewise forms its own menu stratum so the ordinary
    ``candidates`` menus stay uncontaminated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import schemas

logger = logging.getLogger(__name__)

__all__ = [
    "PoolSpec",
    "POOL_SPECS",
    "available_pools",
    "get_pool_spec",
    "load_pool",
    "load_all_pools",
    "items_by_family",
    "matched_index",
    "matched_pairs",
]


_PACKAGE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PACKAGE_DIR / "data"
_CONFIG_DIR = _PACKAGE_DIR / "configs"
_LEGACY_CLAIMS = _PACKAGE_DIR.parent / "temperature_study" / "data" / "claims.json"


@dataclass(frozen=True)
class PoolSpec:
    """Static description of one decision domain."""

    pool_id: str
    framing: str
    families: Tuple[str, ...]
    item_file: Path
    prompts_file: Path
    #: Name of the adapter needed to read a non-canonical source file, or None.
    legacy_adapter: Optional[str] = None
    #: Sidecar supplying authored quality labels when the source file lacks
    #: them (only the legacy insurance file does).
    label_sidecar: Optional[Path] = None
    #: Family whose menus constitute the pool's primary estimand.  Other
    #: families are separate strata (§3.3).
    primary_family: str = ""

    def __post_init__(self) -> None:
        if not self.primary_family:
            object.__setattr__(self, "primary_family", self.families[0])


POOL_SPECS: Dict[str, PoolSpec] = {
    "insurance": PoolSpec(
        pool_id="insurance",
        framing="negative",
        families=("claims",),
        item_file=_LEGACY_CLAIMS,
        prompts_file=_CONFIG_DIR / "prompts_insurance.yaml",
        legacy_adapter="insurance_claims",
        label_sidecar=_DATA_DIR / "insurance_quality_labels.json",
    ),
    "venture": PoolSpec(
        pool_id="venture",
        framing="positive",
        families=("startup", "procurement"),
        item_file=_DATA_DIR / "venture.json",
        prompts_file=_CONFIG_DIR / "prompts_venture.yaml",
        primary_family="startup",
    ),
    "hiring": PoolSpec(
        pool_id="hiring",
        framing="positive",
        families=("candidates", "matched"),
        item_file=_DATA_DIR / "hiring.json",
        prompts_file=_CONFIG_DIR / "prompts_hiring.yaml",
        primary_family="candidates",
    ),
}


def available_pools() -> List[str]:
    """Registered pool ids, in the plan's presentation order."""
    return list(POOL_SPECS)


def get_pool_spec(pool_id: str) -> PoolSpec:
    try:
        return POOL_SPECS[pool_id]
    except KeyError:
        raise KeyError(
            f"Unknown pool {pool_id!r}; registered pools are {available_pools()}"
        ) from None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_pool(pool_id: str, *, validate: bool = True) -> Dict[str, Any]:
    """
    Load one pool in the canonical :mod:`schemas` form.

    Parameters
    ----------
    pool_id:
        A key of :data:`POOL_SPECS`.
    validate:
        Run :func:`schemas.validate_item_pool` and raise on failure.  Only
        disable this when deliberately inspecting a work-in-progress pool.
    """
    spec = get_pool_spec(pool_id)
    if not spec.item_file.exists():
        raise FileNotFoundError(
            f"Pool {pool_id!r} expects its item file at {spec.item_file}, which does "
            f"not exist yet (authored in build phase B)."
        )

    with open(spec.item_file) as handle:
        raw = json.load(handle)

    if spec.legacy_adapter == "insurance_claims":
        pool = _adapt_insurance_claims(raw, spec)
    elif spec.legacy_adapter is None:
        pool = raw
    else:  # pragma: no cover - guard against a mis-registered spec
        raise ValueError(f"Unknown legacy adapter {spec.legacy_adapter!r}")

    if validate:
        schemas.check(
            schemas.validate_item_pool(pool), context=f"item pool {pool_id!r}"
        )

    logger.info(
        "Loaded pool %r: %d items across families %s",
        pool_id,
        len(pool.get("items", [])),
        sorted({item["family"] for item in pool.get("items", [])}),
    )
    return pool


def load_all_pools(*, validate: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load every registered pool, keyed by pool id."""
    return {pid: load_pool(pid, validate=validate) for pid in available_pools()}


def _adapt_insurance_claims(raw: Dict[str, Any], spec: PoolSpec) -> Dict[str, Any]:
    """
    Bring the legacy ``claims.json`` into canonical form **without touching the
    item text** (§6.3: the anchor's text is reused unchanged).

    The legacy file predates the authored strong/ambiguous/weak labels that the
    R4 PC1 validation and the §6.3 difficulty strata require, so the labels are
    supplied by a sidecar mapping ``claim id -> quality label``.  Requiring the
    sidecar rather than inventing labels keeps the omission visible: without
    authored labels there is nothing for PC1 to be validated *against*.
    """
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"{spec.item_file}: expected a non-empty 'claims' list")

    labels = _load_quality_label_sidecar(spec, [claim["id"] for claim in claims])
    family = spec.families[0]

    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": spec.pool_id,
        "framing": spec.framing,
        "consequences": list(raw["consequences"]),
        "families": {
            family: {
                "description": "Insurance claims triage items (replication anchor).",
                "source": str(spec.item_file),
            }
        },
        "items": [
            {
                "id": claim["id"],
                "family": family,
                "text": claim["description"],
                "quality_label": labels[claim["id"]],
                "attributes": {},
                "matched_key": None,
            }
            for claim in claims
        ],
        "provenance": {
            "adapted_from": str(spec.item_file),
            "adapter": spec.legacy_adapter,
            "quality_labels_from": str(spec.label_sidecar),
            "note": (
                "Item text is byte-identical to the methodology-paper pool; only "
                "quality labels and canonical envelope fields were added."
            ),
        },
    }


def _load_quality_label_sidecar(spec: PoolSpec, item_ids: List[str]) -> Dict[str, str]:
    if spec.label_sidecar is None or not spec.label_sidecar.exists():
        raise FileNotFoundError(
            f"Pool {spec.pool_id!r} needs authored quality labels at "
            f"{spec.label_sidecar}. The legacy source file carries none, and the R4 "
            f"PC1 validation (§6.3) has nothing to validate against without them. "
            f"Expected {{\"labels\": {{\"C001\": \"strong\", ...}}}} covering "
            f"{len(item_ids)} item(s)."
        )

    with open(spec.label_sidecar) as handle:
        payload = json.load(handle)
    labels = payload.get("labels", payload)

    missing = [item_id for item_id in item_ids if item_id not in labels]
    if missing:
        raise ValueError(
            f"{spec.label_sidecar}: missing quality label(s) for {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    invalid = {
        item_id: labels[item_id]
        for item_id in item_ids
        if labels[item_id] not in schemas.QUALITY_LABELS
    }
    if invalid:
        raise ValueError(
            f"{spec.label_sidecar}: labels must be one of "
            f"{list(schemas.QUALITY_LABELS)}; got {invalid}"
        )
    return {item_id: labels[item_id] for item_id in item_ids}


# ---------------------------------------------------------------------------
# Views over a loaded pool
# ---------------------------------------------------------------------------


def items_by_family(pool: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a pool's items by family, preserving file order within a family."""
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in pool["families"]}
    for item in pool["items"]:
        grouped.setdefault(item["family"], []).append(item)
    return grouped


def matched_index(pool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map ``matched_key -> item`` for items that carry one (§3.3)."""
    return {
        item["matched_key"]: item
        for item in pool["items"]
        if item.get("matched_key") is not None
    }


def matched_pairs(
    hiring_pool: Dict[str, Any], venture_pool: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Pair hiring-label items with their procurement-label twins (§3.3).

    Returns ``[(hiring_item, procurement_item), ...]`` for every merit vector
    present under both labels.  Keys present on only one side are logged and
    skipped: an unpaired merit vector cannot enter the paired contrast, and
    silently dropping it without a warning would understate the matched set.
    """
    hiring_by_key = matched_index(hiring_pool)
    venture_by_key = matched_index(venture_pool)

    shared = sorted(set(hiring_by_key) & set(venture_by_key))
    unpaired = sorted(set(hiring_by_key) ^ set(venture_by_key))
    if unpaired:
        logger.warning(
            "matched_pairs: %d merit vector(s) present under only one label and "
            "excluded from the paired contrast: %s",
            len(unpaired),
            unpaired[:10],
        )

    return [(hiring_by_key[key], venture_by_key[key]) for key in shared]

"""
Run provenance (study plan §6.5, §3.1).

Results are a perishable model snapshot, so the manifest pins everything a
re-analysis would need and everything a reader would have to take on trust:
endpoint ids and access dates, the **full** per-model request parameters, the
answer-format version, the design seed, embedding and PCA settings, prompt
hashes, and the toolchain versions.

The request parameters are not bookkeeping.  For the reasoning tier
(``reasoning_effort``, ``budget_tokens``) they *are* part of the treatment
(§3.1), so a manifest that omitted them would describe a different experiment
than the one that ran.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import schemas
from .config import (
    REFERENCE_MODEL,
    REFERENCE_PROMPT,
    SEUSensitivityStudyConfig,
    get_model_spec,
)

logger = logging.getLogger(__name__)

__all__ = ["build_run_manifest", "record_substitution", "toolchain_versions"]


def build_run_manifest(
    config: SEUSensitivityStudyConfig,
    *,
    run_id: Optional[str] = None,
    prompt_hashes: Optional[Mapping[str, str]] = None,
    pca_info: Optional[Mapping[str, Any]] = None,
    endpoint_ids: Optional[Mapping[str, str]] = None,
    substitutions: Optional[Mapping[str, Mapping[str, str]]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Assemble the run manifest.

    Parameters
    ----------
    endpoint_ids:
        ``{model_name: endpoint_id}``.  Where a provider exposes a dated
        endpoint the dated form should be supplied; falling back to the bare
        model name is recorded as such, because an undated endpoint cannot
        distinguish a silent provider-side update from a stable one.
    substitutions:
        ``{model_name: {"substituted_for": ..., "reason": ...}}`` recording any
        use of the pre-declared deprecation fallback rule (§3.1).
    """
    started_at = datetime.now(timezone.utc)
    accessed_on = date.today().isoformat()
    endpoint_ids = dict(endpoint_ids or {})
    substitutions = dict(substitutions or {})

    models: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for cell in config.cells:
        if cell.model_name in seen:
            continue
        seen.add(cell.model_name)

        # Prefer an explicitly supplied endpoint id, then the one pinned on the
        # ModelSpec.  The spec's is authoritative for deprecation substitutions,
        # which must be a recorded fact rather than a code comment (§3.1, §6.5).
        try:
            spec = get_model_spec(cell.model_name)
        except KeyError:
            spec = None
        endpoint = endpoint_ids.get(cell.model_name) or (
            spec.endpoint_id if spec else None
        )
        entry: Dict[str, Any] = {
            "model_name": cell.model_name,
            "endpoint_id": endpoint or cell.model_name,
            "endpoint_is_dated": endpoint is not None,
            "provider": cell.provider,
            "accessed_at": accessed_on,
            "request_params": _request_params(cell),
        }
        if spec is not None and spec.tier:
            entry["tier"] = spec.tier
        substitution = substitutions.get(cell.model_name)
        if substitution:
            entry["substituted_for"] = substitution.get("substituted_for")
            entry["substitution_reason"] = substitution.get("reason")
        elif spec is not None and spec.substituted_for:
            entry["substituted_for"] = spec.substituted_for
            entry["substitution_reason"] = spec.substitution_reason
        models.append(entry)

    manifest: Dict[str, Any] = {
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": run_id or f"seu-{started_at.strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": started_at.isoformat(),
        "study_plan_version": "0.5",
        "answer_format_version": schemas.ANSWER_TOKEN_VERSION,
        "menu_sizes": list(config.menu_sizes),
        "num_presentations": config.num_presentations,
        "presentation_mode": config.presentation_mode,
        "design_seed": config.seed,
        "embedding_model": config.embedding_model,
        "pca_target_dim": config.target_dim,
        "pca_info": dict(pca_info or {}),
        "prompt_hashes": dict(prompt_hashes or {}),
        "models": models,
        "pool_ids": list(config.pool_ids),
        "problems_per_family": {
            pool: dict(counts) for pool, counts in config.problems_per_family.items()
        },
        "reference_model": REFERENCE_MODEL,
        "reference_prompt": REFERENCE_PROMPT,
        "config": config.to_dict(),
        "toolchain": toolchain_versions(),
    }

    if validate:
        schemas.check(schemas.validate_run_manifest(manifest), context="run manifest")

    undated = [m["model_name"] for m in models if not m["endpoint_is_dated"]]
    if undated:
        logger.warning(
            "No dated endpoint id pinned for %s; a silent provider-side update "
            "would be indistinguishable from a stable endpoint (§6.5)",
            undated,
        )
    return manifest


def record_substitution(
    manifest: Dict[str, Any], model_name: str, *, substituted_for: str, reason: str
) -> Dict[str, Any]:
    """
    Record a deprecation substitution after the fact (§3.1).

    The pre-declared rule is to substitute the nearest available same-tier,
    same-vendor successor; *reason* should say which model was unavailable and
    what replaced it.
    """
    for entry in manifest.get("models", []):
        if entry["model_name"] == model_name:
            entry["substituted_for"] = substituted_for
            entry["substitution_reason"] = reason
            schemas.check(
                schemas.validate_run_manifest(manifest), context="run manifest"
            )
            return manifest
    raise KeyError(f"Model {model_name!r} is not in the manifest")


def _request_params(cell: Any) -> Dict[str, Any]:
    params = dict(cell.request_params or {})
    # Recorded explicitly, including the None that means "this provider accepts
    # no temperature parameter" -- an absent key would read as an oversight.
    params["temperature"] = cell.temperature
    return params


def toolchain_versions() -> Dict[str, Any]:
    """Versions that can move draws or responses between runs (§6.5)."""
    versions: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "sklearn", "cmdstanpy", "openai", "anthropic"):
        versions[package] = _package_version(package)
    versions["cmdstan"] = _cmdstan_version()
    return versions


def _package_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001 - optional at collection time
        return None
    return getattr(module, "__version__", None)


def _cmdstan_version() -> Optional[str]:
    try:
        import cmdstanpy

        return str(cmdstanpy.cmdstan_version())
    except Exception:  # noqa: BLE001 - CmdStan is not needed to collect data
        return None

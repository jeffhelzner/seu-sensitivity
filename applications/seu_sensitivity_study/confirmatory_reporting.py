"""Artifact loading and report assembly for completed confirmatory fits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from analysis.hierarchical_power import fit_diagnostics

from . import confirmatory_analysis
from .config import SEUSensitivityStudyConfig, build_cells

REQUIRED_VARIANTS = (
    "primary",
    "presentation_1_only",
    "presentation_2_only",
    "utility_035",
    "utility_065",
)


def fit_payload(
    fit: Any,
    cell_ids: list[str],
    gamma_columns: int,
    *,
    max_treedepth: int = 12,
) -> Dict[str, Any]:
    """Extract the anchored parameters and frozen diagnostics from one fit."""
    gamma = np.asarray(fit.stan_variable("gamma"), dtype=float)
    gamma_size = np.asarray(fit.stan_variable("gamma_size"), dtype=float).reshape(-1)
    sigma_cell = np.asarray(fit.stan_variable("sigma_cell"), dtype=float).reshape(-1)
    z_alpha = np.asarray(fit.stan_variable("z_alpha"), dtype=float)
    if gamma.ndim != 2 or gamma.shape[1] != gamma_columns:
        raise ValueError(
            f"gamma draws have shape {gamma.shape}; expected (*, {gamma_columns})"
        )
    if z_alpha.ndim != 2 or z_alpha.shape[1] != len(cell_ids):
        raise ValueError(
            f"z_alpha draws have shape {z_alpha.shape}; expected (*, {len(cell_ids)})"
        )
    draw_counts = {len(gamma), len(gamma_size), len(sigma_cell), len(z_alpha)}
    if len(draw_counts) != 1:
        raise ValueError("Posterior parameter arrays have inconsistent draw counts")
    return {
        "gamma_draws": gamma,
        "gamma_size_draws": gamma_size,
        "sigma_cell_draws": sigma_cell,
        "z_alpha_draws": z_alpha,
        "cell_ids": list(cell_ids),
        "diagnostics": fit_diagnostics(
            fit, seconds=0.0, max_treedepth=max_treedepth
        ),
    }


def build_report_from_manifest(
    manifest: Mapping[str, Any], *, fit_loader: Any = None
) -> Dict[str, Any]:
    """Load saved CmdStan chains and build the frozen complete report."""
    if fit_loader is None:
        from cmdstanpy import from_csv

        fit_loader = from_csv
    fit_paths = manifest.get("fits", {})
    required_groups = {"venture", "hiring", "matched_rq5"}
    if set(fit_paths) != required_groups:
        raise ValueError(f"Fit manifest must contain exactly {sorted(required_groups)}")
    max_treedepth = int(manifest.get("max_treedepth", 12))
    config = SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])

    pool_variants = {}
    pool_contrasts = {}
    artifact_hashes = {}
    for pool_id in ("venture", "hiring"):
        _, columns, cell_ids = config.design_matrix_for_pool(pool_id)
        pool_contrasts[pool_id] = confirmatory_analysis.primary_contrasts(columns)
        pool_variants[pool_id] = _load_variants(
            fit_paths[pool_id],
            cell_ids,
            gamma_columns=len(columns),
            max_treedepth=max_treedepth,
            fit_loader=fit_loader,
            artifact_hashes=artifact_hashes,
        )

    matched_cells = build_cells(["venture", "hiring"])
    matched_contract = confirmatory_analysis.matched_rq5_contract(matched_cells)
    matched_contrasts = tuple(
        confirmatory_analysis.ContrastSpec(**payload)
        for payload in matched_contract["contrasts"]
    )
    matched_variants = _load_variants(
        fit_paths["matched_rq5"],
        [cell.cell_id for cell in matched_cells],
        gamma_columns=len(matched_contract["design_columns"]),
        max_treedepth=max_treedepth,
        fit_loader=fit_loader,
        artifact_hashes=artifact_hashes,
    )
    report = confirmatory_analysis.complete_confirmatory_report(
        pool_variants=pool_variants,
        pool_contrasts=pool_contrasts,
        matched_variants=matched_variants,
        matched_contrasts=matched_contrasts,
    )
    report["fit_artifact_hashes"] = artifact_hashes
    return report


def _load_variants(
    paths: Mapping[str, Any],
    cell_ids: list[str],
    gamma_columns: int,
    *,
    max_treedepth: int,
    fit_loader: Any,
    artifact_hashes: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    if set(paths) != set(REQUIRED_VARIANTS):
        raise ValueError(f"Fit group must contain exactly {list(REQUIRED_VARIANTS)}")
    variants = {}
    for variant in REQUIRED_VARIANTS:
        path = Path(paths[variant]).resolve()
        files = _fit_files(path)
        artifact_hashes[str(path)] = {
            str(file): _sha256_file(file) for file in files
        }
        fit = fit_loader([str(file) for file in files])
        if fit is None:
            raise ValueError(f"No CmdStan fit could be loaded from {path}")
        variants[variant] = fit_payload(
            fit,
            cell_ids,
            gamma_columns,
            max_treedepth=max_treedepth,
        )
    return variants


def _fit_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Fit artifact path does not exist: {path}")
    files = sorted(
        file
        for pattern in ("*.csv", "*.csv.gz")
        for file in path.glob(pattern)
        if file.name not in {"posterior_summary.csv"}
    )
    if not files:
        raise FileNotFoundError(f"Fit directory contains no CmdStan CSV files: {path}")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a machine-readable confirmatory report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
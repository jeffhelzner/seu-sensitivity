"""Approved A3 prior specifications and input identity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

POLICY_VERSION = "amendment7_ceiling_prior_v1"
PRIMARY_MODEL = "h_m01_size_assessment_anchored"
SENSITIVITY_MODEL = PRIMARY_MODEL + "_prior"
PRIOR_VARIANTS = ("prior_L", "prior_H", "prior_S")
_PRIMARY = dict(prior_gamma0_mean=2.5, prior_gamma0_sd=0.5,
                prior_gamma_sd=0.5, prior_sigma_cell_sd=0.3,
                prior_gamma_size_sd=0.2)


def prior_fields(variant: str) -> dict[str, float]:
    fields = dict(_PRIMARY)
    if variant == "prior_L":
        fields["prior_gamma0_sd"] = 1.0
    elif variant == "prior_H":
        fields.update(prior_gamma_sd=1.0, prior_sigma_cell_sd=0.6)
    elif variant == "prior_S":
        fields["prior_gamma_size_sd"] = 0.4
    elif variant != "primary":
        raise ValueError(f"Unknown prior variant: {variant}")
    return fields


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def prior_contract(variant: str) -> dict[str, Any]:
    return {"policy_version": POLICY_VERSION, "variant": variant,
            "model": PRIMARY_MODEL if variant == "primary" else SENSITIVITY_MODEL,
            "fields": prior_fields(variant), "z_alpha": {"mean": 0.0, "sd": 1.0}}


def sensitivity_data(primary: Mapping[str, Any], variant: str) -> dict[str, Any]:
    if variant not in PRIOR_VARIANTS or any(key.startswith("prior_") for key in primary):
        raise ValueError("Prior sensitivity requires unmodified primary input and L/H/S")
    return {**primary, **prior_fields(variant)}


def validate_prior_data(data: Mapping[str, Any], primary: Mapping[str, Any], variant: str) -> None:
    if digest(data) != digest(sensitivity_data(primary, variant)):
        raise ValueError("Prior input differs from primary data or declared prior fields")


def write_prior_inputs(directory: Path, primary, preparation):
    from .confirmatory_reporting import write_report

    if primary.get("utility_values") != [0.0, 0.5, 1.0] or preparation.get("presentation_id") is not None:
        raise ValueError("A3 prior fits require full-data midpoint-0.5 primary input")
    paths = []
    for variant in PRIOR_VARIANTS:
        path = directory / f"stan_data_size_{variant}.json"
        write_report(path, sensitivity_data(primary, variant))
        write_report(directory / f"{path.stem}_assembly_report.json", preparation)
        write_report(directory / f"{variant}_contract.json", prior_contract(variant))
        paths.append(path.name)
    return paths


def fit_plan(results_dir: Path):
    suffixes = {"primary": "", "presentation_1_only": "_presentation_1",
                "presentation_2_only": "_presentation_2", "utility_035": "_u035", "utility_065": "_u065",
                **{variant: f"_{variant}" for variant in PRIOR_VARIANTS}}
    rows = []
    for group in ("venture", "hiring", "matched_rq5"):
        directory = results_dir / ("matched_rq5" if group == "matched_rq5" else f"pools/{group}")
        for variant, suffix in suffixes.items():
            rows.append({"group": group, "variant": variant,
                         "model": SENSITIVITY_MODEL if variant in PRIOR_VARIANTS else PRIMARY_MODEL,
                         "utility_middle": {"utility_035": .35, "utility_065": .65}.get(variant, .5),
                         "presentation_id": {"presentation_1_only": 1, "presentation_2_only": 2}.get(variant),
                         "prior_variant": variant if variant in PRIOR_VARIANTS else "primary",
                         "chains": 4, "max_treedepth": 12,
                         "stan_data": str(directory / f"stan_data_size{suffix}.json"),
                         "preparation_report": str(directory / f"stan_data_size{suffix}_assembly_report.json"),
                         "analysis_contract": str(directory / "analysis_contract.json"),
                         "prior_contract": str(directory / f"{variant}_contract.json") if variant in PRIOR_VARIANTS else None,
                         "status": "planned_not_authorized"})
    return {"schema_version": 2, "policy_version": POLICY_VERSION, "planned_fit_count": 24,
            "additional_prior_fits": 9, "fits": rows,
            "resource_policy": "Separate authorization required; benchmark one authorized fit before scheduling remainder. Fit count is not a runtime estimate."}


def fit_manifest(results_dir: Path):
    from .confirmatory_reporting import _fit_files, _sha256_file

    fits = {group: {} for group in ("venture", "hiring", "matched_rq5")}
    for row in fit_plan(results_dir.resolve())["fits"]:
        group, variant = row["group"], row["variant"]
        chains = Path(row["stan_data"]).parent / variant / "chains"
        try:
            files = _fit_files(chains)
        except FileNotFoundError as error:
            if variant not in PRIOR_VARIANTS:
                raise
            fits[group][variant] = {"status": "missing", "reason": str(error)}
            continue
        entry = {"chain_path": str(chains), "chain_sha256": {path.name: _sha256_file(path) for path in files}}
        names = ["stan_data", "preparation_report", "analysis_contract"]
        if variant in PRIOR_VARIANTS:
            names.append("prior_contract")
            row["model_source"] = str(Path(__file__).resolve().parents[2] / "models" / f"{SENSITIVITY_MODEL}.stan")
            names.append("model_source")
        for name in names:
            path = Path(row[name])
            entry[name] = {"path": str(path), "sha256": _sha256_file(path)}
        fits[group][variant] = entry
    return {"schema_version": 2, "max_treedepth": 12, "fits": fits}
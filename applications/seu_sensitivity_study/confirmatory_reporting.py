"""Artifact loading and report assembly for completed confirmatory fits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from analysis.hierarchical_power import fit_diagnostics

from . import confirmatory_analysis
from . import ceiling_prior, ceiling_diagnostics
from . import predictive_checks as a4_checks
from .config import SEUSensitivityStudyConfig, build_cells

REQUIRED_VARIANTS = (
    "primary",
    "presentation_1_only",
    "presentation_2_only",
    "utility_035",
    "utility_065",
)
MODEL_NAME = "h_m01_size_assessment_anchored"


class SamplerFailure(ValueError):
    """Computationally invalid diagnostics, distinct from malformed bindings."""


def _draws(fit: Any, name: str) -> np.ndarray:
    try:
        values = np.asarray(fit.stan_variable(name), dtype=float)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Fit must contain {name} draws") from error
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise SamplerFailure(f"{name} draws must be nonempty and finite")
    return values


def fit_payload(
    fit: Any,
    cell_ids: list[str],
    gamma_columns: int,
    *,
    max_treedepth: int = 12,
    prior_variant: str = "primary",
) -> Dict[str, Any]:
    """Extract the anchored parameters and frozen diagnostics from one fit."""
    metadata = getattr(getattr(fit, "metadata", None), "cmdstan_config", {})
    if max_treedepth != 12 or metadata.get("max_depth") != 12:
        raise ValueError("Fit metadata must establish the frozen TD12 configuration")
    model_name = ceiling_prior.prior_contract(prior_variant)["model"]
    if metadata.get("model") not in {model_name, f"{model_name}_model"}:
        raise ValueError(f"Fit metadata must establish model {model_name}")
    if prior_variant != "primary":
        try:
            settings = np.asarray(fit.stan_variable("prior_settings"), dtype=float)
        except (KeyError, ValueError) as error:
            raise ValueError("Fit must contain prior_settings draws") from error
        expected = np.asarray(list(ceiling_prior.prior_fields(prior_variant).values()))
        if (settings.ndim != 2 or not len(settings) or settings.shape[1] != 5
                or not np.array_equal(settings, np.broadcast_to(expected, settings.shape))):
            raise ValueError("Saved prior_settings disagree with the declared prior contract")
    gamma = _draws(fit, "gamma")
    if prior_variant != "primary" and len(settings) != len(gamma):
        raise ValueError("Prior settings draw count differs from posterior")
    gamma0 = _draws(fit, "gamma0")
    gamma_size = _draws(fit, "gamma_size")
    sigma_cell = _draws(fit, "sigma_cell")
    z_alpha = _draws(fit, "z_alpha")
    if gamma.ndim != 2 or gamma.shape[1] != gamma_columns:
        raise ValueError(
            f"gamma draws have shape {gamma.shape}; expected (*, {gamma_columns})"
        )
    if z_alpha.ndim != 2 or z_alpha.shape[1] != len(cell_ids):
        raise ValueError(
            f"z_alpha draws have shape {z_alpha.shape}; expected (*, {len(cell_ids)})"
        )
    if any(values.shape != (len(gamma),) for values in (gamma0, gamma_size, sigma_cell)):
        raise ValueError("Posterior parameter arrays have inconsistent draw counts")
    if len(z_alpha) != len(gamma):
        raise ValueError("Posterior parameter arrays have inconsistent draw counts")
    if np.any(sigma_cell < 0):
        raise ValueError("sigma_cell draws must be nonnegative")
    methods = fit.method_variables()
    if getattr(fit, "chains", None) != 4 or any(
        name not in methods
        or np.asarray(methods[name]).ndim != 2
        or np.asarray(methods[name]).shape[1] != 4
        for name in ("treedepth__", "divergent__", "energy__")
    ):
        raise ValueError("Fit must contain four actual chains")
    for name in ("treedepth__", "divergent__", "energy__"):
        values = np.asarray(methods[name], dtype=float)
        if values.size != len(gamma) or not np.all(np.isfinite(values)):
            raise SamplerFailure(f"{name} must contain finite diagnostics for every draw")
    required_parameters = ["gamma0", "gamma_size", "sigma_cell"]
    required_parameters += [f"gamma[{index}]" for index in range(1, gamma_columns + 1)]
    required_parameters += [f"z_alpha[{index}]" for index in range(1, len(cell_ids) + 1)]
    summary = fit.summary()
    if not summary.index.is_unique or not summary.columns.is_unique:
        raise ValueError("Fit summary must have unique parameter and diagnostic labels")
    if not set(required_parameters).issubset(summary.index) or not {
        "ESS_bulk", "ESS_tail", "R_hat"
    }.issubset(summary.columns):
        raise ValueError("Fit summary missing expected structural parameter diagnostics")
    structural = summary.loc[required_parameters, ["ESS_bulk", "ESS_tail", "R_hat"]]
    if not np.all(np.isfinite(structural.to_numpy(dtype=float))):
        raise SamplerFailure("Structural parameter diagnostics must be finite")
    diagnostics = fit_diagnostics(fit, seconds=0.0, max_treedepth=max_treedepth)
    gate_fields = ("max_rhat", "min_ess_bulk", "min_ess_tail", "min_ebfmi", "divergences", "treedepth_saturated_share")
    if any(diagnostics.get(name) is None or not np.isfinite(float(diagnostics[name])) for name in gate_fields):
        raise SamplerFailure("Fit diagnostics must be finite")
    ebfmi = np.asarray(diagnostics.get("ebfmi_by_chain", []), dtype=float)
    if ebfmi.shape != (4,) or not np.all(np.isfinite(ebfmi)):
        raise SamplerFailure("Every chain must have finite E-BFMI diagnostics")
    diagnostics.update(
        max_rhat=max(float(diagnostics["max_rhat"]), float(structural["R_hat"].max())),
        min_ess_bulk=min(float(diagnostics["min_ess_bulk"]), float(structural["ESS_bulk"].min())),
        min_ess_tail=min(float(diagnostics["min_ess_tail"]), float(structural["ESS_tail"].min())),
    )
    return {
        "gamma_draws": gamma,
        "gamma_size_draws": gamma_size,
        "sigma_cell_draws": sigma_cell,
        "z_alpha_draws": z_alpha,
        "cell_ids": list(cell_ids),
        "diagnostics": diagnostics,
    }


def build_report_from_manifest(
    manifest: Mapping[str, Any], *, fit_loader: Any = None
) -> Dict[str, Any]:
    """Load historical v1 or A3 v2 bindings and report saved fits.

    Each fits[group][variant] requires chain_path, chain_sha256 (basename to
    digest), and stan_data/preparation_report/analysis_contract objects with
    path and sha256. Preparation reports require cell_ids, design_columns,
    pool_id, presentation_id and both confirmatory design-rank fields.
    Hashes attest supplied bytes, not which inputs were used in execution.
    V2 prior fits additionally bind prior_contract and model_source. Missing
    prior variants or explicit status/reason entries remain incomplete, while
    primary fits and malformed provenance fail closed.
    """
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2):
        raise ValueError("Fit manifest must declare schema_version 1 or 2; legacy paths are unsupported")
    if set(manifest) != {"schema_version", "max_treedepth", "fits"}:
        raise ValueError("Fit manifest requires schema_version, max_treedepth and fits only")
    if manifest["max_treedepth"] != 12:
        raise ValueError("Fit manifest must use the frozen TD12 configuration")
    if fit_loader is None:
        from cmdstanpy import from_csv

        fit_loader = from_csv
    fit_paths = manifest.get("fits", {})
    required_groups = {"venture", "hiring", "matched_rq5"}
    if not isinstance(fit_paths, Mapping) or set(fit_paths) != required_groups:
        raise ValueError(f"Fit manifest must contain exactly {sorted(required_groups)}")
    max_treedepth = manifest["max_treedepth"]
    config = SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])

    pool_variants = {}
    pool_contrasts = {}
    artifact_hashes = {}
    provenance = {}
    predictive_checks = {}
    prior_fits = {}
    ceiling_reports = {}
    used_chains: set[str] = set()
    for pool_id in ("venture", "hiring"):
        design, columns, cell_ids = config.design_matrix_for_pool(pool_id)
        pool_contrasts[pool_id] = confirmatory_analysis.primary_contrasts(columns)
        pool_variants[pool_id] = _load_variants(
            fit_paths[pool_id],
            cell_ids,
            design,
            columns,
            confirmatory_analysis.contract_manifest(columns, design),
            group=pool_id,
            max_treedepth=max_treedepth,
            fit_loader=fit_loader,
            artifact_hashes=artifact_hashes,
            provenance=provenance,
            predictive_checks=predictive_checks,
            used_chains=used_chains,
            a3=manifest["schema_version"] == 2, prior_fits=prior_fits, ceiling_reports=ceiling_reports,
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
        *confirmatory_analysis.matched_rq5_design(matched_cells),
        matched_contract,
        group="matched_rq5",
        max_treedepth=max_treedepth,
        fit_loader=fit_loader,
        artifact_hashes=artifact_hashes,
        provenance=provenance,
        predictive_checks=predictive_checks,
        used_chains=used_chains,
        a3=manifest["schema_version"] == 2, prior_fits=prior_fits, ceiling_reports=ceiling_reports,
    )
    report = confirmatory_analysis.complete_confirmatory_report(
        pool_variants=pool_variants,
        pool_contrasts=pool_contrasts,
        matched_variants=matched_variants,
        matched_contrasts=matched_contrasts,
        prior_sensitivity_fits=prior_fits,
    )
    report["fit_artifact_hashes"] = artifact_hashes
    report["fit_provenance"] = provenance
    report["posterior_predictive_checks"] = predictive_checks
    report["assessment_scale"] = confirmatory_analysis.assessment_scale.policy()
    report["ceiling_diagnostics"] = ceiling_reports
    report["input_manifest_schema_version"] = manifest["schema_version"]
    report["predictive_check_policy"] = {"policy_version": a4_checks.POLICY_ID, "decision_count": 0,
                                         "planned_fit_count": 24, "primary_emphasis": ["venture", "hiring", "matched_rq5"]}
    for group in ("venture", "hiring", "matched_rq5"):
        section = report["matched_rq5"] if group == "matched_rq5" else report["pools"][group]
        for variant in REQUIRED_VARIANTS:
            _attach_contrast_dependence_text(section[variant], group=group, variant=variant)
            section[variant]["rq6"]["predictive_interpretation"] = a4_checks.interpretation(
                predictive_checks[group][variant]["a4"])
        section["primary"]["rq6"]["predictive_interpretation"]["presentation_comparison"] = {
            variant: {
                "estimate": {key: value for key, value in section[variant]["rq6"].items()
                             if key != "predictive_interpretation"},
                "change_from_primary": {key: section[variant]["rq6"][key] - section["primary"]["rq6"][key]
                                        for key in ("median", "lower_90", "upper_90")},
                "observation_sets_differ": True,
                "primary_observation_set_sha256": predictive_checks[group]["primary"]["a4"]["source_bindings"]["observation_set_sha256"],
                "variant_observation_set_sha256": predictive_checks[group][variant]["a4"]["source_bindings"]["observation_set_sha256"],
            }
            for variant in ("presentation_1_only", "presentation_2_only")}
    report["predictive_check_policy"]["complete_fit_count"] = sum(
        prediction["a4"]["status"] == "descriptive"
        for variants in predictive_checks.values() for prediction in variants.values())
    return report


def _attach_contrast_dependence_text(fit_report, *, group, variant):
    for row in fit_report["contrast_decisions"]["rows"]:
        if row["research_question"] not in ("RQ1", "RQ2", "RQ5"):
            continue
        row["dependence_qualification"] = {
            "policy": "review2_reporting_clarification_2026-10-08",
            "contributing_cell_ids": sorted(cell_id for cell_id, weight in row["cell_weights"].items() if weight),
            "paired_diagnostic_path": ["posterior_predictive_checks", group, variant, "a4"],
            "paired_diagnostic_section_when_available": "pairs",
            "presentation_comparison_path": (["matched_rq5"] if group == "matched_rq5" else ["pools", group])
                + ["presentation_sensitivity"],
            "primary_decision_unchanged": True,
            "text": "An excess same-item repetition flag in contributing cells also qualifies this comparison; read it with the existing presentation-only estimates. The flag raises concern about independent-observation uncertainty but does not identify the cause of repetition or change the interval, threshold or detection decision. Unavailable paired diagnostics are not evidence of no dependence.",
        }


def _load_variants(
    paths: Mapping[str, Any],
    canonical_cell_ids: list[str],
    canonical_design: Any,
    columns: Any,
    expected_contract: Mapping[str, Any],
    *,
    group: str,
    max_treedepth: int,
    fit_loader: Any,
    artifact_hashes: Dict[str, Dict[str, str]],
    provenance: Dict[str, Any],
    predictive_checks: Dict[str, Any],
    used_chains: set[str],
    a3: bool = False,
    prior_fits: Dict[str, Any] | None = None,
    ceiling_reports: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    allowed = set(REQUIRED_VARIANTS) | (set(ceiling_prior.PRIOR_VARIANTS) if a3 else set())
    if not isinstance(paths, Mapping) or not set(REQUIRED_VARIANTS) <= set(paths) or set(paths) - allowed:
        raise ValueError(f"Fit group must contain exactly {list(REQUIRED_VARIANTS)}")
    prior_fits = {} if prior_fits is None else prior_fits
    ceiling_reports = {} if ceiling_reports is None else ceiling_reports
    prior_fits[group] = {}
    variants = {}
    provenance[group] = {}
    predictive_checks[group] = {}
    for variant in (*REQUIRED_VARIANTS, *ceiling_prior.PRIOR_VARIANTS):
        predictive_checks[group][variant] = {"a4": a4_checks.unavailable("Fit not supplied")}
        is_prior = variant in ceiling_prior.PRIOR_VARIANTS
        entry = paths.get(variant)
        if is_prior and entry is None:
            prior_fits[group][variant] = {"status": "missing", "reason": "Prior fit not supplied"}
            continue
        if is_prior and isinstance(entry, Mapping) and set(entry) == {"status", "reason"}:
            if entry["status"] not in {"missing", "failed"} or not isinstance(entry["reason"], str) or not entry["reason"]:
                raise ValueError("Invalid incomplete prior fit declaration")
            prior_fits[group][variant] = dict(entry)
            continue
        required = {"chain_path", "chain_sha256", "stan_data", "preparation_report", "analysis_contract"}
        if is_prior:
            required |= {"prior_contract", "model_source"}
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError(f"{group}/{variant} requires explicit artifact bindings: {sorted(required)}")
        data, data_path = _load_json_binding(entry["stan_data"], "stan_data")
        preparation, _ = _load_json_binding(entry["preparation_report"], "preparation_report")
        if a3 and preparation.get("observation_metadata_version") != 2:
            raise ValueError("A3 manifests require version-2 complete frozen observation evidence for every fit")
        if a3:
            a4_checks.validate_evidence(data, preparation)
        contract, _ = _load_json_binding(entry["analysis_contract"], "analysis_contract")
        if is_prior:
            declared_prior, _ = _load_json_binding(entry["prior_contract"], "prior_contract")
            if declared_prior != ceiling_prior.prior_contract(variant):
                raise ValueError("Prior contract is not canonical")
            model_binding = entry["model_source"]
            expected_model = Path(__file__).resolve().parents[2] / "models" / f"{ceiling_prior.SENSITIVITY_MODEL}.stan"
            if (not isinstance(model_binding, Mapping) or set(model_binding) != {"path", "sha256"}
                    or _sha256_file(_absolute_path(model_binding["path"], "model_source")) != model_binding["sha256"]
                    or model_binding["sha256"] != _sha256_file(expected_model)):
                raise ValueError("Model source binding differs from canonical sensitivity model")
            ceiling_prior.validate_prior_data(data, primary_data, variant)
            if preparation != primary_preparation:
                raise ValueError("Prior preparation differs from primary observation/exclusion binding")
        elif any(key.startswith("prior_") for key in data):
            raise ValueError("Primary-model input must not contain sensitivity prior fields")
        if contract != json.loads(json.dumps(expected_contract)):
            raise ValueError(f"{group}/{variant} analysis_contract differs from the canonical contract/columns")
        cell_ids = _validate_data(
            data, preparation, canonical_cell_ids, canonical_design, columns,
            group=group, variant=variant,
        )
        reference = preparation.get("assessment_scale_reference")
        confirmatory_analysis.assessment_scale.validate_retained_data(reference, data, preparation, group=group)
        if variant == "primary":
            primary_reference = reference
            primary_data, primary_preparation = data, preparation
            if a3:
                ceiling_diagnostics.validate_observations(data, preparation)
        elif reference != primary_reference:
            raise ValueError("assessment_scale reference differs across sibling fit variants")
        if variant != "primary" and preparation.get("predictive_reference") is not None:
            a4_checks.validate_sibling_evidence(primary_preparation, preparation, variant)
        path = _absolute_path(entry["chain_path"], "chain_path")
        try:
            files = _fit_files(path)
        except FileNotFoundError as error:
            if not is_prior:
                raise
            prior_fits[group][variant] = {"status": "missing", "reason": str(error), "artifacts": dict(entry)}
            continue
        if len(files) != 4:
            raise ValueError("Fit binding must identify four actual chain files")
        hashes = {file.name: _sha256_file(file) for file in files}
        if entry["chain_sha256"] != hashes:
            raise ValueError("chain_sha256 must match every chain file exactly")
        for file in files:
            identity = str(file.resolve())
            digest = hashes[file.name]
            if identity in used_chains or digest in used_chains:
                raise ValueError(f"Chain reuse is prohibited across chains, variants and groups: {file}")
            used_chains.update((identity, digest))
        artifact_hashes[str(path)] = {str(file): hashes[file.name] for file in files}
        fit = fit_loader([str(file) for file in files])
        if fit is None:
            if is_prior:
                prior_fits[group][variant] = {"status": "failed", "reason": f"No fit loaded from {path}", "artifacts": dict(entry)}
                continue
            raise ValueError(f"No CmdStan fit could be loaded from {path}")
        metadata = fit.metadata.cmdstan_config
        declared_paths = []
        for key in ("data_file", "data"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                value = value.get("file")
            if value is not None:
                if not isinstance(value, str) or not value or Path(value).resolve() != data_path:
                    raise ValueError("CmdStan metadata data path does not match the bound stan_data path")
                declared_paths.append(value)
        try:
            payload = fit_payload(fit, cell_ids, len(columns), max_treedepth=max_treedepth,
                                  prior_variant=variant if is_prior else "primary")
            try:
                confirmatory_analysis.assert_sampler_gates(payload["diagnostics"])
            except ValueError as error:
                raise SamplerFailure(str(error)) from error
            prediction = _predictive_checks(fit, data, cell_ids, payload)
            prediction["a4"] = (a4_checks.build_predictive_report(
                data, preparation, _draws(fit, "y_pred"), _draws(fit, "alpha_obs"))
                if preparation.get("predictive_reference") is not None else
                a4_checks.unavailable("Historical manifest lacks mandatory A4 canonical role/frozen recipe evidence"))
            prediction["a4"]["source_bindings"] = {
                "stan_data": entry["stan_data"], "preparation_report": entry["preparation_report"],
                "chain_sha256": entry["chain_sha256"], "cryptographic_execution_proof": False,
                "observation_set_sha256": ceiling_prior.digest(preparation.get("observations", [])),
                "limitation": "Declared byte bindings and equation consistency, not proof of execution inputs"}
        except SamplerFailure as error:
            if not is_prior:
                raise
            prior_fits[group][variant] = {"status": "sampler_failed", "reason": str(error), "artifacts": dict(entry)}
            predictive_checks[group][variant] = {"a4": a4_checks.unavailable(f"Sampler-invalid fit: {error}")}
            continue
        if variant == "primary":
            payload["assessment_scale_reference"] = reference
        provenance[group][variant] = {
            "binding": "declared_input_binding",
            "cryptographic_execution_proof": False,
            "limitation": "Hashes verify supplied artifacts, not execution provenance; matching metadata and derived draws are consistency checks only.",
            "model": metadata["model"],
            "chains": fit.chains,
            "max_treedepth": max_treedepth,
            "cell_ids": cell_ids,
            "design_columns": list(columns),
            "artifacts": dict(entry),
            "cmdstan_data_path_check": "matched" if declared_paths else "unavailable",
            "assessment_scale_reference_binding": "preparation_report.sha256",
        }
        predictive_checks[group][variant] = prediction
        if variant == "primary" or is_prior:
            levels = (_draws(fit, "gamma0")[:, None] + payload["gamma_draws"] @ np.asarray(data["X"]).T
                      + payload["sigma_cell_draws"][:, None] * payload["z_alpha_draws"])
            prior_fits[group][variant] = {
                "status": "complete", "payload": payload,
                "cell_quantiles": {cell_id: {"t": ceiling_diagnostics.quantiles(levels[:, index]),
                                             "alpha": ceiling_diagnostics.quantiles(np.exp(levels[:, index]))}
                                   for index, cell_id in enumerate(cell_ids)},
                "size_slope_quantiles": ceiling_diagnostics.quantiles(payload["gamma_size_draws"]),
            }
            if variant == "primary":
                ceiling_reports[group] = (ceiling_diagnostics.retained_ceiling_report(
                    data, preparation, levels, payload["gamma_size_draws"]) if a3 else
                    {"status": "unavailable", "reason": "Historical manifest lacks mandatory A3 observation binding"})
        if not is_prior:
            variants[variant] = payload
    for variant in ceiling_prior.PRIOR_VARIANTS:
        if prior_fits[group][variant].get("status") != "complete":
            incomplete = prior_fits[group][variant]
            predictive_checks[group][variant] = {"a4": {
                **a4_checks.unavailable(incomplete["reason"]), "fit_status": incomplete["status"]}}
    return variants


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute artifact path")
    return Path(value).resolve()


def _load_json_binding(binding: Any, label: str) -> tuple[Dict[str, Any], Path]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind path and sha256")
    path = _absolute_path(binding["path"], label)
    if binding["sha256"] != _sha256_file(path):
        raise ValueError(f"{label} SHA256 mismatch")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return data, path


def _data_array(data: Mapping[str, Any], name: str, shape: tuple[int, ...], *, integer: bool = False) -> np.ndarray:
    try:
        values = np.asarray(data[name], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Stan data requires numeric {name}") from error
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    if integer and np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain integers")
    return values.astype(int) if integer else values


def _validate_data(
    data: Mapping[str, Any], preparation: Mapping[str, Any],
    canonical_cell_ids: list[str], canonical_design: Any, columns: Any,
    *, group: str, variant: str,
) -> list[str]:
    cell_ids = preparation.get("cell_ids")
    if (
        not isinstance(cell_ids, list) or not cell_ids
        or any(not isinstance(cell_id, str) for cell_id in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
        or not set(cell_ids).issubset(canonical_cell_ids)
        or cell_ids != [cell_id for cell_id in canonical_cell_ids if cell_id in cell_ids]
    ):
        raise ValueError("Preparation cell_ids must be a unique canonical subset in canonical order")
    if preparation.get("design_columns") != list(columns):
        raise ValueError("Preparation report must establish canonical design_columns")
    if preparation.get("pool_id") != group:
        raise ValueError("Preparation report pool_id must match its fit group")
    presentation = {"presentation_1_only": 1, "presentation_2_only": 2}.get(variant)
    if "presentation_id" not in preparation or preparation["presentation_id"] != presentation:
        raise ValueError("Preparation report presentation_id must match its fit variant")
    for name, minimum in (("J", 1), ("P", 1), ("M_total", 1), ("R", 2), ("K", 3)):
        if type(data.get(name)) is not int or data[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    cell_count, column_count = data["J"], data["P"]
    observations, items = data["M_total"], data["R"]
    if cell_count != len(cell_ids) or column_count != len(columns) or data["K"] != 3:
        raise ValueError("J/P/K dimensions do not match the canonical retained design")
    design = _data_array(data, "X", (cell_count, column_count))
    canonical = np.asarray(canonical_design, dtype=float)
    indices = [canonical_cell_ids.index(cell_id) for cell_id in cell_ids]
    if not np.array_equal(design, canonical[indices]):
        raise ValueError("X must equal the canonical rows for retained cell_ids")
    rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(cell_count), design])))
    canonical_rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(canonical)), canonical])))
    if rank != canonical_rank or rank != column_count + 1:
        raise ValueError("Retained design rank must be unchanged and full with intercept")
    if preparation.get("confirmatory_design_rank") != rank or preparation.get("confirmatory_design_required_rank") != canonical_rank:
        raise ValueError("Preparation report design rank does not match retained X")
    expected_utility = [0.0, {"utility_035": 0.35, "utility_065": 0.65}.get(variant, 0.5), 1.0]
    utilities = _data_array(data, "utility_values", (3,))
    if not np.array_equal(utilities, expected_utility):
        raise ValueError("utility_values do not match the expected utility variant")
    eta = _data_array(data, "eta", (cell_count, items))
    if np.any((eta < 0) | (eta > 1)):
        raise ValueError("eta must lie in [0, 1]")
    cell = _data_array(data, "cell", (observations,), integer=True)
    indicators = _data_array(data, "I", (observations, items), integer=True)
    choices = _data_array(data, "y", (observations,), integer=True)
    counts = _data_array(data, "M_per_cell", (cell_count,), integer=True)
    size = _data_array(data, "s", (observations,))
    if np.any((cell < 1) | (cell > cell_count)) or np.any(counts < 1):
        raise ValueError("cell/M_per_cell must index every retained cell")
    if not np.array_equal(np.bincount(cell, minlength=cell_count + 1)[1:], counts):
        raise ValueError("M_per_cell must match actual cell observation counts")
    menu_sizes = indicators.sum(axis=1)
    if np.any((indicators != 0) & (indicators != 1)) or np.any(menu_sizes < 2):
        raise ValueError("I must be binary with at least two active alternatives per observation")
    if np.any((choices < 1) | (choices > menu_sizes)):
        raise ValueError("y must index the sorted active alternatives")
    if not np.allclose(size, menu_sizes - menu_sizes.mean(), atol=1e-6, rtol=0):
        raise ValueError("s must equal centered actual menu sizes")
    if preparation.get("observation_metadata_version") == 2:
        ceiling_diagnostics.validate_observations(data, preparation)
    return cell_ids


def _check_draw_shape(values: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if values.shape != shape:
        raise ValueError(f"{name} draws must have shape {shape}; got {values.shape}")


def _require_agreement(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.allclose(actual, expected, rtol=5e-5, atol=5e-6):
        raise ValueError(f"{name} draws disagree with the bound Stan input/parameter equations")


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    return dict(zip(("q05", "q50", "q95"), map(float, np.quantile(values, [0.05, 0.5, 0.95]))))


def _predictive_checks(
    fit: Any, data: Mapping[str, Any], cell_ids: list[str], payload: Mapping[str, Any]
) -> Dict[str, Any]:
    gamma = payload["gamma_draws"]
    draw_count, observation_count = len(gamma), data["M_total"]
    cell = np.asarray(data["cell"], dtype=int) - 1
    choices = np.asarray(data["y"], dtype=int)
    eta = np.asarray(data["eta"], dtype=float)
    indicators = np.asarray(data["I"], dtype=bool)
    utilities = _draws(fit, "upsilon")
    _check_draw_shape(utilities, (draw_count, data["K"]), "upsilon")
    _require_agreement(utilities, np.asarray(data["utility_values"])[None, :], "utility_values/upsilon")
    alpha_cell = _draws(fit, "alpha_cell")
    alpha_obs = _draws(fit, "alpha_obs")
    _check_draw_shape(alpha_cell, (draw_count, len(cell_ids)), "alpha_cell")
    _check_draw_shape(alpha_obs, (draw_count, observation_count), "alpha_obs")
    if np.any(alpha_cell <= 0) or np.any(alpha_obs <= 0):
        raise ValueError("alpha draws must be positive")
    log_alpha = (
        _draws(fit, "gamma0")[:, None] + gamma @ np.asarray(data["X"]).T
        + payload["sigma_cell_draws"][:, None] * payload["z_alpha_draws"]
    )
    _require_agreement(np.log(alpha_cell), log_alpha, "alpha_cell")
    predicted = _draws(fit, "y_pred")
    log_lik = _draws(fit, "log_lik")
    _check_draw_shape(predicted, (draw_count, observation_count), "y_pred")
    _check_draw_shape(log_lik, (draw_count, observation_count), "log_lik")
    if np.any(predicted != np.floor(predicted)):
        raise ValueError("y_pred must contain integer choice positions")
    menu_sizes = indicators.sum(axis=1)
    if np.any((predicted < 1) | (predicted > menu_sizes[None, :])):
        raise ValueError("y_pred must index the sorted active alternatives")
    observed_modal = np.zeros(observation_count)
    tied_eta = np.zeros(observation_count)
    metrics = {name: np.zeros((draw_count, len(cell_ids))) for name in (
        "replicated_modal_fraction", "observed_choice_probability", "replicated_choice_probability",
        "observed_mean_log_score", "replicated_mean_log_score",
    )}
    for observation, cell_index in enumerate(cell):
        _require_agreement(
            np.log(alpha_obs[:, observation]),
            log_alpha[:, cell_index] + payload["gamma_size_draws"] * data["s"][observation],
            "alpha_obs",
        )
        active_eta = eta[cell_index, indicators[observation]]
        logits = alpha_obs[:, observation, None] * (active_eta - active_eta.max())
        log_probabilities = logits - np.log(np.exp(logits).sum(axis=1))[:, None]
        observed_log = log_probabilities[:, choices[observation] - 1]
        _require_agreement(log_lik[:, observation], observed_log, "log_lik/eta/y")
        predicted_positions = predicted[:, observation].astype(int) - 1
        replicated_log = log_probabilities[np.arange(draw_count), predicted_positions]
        modal = active_eta == active_eta.max()
        observed_modal[observation] = modal[choices[observation] - 1]
        tied_eta[observation] = np.count_nonzero(modal) > 1
        metrics["replicated_modal_fraction"][:, cell_index] += modal[predicted_positions]
        metrics["observed_choice_probability"][:, cell_index] += np.exp(observed_log)
        metrics["replicated_choice_probability"][:, cell_index] += np.exp(replicated_log)
        metrics["observed_mean_log_score"][:, cell_index] += observed_log
        metrics["replicated_mean_log_score"][:, cell_index] += replicated_log

    def aggregate(observation_mask: np.ndarray, cell_indices: list[int]) -> Dict[str, Any]:
        count = int(observation_mask.sum())
        return {
            "observation_count": count,
            "observed_modal_fraction": float(observed_modal[observation_mask].mean()),
            "tied_eta_fraction": float(tied_eta[observation_mask].mean()),
            "alpha_cell_quantiles": _quantiles(alpha_cell[:, cell_indices]),
            "alpha_obs_quantiles": _quantiles(alpha_obs[:, observation_mask]),
            **{name: _quantiles(values[:, cell_indices].sum(axis=1) / count) for name, values in metrics.items()},
            "choice_position_distribution": [
                {
                    "position": position,
                    "observed_fraction": float(np.mean(choices[observation_mask] == position)),
                    "replicated_fraction": _quantiles(np.mean(predicted[:, observation_mask] == position, axis=1)),
                }
                for position in range(1, int(menu_sizes[observation_mask].max()) + 1)
            ],
        }

    canonical_pools = {cell.cell_id: cell.pool_id for cell in build_cells(["venture", "hiring"])}
    pools = {cell_id: canonical_pools[cell_id] for cell_id in cell_ids}
    return {
        "status": "descriptive",
        "position_encoding": "one-based rank within sorted active items, not presentation position",
        "tied_eta_definition": "multiple active alternatives attain the exact maximum eta",
        "inferential_ceiling_threshold": None,
        "cells": {cell_id: aggregate(cell == index, [index]) for index, cell_id in enumerate(cell_ids)},
        "pools": {
            pool: aggregate(np.isin(cell, indices), indices)
            for pool in sorted(set(pools.values()))
            for indices in [[index for index, cell_id in enumerate(cell_ids) if pools[cell_id] == pool]]
        },
    }


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
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
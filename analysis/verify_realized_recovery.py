"""Read-only verification of saved assessment-anchored recovery draws."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/applications/seu_sensitivity_study/data/realized_recovery_verification.json")
CAMPAIGNS = {
    "venture": "venture_recovery_pilot_500",
    "hiring": "hiring_recovery_pilot_500",
    "matched_rq5": "matched_rq5_recovery",
}
REPLACEMENTS = {"venture": [3, 13, 16, 28, 33], "hiring": [3, 12, 17, 21, 38],
                "matched_rq5": [2, 4, 6, 8, 13, 14, 16, 17, 26, 38]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def canonical_truth(truth):
    return {**truth, "contrasts": truth.get("contrasts", {})}


class Inputs:
    """Hash only read-only source inputs, using full file bytes (including gzip)."""

    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def record(self, path):
        path = self.root / path
        relative = path.resolve().relative_to(self.root).as_posix()
        if relative not in self.hashes:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            self.hashes[relative] = hasher.hexdigest()
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())


def realized_log_cells(gamma0, gamma, sigma, residual_z, design):
    """Reconstruct one truth or a batch of posterior log sensitivities."""
    design = np.asarray(design, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    residual_z = np.asarray(residual_z, dtype=float)
    gamma0 = np.asarray(gamma0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if design.ndim != 2 or gamma.ndim not in (1, 2):
        raise ValueError("Expected a matrix design and vector or matrix gamma")
    batch = gamma.shape[:-1]
    if (gamma.shape[-1] != design.shape[1]
            or residual_z.shape != batch + (design.shape[0],)
            or gamma0.shape != batch or sigma.shape != batch):
        raise ValueError("Incompatible truth/draw shapes")
    if not all(np.isfinite(value).all() for value in (design, gamma, residual_z, gamma0, sigma)):
        raise ValueError("Nonfinite truth/draw values")
    if np.any(sigma < 0):
        raise ValueError("Negative sigma")
    return gamma0[..., None] + gamma @ design.T + sigma[..., None] * residual_z


def cell_weight_vector(cell_ids: Sequence[str], weights: dict[str, float]) -> np.ndarray:
    """Align named Amendment-5 weights without assuming cell positions."""
    if len(set(cell_ids)) != len(cell_ids) or not set(weights) <= set(cell_ids):
        raise ValueError("Duplicate or missing cell IDs")
    result = np.asarray([weights.get(cell_id, 0.0) for cell_id in cell_ids], dtype=float)
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 0, atol=1e-12):
        raise ValueError("Contrast must be finite and cancel the intercept")
    return result


def chain_metadata(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                columns = line.strip().split(",")
                break
            lines.append(line)
        else:
            raise ValueError(f"Missing CSV header: {path.name}")
    if len(columns) != len(set(columns)) or "lp__" not in columns:
        raise ValueError(f"Malformed CSV header: {path.name}")
    fields = dict(re.findall(r"^#\s*(\w+)\s*=\s*(.*?)\s*$", "".join(lines), re.MULTILINE))
    conversions = {"model": str, "id": int, "seed": int, "num_samples": int,
                   "num_warmup": int, "save_warmup": int, "thin": int,
                   "max_depth": int, "delta": float}
    if not conversions.keys() <= fields.keys():
        raise ValueError(f"Incomplete sampler metadata: {path.name}")
    fields["save_warmup"] = fields["save_warmup"].replace("false", "0").replace("true", "1")
    metadata = {key: convert(fields[key].split(" (Default)")[0])
                for key, convert in conversions.items()}
    if (metadata["num_samples"] < 1 or metadata["num_warmup"] < 0
            or metadata["thin"] < 1 or metadata["save_warmup"] not in (0, 1)):
        raise ValueError("Invalid sampler schedule")
    data_match = re.search(r"^# data\n#\s+file = (.+)$", "".join(lines), re.MULTILINE)
    return {**metadata, "columns": columns,
            "data_file": data_match.group(1).strip() if data_match else None}


def select_chains(directory: Path, references: Sequence[str], *, seed: int,
                  samples: int, warmup: int) -> tuple[list[Path], list[dict], list[str]]:
    if len(references) != 4 or len(set(Path(ref).name for ref in references)) != 4:
        raise ValueError("Exactly four distinct chain references required")
    paths = [directory / Path(reference).name for reference in references]
    if not all(path.is_file() for path in paths):
        raise ValueError("Referenced current chain missing")
    metadata = [chain_metadata(path) for path in paths]
    if sorted(row["id"] for row in metadata) != [1, 2, 3, 4]:
        raise ValueError("Duplicate or invalid chain IDs")
    expected = {"model": "h_m01_size_assessment_anchored_model", "seed": seed,
                "num_samples": samples, "num_warmup": warmup, "max_depth": 12,
                "delta": 0.95, "thin": 1, "save_warmup": 0}
    for row in metadata:
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"Current chain seed/schedule mismatch: {row['id']}")
        if row["columns"] != metadata[0]["columns"]:
            raise ValueError("Inconsistent chain columns")
    order = np.argsort([row["id"] for row in metadata])
    extras = sorted(path.name for path in directory.glob("*.csv*") if path not in paths)
    return [paths[index] for index in order], [metadata[index] for index in order], extras


def read_chain(path: Path, columns: Sequence[str], metadata: dict) -> pd.DataFrame:
    if not set(columns) <= set(metadata["columns"]):
        raise ValueError(f"Missing required draw columns: {path.name}")
    frame = pd.read_csv(path, comment="#", usecols=list(columns), dtype=float)
    thin = metadata["thin"]
    sampling_rows = (metadata["num_samples"] + thin - 1) // thin
    warmup_rows = ((metadata["num_warmup"] + thin - 1) // thin
                   if metadata["save_warmup"] else 0)
    if len(frame) != sampling_rows + warmup_rows:
        raise ValueError(f"Incomplete or excess CSV draws: {path.name}")
    frame = frame.iloc[warmup_rows:].reset_index(drop=True)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Nonfinite posterior draws: {path.name}")
    return frame


def score_draws(draws, truth, rope):
    values = np.asarray(draws, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Invalid contrast draws")
    require(math.isfinite(truth) and math.isfinite(rope) and rope > 0, "Invalid truth or ROPE")
    lower, median, upper = np.quantile(values, [0.05, 0.5, 0.95])
    decision = 1 if lower > 0 and median > rope else -1 if upper < 0 and median < -rope else 0
    return {"truth": float(truth), "mean": float(values.mean()), "median": float(median),
            "lower": float(lower), "upper": float(upper), "decision": decision}


def rate(numerator, denominator):
    return {"count": int(numerator), "denominator": int(denominator),
            "rate": float(numerator / denominator) if denominator else None}


def summarize_scores(rows, rope):
    if not rows:
        return {"n": 0}
    truths = np.asarray([row["truth"] for row in rows])
    decisions = np.asarray([row["decision"] for row in rows])
    lower = np.asarray([row["lower"] for row in rows])
    upper = np.asarray([row["upper"] for row in rows])
    errors = np.asarray([row["mean"] - row["truth"] for row in rows])
    median_errors = np.asarray([row["median"] - row["truth"] for row in rows])
    detected = decisions != 0
    nonnull = truths != 0
    outside = np.abs(truths) > rope
    correct_sign = (decisions == np.sign(truths)) & nonnull
    wrong = detected & nonnull & ~correct_sign
    bins = [0, rope, 2 * rope, np.inf] if rope == math.log(1.05) else [0, rope, math.log(1.5), math.log(2), np.inf]
    labels = ["below_ROPE", "ROPE_to_2ROPE", "above_2ROPE"] if len(bins) == 4 else ["below_1.25", "1.25_to_1.5", "1.5_to_2", "above_2"]
    binned = {}
    for label, left, right in zip(labels, bins[:-1], bins[1:]):
        selected = (np.abs(truths) >= left) & (np.abs(truths) < right)
        binned[label] = rate((detected & selected).sum(), selected.sum())
    return {"n": len(rows), "coverage90": rate(((lower <= truths) & (truths <= upper)).sum(), len(rows)),
            "bias_posterior_mean": float(errors.mean()), "rmse_posterior_mean": float(np.sqrt(np.mean(errors ** 2))),
            "bias_posterior_median": float(median_errors.mean()),
            "rmse_posterior_median": float(np.sqrt(np.mean(median_errors ** 2))),
            "mean_interval_width": float(np.mean(upper - lower)),
            "detection_all_truths": rate(detected.sum(), len(rows)),
            "power_outside_ROPE": rate((detected & outside).sum(), outside.sum()),
            "correct_sign_power_outside_ROPE": rate((correct_sign & outside).sum(), outside.sum()),
            "detections_inside_ROPE_not_false_positives": rate((detected & (np.abs(truths) < rope)).sum(), (np.abs(truths) < rope).sum()),
            "false_positive_exact_null": rate((detected & ~nonnull).sum(), (~nonnull).sum()),
            "type_S_detected_nonzero_truth": rate(wrong.sum(), (detected & nonnull).sum()),
            "median_correct_sign_nonzero_truth": rate(((np.sign([row["median"] for row in rows]) == np.sign(truths)) & nonnull).sum(), nonnull.sum()),
            "detection_bins": binned}


def verify_design_rows(saved, canonical, canonical_ids):
    saved = np.asarray(saved, dtype=float)
    canonical = np.asarray(canonical, dtype=float)
    require(saved.shape == canonical.shape and len(canonical_ids) == len(canonical), "Design shape mismatch")
    require(len(set(canonical_ids)) == len(canonical_ids), "Duplicate canonical IDs")
    lookup = {tuple(row): cell_id for row, cell_id in zip(canonical, canonical_ids)}
    require(len(lookup) == len(canonical), "Canonical rows not unique")
    require(len(set(map(tuple, saved))) == len(saved) and all(tuple(row) in lookup for row in saved), "Unmatched or duplicate design row")
    return [lookup[tuple(row)] for row in saved]


def campaign_geometry(inputs, name, campaign):
    from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig, MODELS, build_cells
    from applications.seu_sensitivity_study import confirmatory_analysis as contract
    from utils.study_design_hierarchical import HierarchicalStudyDesign

    design = inputs.json(campaign / "study_design.json")
    saved_config = inputs.json(campaign / "config_info.json")
    config = inputs.json(Path(f"configs/h_m01_size_assessment_anchored_{name}_validation_config.json"))
    fixed = saved_config["fixed_eta_config"]
    require(fixed == config["fixed_eta_config"], "Fixed eta configuration changed")
    if name == "matched_rq5":
        cells = build_cells(["venture", "hiring"])
        canonical, columns = contract.matched_rq5_design(cells)
        ids = verify_design_rows(design["X"], canonical, [cell.cell_id for cell in cells])
        template_path = Path(fixed["stan_data_template_path"])
        template = inputs.json(template_path)
        assembly = inputs.json(template_path.with_name("assembly_report.json"))
        require(ids == assembly["cell_ids"], "Matched template cell order mismatch")
        for key in ("J", "K", "R", "P", "M_total", "X", "I", "cell", "M_per_cell", "s"):
            require(np.array_equal(design[key], template[key]), f"Matched template mismatch: {key}")
        eta = np.asarray(template["eta"])
        utilities = template["utility_values"]
        specifications = contract.matched_rq5_contract(cells)["contrasts"]
        require(saved_config["linear_contrasts"] == {spec["contrast_id"]: list(spec["coefficients"]) for spec in specifications}, "RQ5 coefficient names changed")
        mapping_method = "Saved matched assembly cell IDs and X verified against matched_rq5_design; template X/I/cell/s identical to saved generation geometry."
    else:
        canonical, columns, canonical_ids = SEUSensitivityStudyConfig(pool_ids=[name]).design_matrix_for_pool(name)
        metadata = design["_metadata"]
        generated, labels, levels = HierarchicalStudyDesign.treatment_design_matrix(
            metadata["factors"], metadata["reference_indices"], metadata["include_interactions"])
        require(np.array_equal(design["X"], generated), "Factorial generation X mismatch")
        require(metadata["column_labels"] == labels and metadata["cell_levels"] == levels, "Factorial labels mismatch")
        require(np.array_equal(generated, canonical), "Factorial to canonical column mapping mismatch")
        ids = verify_design_rows(design["X"], canonical, canonical_ids)
        by_id = {cell.cell_id: cell for cell in build_cells([name])}
        by_model = {model.name: model for model in MODELS}
        pool = inputs.json(Path(fixed["pool_path"]))
        item_ids = [item["id"] for item in pool["items"]]
        utilities = fixed["utility_values"]
        eta = []
        for cell_id, reference in zip(ids, fixed["assessment_files"]):
            require(Path(reference).stem == by_model[by_id[cell_id].model_name].slug, "Assessment file model mapping mismatch")
            assessment = inputs.json(Path(reference))
            probabilities = {row["item_id"]: row["probabilities"] for row in assessment["assessments"] if row.get("parse_ok")}
            require(set(probabilities) == set(item_ids), "Assessment items mismatch")
            eta.append([float(np.asarray(probabilities[item_id]) @ utilities) for item_id in item_ids])
        eta = np.asarray(eta)
        problems = inputs.json(Path(config["study_design_config"]["problem_set_path"]))
        indicators = [[int(item_id in problem["item_ids"]) for item_id in item_ids]
                      for problem in problems["problems"] for _ in problem["presentations"]]
        require(np.array_equal(design["I"], np.tile(indicators, (len(ids), 1))), "Saved menu geometry mismatch")
        require(np.array_equal(design["cell"], np.repeat(np.arange(1, len(ids) + 1), len(indicators))), "Saved observation/cell mapping mismatch")
        specifications = [spec.to_dict() for spec in contract.primary_contrasts(columns)]
        mapping_method = "Factorial metadata f0=model/f1=prompt verified against canonical X; assessment slug per row verifies model identity; problem-set replication verifies I/cell."
    require(eta.shape == (design["J"], design["R"]) and np.isfinite(eta).all(), "Eta shape mismatch")
    require(len(ids) == design["J"] and len(columns) == design["P"], "Dimension mismatch")
    sizes = np.asarray(design["I"]).sum(axis=1)
    require(np.allclose(design["s"], sizes - sizes.mean()), "Incorrect size centering")
    priors = {key: design[key] for key in ("gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd", "gamma_size_mean", "gamma_size_sd")}
    priors.update(saved_config["sim_overrides"])
    require(priors == {"gamma0_mean": 2.5, "gamma0_sd": 0.5, "gamma_sd": 0.5,
                       "sigma_cell_sd": 0.3, "gamma_size_mean": 0.0, "gamma_size_sd": 0.2}, "Unexpected actual generating prior")
    info = {"J": len(ids), "P": len(columns), "R": design["R"], "M_total": design["M_total"],
            "cell_ids": ids, "design_columns": list(columns), "mapping_method": mapping_method,
            "design_X_sha256": digest(design["X"]), "eta_sha256": digest(eta.tolist()),
            "actual_generating_priors": priors, "utility_values": utilities,
            "saved_design_gamma_size_sd_before_override": design["gamma_size_sd"],
            "eta_source": "fixed persisted neutral assessments; no simulated belief map",
            "simulation_seed_rule": "12345 + (iteration - 1)",
            "inference_seed_rule": "54321 + (iteration - 1)",
            "replacement_iterations": REPLACEMENTS[name]}
    return design, ids, specifications, info


def verify_iteration(inputs, name, campaign, iteration, design, ids, specifications):
    from applications.seu_sensitivity_study.confirmatory_analysis import assert_sampler_gates

    directory = campaign / f"iteration_{iteration}"
    truth = inputs.json(directory / "true_parameters.json")
    diagnostics = inputs.json(directory / "diagnostics.json")
    summary = pd.read_csv(inputs.record(directory / "posterior_summary.csv"), index_col=0)
    samples = 1000 if iteration in REPLACEMENTS[name] else 500
    paths, metadata, extras = select_chains(inputs.root / directory / "chains/main", diagnostics["chain_files"],
                                            seed=54320 + iteration, samples=samples, warmup=samples)
    gamma_columns = [f"gamma.{index}" for index in range(1, design["P"] + 1)]
    residual_columns = [f"z_alpha.{index}" for index in range(1, design["J"] + 1)]
    log_columns = [f"log_alpha_cell.{index}" for index in range(1, design["J"] + 1)]
    for prefix, expected_columns in (("gamma.", gamma_columns), ("z_alpha.", residual_columns), ("log_alpha_cell.", log_columns)):
        require({column for column in metadata[0]["columns"] if column.startswith(prefix)} == set(expected_columns),
                f"Saved parameter dimension differs from generation design: {prefix}")
    columns = ["gamma0", "sigma_cell", "gamma_size", "divergent__", "treedepth__", "energy__"] + gamma_columns + residual_columns + log_columns
    frames = [read_chain(inputs.record(path), columns, row) for path, row in zip(paths, metadata)]
    frame = pd.concat(frames, ignore_index=True)
    gamma = frame[gamma_columns].to_numpy()
    realized = realized_log_cells(frame.gamma0.to_numpy(), gamma, frame.sigma_cell.to_numpy(),
                                  frame[residual_columns].to_numpy(), design["X"])
    reconstruction_error = float(np.max(np.abs(realized - frame[log_columns].to_numpy())))
    require(reconstruction_error < 2e-6, "Posterior log-alpha reconstruction differs from saved transformed parameter")
    true_gamma = np.asarray(truth["gamma"])
    true_alpha = np.asarray(truth["alpha"])
    require(true_alpha.shape == (len(ids),) and np.all(true_alpha > 0) and np.isfinite(true_alpha).all(), "Malformed alpha truth")
    require(true_gamma.shape == (design["P"],) and truth["sigma_cell"] > 0, "Malformed generating truth")
    true_log = np.log(true_alpha)
    true_z = (true_log - truth["gamma0"] - np.asarray(design["X"]) @ true_gamma) / truth["sigma_cell"]
    np.testing.assert_allclose(realized_log_cells(truth["gamma0"], true_gamma, truth["sigma_cell"], true_z, design["X"]), true_log, atol=1e-12)
    compared = ["gamma0", "sigma_cell", "gamma_size"] + gamma_columns + residual_columns + log_columns
    summary_mean_error = 0.0
    for column in compared:
        parameter = re.sub(r"\.(\d+)$", r"[\1]", column)
        saved_mean = float(summary.loc[parameter, "Mean"])
        difference = abs(float(frame[column].mean()) - saved_mean)
        summary_mean_error = max(summary_mean_error, difference)
        require(np.isclose(frame[column].mean(), saved_mean, rtol=1e-5, atol=1e-5), f"Current posterior summary mean mismatch: {parameter}")
    computed = {
        "divergences": int(frame.divergent__.sum()),
        "treedepth_saturated_share": float(np.mean(frame.treedepth__ >= 12)),
        "min_ebfmi": min(float(np.mean(np.diff(part.energy__) ** 2) / np.var(part.energy__)) for part in frames),
    }
    for key, value in computed.items():
        require(np.isclose(value, diagnostics[key], rtol=2e-5, atol=2e-6), f"Current diagnostics mismatch: {key}")
    diagnostic_parameters = list(diagnostics["ess_bulk"])
    require(not diagnostics.get("invalid_diagnostic_parameters"), "Invalid diagnostic parameters")
    require(np.isfinite(summary.loc[diagnostic_parameters, ["R_hat", "ESS_bulk", "ESS_tail"]].to_numpy()).all(),
            "Nonfinite saved sampler diagnostics")
    metrics = {"max_rhat": float(summary.loc[diagnostic_parameters, "R_hat"].max()),
               "min_ess_bulk": float(summary.loc[diagnostic_parameters, "ESS_bulk"].min()),
               "min_ess_tail": float(summary.loc[diagnostic_parameters, "ESS_tail"].min()), **computed}
    for key in ("max_rhat", "min_ess_bulk", "min_ess_tail"):
        if key in diagnostics:
            require(np.isclose(metrics[key], diagnostics[key], rtol=1e-5, atol=1e-5), f"Summary diagnostics mismatch: {key}")
    eligibility_failure = None
    try:
        assert_sampler_gates(metrics)
    except ValueError as error:
        eligibility_failure = str(error)
    scores = []
    for spec in specifications:
        weights = cell_weight_vector(ids, spec["realized_cell_weights"][name])
        coefficients = np.asarray(spec["coefficients"])
        require(np.allclose(weights @ np.asarray(design["X"]), coefficients), "Cell/coefficient contrast mismatch")
        for estimand, draws, actual in (("realized", realized @ weights, true_log @ weights),
                                       ("additive_companion", gamma @ coefficients, true_gamma @ coefficients)):
            scores.append({"contrast_id": spec["contrast_id"], "rq": spec["research_question"],
                           "estimand": estimand, **score_draws(draws, float(actual), math.log(1.25))})
    scores.append({"contrast_id": "rq6_gamma_size", "rq": "RQ6", "estimand": "unchanged_slope",
                   **score_draws(frame.gamma_size.to_numpy(), truth["extras"]["gamma_size"], math.log(1.05))})
    archives = []
    for archived_path in sorted((inputs.root / campaign).glob(f"*/iteration_{iteration}/true_parameters.json")):
        archived_truth = inputs.json(archived_path)
        require(canonical_truth(archived_truth) == canonical_truth(truth), "Replacement changed generating truth")
        archives.append(archived_path.relative_to(inputs.root).as_posix())
    data_paths = {row["data_file"] for row in metadata}
    require(len(data_paths) == 1, "Chains reference different inference data")
    reference = next(iter(data_paths))
    data_available = bool(reference and Path(reference).is_file())
    y_hash = None
    if data_available:
        data = inputs.json(Path(reference))
        for key in ("X", "I", "cell", "s"):
            require(np.array_equal(data[key], design[key]), f"Inference input geometry mismatch: {key}")
        require(len(data["y"]) == design["M_total"], "Inference y length mismatch")
        y_hash = digest(data["y"])
    common_truth = [truth["gamma0"], truth["sigma_cell"], truth["extras"]["gamma_size"], *truth["gamma"][:7]]
    return {"iteration": iteration, "simulation_seed": 12344 + iteration, "inference_seed": 54320 + iteration,
            "samples_per_chain": samples, "warmup_per_chain": samples, "chains": 4,
            "selected_chains": [path.relative_to(inputs.root).as_posix() for path in paths],
            "unselected_chain_files": extras, "sampler_eligible": eligibility_failure is None,
            "sampler_failure": eligibility_failure, "sampler_metrics": metrics,
            "summary_mean_max_abs_error": summary_mean_error,
            "log_alpha_max_abs_reconstruction_error": reconstruction_error,
            "truth_sha256": digest(truth), "common_generating_parameters_sha256": digest(common_truth),
            "realized_log_truth_sha256": digest(true_log.tolist()), "residual_z_sha256": digest(true_z.tolist()),
            "true_sigma_cell": truth["sigma_cell"], "archived_truth_matches": archives,
            "inference_data_available": data_available, "simulated_y_sha256": y_hash,
            "scores": scores}


def aggregate_iterations(iterations):
    scores = [score for row in iterations if "scores" in row for score in row["scores"]]
    groups = {}
    for estimand in sorted({score["estimand"] for score in scores}):
        selected = [score for score in scores if score["estimand"] == estimand]
        groups[estimand] = {}
        for research_question in sorted({score["rq"] for score in selected}):
            subset = [score for score in selected if score["rq"] == research_question]
            rope = math.log(1.05) if research_question == "RQ6" else math.log(1.25)
            distinct = [score for score in subset if score["contrast_id"] != "rq1_openai_flagship_minus_small"]
            groups[estimand][research_question] = {
                "named": summarize_scores(subset, rope),
                "distinct_up_to_sign": summarize_scores(distinct, rope),
                "by_contrast": {identifier: summarize_scores([score for score in subset if score["contrast_id"] == identifier], rope)
                                for identifier in sorted({score["contrast_id"] for score in subset})}}
        if estimand != "unchanged_slope":
            distinct = [score for score in selected if score["contrast_id"] != "rq1_openai_flagship_minus_small"]
            groups[estimand]["reviewer_pooled_distinct"] = summarize_scores(distinct, math.log(1.25))
    return groups


def reviewer_comparison(campaigns):
    expected = {
        ("venture", "additive_companion"): [0.15, 0.59, 0.75, 0.90],
        ("venture", "realized"): [0.10, 0.85, 0.99, 1.00],
        ("hiring", "additive_companion"): [0.14, 0.60, 0.79, 0.86],
        ("hiring", "realized"): [0.08, 0.84, 1.00, 1.00],
        ("matched_rq5", "additive_companion"): [0.17, 0.37, 0.82, 0.93],
    }
    comparisons = []
    for (name, estimand), rates in expected.items():
        if name not in campaigns or estimand not in campaigns[name]["summaries"]:
            continue
        summary = campaigns[name]["summaries"][estimand]["reviewer_pooled_distinct"]
        if "detection_bins" not in summary:
            continue
        bins = [summary["detection_bins"][label] for label in ("below_1.25", "1.25_to_1.5", "1.5_to_2", "above_2")]
        observed = [round(row["rate"], 2) if row["rate"] is not None else None for row in bins]
        comparisons.append({"campaign": name, "estimand": estimand, "review_rounded_rates": rates,
                            "audit_rounded_rates": observed, "agrees_at_reported_precision": observed == rates})

    def comparison_narrative(rows, expected_count):
        agreed = sum(row["agrees_at_reported_precision"] for row in rows)
        disagreed = sum(any(observed is not None and observed != expected_rate
                           for observed, expected_rate in zip(row["audit_rounded_rates"], row["review_rounded_rates"]))
                        for row in rows)
        insufficient = expected_count - agreed - disagreed
        status = "reproduced" if agreed == expected_count else "disagreed" if disagreed else "insufficient evidence"
        return (f"{status}: {agreed}/{expected_count} comparisons agreed at reported precision, "
                f"{disagreed} disagreed, {insufficient} with insufficient evidence.")

    primary = [row for row in comparisons if row["campaign"] != "matched_rq5"]
    matched = [row for row in comparisons if row["campaign"] == "matched_rq5"]
    return {"review_table_detection_comparisons": comparisons,
            "A1": "Reviewer RQ1/RQ2 detection " + comparison_narrative(primary, 4) + " Realized contrasts use canonical cell weights; additive companions remain distinct estimands.",
            "B4": "Additive RQ5 detection " + comparison_narrative(matched, 1) + " Realized RQ5 is a separate audit estimand, not supported by the reviewer's RQ5 table. Different truths put different contrast-iterations in effect bins, so bin-rate changes are not paired power gains at identical truths.",
            "B4_prior_geometry": {"additive_reference_task_prior_sd": 0.5,
                                  "additive_nonreference_task_prior_sd": math.sqrt(0.5),
                                  "realized_reference_task_marginal_prior_sd": math.sqrt(0.25 + 0.3 ** 2 * 2 / 3),
                                  "realized_nonreference_task_marginal_prior_sd": math.sqrt(0.5 + 0.3 ** 2 * 2 / 3),
                                  "method": "Independent gamma coefficients with SD .5 plus two three-cell residual means; marginal residual variance uses E[sigma_cell^2]=.3^2. Realized prior is a mixture, not exactly normal."},
            "B5": "Shared-truth conclusions depend on verified pair counts and shifted-residual comparisons in shared_truth_audit; missing evidence does not establish equality. No claim of independent datasets or truths across geometries.",
            "C1": "Current family is 26 named, 24 distinct up to sign, not the review's 25: one OpenAI sign reversal in each primary pool.",
            "not_audited": ["B4 consequence wording", "A3 ceiling/prior alternatives", "A4 predictive checks"]}


def shared_truth_findings(inputs, campaigns):
    paired = {}
    for left, right in (("venture", "hiring"), ("venture", "matched_rq5")):
        if left not in campaigns or right not in campaigns:
            continue
        pairs = [(first, second) for first, second in zip(campaigns[left]["iterations"], campaigns[right]["iterations"])
                 if "scores" in first and "scores" in second]
        paired[f"{left}_vs_{right}"] = {"paired_iterations": len(pairs), **{
            field: sum(first[field] == second[field] for first, second in pairs)
            for field in ("common_generating_parameters_sha256", "realized_log_truth_sha256", "residual_z_sha256")}}
    shifted_errors = []
    if "venture" in campaigns and "matched_rq5" in campaigns:
        verified = [set(row["iteration"] for row in campaigns[name]["iterations"] if "scores" in row)
                    for name in ("venture", "matched_rq5")]
        for iteration in sorted(verified[0] & verified[1]):
            residuals = []
            for name in ("venture", "matched_rq5"):
                directory = Path(campaigns[name]["source_root"])
                truth = inputs.json(directory / f"iteration_{iteration}/true_parameters.json")
                design = inputs.json(directory / "study_design.json")
                residuals.append((np.log(truth["alpha"]) - truth["gamma0"] - np.asarray(design["X"]) @ truth["gamma"]) / truth["sigma_cell"])
            shifted_errors.append(float(np.max(np.abs(residuals[0][6:] - residuals[1][:12]))))
    interpretations = []
    for name, counts in paired.items():
        total = counts["paired_iterations"]
        if not total:
            interpretations.append(f"{name}: insufficient evidence (0 verified pairs).")
            continue
        interpretations.append(f"{name}: {total} verified pairs.")
        for field, label in (("common_generating_parameters_sha256", "gamma0, sigma_cell, gamma_size and leading seven gamma coefficients"),
                             ("realized_log_truth_sha256", "full realized log truths"),
                             ("residual_z_sha256", "full residual vectors")):
            equal = counts[field]
            interpretations.append(f"{label}: {equal}/{total} equal, {total - equal}/{total} disagreed.")
    if not paired:
        interpretations.append("Insufficient evidence: 0 verified pairs across available geometries.")
    if shifted_errors:
        within = sum(error < 1e-5 for error in shifted_errors)
        interpretations.append(f"Shifted residual overlap: {within}/{len(shifted_errors)} within rounding tolerance; "
                               f"{len(shifted_errors) - within}/{len(shifted_errors)} disagreed.")
    else:
        interpretations.append("Shifted residual overlap: insufficient evidence (0 compared iterations).")
    return {"pairwise_equal_counts": paired,
            "shifted_residual_stream": {"mapping": "venture z[7:18] versus matched z[1:12] (1-based); six extra gamma RNG calls precede matched z",
                                        "compared_iterations": len(shifted_errors),
                                        "max_abs_difference": max(shifted_errors) if shifted_errors else None,
                                        "iterations_within_1e_minus5_rounding_tolerance": sum(error < 1e-5 for error in shifted_errors)},
            "interpretation": " ".join(interpretations) + " These comparisons do not establish independent truths across geometries or equality in unverified iterations.",
            "raw_simulated_choices_available": sum(row.get("inference_data_available", False) for camp in campaigns.values() for row in camp["iterations"]),
            "limitation": "Generation code supports conditional independent choices within a dataset, but coupled seeds prevent an independent-across-geometries claim. Missing temporary y inputs cannot be recovered from posterior predictive y_pred; these are not substituted."}


def validate_audit(report):
    require(report["family"]["named_decisions"] == 26 and report["family"]["distinct_up_to_sign"] == 24,
            "Incorrect family counts")
    selected_paths = []
    for name, campaign in report["campaigns"].items():
        rows = campaign["iterations"]
        require([row["iteration"] for row in rows] == list(range(1, 41)), "Missing iteration slots")
        require(campaign["verified_iterations"] == sum("scores" in row for row in rows), "Incorrect verified count")
        require(campaign["sampler_eligible_iterations"] == sum(row["sampler_eligible"] for row in rows), "Incorrect eligibility count")
        require(campaign["summaries"] == aggregate_iterations(rows), "Summaries differ from iteration scores")
        for row in rows:
            if "scores" not in row:
                require("blocker" in row, "Unexplained missing scores")
                continue
            expected_scores = 13 if name == "matched_rq5" else 19
            require(len(row["scores"]) == expected_scores, "Incomplete contrast scores")
            for path in row["selected_chains"]:
                require(path in report["input_sha256"], "Unhashed selected chain")
                selected_paths.append(path)
    require(len(selected_paths) == len(set(selected_paths)), "Chain reused between iterations")
    for path, value in report["input_sha256"].items():
        require(not Path(path).is_absolute() and ".." not in Path(path).parts, "Nonportable input path")
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Malformed input hash")


def check_output(destination, report):
    require(destination.is_file(), "Audit output missing")
    require(digest(json.loads(destination.read_text())) == digest(report), "Saved audit differs from recomputed inputs/results")


def build_audit(root=ROOT, workers=4):
    from applications.seu_sensitivity_study import confirmatory_analysis as contract
    from applications.seu_sensitivity_study.ceiling_prior import fit_plan, PRIOR_VARIANTS
    from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig

    inputs = Inputs(root)
    sources = ["analysis/verify_realized_recovery.py", "analysis/hierarchical_parameter_recovery.py",
               "scripts/run_hierarchical_parameter_recovery.py", "scripts/build_matched_rq5_validation_template.py",
               "utils/study_design_hierarchical.py", "models/h_m01_size_assessment_anchored.stan",
               "models/h_m01_size_assessment_anchored_sim.stan",
               "applications/seu_sensitivity_study/config.py", "applications/seu_sensitivity_study/schemas.py",
               "applications/seu_sensitivity_study/confirmatory_analysis.py",
               "applications/seu_sensitivity_study/ceiling_prior.py",
               "applications/seu_sensitivity_study/data_preparation.py",
               "reports/applications/seu_sensitivity_study/_build_evidence.py",
               "local/seu_sensitivity_precollection_final_review.md", "local/tmp/seu_review/oc.py"]
    for source in sources:
        inputs.record(Path(source))
    for path in sorted((root / "configs").glob("h_m01_size_assessment_anchored_*recovery*config.json")):
        inputs.record(path)
    plan = fit_plan(root / "applications/seu_sensitivity_study/results/production")
    base_plan = [{key: row[key] for key in ("group", "variant", "model", "utility_middle", "presentation_id")}
                 for row in plan["fits"] if row["variant"] not in PRIOR_VARIANTS]
    require(len(base_plan) == 15, "Historical base fit family changed")
    canonical, columns, _ = SEUSensitivityStudyConfig(pool_ids=["venture"]).design_matrix_for_pool("venture")
    manifest = contract.contract_manifest(columns, canonical)
    require(manifest["primary_decision_count"] == 26 and manifest["estimand_contract"]["distinct_primary_decisions_up_to_sign"] == 24, "Decision family changed")
    report = {"schema_version": 1, "scope": "OFFLINE saved-draw recovery; no sampling, collection, preflight or launch authorization",
              "methodology": {
                  "estimand_version": contract.ESTIMAND_VERSION,
                  "truth": "log(saved alpha_cell); z recovered as (log(alpha)-gamma0-X gamma)/sigma because simulator did not persist z. Limited by 8-significant-digit saved truth precision.",
                  "posterior": "gamma0 + X gamma + sigma_cell z_alpha, draw-wise, checked against saved log_alpha_cell to 2e-6; equally weighted named cells from Amendment 5; intercept and common-size slope cancel.",
                  "interval": "central 90%, numpy linear quantiles at .05/.95",
                  "decision": "lower>0 and median>log(1.25), or upper<0 and median<-log(1.25); RQ6 threshold log(1.05); strict inequalities; two-sided regardless of expected direction",
                  "error_metrics": "Bias and RMSE use posterior mean, matching historical exporter; posterior-median versions supplied separately.",
                  "rates": "Every rate carries numerator/denominator. Exact-null false positives only at truth==0; inside-ROPE detections are not null false positives. Power conditions on abs(truth)>ROPE. Type S counts wrong-sign detections at any nonzero truth, denominator detected nonzero truths.",
                  "aggregation": "Named RQ1 includes sign-reversed OpenAI duplicate; distinct summaries omit that name in each pool. No pooling across geometries as independent truths. Additive companions never mixed with realized decisions.",
                  "chain_selection": "Four diagnostics-referenced basenames; distinct IDs 1..4; seed 54320+iteration; 500/500 or recorded replacement 1000/1000; venture16 retains seed54336; compare means and sampler diagnostics to current posterior_summary.csv. Extras disclosed, never concatenated.",
                  "eligibility": "All 40 current iteration slots retained per campaign. Failed sampler gates are explicit and never silently dropped. Rhat/bulk/tail from full saved non-observation parameter summaries; divergence/depth/E-BFMI recomputed from selected draws.",
                  "independence": "Simulator makes a separate categorical_rng call per observation, with no copy mechanism. Same simulation seeds across geometries couple RNG streams: neither distinct y bytes nor different eta imply cross-campaign independence. No new RNG simulation performed.",
                  "environment": "Python 3.10+, numpy and pandas in existing seu-sensitivity conda environment; original ignored recovery/assessment/template artifacts and source files required. No Stan executable or provider access required.",
                  "check": "--check rebuilds the complete JSON from saved inputs and compares canonical content, including full-byte input SHA256 hashes; never writes output.",
              }, "family": {"named_decisions": 26, "distinct_up_to_sign": 24, "historical_base_fit_count": 15,
                            "base_fit_plan": base_plan, "excluded_prior_alternative_count": plan["additional_prior_fits"],
                            "rq3_rq4": "descriptive; no recovery decision family",
                            "matched_RQ6": "unchanged slope companion only; not one of 26 decisions",
                            "not_validated": ["A3 prior/ceiling sensitivity", "A4 predictive checks", "15 actual production fits", "24 current planned alternatives"]},
              "campaigns": {}, "blockers": []}
    for name, suffix in CAMPAIGNS.items():
        campaign = Path("results/parameter_recovery") / ("h_m01_size_assessment_anchored_" + suffix)
        try:
            design, ids, specifications, geometry = campaign_geometry(inputs, name, campaign)
            all_truths = inputs.json(campaign / "all_true_parameters.json")
            require(len(all_truths) == 40, "Expected 40 aggregate truths")
            for iteration, aggregate_truth in enumerate(all_truths, 1):
                require(canonical_truth(aggregate_truth) == canonical_truth(inputs.json(campaign / f"iteration_{iteration}/true_parameters.json")), "Aggregate/current generating truths differ")
        except (ValueError, KeyError, OSError) as error:
            report["blockers"].append({"campaign": name, "error": str(error)})
            continue
        def run_iteration(iteration):
            try:
                return verify_iteration(inputs, name, campaign, iteration, design, ids, specifications)
            except (ValueError, KeyError, OSError, AssertionError) as error:
                return {"iteration": iteration, "blocker": str(error), "sampler_eligible": False}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterations = list(executor.map(run_iteration, range(1, 41)))
        for row in iterations:
            if "blocker" in row:
                report["blockers"].append({"campaign": name, "iteration": row["iteration"], "error": row["blocker"]})
        report["campaigns"][name] = {"source_root": campaign.as_posix(), "geometry": geometry,
                                     "expected_iterations": 40, "verified_iterations": sum("scores" in row for row in iterations),
                                     "sampler_eligible_iterations": sum(row["sampler_eligible"] for row in iterations),
                                     "contrasts": specifications, "iterations": iterations,
                                     "summaries": aggregate_iterations(iterations)}
        print(f"{name}: {report['campaigns'][name]['verified_iterations']}/40 verified, "
              f"{report['campaigns'][name]['sampler_eligible_iterations']}/40 sampler eligible", flush=True)
    campaigns = report["campaigns"]
    report["shared_truth_audit"] = shared_truth_findings(inputs, campaigns)
    report["reviewer_comparison"] = reviewer_comparison(campaigns)
    report["provenance_blockers"] = [{
        "claim_not_established": "Exact regeneration of every saved simulated choice vector from its recorded seed; independent simulated choices across geometries",
        "missing_inference_data_count": sum("scores" in row and not row["inference_data_available"]
                                            for campaign in campaigns.values() for row in campaign["iterations"]),
        "reason": "Original temporary inference JSONs and simulator CSVs are not retained at referenced paths. Source/config/truth agreement is verified, but y equality on reruns is not directly verified. Coupled seeds additionally rule out claiming independently seeded geometries.",
        "effect_on_saved_draw_recovery": "Does not prevent contrasts of verified saved posterior draws against canonical saved truths; bounds the generative provenance claim."}]
    report["limitations"] = ["Simulation-model recovery under fitting priors and fixed assessed utilities, not real production choice behavior or arbitrary ceiling/misfit validation.",
                              "40 seed/truth clusters; coarse rates and small effect bins, correlated contrasts and geometries; no independent-binomial precision claims.",
                              "Saved generating z is reconstructed from rounded alpha, not independently observed.",
                              "Input manifests bind available historical artifacts and current generating source; absent original simulation CSV/y means exact seed-to-y replay cannot be established offline without simulation."]
    report["status"] = "complete_saved_draw_verification" if not report["blockers"] else "partial_with_blockers"
    report["input_sha256"] = dict(sorted(inputs.hashes.items()))
    validate_audit(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and compare; write nothing")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent read-only iteration readers")
    args = parser.parse_args()
    require(1 <= args.workers <= 8, "workers must be between 1 and 8")
    report = build_audit(workers=args.workers)
    destination = ROOT / OUTPUT
    if args.check:
        check_output(destination, report)
        print("Exact saved-input reproduction passed", flush=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"Wrote {OUTPUT} ({destination.stat().st_size} bytes)", flush=True)
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
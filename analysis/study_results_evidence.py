"""Generated evidence for the SEU sensitivity study results report (no hand-typed numbers).

Two stages, following the design report's exporter:

* ``--refresh`` reads the local, gitignored result artifacts once (the hash-verified complete confirmatory
  report, its manifests, the 24 fit records, the post-collection audit, the collection ledger, the venture
  primary attempt records and the position-stable subset fits) and writes the compact portable supplement
  ``reports/applications/seu_sensitivity_study/data/results_evidence.json``.
* ``--write`` renders the report includes (``_results/*.qmd``), the prose values file
  (``data/results_values.yml``) and the figures from that supplement alone. ``--check`` (the default)
  regenerates them in memory and requires byte-identical text and identical figure-input digests.
  ``--check-sources`` additionally re-extracts from the local sources and requires an identical supplement.

Run from the repository root:
  conda run --no-capture-output -n seu-sensitivity python -m analysis.study_results_evidence [--refresh|--write|--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "applications/seu_sensitivity_study"
RESULTS = STUDY / "results"
COMPLETE = RESULTS / "complete_report_20261102_v1"
AUDIT = RESULTS / "post_collection_audit_20261010_v1"
CAMPAIGN = RESULTS / "production_20261010"
STABLE = RESULTS / "position_stable_fits_20261104_v1"
TRACE = RESULTS / "trace_inspection_20261011_v1"
MIGRATION = ROOT / "local/migration_20261101"
ARM_EQUIVALENCE = RESULTS / "prior_equivalence_arm64_20261102_v1"
REPORT = ROOT / "reports/applications/seu_sensitivity_study"
SUPPLEMENT = REPORT / "data/results_evidence.json"
SCHEMA = "seu_results_evidence_v1"

GROUPS = ("venture", "hiring", "matched_rq5")
POOLS = ("venture", "hiring")
VARIANTS = ("primary", "presentation_1_only", "presentation_2_only", "utility_035", "utility_065",
            "prior_L", "prior_H", "prior_S")
PRIOR_VARIANTS = ("prior_L", "prior_H", "prior_S")
MODELS = ("gpt-4o", "gpt-4o-mini", "o3-mini", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-sonnet-4-5-thinking")
PROMPTS = ("neutral", "seu_maximizing", "deliberative")
VENTURE_ATTEMPTS = ("benchmark_venture_primary_20261010_v1", "benchmark_venture_primary_20261011_retry01",
                    "benchmark_venture_primary_20261011_long01")
STATISTICS = ("maximizer_fraction", "mean_regret", "filler_fraction", "mean_selected_probability", "mean_log_score")
SLICE_CELLS = ("gpt_4o_neutral_hiring", "gpt_4o_seu_maximizing_hiring", "gpt_4o_deliberative_hiring",
               "o3_mini_seu_maximizing_hiring", "claude_sonnet_4_5_thinking_deliberative_hiring",
               "gpt_4o_neutral_venture", "claude_haiku_4_5_neutral_hiring", "claude_haiku_4_5_neutral_venture")
SUMMARY_FIELDS = ("median", "lower_90", "upper_90", "mean")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def load(path: Path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pick(row, fields=SUMMARY_FIELDS):
    return {field: row[field] for field in fields if field in row}


class Sources:
    """Read local artifacts and record the SHA-256 of every file consumed."""

    def __init__(self):
        self.hashes = {}

    def json(self, path: Path):
        self.hashes[rel(path)] = digest(path)
        return load(path)

    def touch(self, path: Path):
        self.hashes[rel(path)] = digest(path)


def cell_metadata():
    from applications.seu_sensitivity_study.config import build_cells

    return {cell.cell_id: {"model": cell.model_name, "prompt": cell.prompt_condition, "pool": cell.pool_id}
            for cell in build_cells(list(POOLS))}


def task_of(group, cell):
    if group != "matched_rq5":
        return group
    return "hiring" if cell["pool"] == "hiring" else "procurement"


def fit_directories(fit_manifest):
    directories = {}
    for group, variants in fit_manifest["fits"].items():
        for variant, entry in variants.items():
            parts = Path(entry["chain_path"]).parts
            directories[(group, variant)] = ROOT.joinpath(*parts[parts.index("applications"):]).parent
    require(sorted(directories) == sorted((g, v) for g in GROUPS for v in VARIANTS),
            "Fit manifest must contain exactly the 24 planned fits")
    return directories


def sampler_row(diagnostics):
    return {key: diagnostics[key] for key in ("max_rhat", "min_ess_bulk", "min_ess_tail", "min_ebfmi",
                                               "divergences", "treedepth_saturated_share")}


def extract_fits(sources, fit_manifest):
    rows = []
    for (group, variant), directory in sorted(fit_directories(fit_manifest).items(),
                                              key=lambda item: (GROUPS.index(item[0][0]), VARIANTS.index(item[0][1]))):
        result = sources.json(directory / "result.json")
        data = sources.json(directory / "stan_data_size.json")
        preparation = sources.json(directory / "stan_data_size_assembly_report.json")
        manifest = sources.json(directory / "launch_manifest.json")
        settings = manifest["settings"]
        require(result["status"] == "completed_sampler_passed", f"{group}/{variant} is not sampler-valid")
        rows.append({"group": group, "variant": variant, "observations": data["M_total"],
                     "cells": data["J"], "mean_menu_size": preparation["mean_menu_size"],
                     # Archived fits predate the toolchain record and ran on the original x86_64 Mac.
                     "machine": result.get("toolchain_machine", "x86_64"),
                     "sample_minutes": result["sample_seconds"] / 60, "draws": result["posterior_draws"],
                     "settings": {key: settings[key] for key in ("chains", "iter_warmup", "iter_sampling",
                                                                 "adapt_delta", "max_treedepth", "seed")},
                     **sampler_row(result["diagnostics"])})
    return rows


def extract_attempts(sources):
    rows = []
    for name in VENTURE_ATTEMPTS:
        directory = RESULTS / name
        manifest = sources.json(directory / "launch_manifest.json")
        result = sources.json(directory / "result.json")
        if (directory / "diagnostic_supplement.json").exists():
            supplement = sources.json(directory / "diagnostic_supplement.json")
            diagnostics, error = supplement["diagnostics"], supplement["sampler_gates"].get("error")
        else:
            diagnostics, error = result["diagnostics"], result.get("sampler_error")
        settings = manifest["settings"]
        rows.append({"attempt": name, "created_at": manifest["created_at"], "status": result["status"],
                     "adapt_delta": settings["adapt_delta"], "iter_warmup": settings["iter_warmup"],
                     "iter_sampling": settings["iter_sampling"], "seed": settings["seed"],
                     "gate_error": error, "max_rhat": diagnostics.get("max_rhat"),
                     "divergences": diagnostics.get("divergences")})
    sources.touch(TRACE / "ASSESSMENT.md")
    return rows


def extract_collection(sources):
    audit = sources.json(AUDIT / "audit_summary.json")
    ledger = sources.json(CAMPAIGN / "batch_budget_report.json")
    checks = sorted((CAMPAIGN / "reconciliation").glob("provider_check_*.json"))
    require(len(checks) == 1, "Expected exactly one provider reconciliation check")
    provider = sources.json(checks[0])
    attempts = ledger["attempts"]
    ambiguous = [row for row in attempts if row["status"] != "completed"]
    require(len(ambiguous) == 1 and ambiguous[0]["status"] == "submission_ambiguous",
            "Expected exactly one ambiguous submission")
    replacement = [row for row in attempts if row["cell_id"] == ambiguous[0]["cell_id"] and row["status"] == "completed"]
    require(len(replacement) == 1, "Expected exactly one completed replacement")
    remote = provider["cells"][ambiguous[0]["cell_id"]]
    cells = audit["cells"]
    return {
        "cells": len(cells),
        "records": sum(row["replayed_records"] for row in cells),
        "unresolved_by_cell": {row["cell_id"]: row["unresolved"] for row in cells if row["unresolved"]},
        "unresolved_by_size": audit["unresolved_by_size"],
        "unresolved_by_finish_reason": audit["unresolved_by_finish_reason"],
        "provider_finish_reasons": audit["provider_finish_reasons"],
        "variants": {f"{row['group']}/{row['variant']}": {key: row[key] for key in (
            "observations", "cells", "rank", "excluded_cells", "excluded_observations", "retained_by_size")}
            for row in audit["variants"]},
        "usage_estimated_usd": audit["usage_estimated_usd"],
        "retained_reservation_usd": audit["retained_reservation_usd"],
        "budget_ceiling_usd": ledger["budget_ceiling_usd"],
        "completed_batches": sum(row["status"] == "completed" for row in attempts),
        "ambiguous_submission": {"cell_id": ambiguous[0]["cell_id"], "recorded_at": ambiguous[0]["recorded_at"],
                                 "reservation_usd": ambiguous[0]["reservation_usd"],
                                 "error": remote["local_state"].get("submission_error"),
                                 "provider_matching_batches": len(remote["matching_batches"]),
                                 "provider_checked_at": provider["checked_at"]},
        "replacement_recorded_at": replacement[0]["recorded_at"],
        "stability": {pool: {key: value for key, value in audit["stability"][pool].items() if key != "problem_ids"}
                      for pool in POOLS},
    }


def contributing_flags(a4):
    """Cells whose overall same-item repetition exceeds the predictive band (existing A4 flags)."""
    return sorted(row["id"] for row in a4["pairs"]
                  if row["scope"] == "cell" and "menu_size" not in row and "family" not in row
                  and row["same_item"].get("descriptive_review_flag")
                  and row["same_item"]["difference_from_predictive_median"] > 0)


def decision_rows(section, probabilities, flagged_cells):
    rows = []
    for row in section["rows"]:
        identifier = row.get("contrast_id") or row.get("parameter_id")
        entry = {"id": identifier, "research_question": row.get("research_question", "RQ6"),
                 "decision": row["decision"], "interpretation": row["substantive_interpretation"],
                 "rope_half_width": row["rope_half_width"], "status": row.get("status", "available"),
                 **pick(row), **probabilities.get(identifier, {})}
        if "dependence_qualification" in row:
            contributing = sorted(row["dependence_qualification"]["contributing_cell_ids"])
            entry["repetition_flagged_cells"] = [cell for cell in contributing if cell in flagged_cells]
            entry["contributing_cells"] = len(contributing)
        rows.append(entry)
    return rows


def prior_probabilities(prior_group):
    comparisons = prior_group["variants"]["prior_L"]["comparisons"]
    return {row["contrast_id"]: {"probability_positive": row["primary"]["probability_positive"],
                                 "probability_negative": row["primary"]["probability_negative"]}
            for row in comparisons}


def fit_section(report, group, variant):
    if variant in PRIOR_VARIANTS:
        return report["ceiling_prior_sensitivity"]["groups"][group]["variants"][variant]["report"]
    return (report["matched_rq5"] if group == "matched_rq5" else report["pools"][group])[variant]


def comparison_rows(comparison, variants):
    rows = []
    for row in comparison["rows"]:
        rows.append({"id": row["contrast_id"], "sign_changed": row["sign_changed"],
                     "decision_changed": row["interval_decision_changed"],
                     "interpretation_changed": row["substantive_interpretation_changed"],
                     "estimates": {variant: {"decision": row["estimates"][variant]["decision"],
                                             **pick(row["estimates"][variant])} for variant in variants}})
    return rows


def a4_available(row, name):
    value = row.get("statistics", row).get(name) or {}
    return value.get("status") == "available"


def flag_counts(a4):
    """Flagged and evaluated rows per qualification, using the predictive_checks.interpretation rules."""
    def count(section, fields):
        evaluated = flagged = 0
        for row in a4[section]:
            for name in fields:
                if a4_available(row, name):
                    evaluated += 1
                    flagged += bool(row.get("statistics", row)[name].get("descriptive_review_flag"))
        return evaluated, flagged

    trend = count("trends", ("maximizer_fraction", "mean_regret", "filler_fraction"))
    groups = count("groups", ("maximizer_fraction", "mean_regret", "filler_fraction"))
    items = count("items", ("conditional_rate", "unconditional_share"))
    positions = count("positions", ("fraction",))
    pairs_evaluated = sum(a4_available(row, "same_item") for row in a4["pairs"])
    pairs_excess = sum(1 for row in a4["pairs"] if a4_available(row, "same_item")
                       and row["same_item"]["descriptive_review_flag"]
                       and row["same_item"]["difference_from_predictive_median"] > 0)
    return {"trend": {"evaluated": trend[0], "flagged": trend[1]},
            "choice_pattern": {"evaluated": groups[0] + items[0], "flagged": groups[1] + items[1]},
            "repetition": {"evaluated": pairs_evaluated, "flagged": pairs_excess},
            "position": {"evaluated": positions[0], "flagged": positions[1]}}


def statistic_summary(value):
    if value.get("status") != "available":
        return {"status": value.get("status")}
    out = {"flag": value["descriptive_review_flag"], "replicated": value["replicated"],
           "draw_dependent": value["draw_dependent"], "structural_zero": value["structural_zero"]}
    if value["draw_dependent"]:
        out["observed"] = value["observed"]
        out["observed_minus_replicated"] = value["observed_minus_replicated"]
    else:
        out["observed"] = value["observed"]
    return out


def extract_a4(report, summary):
    out = {"flags": {}, "pool": {}, "cells": {}, "positions": {}, "missingness": {}}
    for group in GROUPS:
        for variant in VARIANTS:
            a4 = report["posterior_predictive_checks"][group][variant]["a4"]
            require(summary["predictive_qualifications"][group][variant]["status"] == "descriptive",
                    f"{group}/{variant}: predictive qualifications unavailable")
            counts = flag_counts(a4)
            recorded = summary["predictive_qualifications"][group][variant]["qualifications"]
            for key, name in (("trend", "behavioral_trend_not_reproduced"), ("choice_pattern", "choice_pattern_discrepancies"),
                              ("repetition", "excess_same_item_repetition"), ("position", "display_position_discrepancies")):
                require(counts[key]["flagged"] == recorded[name]["flagged_rows"],
                        f"{group}/{variant}/{key}: recount differs from the recorded qualification count")
            out["flags"][f"{group}/{variant}"] = {
                **counts, "draws": a4["draw_count"],
                "unavailable_sizes_or_pairs": recorded["unavailable_sizes_or_pairs"]["applies"],
                "conditional_on_retention": recorded["conditional_on_retention"]["applies"]}
        a4 = report["posterior_predictive_checks"][group]["primary"]["a4"]
        pool_rows = []
        for row in a4["groups"]:
            if row["scope"] != "pool_task" or "family" in row:
                continue
            pool_rows.append({"task": row["pool_id"], "menu_size": row.get("menu_size"),
                              "observations": row["observation_count"],
                              "statistics": {name: statistic_summary(row["statistics"][name]) for name in STATISTICS}})
        pairs = [{"task": row["pool_id"], "complete_pairs": row["complete_pair_count"],
                  "same_item": statistic_summary(row["same_item"]), "same_position": statistic_summary(row["same_position"])}
                 for row in a4["pairs"] if row["scope"] == "pool_task" and "family" not in row and "menu_size" not in row]
        out["pool"][group] = {"groups": pool_rows, "pairs": pairs,
                              "trends": [{"task": row["pool_id"], "family": row["family"],
                                          "stratum": row["difficulty_stratum"],
                                          "statistics": {name: statistic_summary(row["statistics"][name])
                                                         for name in ("maximizer_fraction", "mean_regret", "filler_fraction")}}
                                         for row in a4["trends"] if row["scope"] == "pool_task"]}
        out["positions"][group] = [
            {"task": row["pool_id"], "menu_size": row["menu_size"], "presentation": row["presentation_id"],
             "position": row["position"], "observations": row["observation_count"],
             "fraction": statistic_summary(row["fraction"])}
            for row in a4["positions"] if row["scope"] == "pool_task"]
        cells = {}
        for row in a4["groups"]:
            if row["scope"] == "cell" and "menu_size" not in row and "family" not in row:
                cells[row["id"]] = {"observations": row["observation_count"],
                                    **{name: statistic_summary(row["statistics"][name])
                                       for name in ("maximizer_fraction", "mean_regret", "filler_fraction")}}
        for row in a4["pairs"]:
            if row["scope"] == "cell" and "menu_size" not in row and "family" not in row:
                cells[row["id"]]["same_item"] = statistic_summary(row["same_item"])
                cells[row["id"]]["complete_pairs"] = row["complete_pair_count"]
        out["cells"][group] = cells
        missing = a4["missingness"]
        out["missingness"][group] = {
            "cells": {row["id"]: {key: row[key] for key in ("eligible", "retained", "unresolved", "resolved_whole_cell_removed")}
                      for row in missing["groups"] if row["scope"] == "cell"
                      and not any(key in row for key in ("family", "menu_size", "presentation_id"))},
            "cell_size_unresolved": {row["id"]: {size: value["unresolved"] for size, value in row["by_size"].items()}
                                     for row in missing["size_unresolved"] if row["scope"] == "cell"},
        }
    return out


def extract_ceiling(report, metadata):
    out = {}
    for group in GROUPS:
        prior_group = report["ceiling_prior_sensitivity"]["groups"][group]
        cells = {}
        for cell_id, value in report["ceiling_diagnostics"][group]["cells"].items():
            individual = value["individual"]
            regrets = [row["regret"] for row in individual]
            by_size = {size: {"observations": entry["observations"],
                              "maximizer_choices": entry["exact"]["maximizer_choices"],
                              "below_maximum_choices": entry["exact"]["below_maximum_choices"],
                              "median_top_two_gap": entry["top_two_gap"]["q50"]}
                       for size, entry in value["by_size"].items()}
            likelihood = value["likelihood"]
            quantiles = {"primary": prior_group["primary_cell_quantiles"][cell_id]}
            for variant in PRIOR_VARIANTS:
                quantiles[variant] = prior_group["variants"][variant]["cell_quantiles"][cell_id]
            cells[cell_id] = {
                **metadata[cell_id], "task": task_of(group, metadata[cell_id]),
                "observations": len(individual), "distinct_menus": value["distinct_menu_ids"],
                "exclusions": value["exclusions"], "exact": value["exact"],
                "near_maximizer_choices": sum(row["chosen_near_maximizer"] for row in individual),
                "mean_regret": statistics.fmean(regrets), "max_regret": max(regrets),
                "by_size": by_size, "classification": likelihood["classification"],
                "analytic_limit": likelihood.get("analytic_limit"), "quantiles": quantiles,
            }
            if group == "matched_rq5" and cell_id in SLICE_CELLS:
                median_slice = next(entry for entry in likelihood["slices"] if entry["slope_quantile"] == "q50")
                cells[cell_id]["slice"] = {
                    "grid": [round(x, 6) for x in median_slice["grid"]],
                    "log_likelihood_difference": [round(x, 6) for x in median_slice["log_likelihood_difference"]],
                    "slope": median_slice["slope"], "grid_maximum_t": median_slice["grid_maximum_t"],
                    "status": median_slice["status"]}
        out[group] = {"cells": cells,
                      "size_slope_quantiles": {"primary": prior_group["primary_size_slope_quantiles"],
                                               **{variant: prior_group["variants"][variant]["size_slope_quantiles"]
                                                  for variant in PRIOR_VARIANTS}}}
    return out


def extract_stable(sources):
    report = sources.json(STABLE / "stable_report.json")
    sources.touch(STABLE / "authorization.json")
    out = {"interpretation": report["interpretation"], "pools": {}}
    for pool in POOLS:
        value = report["pools"][pool]
        result = sources.json(STABLE / pool / "position_stable" / "result.json")
        preparation = sources.json(STABLE / pool / "position_stable" / "stan_data_size_assembly_report.json")
        manifest = sources.json(STABLE / pool / "position_stable" / "launch_manifest.json")
        choices = {}
        for row in preparation["observations"]:
            choices.setdefault(row["problem_id"], set()).add(row["chosen_item_id"])
        stability = value["stability_subset"]
        out["pools"][pool] = {
            "observations": value["observations"], "menus": len(stability["problem_ids"]),
            "unanimous_menus": sum(len(items) == 1 for items in choices.values()),
            "retention": {key: stability[key] for key in stability if key != "problem_ids"},
            "machine": result["toolchain_machine"], "sample_minutes": result["sample_seconds"] / 60,
            "draws": result["posterior_draws"],
            "settings": {key: manifest["settings"][key] for key in ("chains", "iter_warmup", "iter_sampling",
                                                                   "adapt_delta", "max_treedepth", "seed")},
            **sampler_row(result["diagnostics"]),
            "sigma_cell": pick(value["report"]["rq3"]["sigma_cell"]),
            "comparison": comparison_rows(value["comparison"], ("primary", "position_stable_subset")),
        }
    return out


def extract(sources=None):
    sources = sources or Sources()
    artifact = sources.json(COMPLETE / "artifact_manifest.json")
    for name in ("complete_report.json", "summary.json", "fit_manifest.json", "input_manifest.json"):
        require(digest(COMPLETE / name) == artifact[name], f"{name} differs from the complete-report artifact manifest")
    report = sources.json(COMPLETE / "complete_report.json")
    summary = sources.json(COMPLETE / "summary.json")
    fit_manifest = sources.json(COMPLETE / "fit_manifest.json")
    sources.touch(COMPLETE / "input_manifest.json")
    acceptance = sources.json(MIGRATION / "migration_acceptance.json")
    equivalence = sources.json(ARM_EQUIVALENCE / "arm64_record.json")
    metadata = cell_metadata()
    multiplicity = report["multiplicity"]
    require(multiplicity["primary_decision_count"] == 26 and multiplicity["available_primary_decision_count"] == 26,
            "Expected 26 available named primary decisions")

    a4 = extract_a4(report, summary)
    decisions, companions, rq3, sonnet, scale, sensitivity, prior = {}, {}, {}, {}, {}, {}, {}
    for group in GROUPS:
        section = report["matched_rq5"] if group == "matched_rq5" else report["pools"][group]
        prior_group = report["ceiling_prior_sensitivity"]["groups"][group]
        flagged = contributing_flags(report["posterior_predictive_checks"][group]["primary"]["a4"])
        decisions[group] = decision_rows(section["primary"], prior_probabilities(prior_group), set(flagged))
        companions[group] = [{"id": row["contrast_id"], **pick(row)} for row in section["primary"]["gamma_companion"]["rows"]]
        value = section["primary"]["rq3"]
        rq3[group] = {"sigma_cell": pick(value["sigma_cell"]),
                      "residuals": {row["cell_id"]: pick(row) for row in value["cell_residuals"]},
                      "dids": [{"id": row["contrast_id"], "model": row["model"], "prompt": row["prompt"],
                                **({"task": row["task"]} if "task" in row else {}), **pick(row)}
                               for row in value["model_by_prompt_dids"]]}
        scale_section = section["primary"]["assessment_scale"]
        scale[group] = {"rq6_slope_unchanged": scale_section["rq6"].get("slope_unchanged"),
                        "rows": [{"id": row["contrast_id"], "research_question": row["research_question"],
                                  "offset": row["deterministic_offset"], "sign_reversal": row["sign_reversal"],
                                  "zero_exclusion_changed": row["interval_zero_exclusion_changed"],
                                  "zero_sd_cells": row["zero_sd_cell_ids"], "status": row["status"],
                                  "original": pick(row["original"]), "standardized": pick(row["standardized"])}
                                 for row in scale_section["rows"]]}
        sensitivity[group] = {
            "presentation": comparison_rows(section["presentation_sensitivity"],
                                            ("primary", "presentation_1_only", "presentation_2_only")),
            "utility": comparison_rows(section["utility_sensitivity"], ("utility_035", "primary", "utility_065"))}
        prior[group] = {
            "annotations": {row["contrast_id"]: row["interpretation"] for row in prior_group["annotations"]},
            "variants": {variant: {"status": prior_group["variants"][variant]["status"],
                                   "comparisons": [{"id": row["contrast_id"], "decision_changed": row["decision_rule_changed"],
                                                    "sign_changed": row["median_sign_changed"],
                                                    "median_shift": row["median_shift"],
                                                    "width_ratio": row["interval_width_ratio"],
                                                    "alternative": {"decision": row["alternative"]["decision"],
                                                                    **pick(row["alternative"])}}
                                                   for row in prior_group["variants"][variant]["comparisons"]]}
                         for variant in PRIOR_VARIANTS}}
        if group in POOLS:
            sonnet[group] = {}
            for variant in VARIANTS:
                value = fit_section(report, group, variant)["sonnet_thinking_descriptive"]
                sonnet[group][variant] = {**pick(value), "probability_negative": value["probability_negative"],
                                          "probability_positive": value["probability_positive"],
                                          "ratio": pick(value["geometric_mean_sensitivity_ratio"])}

    rq4 = {"combination": report["rq4"]["independent_posterior_combination"],
           "rows": [{"id": row["contrast_id"], "venture": pick(row["venture"]), "hiring": pick(row["hiring"]),
                     "difference": pick(row["hiring_minus_venture"]),
                     "probability_same_sign": row["posterior_probability_same_sign"],
                     "median_sign_agrees": row["median_sign_agrees"]} for row in report["rq4"]["rows"]],
           "orderings": [{"first": row["first_model"], "second": row["second_model"],
                          "venture": {**pick(row["venture"]),
                                      "probability_first_greater": row["venture"]["probability_first_model_greater"]},
                          "hiring": {**pick(row["hiring"]),
                                     "probability_first_greater": row["hiring"]["probability_first_model_greater"]},
                          "probability_same_sign": row["posterior_probability_same_sign"],
                          "median_sign_agrees": row["median_sign_agrees"]} for row in report["rq4"]["model_orderings"]]}
    prior_rq4 = {variant: {"orderings_agree": sum(row["median_sign_agrees"]
                                                  for row in report["ceiling_prior_sensitivity"]["rq4"][variant]["model_orderings"]),
                           "orderings": len(report["ceiling_prior_sensitivity"]["rq4"][variant]["model_orderings"])}
                 for variant in PRIOR_VARIANTS}

    evidence = {
        "schema": SCHEMA,
        "status": "Generated from local result artifacts by analysis/study_results_evidence.py --refresh; do not edit.",
        "multiplicity": {key: multiplicity[key] for key in (
            "primary_decision_count", "available_primary_decision_count", "unavailable_primary_decision_count",
            "distinct_primary_decisions_up_to_sign", "adjustment", "family_definition")},
        "estimand_contract": {key: report["estimand_contract"][key] for key in (
            "version", "primary_estimand", "rq1_weights", "rq2_weights", "rq5_weights", "rq4_estimand",
            "size_reference", "fit_count", "base_fit_count", "additional_prior_fit_count")},
        "report_substitutions": report["report_substitutions"],
        "summary_qualifications": summary["qualifications"],
        "cross_checks": summary["cross_checks"],
        "a4_probes": {key: {"probe_draws": value["probe_draws"], "probe_exact": value["probe_exact"],
                            "draws": value["draws"]} for key, value in summary["a4_records"].items()},
        "migration": {"accepted_result": acceptance["accepted_result"], "qualification": acceptance["qualification"],
                      "created_at": acceptance["created_at"]},
        "prior_equivalence": {"status": equivalence["status"]},
        "collection": extract_collection(sources),
        "fits": extract_fits(sources, fit_manifest),
        "venture_attempts": extract_attempts(sources),
        "decisions": decisions, "gamma_companions": companions, "rq3": rq3, "rq4": rq4, "prior_rq4": prior_rq4,
        "sonnet": sonnet, "assessment_scale": scale, "sensitivity": sensitivity, "prior": prior,
        "prior_status": {"status": report["ceiling_prior_sensitivity"]["status"],
                         "complete": report["ceiling_prior_sensitivity"]["complete_fit_count"],
                         "required": report["ceiling_prior_sensitivity"]["required_fit_count"]},
        "ceiling": extract_ceiling(report, metadata),
        "a4": a4,
        "stable": extract_stable(sources),
    }
    evidence["sources"] = dict(sorted(sources.hashes.items()))
    evidence["source_count"] = len(evidence["sources"])
    return evidence


def serialize(evidence) -> str:
    return json.dumps(evidence, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="Read local sources and replace the supplement")
    mode.add_argument("--write", action="store_true", help="Render includes, values and figures from the supplement")
    mode.add_argument("--check", action="store_true", help="Verify generated outputs from the supplement (default)")
    parser.add_argument("--check-sources", action="store_true", help="Also re-extract from local sources")
    args = parser.parse_args(argv)
    from analysis import study_results_render as render

    if args.refresh:
        SUPPLEMENT.write_text(serialize(extract()))
        print(f"Wrote {rel(SUPPLEMENT)}", flush=True)
        return
    evidence = json.loads(SUPPLEMENT.read_text())
    require(evidence.get("schema") == SCHEMA, "Unknown supplement schema")
    if args.write:
        render.write_outputs(evidence)
        print("Wrote includes, values and figures", flush=True)
    render.check_outputs(evidence)
    if args.check_sources:
        require(serialize(extract()) == SUPPLEMENT.read_text(), "Supplement differs from re-extracted local sources")
        print("Supplement reproduced from local sources", flush=True)
    print("Results evidence validated", flush=True)


if __name__ == "__main__":
    main()

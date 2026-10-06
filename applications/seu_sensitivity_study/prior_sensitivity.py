"""A3 comparisons of independent fits, kept outside the primary family."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .ceiling_prior import POLICY_VERSION, PRIOR_VARIANTS


def _with_sign_probabilities(report, payload, contrasts):
    from . import confirmatory_analysis as analysis

    report = deepcopy(report)
    values, group = analysis._realized_fit_values(
        payload["gamma_draws"], payload["sigma_cell_draws"], payload["z_alpha_draws"], contrasts, payload["cell_ids"])
    by_name = {contrast.contrast_id: contrast for contrast in contrasts}
    for row in report["rows"]:
        name = row.get("contrast_id", row.get("parameter_id"))
        if name == "rq6_gamma_size":
            draws = np.asarray(payload["gamma_size_draws"])
        else:
            draws, _ = analysis._cell_contrast_values(values, payload["cell_ids"], by_name[name].realized_cell_weights[group])
        row["probability_positive"] = float(np.mean(draws > 0)) if draws is not None else None
        row["probability_negative"] = float(np.mean(draws < 0)) if draws is not None else None
    return report


def compare_rows(primary, alternative):
    baseline = {row.get("contrast_id", row.get("parameter_id")): row for row in primary["rows"]}
    comparisons = []
    for row in alternative["rows"]:
        name = row.get("contrast_id", row.get("parameter_id"))
        original = baseline[name]
        result = {"contrast_id": name, "primary": original, "alternative": row,
                  "included_in_primary_family": False, "status": "sensitivity_annotation"}
        if original.get("status") == "unavailable" or row.get("status") == "unavailable":
            result.update(status="unavailable", reason="Required cells missing; no weight renormalization")
        else:
            width = original["upper_90"] - original["lower_90"]
            result.update(
                median_shift=row["median"] - original["median"],
                lower_endpoint_shift=row["lower_90"] - original["lower_90"],
                upper_endpoint_shift=row["upper_90"] - original["upper_90"],
                interval_width_ratio=(row["upper_90"] - row["lower_90"]) / width if width > 0 else None,
                width_ratio_status="available" if width > 0 else "undefined_zero_primary_width",
                median_sign_changed=row["median_sign"] != original["median_sign"],
                zero_exclusion_changed=((row["lower_90"] > 0 or row["upper_90"] < 0)
                                        != (original["lower_90"] > 0 or original["upper_90"] < 0)),
                decision_rule_changed=row["decision"] != original["decision"],
            )
        comparisons.append(result)
    return comparisons


def sensitivity_report(primary_reports, fits, contrasts):
    from . import confirmatory_analysis as analysis

    groups = ("venture", "hiring", "matched_rq5")
    if fits is not None and set(fits) != set(groups):
        raise ValueError("Prior sensitivity requires all three group statuses")
    output = {"policy_version": POLICY_VERSION, "decision_count": 0,
              "included_in_primary_family": False, "cross_prior_draw_pairing": False,
              "assessment_scale_grid": False, "groups": {}, "rq4": {}}
    valid = {}
    for group in groups:
        supplied = fits[group] if fits is not None else {}
        if set(supplied) - {"primary", *PRIOR_VARIANTS}:
            raise ValueError("Unexpected crossed prior variant")
        entries = {}
        valid[group] = {}
        primary = (_with_sign_probabilities(primary_reports[group], supplied["primary"]["payload"], contrasts[group])
               if "primary" in supplied else primary_reports[group])
        for variant in PRIOR_VARIANTS:
            entry = supplied.get(variant, {"status": "missing", "reason": "Prior fit not supplied"})
            if entry.get("status") != "complete":
                if entry.get("status") not in {"missing", "failed", "sampler_failed"}:
                    raise ValueError("Unknown prior fit completion status")
                entries[variant] = dict(entry)
                continue
            payload = entry["payload"]
            try:
                analysis.assert_sampler_gates(payload["diagnostics"])
            except ValueError as error:
                entries[variant] = {"status": "sampler_failed", "reason": str(error)}
                continue
            report = analysis.posterior_fit_report(contrasts=contrasts[group], **payload)
            report = _with_sign_probabilities(report, payload, contrasts[group])
            rows = compare_rows(primary, report)
            for row in report["rows"]:
                row["included_in_primary_family"] = False
                if row.get("status") != "unavailable":
                    row["status"] = "sensitivity_annotation"
            report["decision_count"] = 0
            report["contrast_decisions"]["decision_count"] = 0
            report["contrast_decisions"]["available_decision_count"] = 0
            report["contrast_decisions"]["unavailable_decision_count"] = 0
            entries[variant] = {"status": "complete", "report": report, "comparisons": rows,
                                "cell_quantiles": entry["cell_quantiles"],
                                "size_slope_quantiles": entry["size_slope_quantiles"]}
            valid[group][variant] = payload
        annotations = []
        for row in primary_reports[group]["rows"]:
            name = row.get("contrast_id", row.get("parameter_id"))
            changes = [comparison for entry in entries.values() if entry["status"] == "complete"
                       for comparison in entry["comparisons"] if comparison["contrast_id"] == name]
            complete = len(changes) == 3 and all(change["status"] != "unavailable" for change in changes)
            changed = any(change.get("decision_rule_changed") for change in changes)
            annotations.append({"contrast_id": name, "included_in_primary_family": False,
                                "assessment_complete": complete,
                                "interpretation": "prior-sensitive under the specified checks" if changed else
                                "unchanged under specified checks; disclose magnitude and upper tails" if complete else
                                "incomplete sensitivity assessment; robustness not established"})
        output["groups"][group] = {"variants": entries, "annotations": annotations,
                                  "primary_cell_quantiles": supplied.get("primary", {}).get("cell_quantiles"),
                                  "primary_size_slope_quantiles": supplied.get("primary", {}).get("size_slope_quantiles")}
    for variant in PRIOR_VARIANTS:
        if not all(variant in valid[group] for group in ("venture", "hiring")):
            output["rq4"][variant] = {"status": "incomplete", "decision_count": 0}
            continue
        venture, hiring = valid["venture"][variant], valid["hiring"][variant]
        output["rq4"][variant] = analysis.cross_pool_descriptive_report(
            venture["gamma_draws"], hiring["gamma_draws"], contrasts["venture"],
            venture["diagnostics"], hiring["diagnostics"],
            venture_sigma_cell_draws=venture["sigma_cell_draws"], venture_z_alpha_draws=venture["z_alpha_draws"],
            venture_cell_ids=venture["cell_ids"], hiring_sigma_cell_draws=hiring["sigma_cell_draws"],
            hiring_z_alpha_draws=hiring["z_alpha_draws"], hiring_cell_ids=hiring["cell_ids"])
    output["complete_fit_count"] = sum(len(values) for values in valid.values())
    output["required_fit_count"] = 9
    output["status"] = "complete" if output["complete_fit_count"] == 9 else "incomplete"
    return output
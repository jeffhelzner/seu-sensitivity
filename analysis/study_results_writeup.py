"""Generate the interim results write-up from the complete confirmatory report (no hand-typed numbers).

Reads results/complete_report_20261102_v1/complete_report.json, summary.json and fit_manifest.json (each
hash-checked against the report's artifact manifest) plus the 24 fit result.json files named by that fit
manifest; fills {{TABLE:name}}, {{VALUE:name}} and {{RATIO:group:contrast}} markers in
applications/seu_sensitivity_study/RESULTS_WRITEUP_TEMPLATE.md and writes
applications/seu_sensitivity_study/RESULTS_WRITEUP_20261103.md.

Run from the repository root:
  conda run --no-capture-output -n seu-sensitivity python -m analysis.study_results_writeup [--tables-only]
"""

import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "applications/seu_sensitivity_study/results/complete_report_20261102_v1"
TEMPLATE = ROOT / "applications/seu_sensitivity_study/RESULTS_WRITEUP_TEMPLATE.md"
OUTPUT = ROOT / "applications/seu_sensitivity_study/RESULTS_WRITEUP_20261103.md"
GROUP_TITLES = {"venture": "Venture", "hiring": "Hiring", "matched_rq5": "Matched RQ5"}
VARIANT_ORDER = ("primary", "presentation_1_only", "presentation_2_only", "utility_035", "utility_065",
                 "prior_L", "prior_H", "prior_S")
QUALIFICATIONS = (("behavioral_trend_not_reproduced", "Trend"), ("choice_pattern_discrepancies", "Choice pattern"),
                  ("excess_same_item_repetition", "Repetition"), ("display_position_discrepancies", "Position"),
                  ("unavailable_sizes_or_pairs", "Unavailable"), ("conditional_on_retention", "Retention"))
DECISIONS = {"detected_positive": "detected +", "detected_negative": "detected −", "not_detected": "not detected"}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_report():
    manifest = json.loads((REPORT_DIR / "artifact_manifest.json").read_text())
    for name in ("complete_report.json", "summary.json", "fit_manifest.json"):
        if digest(REPORT_DIR / name) != manifest[name]:
            raise ValueError(f"{name} differs from the complete-report artifact manifest")
    return (json.loads((REPORT_DIR / "complete_report.json").read_text()),
            json.loads((REPORT_DIR / "summary.json").read_text()), manifest)


def num(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace("-", "−")


def interval(row, digits=3, transform=None):
    lower, upper = row["lower_90"], row["upper_90"]
    if transform:
        lower, upper = transform(lower), transform(upper)
    return f"[{num(lower, digits)}, {num(upper, digits)}]"


def label(row):
    if row.get("label"):
        return row["label"]
    return {"rq6_gamma_size": "menu-size slope (per added alternative)"}.get(row.get("parameter_id"), row.get("parameter_id"))


def row_id(row):
    return row.get("contrast_id") or row.get("parameter_id")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def decision_table(fit_report):
    rows = []
    for row in fit_report["rows"]:
        rows.append([row.get("research_question", "RQ6"), label(row), num(row["median"]), interval(row),
                     num(math.exp(row["median"]), 2), interval(row, 2, math.exp), DECISIONS[row["decision"]]])
    return table(["RQ", "Contrast", "Median (log)", "90% interval", "Ratio", "Ratio 90%", "Decision"], rows)


def fit_directories():
    """Fit directories named by the hash-checked fit manifest; paths re-rooted at this checkout."""
    manifest = json.loads((REPORT_DIR / "fit_manifest.json").read_text())
    directories = {}
    for group, variants in manifest["fits"].items():
        for variant, entry in variants.items():
            parts = Path(entry["chain_path"]).parts
            directories[(group, variant)] = ROOT.joinpath(*parts[parts.index("applications"):]).parent
    return directories


def sampler_table():
    rows = []
    for (group, variant), directory in sorted(fit_directories().items(), key=lambda item: (
            list(GROUP_TITLES).index(item[0][0]), VARIANT_ORDER.index(item[0][1]))):
        result = json.loads((directory / "result.json").read_text())
        data = json.loads((directory / "stan_data_size.json").read_text())
        # Archived fits predate the toolchain record and ran on the original x86_64 Mac.
        machine = result.get("toolchain_machine", "x86_64")
        d = result["diagnostics"]
        rows.append([GROUP_TITLES[group], variant, data["M_total"], machine, num(d["max_rhat"], 4),
                     f"{d['min_ess_bulk']:.0f} / {d['min_ess_tail']:.0f}", num(d["min_ebfmi"], 2),
                     d["divergences"], num(d["treedepth_saturated_share"], 1), f"{result['sample_seconds'] / 60:.1f}"])
    return table(["Group", "Variant", "N", "Machine", "Max R̂", "Min ESS bulk / tail", "Min E-BFMI",
                  "Divergences", "TD12 share", "Sampling min"], rows)


def sensitivity_table(section, name, variants):
    comparison = section[name]
    rows = []
    for row in comparison["rows"]:
        if not (row["sign_changed"] or row["interval_decision_changed"] or row["substantive_interpretation_changed"]):
            continue
        estimates = row["estimates"]
        flags = ", ".join(flag for flag, key in (("sign", "sign_changed"), ("decision", "interval_decision_changed"),
                                                 ("interpretation", "substantive_interpretation_changed")) if row[key])
        rows.append([label(estimates["primary"])] + [
            f"{num(estimates[variant]['median'])} {interval(estimates[variant])} {DECISIONS[estimates[variant]['decision']]}"
            for variant in variants] + [flags])
    if not rows:
        return "_No sign, decision or interpretation changes._"
    return table(["Contrast", *variants, "Changed"], rows)


def prior_tables(report):
    groups = report["ceiling_prior_sensitivity"]["groups"]
    flagged_rows, shift_rows = [], []
    for group in GROUP_TITLES:
        value = groups[group]
        sensitive = [a["contrast_id"] for a in value["annotations"] if "prior-sensitive" in a["interpretation"]]
        for variant in ("prior_L", "prior_H", "prior_S"):
            comparisons = value["variants"][variant]["comparisons"]
            shifts = [abs(c["median_shift"]) for c in comparisons if c.get("median_shift") is not None]
            ratios = [c["interval_width_ratio"] for c in comparisons if c.get("interval_width_ratio") is not None]
            changed = [c["contrast_id"] for c in comparisons if c.get("decision_rule_changed")]
            shift_rows.append([GROUP_TITLES[group], variant, len(comparisons), num(max(shifts)),
                               f"{num(min(ratios), 2)}–{num(max(ratios), 2)}", ", ".join(changed) or "none"])
            for comparison in comparisons:
                if comparison["contrast_id"] in sensitive and comparison.get("decision_rule_changed"):
                    primary, alternative = comparison["primary"], comparison["alternative"]
                    flagged_rows.append([GROUP_TITLES[group], label(primary) if primary.get("label") else comparison["contrast_id"],
                                         variant, f"{num(primary['median'])} {interval(primary)} {DECISIONS[primary['decision']]}",
                                         f"{num(alternative['median'])} {interval(alternative)} {DECISIONS[alternative['decision']]}"])
    return (table(["Group", "Prior", "Estimands", "Max |median shift|", "Width ratio range", "Decision changed"], shift_rows),
            table(["Group", "Contrast", "Prior", "Primary prior", "Alternative prior"], flagged_rows))


def rq3_table(report):
    rows = []
    for group, section in (("venture", report["pools"]["venture"]), ("hiring", report["pools"]["hiring"]),
                           ("matched_rq5", report["matched_rq5"])):
        rq3 = section["primary"]["rq3"]
        dids = rq3["model_by_prompt_dids"]
        excluding = [d for d in dids if d["lower_90"] > 0 or d["upper_90"] < 0]
        rows.append([GROUP_TITLES[group], num(rq3["sigma_cell"]["median"]), interval(rq3["sigma_cell"]),
                     len(dids), len(excluding),
                     num(max(abs(d["median"]) for d in dids))])
    return table(["Group", "σ_cell median", "σ_cell 90%", "Model×prompt DiDs", "DiD intervals excluding 0",
                  "Max |DiD median|"], rows)


def rq4_table(report):
    rows = []
    for row in report["rq4"]["rows"]:
        rows.append([row["contrast_id"].replace("rq1_", "").replace("rq2_", "").replace("_", " "),
                     num(row["venture"]["median"]), num(row["hiring"]["median"]),
                     f"{num(row['hiring_minus_venture']['median'])} {interval(row['hiring_minus_venture'])}",
                     num(row["posterior_probability_same_sign"], 3), "yes" if row["median_sign_agrees"] else "no"])
    return table(["Contrast", "Venture median", "Hiring median", "Hiring − venture [90%]", "P(same sign)",
                  "Median signs agree"], rows)


def ordering_summary(report):
    orderings = report["rq4"]["model_orderings"]
    agree = sum(1 for row in orderings if row["median_sign_agrees"])
    return f"{agree} of {len(orderings)}"


def sonnet_variants_table(report):
    rows = []
    for group in ("venture", "hiring"):
        priors = report["ceiling_prior_sensitivity"]["groups"][group]["variants"]
        for variant in VARIANT_ORDER:
            fit = (priors[variant]["report"] if variant.startswith("prior_") else report["pools"][group][variant])
            s = fit["sonnet_thinking_descriptive"]
            ratio = s["geometric_mean_sensitivity_ratio"]
            excludes = "above 0" if s["lower_90"] > 0 else "below 0" if s["upper_90"] < 0 else "includes 0"
            rows.append([GROUP_TITLES[group], variant, num(s["median"]), interval(s), excludes,
                         num(ratio["median"], 2), interval(ratio, 2)])
    return table(["Pool", "Variant", "Median (log)", "90% interval", "Interval vs 0", "Geometric-mean ratio",
                  "Ratio 90%"], rows)


def amendment6_checks(report):
    """Return the invariance statement only if every stated property holds in the report."""
    for group, section in (("venture", report["pools"]["venture"]), ("hiring", report["pools"]["hiring"]),
                           ("matched_rq5", report["matched_rq5"])):
        scale = section["primary"]["assessment_scale"]
        if any(row.get("zero_sd_cell_ids") for row in scale["rows"]):
            raise ValueError(f"{group}: zero reference SD present; invariance wording would be wrong")
        if any(row["research_question"] == "RQ2" and row["deterministic_offset"] != 0 for row in scale["rows"]):
            raise ValueError(f"{group}: RQ2 offsets do not cancel")
        if scale["rq6"].get("slope_unchanged") is not True:
            raise ValueError(f"{group}: RQ6 slope changed under the assessment scale")
    return ("All reference SDs are positive; within-model RQ2 offsets cancel exactly (offset 0) and the RQ6 "
            "menu-size slope is unchanged in every group.")


def sonnet_table(report):
    rows = []
    for group in ("venture", "hiring"):
        s = report["pools"][group]["primary"]["sonnet_thinking_descriptive"]
        ratio = s["geometric_mean_sensitivity_ratio"]
        rows.append([GROUP_TITLES[group], num(s["median"]), interval(s), num(ratio["median"], 2), interval(ratio, 2),
                     num(s["probability_negative"], 3)])
    return table(["Pool", "Median (log)", "90% interval", "Geometric-mean ratio", "Ratio 90%", "P(negative)"], rows)


def assessment_scale_table(report):
    rows = []
    for group, section in (("venture", report["pools"]["venture"]), ("hiring", report["pools"]["hiring"]),
                           ("matched_rq5", report["matched_rq5"])):
        scale = section["primary"]["assessment_scale"]
        items = scale["rows"]
        reversals = [r["contrast_id"] for r in items if r.get("sign_reversal")]
        zero = [r["contrast_id"] for r in items if r.get("interval_zero_exclusion_changed")]
        unavailable = [r["contrast_id"] for r in items if r.get("status") == "unavailable"]
        rows.append([GROUP_TITLES[group], len(items), ", ".join(reversals) or "none", ", ".join(zero) or "none",
                     ", ".join(unavailable) or "none"])
    return table(["Group", "Rows", "Sign reversal", "Zero-exclusion changed", "Unavailable"], rows)


def repetition_table(report):
    rows = []
    for group in GROUP_TITLES:
        a4 = report["posterior_predictive_checks"][group]["primary"]["a4"]
        for pair in a4["pairs"]:
            if pair["scope"] != "pool_task" or "menu_size" in pair or "family" in pair:
                continue
            same = pair["same_item"]
            rows.append([GROUP_TITLES[group], pair["id"], pair["complete_pair_count"], num(same["observed"], 4),
                         f"[{num(same['replicated']['q05'], 4)}, {num(same['replicated']['q95'], 4)}]",
                         "yes" if same["descriptive_review_flag"] else "no"])
    return table(["Fit group", "Pool/task", "Complete pairs", "Observed same-item", "Predictive 90%", "Review flag"], rows)


def qualification_table(summary):
    rows = []
    for group in GROUP_TITLES:
        for variant in VARIANT_ORDER:
            value = summary["predictive_qualifications"][group][variant]
            cells = []
            for key, _ in QUALIFICATIONS:
                item = value["qualifications"].get(key, {})
                cells.append(f"{'yes' if item.get('applies') else 'no'} ({item.get('flagged_rows', 0)})"
                             if item.get("flagged_rows") else ("yes" if item.get("applies") else "no"))
            rows.append([GROUP_TITLES[group], variant] + cells)
    return table(["Group", "Variant", *(title for _, title in QUALIFICATIONS)], rows)


def ceiling_table(report):
    rows = []
    for group in GROUP_TITLES:
        cells = report["ceiling_diagnostics"][group]["cells"]
        counts = {}
        for value in cells.values():
            counts[value["likelihood"].get("classification", "unresolved")] = counts.get(
                value["likelihood"].get("classification", "unresolved"), 0) + 1
        infinite = [name for name, value in cells.items()
                    if value["likelihood"].get("classification") == "finite_supremum_at_infinity"]
        rows.append([GROUP_TITLES[group], len(cells), ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
                     ", ".join(infinite) or "none"])
    return table(["Group", "Cells", "Classifications", "Supremum at infinite sensitivity"], rows)


def build(report, summary, manifest):
    multiplicity = report["multiplicity"]
    tables = {
        "sampler": sampler_table(),
        "venture_primary": decision_table(report["pools"]["venture"]["primary"]),
        "hiring_primary": decision_table(report["pools"]["hiring"]["primary"]),
        "matched_primary": decision_table(report["matched_rq5"]["primary"]),
        "rq3": rq3_table(report), "rq4": rq4_table(report), "sonnet": sonnet_table(report),
        "sonnet_variants": sonnet_variants_table(report),
        "assessment_scale": assessment_scale_table(report), "repetition": repetition_table(report),
        "qualifications": qualification_table(summary), "ceiling": ceiling_table(report),
    }
    for group, section in (("venture", report["pools"]["venture"]), ("hiring", report["pools"]["hiring"]),
                           ("matched", report["matched_rq5"])):
        tables[f"{group}_presentation"] = sensitivity_table(section, "presentation_sensitivity",
                                                            ("primary", "presentation_1_only", "presentation_2_only"))
        tables[f"{group}_utility"] = sensitivity_table(section, "utility_sensitivity", ("utility_035", "primary", "utility_065"))
    tables["prior_overview"], tables["prior_flagged"] = prior_tables(report)
    values = {
        "primary_decision_count": multiplicity["primary_decision_count"],
        "available_primary_decision_count": multiplicity["available_primary_decision_count"],
        "distinct_up_to_sign": multiplicity["distinct_primary_decisions_up_to_sign"],
        "prior_complete": f"{report['ceiling_prior_sensitivity']['complete_fit_count']} of "
                          f"{report['ceiling_prior_sensitivity']['required_fit_count']}",
        "rq4_orderings_agree": ordering_summary(report),
        "complete_report_sha256": manifest["complete_report.json"],
        "summary_sha256": manifest["summary.json"], "fit_manifest_sha256": manifest["fit_manifest.json"],
        "generator_sha256": digest(Path(__file__)),
        "template_sha256": digest(TEMPLATE) if TEMPLATE.exists() else "none",
        "a4_probes": f"{sum(1 for r in summary['a4_records'].values() if r['probe_exact'])} of {len(summary['a4_records'])}",
        "cross_venture": f"{summary['cross_checks']['venture']['max_relative_difference']:.1e}",
        "cross_hiring": f"{summary['cross_checks']['hiring']['max_relative_difference']:.1e}",
    }
    for group, rows in (("venture", report["pools"]["venture"]["primary"]["rows"]),
                        ("hiring", report["pools"]["hiring"]["primary"]["rows"]),
                        ("matched", [row for row in report["matched_rq5"]["primary"]["rows"]
                                     if row.get("research_question") == "RQ5"])):
        values[f"{group}_detected"] = f"{sum(row['decision'] != 'not_detected' for row in rows)} of {len(rows)}"
    values["total_detected"] = str(sum(int(values[f"{group}_detected"].split()[0]) for group in ("venture", "hiring", "matched")))
    values["amendment6_invariance"] = amendment6_checks(report)
    sigmas = [section["primary"]["rq3"]["sigma_cell"]["median"] for section in
              (report["pools"]["venture"], report["pools"]["hiring"], report["matched_rq5"])]
    values["sigma_min"], values["sigma_max"] = num(min(sigmas)), num(max(sigmas))
    for group in ("venture", "hiring"):
        s = report["pools"][group]["primary"]["sonnet_thinking_descriptive"]
        values[f"sonnet_{group}"] = f"{num(s['median'])} {interval(s)}"
    return tables, values


def render(template, tables, values, report=None):
    def ratio(match):
        group, contrast = match.group(1), match.group(2)
        section = report["matched_rq5"] if group == "matched_rq5" else report["pools"][group]
        rows = {row_id(row): row for row in section["primary"]["rows"]}
        row = rows[contrast]
        return f"{num(math.exp(row['median']), 2)} [{num(math.exp(row['lower_90']), 2)}, {num(math.exp(row['upper_90']), 2)}]"

    def replace(match):
        kind, name = match.group(1), match.group(2)
        source = tables if kind == "TABLE" else values
        if name not in source:
            raise KeyError(f"Unknown {kind} marker {name}")
        return str(source[name])

    text = re.sub(r"\{\{RATIO:([a-z0-9_]+):([a-z0-9_]+)\}\}", ratio, template)
    text = re.sub(r"\{\{(TABLE|VALUE):([a-z0-9_]+)\}\}", replace, text)
    if "{{" in text:
        raise ValueError("Unresolved template marker")
    return text


def main(argv):
    report, summary, manifest = load_report()
    tables, values = build(report, summary, manifest)
    if "--tables-only" in argv:
        for name, value in tables.items():
            print(f"\n### {name}\n\n{value}")
        print("\n" + json.dumps(values, indent=1))
        return
    OUTPUT.write_text(render(TEMPLATE.read_text(), tables, values, report))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main(sys.argv)

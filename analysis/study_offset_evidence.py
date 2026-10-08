"""Publish frozen assessment-scale offsets without choices or posterior fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from applications.seu_sensitivity_study.assessment_scale import build_reference, contrast_offset
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig, build_cells
from applications.seu_sensitivity_study.confirmatory_analysis import (
    matched_rq5_contract,
    primary_contrasts,
)
from applications.seu_sensitivity_study.pools import get_pool_spec


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/applications/seu_sensitivity_study"
BUNDLE = REPORT / "data/design_evidence.yml"
OUTPUT = REPORT / "data/offset_evidence.json"
INCLUDE = REPORT / "_offset_evidence.qmd"
POOL_FAMILIES = {"venture": "procurement", "hiring": "matched"}


def source_paths():
    package = ROOT / "applications/seu_sensitivity_study"
    return [BUNDLE, Path(__file__),
            *(package / name for name in (
                "assessment_scale.py", "config.py", "confirmatory_analysis.py",
                "data_preparation.py", "pools.py", "schemas.py")),
            *(get_pool_spec(pool).item_file for pool in POOL_FAMILIES)]


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths()}


def build_references(bundle):
    references = {}
    matched_items = []
    matched_menus = []
    matched_probabilities = {}
    for pool_id, family in POOL_FAMILIES.items():
        pool = bundle["pools"][pool_id]
        references[pool_id] = build_reference(
            group=pool_id, items=pool["items"], problems=pool["menus"],
            probabilities=pool["probabilities"])
        matched_items.extend(item for item in pool["items"] if item["family"] == family)
        matched_menus.extend(menu for menu in pool["menus"] if menu["family"] == family)
        for model, probabilities in pool["probabilities"].items():
            matched_probabilities.setdefault(model, {}).update(probabilities)
    references["matched_rq5"] = build_reference(
        group="matched_rq5", items=matched_items, problems=matched_menus,
        probabilities=matched_probabilities)
    return references


def offset_row(reference, *, contrast_id, research_question, label, weights, context=False):
    offset, zero_cells = contrast_offset(reference, weights)
    return {
        "group": reference["group"], "contrast_id": contrast_id,
        "research_question": research_question, "label": label,
        "direction": label, "realized_cell_weights": weights,
        "offset": offset, "status": "unavailable_zero_sd" if zero_cells else "available",
        "zero_sd_cell_ids": zero_cells,
        "scope": "fixed_geometry_context_only" if context else "named_contrast_fixed_geometry",
        "included_in_primary_family": False,
        "posterior_extension": False,
    }


def build_evidence(bundle):
    references = build_references(bundle)
    rows = []
    config = SEUSensitivityStudyConfig(pool_ids=list(POOL_FAMILIES))
    for pool_id in POOL_FAMILIES:
        _, columns, _ = config.design_matrix_for_pool(pool_id)
        contrasts = primary_contrasts(columns)
        for contrast in contrasts:
            rows.append(offset_row(
                references[pool_id], contrast_id=contrast.contrast_id,
                research_question=contrast.research_question, label=contrast.label,
                weights=dict(contrast.realized_cell_weights[pool_id])))
        by_id = {contrast.contrast_id: contrast for contrast in contrasts}
        thinking = by_id["rq1_claude_sonnet_4_5_thinking_minus_gpt_4o"].realized_cell_weights[pool_id]
        sonnet = by_id["rq1_claude_sonnet_4_5_minus_gpt_4o"].realized_cell_weights[pool_id]
        weights = {cell_id: thinking.get(cell_id, 0.0) - sonnet.get(cell_id, 0.0)
                   for cell_id in sorted(set(thinking) | set(sonnet))
                   if thinking.get(cell_id, 0.0) != sonnet.get(cell_id, 0.0)}
        rows.append(offset_row(
            references[pool_id], contrast_id="context_thinking_minus_sonnet",
            research_question="context", label="claude-sonnet-4-5-thinking minus claude-sonnet-4-5",
            weights=weights, context=True))
    for contrast in matched_rq5_contract(build_cells(list(POOL_FAMILIES)))["contrasts"]:
        rows.append(offset_row(
            references["matched_rq5"], contrast_id=contrast["contrast_id"],
            research_question=contrast["research_question"], label=contrast["label"],
            weights=contrast["realized_cell_weights"]["matched_rq5"]))
    return {
        "schema_version": 1, "date": "2026-10-08", "snapshot_date": bundle["snapshot_date"],
        "utility_values": [0.0, 0.5, 1.0], "ddof": 0,
        "eta_arithmetic": "assessment_expected_utilities; probabilities normalized before eta",
        "weighting": "equal fixed items; never choices, menu exposures or retained observations",
        "transformation": "standardized contrast = raw contrast + offset; offset = sum(weight * log(sd))",
        "zero_sd": "affected rows unavailable; null offset; no epsilon",
        "interpretation": "Fixed assessment geometry, not posterior effects or decisions; a negative offset does not establish a negative production effect.",
        "amendment9_standardized_posterior_extension": False,
        "choice_records_read": False, "posterior_fits": 0, "provider_calls": 0,
        "source_hashes": source_hashes(), "references": references, "rows": rows,
    }


def render_include(evidence):
    lines = [
        "<!-- Generated by python -m analysis.study_offset_evidence --write. -->",
        "### Frozen Assessment-Scale Offsets {#sec-assessment-scale-offsets}", "",
        "This October 8 supplement reports fixed assessment geometry from the frozen September "
        "probability/menu bundle, not choices or posterior results. For each named direction, "
        "$\\Delta^* = \\Delta + c$, where $c=\\sum_j w_j\\log(s_j)$ and "
        "$s_j$ is the population SD of normalized-probability expected utilities at $(0,0.5,1)$.", "",
        "Primary-pool references contain all 60 fixed items per arm, including the matched family, "
        "not only the 36 primary-family items. Matched RQ5 references contain each arm's own "
        "24 items per task, not a pooled 48-item SD. Items have equal weight; `ddof=0`. "
        "No choice, exposure or retained-observation weighting is used. Zero SD makes an affected "
        "row unavailable, with no epsilon substitution.", "",
        "All seven named RQ1 contrasts per pool are retained, including the sign-reversed OpenAI "
        "duplicate. Both named RQ2 offsets per pool are exactly zero because within-model prompt "
        "offsets cancel. Six matched RQ5 rows use hiring minus procurement.", "",
        "| Group | Named direction | Scope | Offset added to raw log contrast |",
        "|---|---|---|---:|",
    ]
    for row in evidence["rows"]:
        value = "unavailable (zero SD)" if row["offset"] is None else f"{row['offset']:.6f}"
        lines.append(f"| {row['group']} | {row['label']} | {row['research_question']} | {value} |")
    lines.extend([
        "",
        "The two thinking-minus-Sonnet rows are fixed-geometry context only: this is explicitly "
        "**not an Amendment 9 standardized posterior extension**. No posterior interval, detection "
        "rule or primary decision is added or changed. Negative RQ5 offsets describe the direction "
        "of the scale transformation, not the sign of an unknown production RQ5 effect.", "",
        "The [machine-readable supplement](data/offset_evidence.json) preserves unrounded offsets, "
        "directions and cell weights, full references with item IDs, counts, probabilities, expected "
        "utilities and SDs, and SHA-256 hashes of every consumed input and calculation module "
        "including the canonical pool files. Table values alone are rounded. Reproduce both files "
        "exactly with `python -m analysis.study_offset_evidence --check`. Historical evidence is unchanged.",
    ])
    return "\n".join(lines) + "\n"


def check_outputs(evidence, *, output=OUTPUT, include=INCLUDE):
    serialized = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.read_bytes() != serialized.encode() or include.read_bytes() != render_include(evidence).encode():
        raise ValueError("Offset evidence differs from recomputed frozen-input results or source hashes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    evidence = build_evidence(yaml.safe_load(BUNDLE.read_text()))
    if args.write:
        OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
        INCLUDE.write_text(render_include(evidence))
    check_outputs(evidence)
    print("Offset evidence validated: 14 RQ1, 4 RQ2, 6 RQ5 and 2 context rows; 24 arm references.")


if __name__ == "__main__":
    main()
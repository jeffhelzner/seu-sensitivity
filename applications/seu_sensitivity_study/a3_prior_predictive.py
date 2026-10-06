"""Reproducible offline A3 prior draws on persisted frozen assessments and menus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax, xlogy

from .ceiling_diagnostics import quantiles
from .ceiling_prior import POLICY_VERSION, PRIOR_VARIANTS, digest, prior_contract, prior_fields
from .config import MODELS, SEUSensitivityStudyConfig, build_cells
from .confirmatory_analysis import ContrastSpec, matched_rq5_contract, matched_rq5_design, primary_contrasts
from .data_preparation import assessment_expected_utilities

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = ROOT / "applications/seu_sensitivity_study/results/production/production_stages/production-20260911-wave01"
DEFAULT_OUTPUT = ROOT / "reports/applications/seu_sensitivity_study/data/a3_prior_predictive.json"
SEED = 20261006


def load_frozen(stage):
    sources = {}

    def read(relative):
        path = stage / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    manifest = read("preflight_manifest.json")
    pools = {}
    for pool_id in ("venture", "hiring"):
        prefix = f"pools/{pool_id}/"
        pool = read(prefix + "pool.json")
        problems = read(prefix + "problems.json")["problems"]
        probabilities = {}
        for model in MODELS:
            assessment = read(prefix + f"assessments/{model.slug}.json")
            if assessment["instruction"] != "neutral" or assessment["model_name"] != model.name:
                raise ValueError("Frozen assessment identity mismatch")
            rows = assessment["assessments"]
            if not all(row["parse_ok"] for row in rows):
                raise ValueError("Unparsed frozen assessment")
            values = {row["item_id"]: row["probabilities"] for row in rows}
            if len(values) != len(rows) or set(values) != {item["id"] for item in pool["items"]}:
                raise ValueError("Frozen assessment item mapping mismatch")
            matrix = np.asarray(list(values.values()), dtype=float)
            if (matrix.shape != (len(values), 3) or not np.all(np.isfinite(matrix))
                    or np.any(matrix < 0) or np.any(matrix > 1)
                    or not np.allclose(matrix.sum(axis=1), 1, atol=1e-6, rtol=0)):
                raise ValueError("Invalid frozen probability simplex")
            probabilities[model.name] = values
        pools[pool_id] = {"pool": pool, "menus": problems, "probabilities": probabilities}
    for relative, value in sources.items():
        if relative != "preflight_manifest.json" and manifest["source_hashes"].get(relative) != value:
            raise ValueError(f"Frozen input hash mismatch: {relative}")
    return pools, sources


def prior_draws(design, variant, draws, rng):
    fields = prior_fields(variant)
    intercept = rng.normal(fields["prior_gamma0_mean"], fields["prior_gamma0_sd"], draws)
    gamma = rng.normal(0, fields["prior_gamma_sd"], (draws, design.shape[1]))
    sigma = np.abs(rng.normal(0, fields["prior_sigma_cell_sd"], draws))
    residual = sigma[:, None] * rng.normal(size=(draws, design.shape[0]))
    slope = rng.normal(0, fields["prior_gamma_size_sd"], draws)
    return intercept[:, None] + gamma @ design.T + residual, slope, gamma


def menu_utilities(probabilities, menus):
    return np.asarray([assessment_expected_utilities(
        probabilities, item_ids=menu["item_ids"], utilities=[0.0, 0.5, 1.0]) for menu in menus])


def build_check(stage=DEFAULT_STAGE, *, draws=10000, seed=SEED):
    if draws < 100 or seed < 0:
        raise ValueError("Use at least 100 prior draws and a nonnegative seed")
    pools, sources = load_frozen(Path(stage))
    config = SEUSensitivityStudyConfig(pool_ids=["venture", "hiring"])
    matched_cells = build_cells(["venture", "hiring"])
    matched = matched_rq5_contract(matched_cells)
    matched_keys = {}
    for pool_id, family in (("venture", "procurement"), ("hiring", "matched")):
        lookup = {item["id"]: item["matched_key"] for item in pools[pool_id]["pool"]["items"] if item["family"] == family}
        matched_keys[pool_id] = sorted(tuple(sorted(lookup[item] for item in menu["item_ids"]))
                                       for menu in pools[pool_id]["menus"] if menu["family"] == family)
    if matched_keys["venture"] != matched_keys["hiring"] or not matched_keys["venture"]:
        raise ValueError("Frozen matched menus do not pair by merit keys")
    groups = {}
    streams = iter(np.random.SeedSequence(seed).spawn(12))
    for group in ("venture", "hiring", "matched_rq5"):
        if group == "matched_rq5":
            cells = matched_cells
            design, columns = matched_rq5_design(cells)
            contrasts = [ContrastSpec(**row) for row in matched["contrasts"]]
        else:
            cells = config.cells_for_pool(group)
            design, columns, _ = config.design_matrix_for_pool(group)
            contrasts = primary_contrasts(columns)
        cell_ids = [cell.cell_id for cell in cells]
        menus_by_cell = []
        for cell in cells:
            menus = pools[cell.pool_id]["menus"]
            if group == "matched_rq5":
                family = "procurement" if cell.pool_id == "venture" else "matched"
                menus = [menu for menu in menus if menu["family"] == family]
            if {menu["menu_size"] for menu in menus} != {2, 4, 6, 8}:
                raise ValueError("Frozen menus must cover sizes 2/4/6/8")
            menus_by_cell.append(menus)
        size_center = float(np.mean([menu["menu_size"] for menus in menus_by_cell for menu in menus
                                    for _ in menu["presentations"]]))
        variants = {}
        for variant in ("primary", *PRIOR_VARIANTS):
            levels, slope, gamma = prior_draws(design, variant, draws, np.random.default_rng(next(streams)))
            cell_summaries = {cell_id: {"t": quantiles(levels[:, index]), "alpha": quantiles(np.exp(levels[:, index]))}
                              for index, cell_id in enumerate(cell_ids)}
            contrast_summaries = {}
            for contrast in contrasts:
                weights = np.array([contrast.realized_cell_weights[group].get(cell_id, 0) for cell_id in cell_ids])
                realized = levels @ weights
                contrast_summaries[contrast.contrast_id] = {
                    "realized": quantiles(realized), "additive_companion": quantiles(gamma @ contrast.coefficients),
                    "probability_positive": float(np.mean(realized > 0)), "cell_weights": weights.tolist()}
            by_size = {}
            for size in (2, 4, 6, 8):
                metrics = {name: np.zeros(draws) for name in ("max_choice_probability", "maximizer_set_probability", "entropy", "expected_regret")}
                observations = 0
                for cell_index, (cell, menus) in enumerate(zip(cells, menus_by_cell)):
                    selected = [menu for menu in menus if menu["menu_size"] == size]
                    probabilities = pools[cell.pool_id]["probabilities"][cell.model_name]
                    utilities = menu_utilities(probabilities, selected)
                    if utilities.shape != (len(selected), size):
                        raise ValueError("Menu size disagrees with frozen composition")
                    centered = utilities - utilities.max(axis=1, keepdims=True)
                    scale = np.exp(levels[:, cell_index] + slope * (size - size_center))
                    choice = softmax(scale[:, None, None] * centered[None, :, :], axis=2)
                    if not np.all(np.isfinite(choice)):
                        raise ValueError("Nonfinite prior predictive probability; no clipping permitted")
                    metrics["max_choice_probability"] += choice.max(axis=2).sum(axis=1)
                    metrics["maximizer_set_probability"] += (choice * (centered == 0)).sum(axis=(1, 2))
                    metrics["entropy"] += -xlogy(choice, choice).sum(axis=(1, 2))
                    metrics["expected_regret"] += (choice * -centered).sum(axis=(1, 2))
                    observations += len(selected)
                by_size[str(size)] = {"cell_menu_count": observations,
                                      **{name: quantiles(values / observations) for name, values in metrics.items()}}
            variants[variant] = {"prior": prior_contract(variant), "cells": cell_summaries,
                                 "contrasts": contrast_summaries, "size_slope": quantiles(slope),
                                 "choice_by_size": by_size}
        changes = {}
        for variant in PRIOR_VARIANTS:
            changes[variant] = {
                "cell_t_q95_shift_range": [float(function([variants[variant]["cells"][cell]["t"]["q95"] - variants["primary"]["cells"][cell]["t"]["q95"] for cell in cell_ids])) for function in (min, max)],
                "realized_contrast_width_ratios": {name: (summary["realized"]["q95"] - summary["realized"]["q05"]) /
                                                    (variants["primary"]["contrasts"][name]["realized"]["q95"] - variants["primary"]["contrasts"][name]["realized"]["q05"])
                                                    for name, summary in variants[variant]["contrasts"].items()},
                "max_choice_probability_median_shifts": {size: variants[variant]["choice_by_size"][size]["max_choice_probability"]["q50"] -
                                                         variants["primary"]["choice_by_size"][size]["max_choice_probability"]["q50"] for size in ("2", "4", "6", "8")}}
        groups[group] = {"cell_ids": cell_ids, "design_columns": list(columns), "design_sha256": digest(design.tolist()),
                         "size_center": size_center, "variants": variants, "changes_from_primary": changes}
    code_paths = [Path(__file__), Path(__file__).with_name("ceiling_prior.py"), Path(__file__).with_name("ceiling_diagnostics.py"),
                  Path(__file__).with_name("data_preparation.py"),
                  Path(__file__).with_name("confirmatory_analysis.py"), Path(__file__).with_name("config.py"),
                  ROOT / "models/h_m01_size_assessment_anchored.stan",
                  ROOT / "models/h_m01_size_assessment_anchored_prior.stan"]
    return {"schema_version": 1, "policy_version": POLICY_VERSION, "seed": seed, "draws_per_group_prior": draws,
            "status": "finite_offline_prior_check", "posterior_fits": 0, "provider_calls": 0,
            "choice_collection_read": False, "clipping": False, "groups": groups,
            "probability_summary_unit": "Quantiles across prior draws of equally weighted cell-menu means, within each size; repeated presentations have identical predictions.",
            "comparison_policy": "Independent prior streams; shifts compare marginal summaries, not paired draws.",
            "source_hashes": sources,
            "code_hashes": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
            "source_binding": "Supplied bytes checked against frozen stage manifest; hashes are not execution proof."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = build_check(args.stage, draws=args.draws, seed=args.seed)
    if args.check:
        if json.loads(args.output.read_text()) != result:
            raise ValueError("Prior predictive artifact differs from reproduced check")
        print(f"Reproduced {args.output.name}: {args.draws} draws per group/prior")
    else:
        from .confirmatory_reporting import write_report

        write_report(args.output, result)
        print(f"Wrote {args.output.name}: {args.output.stat().st_size} bytes; prior draws only")


if __name__ == "__main__":
    main()
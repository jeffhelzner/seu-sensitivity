"""Prior-predictive diagnostics for assessment-anchored SEU sensitivity.

This analysis uses persisted neutral assessment probabilities and frozen menus.
It makes no provider calls and does not require observed choices or Stan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from applications.seu_sensitivity_study.config import (
    MODELS,
    SEUSensitivityStudyConfig,
)
from applications.seu_sensitivity_study.data_preparation import (
    assessment_expected_utilities,
)


DEFAULT_RESULTS_ROOT = ROOT / "applications" / "seu_sensitivity_study" / "results"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "sd": float(np.std(values)),
    }


def _load_probabilities(path: Path) -> dict[str, Sequence[float]]:
    payload = json.loads(path.read_text())
    parsed = {
        record["item_id"]: record["probabilities"]
        for record in payload["assessments"]
        if record.get("parse_ok") and record.get("probabilities") is not None
    }
    if len(parsed) != len(payload["assessments"]):
        raise ValueError(f"Assessment artifact has unparsed rows: {path}")
    return parsed


def _menu_metrics(
    eta_by_item: Mapping[str, float], problems: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, dict[int, list[float]]]:
    gaps = []
    by_size: dict[int, list[float]] = {}
    for problem in problems:
        values = sorted(
            (eta_by_item[item_id] for item_id in problem["item_ids"]), reverse=True
        )
        gap = values[0] - values[1]
        gaps.append(gap)
        by_size.setdefault(int(problem["menu_size"]), []).append(gap)
    return np.asarray(gaps), by_size


def _prior_predictive(
    *,
    eta_by_cell: Sequence[Mapping[str, float]],
    problems: Sequence[Mapping[str, Any]],
    design_matrix: np.ndarray,
    draws: int,
    seed: int,
    gamma_size_sd: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cell_count, predictor_count = design_matrix.shape
    gamma0 = rng.normal(2.5, 0.5, size=draws)
    gamma = rng.normal(0.0, 0.5, size=(draws, predictor_count))
    gamma_size = rng.normal(0.0, gamma_size_sd, size=draws)
    sigma_cell = np.abs(rng.normal(0.0, 0.3, size=draws))
    z_alpha = rng.normal(size=(draws, cell_count))
    log_alpha_cell = (
        gamma0[:, None]
        + gamma @ design_matrix.T
        + sigma_cell[:, None] * z_alpha
    )

    mean_size = float(np.mean([problem["menu_size"] for problem in problems]))
    modal_sum = np.zeros(draws)
    normalized_entropy_sum = np.zeros(draws)
    total = 0
    modal_by_size = {int(size): np.zeros(draws) for size in sorted({p["menu_size"] for p in problems})}
    count_by_size = {size: 0 for size in modal_by_size}

    for cell_index, eta_by_item in enumerate(eta_by_cell):
        for problem in problems:
            size = int(problem["menu_size"])
            log_alpha = log_alpha_cell[:, cell_index] + gamma_size * (size - mean_size)
            alpha = np.exp(np.clip(log_alpha, -30.0, 30.0))
            eta = np.asarray([eta_by_item[item] for item in problem["item_ids"]])
            logits = alpha[:, None] * eta[None, :]
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            maxima = np.isclose(eta, np.max(eta), atol=1e-12)
            modal_probability = np.sum(probabilities[:, maxima], axis=1)
            entropy = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, None)), axis=1
            ) / np.log(size)
            modal_sum += modal_probability
            normalized_entropy_sum += entropy
            modal_by_size[size] += modal_probability
            count_by_size[size] += 1
            total += 1

    alpha_by_size = {}
    sizes = sorted(modal_by_size)
    for size in sizes:
        alpha = np.exp(
            np.clip(
                log_alpha_cell + gamma_size[:, None] * (size - mean_size),
                -30.0,
                30.0,
            )
        )
        alpha_by_size[str(size)] = _quantiles(alpha.ravel())

    size_log_alpha_change = gamma_size * (sizes[-1] - sizes[0])
    size_alpha_ratio = np.exp(np.clip(size_log_alpha_change, -30.0, 30.0))

    return {
        "draws": draws,
        "seed": seed,
        "mean_menu_size": mean_size,
        "alpha_at_mean_size": _quantiles(np.exp(np.clip(log_alpha_cell, -30.0, 30.0)).ravel()),
        "alpha_by_menu_size": alpha_by_size,
        "alpha_ratio_largest_to_smallest_menu": {
            **_quantiles(size_alpha_ratio),
            "probability_below_0.1": float(np.mean(size_alpha_ratio < 0.1)),
            "probability_above_10": float(np.mean(size_alpha_ratio > 10.0)),
        },
        "mean_modal_choice_probability": _quantiles(modal_sum / total),
        "mean_normalized_entropy": _quantiles(normalized_entropy_sum / total),
        "modal_probability_by_menu_size": {
            str(size): _quantiles(values / count_by_size[size])
            for size, values in modal_by_size.items()
        },
    }


def analyze(
    *,
    results_root: Path,
    pools: Sequence[str],
    utility_middles: Sequence[float],
    draws: int,
    seed: int,
    gamma_size_sd: float,
) -> dict[str, Any]:
    config = SEUSensitivityStudyConfig(pool_ids=list(pools))
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "assessment_anchored_prior_predictive",
        "prior": {
            "gamma0": {"distribution": "normal", "mean": 2.5, "sd": 0.5},
            "gamma": {"distribution": "normal", "mean": 0.0, "sd": 0.5},
            "gamma_size": {
                "distribution": "normal",
                "mean": 0.0,
                "sd": gamma_size_sd,
            },
            "sigma_cell": {"distribution": "half_normal", "sd": 0.3},
        },
        "pools": {},
    }

    for pool_offset, pool_id in enumerate(pools):
        pool_dir = results_root / "pools" / pool_id
        pool = json.loads((pool_dir / "pool.json").read_text())
        problem_set = json.loads((pool_dir / "problems.json").read_text())
        item_ids = [item["id"] for item in pool["items"]]
        design_matrix, _, cells = config.design_matrix_for_pool(pool_id)
        model_probabilities = {
            model.name: _load_probabilities(
                pool_dir / "assessments" / f"{model.slug}.json"
            )
            for model in MODELS
        }
        pool_report: dict[str, Any] = {}

        for utility_offset, middle in enumerate(utility_middles):
            eta_by_model = {}
            model_report = {}
            for model in MODELS:
                eta = assessment_expected_utilities(
                    model_probabilities[model.name],
                    item_ids=item_ids,
                    utilities=[0.0, middle, 1.0],
                )
                eta_by_item = dict(zip(item_ids, eta))
                eta_by_model[model.name] = eta_by_item
                gaps, gaps_by_size = _menu_metrics(
                    eta_by_item, problem_set["problems"]
                )
                model_report[model.name] = {
                    "eta": _quantiles(np.asarray(eta)),
                    "top_two_menu_gap": {
                        **_quantiles(gaps),
                        "fraction_zero": float(np.mean(np.isclose(gaps, 0.0))),
                        "fraction_below_0.01": float(np.mean(gaps < 0.01)),
                    },
                    "top_two_gap_by_menu_size": {
                        str(size): _quantiles(np.asarray(values))
                        for size, values in sorted(gaps_by_size.items())
                    },
                }

            eta_by_cell = [
                eta_by_model[cell.model_name] for cell in config.cells_for_pool(pool_id)
            ]
            pool_report[f"u={middle:.2f}"] = {
                "models": model_report,
                "prior_predictive": _prior_predictive(
                    eta_by_cell=eta_by_cell,
                    problems=problem_set["problems"],
                    design_matrix=design_matrix,
                    draws=draws,
                    seed=seed + 1000 * pool_offset + utility_offset,
                    gamma_size_sd=gamma_size_sd,
                ),
            }
        report["pools"][pool_id] = pool_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--pools", default="venture,hiring")
    parser.add_argument("--utility-middles", default="0.35,0.5,0.65")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--gamma-size-sd", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "assessment_anchored_prior_predictive.json",
    )
    args = parser.parse_args()

    report = analyze(
        results_root=args.results_root,
        pools=[value.strip() for value in args.pools.split(",")],
        utility_middles=[float(value) for value in args.utility_middles.split(",")],
        draws=args.draws,
        seed=args.seed,
        gamma_size_sd=args.gamma_size_sd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
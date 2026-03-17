#!/usr/bin/env python3
"""
Generate primary_analysis.json for any factorial cell.

Loads alpha posterior draws from fit_T* directories, computes:
  - Summary table (median, mean, SD, 90% CI per temperature)
  - Slope distribution (Δα/ΔT per posterior draw + P(slope < 0))
  - Pairwise P(α_i > α_j) for all i < j
  - Strict monotonicity P(α_T1 > α_T2 > ... > α_Tn)

Usage:
    python scripts/generate_primary_analysis.py <results_dir>

    where <results_dir> contains fit_T* subdirectories with alpha_draws.npz files
    and a fit_summary.json.

Examples:
    python scripts/generate_primary_analysis.py applications/temperature_study/results
    python scripts/generate_primary_analysis.py applications/ellsberg_study/results
    python scripts/generate_primary_analysis.py applications/claude_insurance_study/results
    python scripts/generate_primary_analysis.py applications/gpt4o_ellsberg_study/results
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def load_alpha_draws(results_dir: Path) -> dict[float, np.ndarray]:
    """Load alpha posterior draws from fit_T* subdirectories."""
    alpha_draws = {}
    for fit_dir in sorted(results_dir.glob("fit_T*")):
        npz_path = fit_dir / "alpha_draws.npz"
        if not npz_path.exists():
            continue
        # Parse temperature from directory name, e.g. fit_T0_3 -> 0.3
        tag = fit_dir.name.removeprefix("fit_T")
        temp = float(tag.replace("_", "."))
        data = np.load(npz_path)
        alpha_draws[temp] = data["alpha"]
    return dict(sorted(alpha_draws.items()))


def compute_summary_table(alpha_draws: dict[float, np.ndarray]) -> list[dict]:
    """Per-temperature summary statistics."""
    table = []
    for t, draws in alpha_draws.items():
        table.append({
            "temperature": t,
            "median": float(np.median(draws)),
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws)),
            "ci_low": float(np.percentile(draws, 5)),
            "ci_high": float(np.percentile(draws, 95)),
        })
    return table


def compute_pairwise(alpha_draws: dict[float, np.ndarray]) -> dict[str, float]:
    """P(α_i > α_j) for all pairs i < j."""
    temps = list(alpha_draws.keys())
    pairwise = {}
    for i, t1 in enumerate(temps):
        for j, t2 in enumerate(temps):
            if i < j:
                n = min(len(alpha_draws[t1]), len(alpha_draws[t2]))
                prob = float(np.mean(alpha_draws[t1][:n] > alpha_draws[t2][:n]))
                pairwise[f"{t1}_vs_{t2}"] = prob
    return pairwise


def compute_monotonicity(alpha_draws: dict[float, np.ndarray]) -> float:
    """P(α strictly decreasing across all temperatures)."""
    temps = list(alpha_draws.keys())
    n = min(len(alpha_draws[t]) for t in temps)
    strictly_decreasing = 0
    for i in range(n):
        vals = [alpha_draws[t][i] for t in temps]
        if all(vals[j] > vals[j + 1] for j in range(len(vals) - 1)):
            strictly_decreasing += 1
    return strictly_decreasing / n


def compute_slope(alpha_draws: dict[float, np.ndarray]) -> dict:
    """Posterior distribution of slope Δα/ΔT via OLS per draw."""
    temps = list(alpha_draws.keys())
    temp_array = np.array(temps)
    n = min(len(alpha_draws[t]) for t in temps)

    slope_draws = np.empty(n)
    temp_var = np.var(temp_array)
    temp_centered = temp_array - np.mean(temp_array)

    for i in range(n):
        alphas = np.array([alpha_draws[t][i] for t in temps])
        alpha_centered = alphas - np.mean(alphas)
        slope_draws[i] = np.dot(temp_centered, alpha_centered) / (temp_var * len(temps))

    return {
        "median": float(np.median(slope_draws)),
        "mean": float(np.mean(slope_draws)),
        "sd": float(np.std(slope_draws)),
        "ci_low": float(np.percentile(slope_draws, 5)),
        "ci_high": float(np.percentile(slope_draws, 95)),
        "p_negative": float(np.mean(slope_draws < 0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate primary_analysis.json for a factorial cell"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory containing fit_T* subdirectories",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    alpha_draws = load_alpha_draws(results_dir)
    if not alpha_draws:
        print(f"Error: no fit_T*/alpha_draws.npz found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    temps = list(alpha_draws.keys())
    n_draws = min(len(v) for v in alpha_draws.values())
    print(f"Loaded {len(temps)} temperatures: {temps}")
    print(f"  {n_draws} posterior draws per temperature")

    analysis = {
        "summary_table": compute_summary_table(alpha_draws),
        "pairwise_comparisons": compute_pairwise(alpha_draws),
        "monotonicity_prob": compute_monotonicity(alpha_draws),
        "slope": compute_slope(alpha_draws),
    }

    out_path = results_dir / "primary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print results
    print(f"\n=== Summary Table ===")
    for row in analysis["summary_table"]:
        print(
            f"  T={row['temperature']:.1f}  "
            f"median={row['median']:.1f}  mean={row['mean']:.1f}  "
            f"SD={row['sd']:.1f}  90% CI=[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        )

    print(f"\n=== Slope ===")
    s = analysis["slope"]
    print(
        f"  median={s['median']:.2f}  mean={s['mean']:.2f}  "
        f"SD={s['sd']:.2f}  90% CI=[{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
    )
    print(f"  P(slope < 0) = {s['p_negative']:.3f}")

    print(f"\n=== Pairwise P(α_i > α_j) ===")
    for pair, prob in analysis["pairwise_comparisons"].items():
        print(f"  {pair}: {prob:.3f}")

    print(f"\n=== Strict Monotonicity ===")
    print(f"  P(α strictly decreasing) = {analysis['monotonicity_prob']:.3f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""
Temperature Study — Primary Analysis (DESIGN.md §6.1)

This script produces the main quantitative results and visualisations for the
temperature study, which investigates how LLM sampling temperature affects the
estimated sensitivity parameter α.

Prerequisites:
    The temperature study pipeline must have been run first so that fitted
    α posterior draws exist in the results directory:
        applications/temperature_study/results/fit_T<temp>/alpha_draws.npz
    See applications/temperature_study/README.md for the full collection workflow.

Purpose:
    - Load α posterior draws from each temperature condition
    - Compute summary statistics (median, mean, SD, 90 %% CI)
    - Test for monotonic decrease of α with temperature
    - Compute pairwise P(α_i > α_j) comparisons
    - Estimate linear slope Δα / ΔT with uncertainty
    - Generate diagnostic plots (forest plot, density overlay)

Usage:
    python scripts/run_temperature_analysis.py [--results-dir PATH]

Examples:
    # Run with default results directory
    python scripts/run_temperature_analysis.py

    # Point to an alternative results directory
    python scripts/run_temperature_analysis.py --results-dir path/to/results

Outputs (saved to the results directory):
    - primary_analysis.json   — summary table, monotonicity, pairwise, slope
    - forest_plot.png         — forest plot of α posteriors by temperature
    - alpha_density_plot.png  — overlaid density curves
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from applications.temperature_study.visualization import (
    alpha_density_plot,
    alpha_slope,
    alpha_summary_table,
    forest_plot,
    posterior_monotonicity_prob,
)

RESULTS_DIR = (
    PROJECT_ROOT / "applications" / "temperature_study" / "results"
)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.5]


def load_alpha_posteriors(results_dir: Path) -> dict[float, np.ndarray]:
    """Load α posterior draws from alpha_draws.npz files."""
    posteriors = {}
    for temp in TEMPERATURES:
        tag = f"T{temp:.1f}".replace(".", "_")
        path = results_dir / f"fit_{tag}" / "alpha_draws.npz"
        if not path.exists():
            print(f"  ⚠  Missing: {path}")
            continue
        data = np.load(path)
        posteriors[temp] = data["alpha"]
        print(f"  T={temp}:  {len(posteriors[temp])} draws loaded")
    return posteriors


def main():
    parser = argparse.ArgumentParser(
        description="Primary analysis of temperature study results (§6.1)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Path to temperature study results directory",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # ── Load draws ──
    print("Loading α posterior draws ...")
    posteriors = load_alpha_posteriors(results_dir)
    if len(posteriors) < 2:
        print("Need at least 2 temperature conditions. Exiting.")
        sys.exit(1)

    # ── Summary table ──
    print("\n" + "=" * 72)
    print("α POSTERIOR SUMMARY")
    print("=" * 72)
    table = alpha_summary_table(posteriors, ci=0.90)
    header = f"{'T':>5s}  {'median':>8s}  {'mean':>8s}  {'sd':>8s}  {'90% CI':>20s}"
    print(header)
    print("-" * len(header))
    for row in table:
        print(
            f"{row['temperature']:5.1f}  "
            f"{row['median']:8.1f}  "
            f"{row['mean']:8.1f}  "
            f"{row['sd']:8.1f}  "
            f"[{row['ci_low']:7.1f}, {row['ci_high']:7.1f}]"
        )

    # ── Monotonicity test ──
    mono_prob = posterior_monotonicity_prob(posteriors)
    print(f"\nP(α strictly decreasing in T) = {mono_prob:.4f}")

    # ── Pairwise comparisons ──
    temps = sorted(posteriors)
    min_len = min(len(posteriors[t]) for t in temps)
    print("\nPairwise P(α_i > α_j):")
    pairwise = {}
    for i in range(len(temps)):
        for j in range(i + 1, len(temps)):
            ti, tj = temps[i], temps[j]
            draws_i = posteriors[ti][:min_len]
            draws_j = posteriors[tj][:min_len]
            prob = float(np.mean(draws_i > draws_j))
            pairwise[f"{ti}_vs_{tj}"] = round(prob, 4)
            print(f"  P(α(T={ti}) > α(T={tj})) = {prob:.4f}")

    # ── Slope ──
    slope_result = alpha_slope(posteriors)
    print(
        f"\nEstimated slope: Δα/ΔT = {slope_result['slope']:.2f}  "
        f"90% CI [{slope_result['ci_low']:.2f}, {slope_result['ci_high']:.2f}]"
    )

    # ── Save quantitative results ──
    analysis_output = {
        "summary_table": table,
        "monotonicity_prob": mono_prob,
        "pairwise_comparisons": pairwise,
        "slope": slope_result,
    }
    out_json = results_dir / "primary_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Forest plot ──
    forest_path = results_dir / "forest_plot.png"
    fig = forest_plot(posteriors, ci=0.90, save_path=str(forest_path))
    if fig is not None:
        print(f"Saved: {forest_path}")

    # ── Density plot ──
    density_path = results_dir / "alpha_density_plot.png"
    fig = alpha_density_plot(posteriors, save_path=str(density_path))
    if fig is not None:
        print(f"Saved: {density_path}")

    print("\n✅  Primary analysis complete.")


if __name__ == "__main__":
    main()

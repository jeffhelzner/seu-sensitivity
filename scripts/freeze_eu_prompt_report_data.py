"""
Freeze data snapshots for the EU-prompt temperature study report.

Copies relevant outputs from the pipeline results directory into the report's
data/ directory and computes cross-study comparison quantities (posterior
P(α_base > α_EU), slope comparisons, etc.).

Usage:
    python scripts/freeze_eu_prompt_report_data.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EU_RESULTS = PROJECT_ROOT / "applications" / "temperature_study_with_eu_prompt" / "results"
BASE_DATA = PROJECT_ROOT / "reports" / "applications" / "temperature_study" / "data"
REPORT_DATA = PROJECT_ROOT / "reports" / "applications" / "temperature_study_with_eu_prompt" / "data"

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.5]


def temp_key(t: float) -> str:
    return f"T{str(t).replace('.', '_')}"


def freeze_snapshots() -> None:
    """Copy pipeline outputs into the report data directory."""
    REPORT_DATA.mkdir(parents=True, exist_ok=True)

    for t in TEMPERATURES:
        tk = temp_key(t)
        fit_dir = EU_RESULTS / f"fit_{tk}"

        # Alpha draws
        src = fit_dir / "alpha_draws.npz"
        dst = REPORT_DATA / f"alpha_draws_{tk}.npz"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

        # Diagnostics
        src = fit_dir / "diagnostics.txt"
        dst = REPORT_DATA / f"diagnostics_{tk}.txt"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

        # PPC
        src = fit_dir / "ppc.json"
        dst = REPORT_DATA / f"ppc_{tk}.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

        # Stan data
        src = EU_RESULTS / f"stan_data_{tk}.json"
        dst = REPORT_DATA / f"stan_data_{tk}.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

    # Aggregate files
    for fname in ["fit_summary.json", "run_summary.json"]:
        src = EU_RESULTS / fname
        dst = REPORT_DATA / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

    # Study config
    src = PROJECT_ROOT / "applications" / "temperature_study_with_eu_prompt" / "configs" / "study_config.yaml"
    dst = REPORT_DATA / "study_config.yaml"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  ✓ {dst.name}")

    # Prompts
    src = PROJECT_ROOT / "applications" / "temperature_study_with_eu_prompt" / "configs" / "prompts.yaml"
    dst = REPORT_DATA / "prompts.yaml"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  ✓ {dst.name}")


def compute_cross_study_analysis() -> dict:
    """Compute cross-study comparison quantities from frozen alpha draws."""
    # Load EU-prompt alpha draws
    eu_draws = {}
    for t in TEMPERATURES:
        tk = temp_key(t)
        data = np.load(REPORT_DATA / f"alpha_draws_{tk}.npz")
        eu_draws[t] = data["alpha"]

    # Load base study alpha draws
    base_draws = {}
    for t in TEMPERATURES:
        tk = temp_key(t)
        data = np.load(BASE_DATA / f"alpha_draws_{tk}.npz")
        base_draws[t] = data["alpha"]

    n_draws = min(len(eu_draws[TEMPERATURES[0]]), len(base_draws[TEMPERATURES[0]]))

    # --- Per-temperature comparison ---
    per_temperature = {}
    for t in TEMPERATURES:
        base = base_draws[t][:n_draws]
        eu = eu_draws[t][:n_draws]
        diff = base - eu

        per_temperature[str(t)] = {
            "base_median": float(np.median(base)),
            "base_mean": float(np.mean(base)),
            "base_q05": float(np.percentile(base, 5)),
            "base_q95": float(np.percentile(base, 95)),
            "eu_median": float(np.median(eu)),
            "eu_mean": float(np.mean(eu)),
            "eu_q05": float(np.percentile(eu, 5)),
            "eu_q95": float(np.percentile(eu, 95)),
            "p_base_gt_eu": float(np.mean(base > eu)),
            "diff_median": float(np.median(diff)),
            "diff_mean": float(np.mean(diff)),
            "diff_q05": float(np.percentile(diff, 5)),
            "diff_q95": float(np.percentile(diff, 95)),
        }

    # --- Slope comparison ---
    temp_array = np.array(TEMPERATURES)
    t_var = np.var(temp_array)

    base_slopes = []
    eu_slopes = []
    for i in range(n_draws):
        base_alphas = np.array([base_draws[t][i] for t in TEMPERATURES])
        eu_alphas = np.array([eu_draws[t][i] for t in TEMPERATURES])

        base_slopes.append(np.cov(temp_array, base_alphas)[0, 1] / t_var)
        eu_slopes.append(np.cov(temp_array, eu_alphas)[0, 1] / t_var)

    base_slopes = np.array(base_slopes)
    eu_slopes = np.array(eu_slopes)
    slope_diff = eu_slopes - base_slopes

    slope_comparison = {
        "base_slope_median": float(np.median(base_slopes)),
        "base_slope_q05": float(np.percentile(base_slopes, 5)),
        "base_slope_q95": float(np.percentile(base_slopes, 95)),
        "eu_slope_median": float(np.median(eu_slopes)),
        "eu_slope_q05": float(np.percentile(eu_slopes, 5)),
        "eu_slope_q95": float(np.percentile(eu_slopes, 95)),
        "p_eu_slope_steeper": float(np.mean(eu_slopes < base_slopes)),
        "slope_diff_median": float(np.median(slope_diff)),
        "slope_diff_q05": float(np.percentile(slope_diff, 5)),
        "slope_diff_q95": float(np.percentile(slope_diff, 95)),
    }

    # --- Monotonicity comparison ---
    base_mono = 0
    eu_mono = 0
    for i in range(n_draws):
        base_vals = [base_draws[t][i] for t in TEMPERATURES]
        eu_vals = [eu_draws[t][i] for t in TEMPERATURES]
        if all(base_vals[j] > base_vals[j + 1] for j in range(len(TEMPERATURES) - 1)):
            base_mono += 1
        if all(eu_vals[j] > eu_vals[j + 1] for j in range(len(TEMPERATURES) - 1)):
            eu_mono += 1

    monotonicity = {
        "base_strict_monotonicity": float(base_mono / n_draws),
        "eu_strict_monotonicity": float(eu_mono / n_draws),
    }

    # --- Interaction: does EU-prompt hurt more at low vs high temperature? ---
    # Compare the drop (base - EU) at T=0.0 vs T=1.5
    diff_low = base_draws[0.0][:n_draws] - eu_draws[0.0][:n_draws]
    diff_high = base_draws[1.5][:n_draws] - eu_draws[1.5][:n_draws]
    interaction = {
        "diff_at_T0_0_median": float(np.median(diff_low)),
        "diff_at_T0_0_q05": float(np.percentile(diff_low, 5)),
        "diff_at_T0_0_q95": float(np.percentile(diff_low, 95)),
        "diff_at_T1_5_median": float(np.median(diff_high)),
        "diff_at_T1_5_q05": float(np.percentile(diff_high, 5)),
        "diff_at_T1_5_q95": float(np.percentile(diff_high, 95)),
        "p_larger_drop_at_low_temp": float(np.mean(diff_low > diff_high)),
    }

    return {
        "n_draws": n_draws,
        "per_temperature": per_temperature,
        "slope_comparison": slope_comparison,
        "monotonicity": monotonicity,
        "interaction": interaction,
    }


def main() -> None:
    print("═══ Freezing EU-Prompt Report Data ═══\n")

    print("Step 1: Copying pipeline outputs …")
    freeze_snapshots()

    print("\nStep 2: Computing cross-study analysis …")
    analysis = compute_cross_study_analysis()

    out_path = REPORT_DATA / "cross_study_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  ✓ {out_path.name}")

    # Print summary
    print("\n═══ Cross-Study Summary ═══\n")
    print(f"{'Temp':>6}  {'Base α':>8}  {'EU α':>8}  {'Δ':>8}  {'P(base>EU)':>10}")
    print("─" * 50)
    for t in TEMPERATURES:
        s = analysis["per_temperature"][str(t)]
        print(
            f"{t:6.1f}  {s['base_median']:8.1f}  {s['eu_median']:8.1f}  "
            f"{s['diff_median']:+8.1f}  {s['p_base_gt_eu']:10.3f}"
        )

    sc = analysis["slope_comparison"]
    print(f"\nSlope (Δα/ΔT):")
    print(f"  Base:  {sc['base_slope_median']:.1f}  90% CI [{sc['base_slope_q05']:.1f}, {sc['base_slope_q95']:.1f}]")
    print(f"  EU:    {sc['eu_slope_median']:.1f}  90% CI [{sc['eu_slope_q05']:.1f}, {sc['eu_slope_q95']:.1f}]")
    print(f"  P(EU slope steeper): {sc['p_eu_slope_steeper']:.3f}")

    print(f"\nStrict monotonicity:")
    m = analysis["monotonicity"]
    print(f"  Base: {m['base_strict_monotonicity']:.3f}")
    print(f"  EU:   {m['eu_strict_monotonicity']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

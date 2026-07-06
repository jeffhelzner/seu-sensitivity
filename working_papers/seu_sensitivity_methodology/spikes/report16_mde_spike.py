"""
Report-16 MDE / power spike (methodological paper, claims-ledger row C16).

Calibrates the Claude x insurance NULL (plan SS7.5.2 / SS7.6.1). The application
found a slope of alpha on temperature that is statistically indistinguishable
from zero: posterior median ~ -2.9, sd ~ 22.1, P(slope < 0) ~ 0.56. The plan
requires that this null be *calibrated* rather than read as a positive claim of
no effect:

    "...the slope magnitude the design *could* have resolved (e.g., the smallest
    |Dalpha/DT| for which P(slope < 0) would have exceeded ~0.95). This
    distinguishes 'no effect at the achievable resolution' from 'effect too
    small to license the comparative reading.'  (Claims-ledger row C16.)"

This script computes that minimum-detectable-effect (MDE) directly from the
saved per-condition posterior alpha draws, using the SAME draw-wise
population-OLS slope functional the application used:

    for each posterior draw i:
        alphas_i = [alpha_{T0,i}, alpha_{T0.2,i}, ..., alpha_{T1.0,i}]
        b_i = Cov(T, alphas_i) / Var(T)          (population moments)
    slope posterior = { b_i },   P(slope < 0) = mean(b_i < 0).

b_i is a linear functional of the five per-condition posteriors, so the design's
*resolving power* is fixed by the per-condition posterior dispersions and the
temperature weights. The MDE is the smallest counterfactual true slope |beta|
for which P(slope < 0) would reach the criterion.

Criterion (primary): one-sided P(slope < 0) >= 0.95  (z_0.95 = 1.6449).
Criterion (secondary, reported): P(slope < 0) >= 0.975 (two-sided-equivalent).

Three independent MDE estimators (cross-checked):
  (A) Analytic Gaussian        MDE = z * sigma_slope.
  (B) Empirical-quantile       re-center the actual slope draws to mean beta;
      (additive-noise)         read the 95th percentile of |re-centered slope|.
                               (Faithful to the non-Gaussian slope shape, but
                               assumes per-condition dispersion is unchanged by
                               the counterfactual mean shift.)
  (C) Constant-CV Monte Carlo  impose a counterfactual mean line
                               mu_t = alpha_ref + beta*(T_t - Tbar), preserve
                               each condition's RELATIVE residuals (CV ~ 0.28
                               near-constant across conditions), recompute the
                               slope through the same Cov/Var functional, and
                               root-find beta where P(slope < 0) = criterion.
                               (Accounts for posterior dispersion scaling with
                               the alpha level.)

Outputs:
  report16_mde_results.json                   (next to this script)
  ../figures/report16_mde_power_curve.png     (power curve + MDE markers)

Run:
  python working_papers/seu_sensitivity_methodology/spikes/report16_mde_spike.py
"""
from __future__ import annotations

import json
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))

DATA_DIR = os.path.join(
    PROJECT_ROOT, "reports", "applications", "claude_insurance_study", "data"
)

TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]

# Criterion z-multipliers for P(slope < 0) thresholds.
Z_95 = 1.6448536269514722   # one-sided 0.95
Z_975 = 1.959963984540054   # one-sided 0.975 (= two-sided 0.95)

# GPT-4o reference slope (full, unequal temperature grid), reported in plan
# SS7.5.2a for context only. NOT computed here; carries the unequal-grid caveat.
# Canonical draw-level population-OLS median (spikes/report_2x2_forest_results.json,
# GPT-4o x insurance). The earlier -31 figure was inflated by the ddof
# covariance/variance mismatch documented in report_2x2_forest_spike.py.
GPT4O_SLOPE_FULLGRID_REF = -24.6

SEED = 20260618
N_MC = 200_000  # Monte-Carlo resamples for estimator (C)


def temp_key(t: float) -> str:
    return f"T{str(t).replace('.', '_')}"


def load_alpha_draws() -> dict[float, np.ndarray]:
    draws = {}
    for t in TEMPERATURES:
        path = os.path.join(DATA_DIR, f"alpha_draws_{temp_key(t)}.npz")
        with np.load(path) as z:
            draws[t] = np.asarray(z["alpha"], dtype=float)
    n = {t: a.shape[0] for t, a in draws.items()}
    if len(set(n.values())) != 1:
        raise ValueError(f"Unequal draw counts across conditions: {n}")
    return draws


def slope_weights(temps: np.ndarray) -> np.ndarray:
    """Population-OLS weights w_t = (T_t - Tbar) / sum_j (T_j - Tbar)^2.

    slope = sum_t w_t * alpha_t  reproduces Cov(T, alpha) / Var(T).
    """
    tc = temps - temps.mean()
    return tc / np.sum(tc**2)


def slope_draws_from(draws: dict[float, np.ndarray], temps: np.ndarray) -> np.ndarray:
    w = slope_weights(temps)
    A = np.column_stack([draws[t] for t in TEMPERATURES])  # (n_draws, 5)
    return A @ w


def main() -> None:
    rng = np.random.default_rng(SEED)

    draws = load_alpha_draws()
    temps = np.asarray(TEMPERATURES, dtype=float)
    n_draws = draws[TEMPERATURES[0]].shape[0]

    # ---- recompute slope posterior; self-check vs application output ----
    slope = slope_draws_from(draws, temps)
    slope_median = float(np.median(slope))
    slope_mean = float(np.mean(slope))
    slope_sd = float(np.std(slope, ddof=1))
    p_negative = float(np.mean(slope < 0))

    ref_path = os.path.join(DATA_DIR, "primary_analysis.json")
    ref = json.load(open(ref_path))
    ref_slope = ref["slope"]
    selfcheck = {
        "recomputed": {
            "median": slope_median, "mean": slope_mean,
            "sd": slope_sd, "p_negative": p_negative,
        },
        "primary_analysis_json": {
            "median": ref_slope["median"], "mean": ref_slope["mean"],
            "sd": ref_slope["sd"], "p_negative": ref_slope["p_negative"],
        },
        "abs_diff": {
            "median": abs(slope_median - ref_slope["median"]),
            "mean": abs(slope_mean - ref_slope["mean"]),
            "sd": abs(slope_sd - ref_slope["sd"]),
            "p_negative": abs(p_negative - ref_slope["p_negative"]),
        },
    }
    ok = (selfcheck["abs_diff"]["sd"] < 0.5
          and selfcheck["abs_diff"]["p_negative"] < 0.01
          and selfcheck["abs_diff"]["median"] < 1.0)
    selfcheck["passed"] = bool(ok)

    # per-condition summaries
    cond_mean = {t: float(np.mean(draws[t])) for t in TEMPERATURES}
    cond_sd = {t: float(np.std(draws[t], ddof=1)) for t in TEMPERATURES}
    cond_cv = {t: cond_sd[t] / cond_mean[t] for t in TEMPERATURES}
    grand_mean_alpha = float(np.mean([cond_mean[t] for t in TEMPERATURES]))
    mean_cv = float(np.mean([cond_cv[t] for t in TEMPERATURES]))
    dT = float(temps.max() - temps.min())

    # ===========================================================
    # (A) Analytic Gaussian MDE
    # ===========================================================
    mde_A_95 = Z_95 * slope_sd
    mde_A_975 = Z_975 * slope_sd

    # ===========================================================
    # (B) Empirical-quantile (additive-noise) MDE
    #   slope^cf_i = (b_i - mean_b) + beta.  For a decline (beta<0):
    #   P(slope^cf<0) = P(eps < -beta) = ECDF_eps(|beta|) = crit
    #   => |beta| = quantile_crit( eps ),   eps = b_i - mean_b.
    # ===========================================================
    eps = slope - slope_mean
    mde_B_95 = float(np.quantile(eps, 0.95))
    mde_B_975 = float(np.quantile(eps, 0.975))

    # ===========================================================
    # (C) Constant-CV Monte-Carlo MDE
    #   relative residuals preserve each condition's dispersion shape;
    #   impose mu_t(beta) = alpha_ref + beta*(T_t - Tbar); recompute slope.
    # ===========================================================
    rel_resid = {  # (n_draws,) per condition, mean ~ 0
        t: (draws[t] - cond_mean[t]) / cond_mean[t] for t in TEMPERATURES
    }
    alpha_ref = grand_mean_alpha
    tc = temps - temps.mean()
    w = slope_weights(temps)

    def p_negative_cf(beta: float, n_mc: int) -> float:
        mu = alpha_ref + beta * tc  # (5,)
        idx = rng.integers(0, n_draws, size=(n_mc, len(TEMPERATURES)))
        cf = np.empty((n_mc, len(TEMPERATURES)))
        for j, t in enumerate(TEMPERATURES):
            cf[:, j] = mu[j] * (1.0 + rel_resid[t][idx[:, j]])
        s = cf @ w
        return float(np.mean(s < 0.0))

    def mde_C(target: float) -> float:
        # root-find |beta| (beta<0) s.t. P(slope<0) = target on a grid + bisection
        lo, hi = 0.0, 3.0 * max(mde_A_975, 1.0)  # search |beta| in [lo, hi]
        f = lambda b: p_negative_cf(-b, N_MC)    # noqa: E731
        # ensure bracket
        if f(hi) < target:
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-3:
                break
        return 0.5 * (lo + hi)

    mde_C_95 = mde_C(0.95)
    mde_C_975 = mde_C(0.975)

    # ---- power curve (estimator C, the realistic one) ----
    beta_grid = np.linspace(0.0, 1.6 * max(mde_A_95, mde_B_95, mde_C_95), 33)
    power_curve = [p_negative_cf(-b, 60_000) for b in beta_grid]

    # ===========================================================
    # contextualization
    # ===========================================================
    obs_slope_abs = abs(slope_median)
    headline_mde = mde_C_95  # report the constant-CV estimate as headline

    def contextualize(mde: float) -> dict:
        return {
            "slope_units_alpha_per_unitT": mde,
            "end_to_end_dalpha_over_T_range": mde * dT,
            "multiple_of_observed_abs_slope": mde / obs_slope_abs,
            "fraction_of_grand_mean_alpha": mde / grand_mean_alpha,
        }

    results = {
        "spec": {
            "row": "C16",
            "sections": ["7.5.2", "7.6.1"],
            "data_dir": DATA_DIR,
            "temperatures": TEMPERATURES,
            "n_draws_per_condition": n_draws,
            "criterion_primary": "P(slope < 0) >= 0.95 (one-sided)",
            "criterion_secondary": "P(slope < 0) >= 0.975",
            "seed": SEED,
            "n_mc": N_MC,
        },
        "slope_selfcheck": selfcheck,
        "per_condition": {
            "mean": cond_mean, "sd": cond_sd, "cv": cond_cv,
            "grand_mean_alpha": grand_mean_alpha, "mean_cv": mean_cv,
        },
        "observed_slope": {
            "median": slope_median, "mean": slope_mean, "sd": slope_sd,
            "p_negative": p_negative, "abs_median": obs_slope_abs,
        },
        "mde": {
            "criterion_0.95": {
                "A_analytic_gaussian": mde_A_95,
                "B_empirical_quantile": mde_B_95,
                "C_constant_cv_montecarlo": mde_C_95,
            },
            "criterion_0.975": {
                "A_analytic_gaussian": mde_A_975,
                "B_empirical_quantile": mde_B_975,
                "C_constant_cv_montecarlo": mde_C_975,
            },
        },
        "headline": {
            "mde_estimator": "C_constant_cv_montecarlo",
            "criterion": "P(slope<0) >= 0.95",
            "value": headline_mde,
            "context": contextualize(headline_mde),
        },
        "context_all_0.95": {
            "A_analytic_gaussian": contextualize(mde_A_95),
            "B_empirical_quantile": contextualize(mde_B_95),
            "C_constant_cv_montecarlo": contextualize(mde_C_95),
        },
        "gpt4o_reference": {
            "slope_fullgrid": GPT4O_SLOPE_FULLGRID_REF,
            "note": "reference only (plan SS7.5.2a); full, unequal grid; "
                    "carries the unequal-grid caveat; not computed here",
            "abs_slope_vs_mde_C95": abs(GPT4O_SLOPE_FULLGRID_REF) / headline_mde,
        },
        "power_curve": {
            "beta_abs": beta_grid.tolist(),
            "p_negative": power_curve,
        },
    }

    out_path = os.path.join(THIS_DIR, "report16_mde_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- print ----
    print("=== C16: MDE / power for the Claude x insurance null ===\n")
    print("Slope self-check (recomputed vs primary_analysis.json):")
    sc = selfcheck
    print(f"  median  {sc['recomputed']['median']:+.4f} vs "
          f"{sc['primary_analysis_json']['median']:+.4f}")
    print(f"  sd      {sc['recomputed']['sd']:.4f} vs "
          f"{sc['primary_analysis_json']['sd']:.4f}")
    print(f"  p(<0)   {sc['recomputed']['p_negative']:.4f} vs "
          f"{sc['primary_analysis_json']['p_negative']:.4f}")
    print(f"  PASSED: {sc['passed']}\n")

    print(f"Per-condition CV: " + ", ".join(
        f"T{t}={cond_cv[t]:.3f}" for t in TEMPERATURES))
    print(f"  mean CV = {mean_cv:.3f}, grand-mean alpha = {grand_mean_alpha:.2f}, "
          f"slope sd = {slope_sd:.2f}\n")

    print("MDE  (smallest |Dalpha/DT| reaching the criterion), alpha-units per unit T:")
    print(f"  criterion P(slope<0) >= 0.95:")
    print(f"     (A) analytic Gaussian       {mde_A_95:6.2f}")
    print(f"     (B) empirical quantile      {mde_B_95:6.2f}")
    print(f"     (C) constant-CV Monte Carlo {mde_C_95:6.2f}   <- headline")
    print(f"  criterion P(slope<0) >= 0.975:")
    print(f"     (A) analytic Gaussian       {mde_A_975:6.2f}")
    print(f"     (B) empirical quantile      {mde_B_975:6.2f}")
    print(f"     (C) constant-CV Monte Carlo {mde_C_975:6.2f}\n")

    ctx = contextualize(headline_mde)
    print(f"Headline (C, P>=0.95)  MDE = {headline_mde:.1f} alpha-units / unit T")
    print(f"  end-to-end Dalpha over T in [0,1]: {ctx['end_to_end_dalpha_over_T_range']:.1f}")
    print(f"  = {ctx['multiple_of_observed_abs_slope']:.1f}x the observed |slope| "
          f"({obs_slope_abs:.1f})")
    print(f"  = {ctx['fraction_of_grand_mean_alpha']:.2f} of grand-mean alpha "
          f"({grand_mean_alpha:.1f})")
    print(f"  GPT-4o full-grid ref |slope| {abs(GPT4O_SLOPE_FULLGRID_REF):.0f} "
          f"= {abs(GPT4O_SLOPE_FULLGRID_REF)/headline_mde:.2f}x the MDE "
          f"(reference only; unequal-grid caveat)\n")

    print(f"Results written to {out_path}")

    make_figure(beta_grid, power_curve, mde_A_95, mde_B_95, mde_C_95,
                obs_slope_abs, headline_mde)


def make_figure(beta_grid, power_curve, mde_A, mde_B, mde_C,
                obs_slope_abs, headline_mde):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(beta_grid, power_curve, "-", lw=2.0, color="#1f77b4",
            label="P(slope < 0)  (constant-CV MC)")
    ax.axhline(0.95, color="#7f7f7f", ls="--", lw=1.2)
    ax.text(beta_grid[1], 0.955, "criterion 0.95", fontsize=9, color="#555")

    ax.axvline(mde_C, color="#d62728", ls="-", lw=1.6,
               label=f"MDE (C, MC) = {mde_C:.1f}")
    ax.axvline(mde_A, color="#9467bd", ls=":", lw=1.4,
               label=f"MDE (A, Gaussian) = {mde_A:.1f}")
    ax.axvline(mde_B, color="#2ca02c", ls=":", lw=1.4,
               label=f"MDE (B, empirical) = {mde_B:.1f}")
    ax.axvline(obs_slope_abs, color="#ff7f0e", ls="-.", lw=1.6,
               label=f"observed |slope| = {obs_slope_abs:.1f}")
    if abs(GPT4O_SLOPE_FULLGRID_REF) <= beta_grid.max():
        ax.axvline(abs(GPT4O_SLOPE_FULLGRID_REF), color="#8c564b", ls="-.",
                   lw=1.2,
                   label=f"GPT-4o full-grid |slope| ~ {abs(GPT4O_SLOPE_FULLGRID_REF):.0f} (ref)")

    ax.set_xlim(0, beta_grid.max())
    ax.set_ylim(0.45, 1.005)
    ax.set_xlabel(r"counterfactual true $|\Delta\alpha/\Delta T|$  ($\alpha$-units per unit temperature)")
    ax.set_ylabel("P(posterior slope < 0)")
    ax.set_title("Claude x insurance: power to resolve a temperature-on-$\\alpha$ slope\n"
                 "(minimum detectable effect at the achievable resolution)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()

    fig_dir = os.path.join(PAPER_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "report16_mde_power_curve.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()

"""
Visualization for the Temperature Study.

Implements DESIGN.md §6.1 (forest plot), §6.2 (position bias charts),
§6.3 (consistency plots), and §6.4 (NA rate display).

Matplotlib is an optional dependency; all public functions gracefully
degrade (log a warning and return ``None``) when it is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Plotting disabled.")

# Colorblind-friendly palette (Okabe-Ito)
TEMP_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
]


def _check_matplotlib() -> bool:
    if not HAS_MATPLOTLIB:
        logger.warning(
            "matplotlib not installed — skipping plot."
        )
        return False
    return True


# ------------------------------------------------------------------
# §6.1  Primary analysis: forest plot of α posteriors
# ------------------------------------------------------------------


def forest_plot(
    alpha_posteriors: Dict[float, np.ndarray],
    ci: float = 0.90,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """
    Forest plot of α posterior distributions by temperature.

    Args:
        alpha_posteriors: Mapping temperature → 1-D array of posterior
            draws for α.
        ci: Width of the credible interval to display (default 90%).
        figsize: Figure size in inches.
        save_path: If given, save figure to this path.

    Returns:
        The matplotlib Figure, or None if matplotlib unavailable.
    """
    if not _check_matplotlib():
        return None

    temps = sorted(alpha_posteriors)
    n = len(temps)

    fig, ax = plt.subplots(figsize=figsize)

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    y_positions = list(range(n))

    for i, temp in enumerate(temps):
        draws = alpha_posteriors[temp]
        median = float(np.median(draws))
        lo = float(np.quantile(draws, lower_q))
        hi = float(np.quantile(draws, upper_q))

        color = TEMP_COLORS[i % len(TEMP_COLORS)]
        ax.plot(
            [lo, hi], [i, i], color=color, linewidth=2.5, solid_capstyle="round"
        )
        ax.plot(median, i, "o", color=color, markersize=8, zorder=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"T = {t}" for t in temps])
    ax.set_xlabel(r"$\alpha$ (sensitivity parameter)")
    ax.set_title(f"Posterior {int(ci * 100)}% CIs for α by Temperature")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Forest plot saved to %s", save_path)
    return fig


def alpha_density_plot(
    alpha_posteriors: Dict[float, np.ndarray],
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """
    Overlaid kernel-density plots of α posteriors for each temperature.
    """
    if not _check_matplotlib():
        return None

    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=figsize)

    for i, temp in enumerate(sorted(alpha_posteriors)):
        draws = alpha_posteriors[temp]
        kde = gaussian_kde(draws)
        x = np.linspace(draws.min(), draws.max(), 300)
        color = TEMP_COLORS[i % len(TEMP_COLORS)]
        ax.plot(x, kde(x), color=color, label=f"T = {temp}", linewidth=1.5)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Density")
    ax.set_title("α Posterior Densities by Temperature")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# §6.2  Position bias visualisation
# ------------------------------------------------------------------


def position_bias_bars(
    per_temp_rates: Dict[str, Dict[int, float]],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """
    Grouped bar chart of position choice rates per temperature.

    Args:
        per_temp_rates: Mapping ``str(temp)`` → {position: rate}.
        figsize: Figure size.
        save_path: If given, save figure.
    """
    if not _check_matplotlib():
        return None

    temps = sorted(per_temp_rates, key=float)
    positions = sorted(
        {p for rates in per_temp_rates.values() for p in rates}
    )
    n_temps = len(temps)
    n_pos = len(positions)

    x = np.arange(n_pos)
    width = 0.8 / n_temps

    fig, ax = plt.subplots(figsize=figsize)

    for i, temp in enumerate(temps):
        rates = per_temp_rates[temp]
        vals = [rates.get(p, 0.0) for p in positions]
        color = TEMP_COLORS[i % len(TEMP_COLORS)]
        ax.bar(x + i * width, vals, width, label=f"T = {temp}", color=color)

    # Uniform reference line
    if positions:
        ax.axhline(
            1.0 / max(positions),
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label="Uniform",
        )

    ax.set_xticks(x + width * (n_temps - 1) / 2)
    ax.set_xticklabels([f"Position {p}" for p in positions])
    ax.set_ylabel("Choice rate")
    ax.set_title("Position Choice Rates by Temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# §6.3  Consistency visualisation
# ------------------------------------------------------------------


def consistency_line_plot(
    consistency_summary: Dict[str, Any],
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """
    Line plot of unanimity and modal-agreement rates vs. temperature.

    Args:
        consistency_summary: Output of
            ``consistency_analysis.consistency_by_temperature()``.
    """
    if not _check_matplotlib():
        return None

    per_temp = consistency_summary["per_temperature"]
    temps = sorted(float(t) for t in per_temp)
    una = [per_temp[str(t)]["unanimity"]["rate"] for t in temps]
    modal = [per_temp[str(t)]["modal_agreement"]["mean_rate"] for t in temps]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(temps, una, "o-", color=TEMP_COLORS[0], label="Unanimity rate")
    ax.plot(temps, modal, "s-", color=TEMP_COLORS[1], label="Modal agreement")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Choice Consistency vs. Temperature")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# §6.4  NA rate visualisation
# ------------------------------------------------------------------


def na_rate_bar_chart(
    na_summary: Dict[str, Any],
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """
    Bar chart of NA (non-parseable response) rates per temperature.

    Args:
        na_summary: Output of ``na_analysis.na_rates_by_temperature()``.
    """
    if not _check_matplotlib():
        return None

    per_temp = na_summary["per_temperature"]
    temps = sorted(per_temp, key=float)
    rates = [per_temp[t]["na_rate"] for t in temps]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [TEMP_COLORS[i % len(TEMP_COLORS)] for i in range(len(temps))]
    ax.bar([f"T={t}" for t in temps], rates, color=colors)
    ax.set_ylabel("NA Rate")
    ax.set_title("Non-Parseable Response Rates by Temperature")
    ax.set_ylim(0, max(max(rates) * 1.3, 0.05) if rates else 0.1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# §6.1  Quantitative summaries (non-visual)
# ------------------------------------------------------------------


def posterior_monotonicity_prob(
    alpha_posteriors: Dict[float, np.ndarray],
) -> float:
    """
    Posterior probability that α is strictly decreasing in temperature:
    α(T₁) > α(T₂) > … > α(Tₖ).

    Works sample-by-sample across the posterior draws.  All arrays
    must have the same length.
    """
    temps = sorted(alpha_posteriors)
    if len(temps) < 2:
        return 1.0

    # Align draws (take min length across temperatures)
    min_len = min(len(alpha_posteriors[t]) for t in temps)
    draws = np.column_stack(
        [alpha_posteriors[t][:min_len] for t in temps]
    )  # shape (S, T)

    # Check strict decrease: every adjacent pair
    decreasing = np.all(np.diff(draws, axis=1) < 0, axis=1)
    return round(float(np.mean(decreasing)), 4)


def alpha_slope(
    alpha_posteriors: Dict[float, np.ndarray],
) -> Dict[str, float]:
    """
    Estimated slope Δα per unit increase in temperature, computed by
    OLS regression of posterior medians on temperature.

    Also returns 90% CI on the slope from the posterior draws.
    """
    temps = sorted(alpha_posteriors)
    if len(temps) < 2:
        return {"slope": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    min_len = min(len(alpha_posteriors[t]) for t in temps)
    t_arr = np.array(temps)

    # Compute slope for each posterior draw
    draws = np.column_stack(
        [alpha_posteriors[t][:min_len] for t in temps]
    )
    # Simple OLS: slope = cov(T, alpha) / var(T)  per draw
    t_mean = t_arr.mean()
    t_var = np.var(t_arr, ddof=0)
    if t_var == 0:
        return {"slope": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    slopes = np.array([
        np.sum((t_arr - t_mean) * (draws[s, :] - draws[s, :].mean())) / (t_var * len(t_arr))
        for s in range(min_len)
    ])

    return {
        "slope": round(float(np.median(slopes)), 4),
        "ci_low": round(float(np.quantile(slopes, 0.05)), 4),
        "ci_high": round(float(np.quantile(slopes, 0.95)), 4),
    }


def alpha_summary_table(
    alpha_posteriors: Dict[float, np.ndarray],
    ci: float = 0.90,
) -> List[Dict[str, Any]]:
    """
    Tabular summary of α posteriors for each temperature.

    Returns a list of dicts, one per temperature, with keys:
    ``temperature``, ``median``, ``mean``, ``sd``, ``ci_low``, ``ci_high``.
    """
    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q
    rows = []
    for temp in sorted(alpha_posteriors):
        draws = alpha_posteriors[temp]
        rows.append({
            "temperature": temp,
            "median": round(float(np.median(draws)), 4),
            "mean": round(float(np.mean(draws)), 4),
            "sd": round(float(np.std(draws, ddof=1)), 4),
            "ci_low": round(float(np.quantile(draws, lower_q)), 4),
            "ci_high": round(float(np.quantile(draws, upper_q)), 4),
        })
    return rows

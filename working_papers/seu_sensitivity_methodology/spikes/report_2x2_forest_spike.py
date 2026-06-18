"""
2x2 forest plot of the per-cell global-slope posteriors (methodological paper,
section 7.5.5; companion to @tbl-2x2).

The application's LLM x task factorial summarises each cell by the posterior of
the population-OLS global slope of alpha on sampling temperature, Dalpha/DT. The
body presents the four cells as a table (@tbl-2x2); section 7.5.5 notes the
reading "is best presented as a 2x2 forest plot of the per-cell global-slope
posteriors." This script renders that forest plot.

PROVENANCE (single source of truth = claims_ledger.md). The plotted point +
90% CI + P(slope<0) for each cell are the canonical *report-level* slope
summaries already cited in the body, NOT a re-derivation:

  cell                 ledger  source report
  GPT-4o x Insurance   C9      temperature_study/01_initial_study.qmd
  GPT-4o x Ellsberg    C12     gpt4o_ellsberg_study/data/primary_analysis.json
  Claude x Insurance   C10     claude_insurance_study/01_claude_insurance_study.qmd
  Claude x Ellsberg    C13     ellsberg_study/data/primary_analysis.json

These are exactly the numbers in @tbl-2x2, so the figure and the table agree by
construction.

NOTE ON SLOPE CONVENTIONS (honest-reporting). The report-level slope summaries
above are the canonical paper numbers. A *separate* draw-wise population-OLS
functional b_i = Cov(T, alpha_i)/Var(T) applied to the committed per-condition
alpha draws (the functional used by the C16 MDE spike) reproduces the report
summaries for the two Ellsberg cells but gives smaller-magnitude medians for the
two insurance cells (GPT-4o ~ -24.6 vs -31; Claude ~ -2.9 vs -3.6) at an
identical P(slope<0). The discrepancy is a fixed per-study scale on the insurance
cells and is already documented in C16 (Claude insurance: population-OLS -2.89 vs
report-level -3.6). This script PLOTS the canonical report-level values (so it
matches @tbl-2x2) and, when the draws are present, records the population-OLS
cross-check in the results JSON for transparency only.

Outputs:
  report_2x2_forest_results.json              (next to this script)
  ../figures/report_2x2_forest.png            (2x2 forest plot)

Run:
  python working_papers/seu_sensitivity_methodology/spikes/report_2x2_forest_spike.py
"""
from __future__ import annotations

import json
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))

# ---------------------------------------------------------------------------
# Canonical per-cell slope summaries (claims_ledger.md rows C9, C10, C12, C13).
# slope = global OLS Dalpha/DT; negative => alpha declines with temperature.
# ci is the 90% credible interval; p_neg = P(slope < 0).
# ---------------------------------------------------------------------------
CELLS = [
    {
        "llm": "GPT-4o", "task": "Insurance", "ledger": "C9",
        "median": -31.0, "ci_low": -66.0, "ci_high": -8.0, "p_neg": 0.99,
        "source": "temperature_study/01_initial_study.qmd",
    },
    {
        "llm": "GPT-4o", "task": "Ellsberg", "ledger": "C12",
        "median": -38.4, "ci_low": -72.1, "ci_high": -10.0, "p_neg": 0.984,
        "source": "gpt4o_ellsberg_study/data/primary_analysis.json",
    },
    {
        "llm": "Claude 3.5", "task": "Insurance", "ledger": "C10",
        "median": -3.6, "ci_low": -54.0, "ci_high": 39.0, "p_neg": 0.56,
        "source": "claude_insurance_study/01_claude_insurance_study.qmd",
    },
    {
        "llm": "Claude 3.5", "task": "Ellsberg", "ledger": "C13",
        "median": -18.8, "ci_low": -65.3, "ci_high": 24.5, "p_neg": 0.766,
        "source": "ellsberg_study/data/primary_analysis.json",
    },
]

# Per-cell committed alpha-draw locations, for the optional population-OLS
# cross-check only (NOT the plotted values).
DRAW_SPEC = {
    "C9": ("temperature_study", [0.0, 0.3, 0.7, 1.0, 1.5],
           ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"]),
    "C12": ("gpt4o_ellsberg_study", [0.0, 0.3, 0.7, 1.0, 1.5],
            ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"]),
    "C10": ("claude_insurance_study", [0.0, 0.2, 0.5, 0.8, 1.0],
            ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"]),
    "C13": ("ellsberg_study", [0.0, 0.2, 0.5, 0.8, 1.0],
            ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"]),
}
DATA_BASE = os.path.join(PROJECT_ROOT, "reports", "applications")


def population_ols_slope(ledger: str):
    """Draw-wise b_i = Cov(T, alpha_i)/Var(T); cross-check only. None if absent."""
    study, temps, keys = DRAW_SPEC[ledger]
    temps = np.asarray(temps, dtype=float)
    tc = temps - temps.mean()
    w = tc / np.sum(tc**2)
    cols = []
    for k in keys:
        path = os.path.join(DATA_BASE, study, "data", f"alpha_draws_{k}.npz")
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            cols.append(np.asarray(z["alpha"], dtype=float))
    n = min(c.shape[0] for c in cols)
    s = np.column_stack([c[:n] for c in cols]) @ w
    return {
        "median": float(np.median(s)),
        "ci_low": float(np.quantile(s, 0.05)),
        "ci_high": float(np.quantile(s, 0.95)),
        "p_neg": float(np.mean(s < 0.0)),
        "n_draws": int(n),
    }


def main() -> None:
    crosscheck = {}
    for c in CELLS:
        cc = population_ols_slope(c["ledger"])
        if cc is not None:
            crosscheck[c["ledger"]] = {
                "cell": f"{c['llm']} x {c['task']}",
                "report_level_median": c["median"],
                "population_ols": cc,
                "p_neg_agrees": abs(cc["p_neg"] - c["p_neg"]) < 0.02,
            }

    results = {
        "spec": {
            "section": "7.5.5",
            "companion_table": "@tbl-2x2",
            "quantity": "global OLS slope Dalpha/DT (alpha-units per unit T)",
            "plotted_values": "canonical report-level summaries "
                              "(claims_ledger C9/C10/C12/C13)",
            "ci_level": 0.90,
        },
        "cells": CELLS,
        "population_ols_crosscheck": {
            "note": "diagnostic only; NOT plotted. Reproduces the report-level "
                    "summaries for the Ellsberg cells; smaller-magnitude medians "
                    "for the insurance cells at identical P(slope<0) (see C16).",
            "by_ledger": crosscheck,
        },
    }
    out_path = os.path.join(THIS_DIR, "report_2x2_forest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== 7.5.5: 2x2 forest plot of per-cell global-slope posteriors ===\n")
    print(f"{'cell':22s} {'median':>8s}  {'90% CI':>18s}  {'P(<0)':>6s}  ledger")
    for c in CELLS:
        ci = f"[{c['ci_low']:.1f}, {c['ci_high']:.1f}]"
        print(f"{c['llm'] + ' x ' + c['task']:22s} {c['median']:8.1f}  "
              f"{ci:>18s}  {c['p_neg']:6.3f}  {c['ledger']}")
    if crosscheck:
        print("\npopulation-OLS cross-check (diagnostic only, not plotted):")
        for lid, cc in crosscheck.items():
            o = cc["population_ols"]
            print(f"  {cc['cell']:22s} pop-OLS median {o['median']:7.2f} "
                  f"P(<0) {o['p_neg']:.3f}  (report-level {cc['report_level_median']:.1f}; "
                  f"P agrees: {cc['p_neg_agrees']})")
    print(f"\nResults written to {out_path}")

    make_figure(CELLS)


def make_figure(cells) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Order top-to-bottom; group by LLM.
    order = ["GPT-4o", "Claude 3.5"]
    rows = []
    for llm in order:
        for c in cells:
            if c["llm"] == llm:
                rows.append(c)
    rows = rows[::-1]  # matplotlib y increases upward; reverse for top-down read
    y = np.arange(len(rows))

    detected = "#1f77b4"   # CI excludes 0
    inconclusive = "#7f7f7f"  # CI straddles 0

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for yi, c in zip(y, rows):
        excludes_zero = (c["ci_low"] < 0 and c["ci_high"] < 0) or \
                        (c["ci_low"] > 0 and c["ci_high"] > 0)
        col = detected if excludes_zero else inconclusive
        ax.plot([c["ci_low"], c["ci_high"]], [yi, yi], "-", color=col, lw=2.2,
                solid_capstyle="round")
        ax.plot(c["median"], yi, "o", color=col, ms=9, zorder=3)
        # P(slope<0) annotation at the right margin
        ax.text(1.012, yi, f"$P_{{<0}}={c['p_neg']:.2f}$",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9.5, color=col)

    ax.axvline(0.0, color="#d62728", ls="--", lw=1.3, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['llm']} $\\times$ {c['task']}" for c in rows],
                       fontsize=10.5)
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.7)
    ax.set_xlabel(r"global slope $\Delta\alpha/\Delta T$  "
                  r"($\alpha$-units per unit temperature)")
    ax.set_title(r"Per-cell global-slope posteriors (median, 90% CI)"
                 "\nGPT-4o declines in both tasks; Claude in neither",
                 fontsize=11.5)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=detected, lw=2.2, marker="o", ms=8,
               label="90% CI excludes 0 (effect detected)"),
        Line2D([0], [0], color=inconclusive, lw=2.2, marker="o", ms=8,
               label="90% CI straddles 0 (does not detect)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.subplots_adjust(right=0.80)
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    fig_dir = os.path.join(PAPER_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "report_2x2_forest.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()

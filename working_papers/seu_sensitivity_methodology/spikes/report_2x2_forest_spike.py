"""
2x2 forest plot of the per-cell global-slope posteriors (methodological paper,
section 7.5.5; companion to @tbl-2x2), plus the canonical slope computation.

The application's LLM x task factorial summarises each cell by the posterior of
the population-OLS global slope of alpha on sampling temperature, Dalpha/DT.
This script COMPUTES the canonical per-cell slope posteriors directly from the
committed per-condition alpha draws and renders the section 7.5.5 forest plot,
so the table (@tbl-2x2), the figure, and the ledger rows C9/C10/C12/C13 agree
by construction and share a single estimator.

CANONICAL ESTIMATOR (single convention, paper-wide). For posterior draw s,
  b^(s) = Cov(T, alpha^(s)) / Var(T)   (population moments, equivalently the
          OLS weights w = (T - Tbar) / sum((T - Tbar)^2))
applied draw-wise to the five per-condition alpha draws, pairing draws by the
exchangeable draw index. Reported: median, 90% CI (5-95%), P(slope < 0).
This is the same functional used by the C16 MDE spike (report16_mde_spike.py),
the cross-LLM spike (report11), and the prior-sensitivity spike.

NOTE ON THE SUPERSEDED ESTIMATOR (honest-reporting; see claims_ledger C16).
Earlier report-level slope summaries (temperature_study /
claude_insurance_study / ellsberg_study report qmds and
scripts/generate_ellsberg_primary_analysis.py) computed the per-draw slope as
  np.cov(T, alpha)[0, 1] / np.var(T),
which mixes a ddof=1 covariance with a ddof=0 variance and therefore inflates
every draw's slope -- hence the posterior median AND the CI endpoints -- by
exactly n/(n-1) = 5/4 = 1.25 for the five-point temperature grids. P(slope<0)
is scale-invariant, which is why it always agreed across the two computations.
The previously reported -31 / -3.6 / -18.8 medians are the inflated values
(-24.6 x 1.25, -2.89 x 1.25, -15.02 x 1.25); GPT-4o x Ellsberg used the correct
formula (scripts/generate_primary_analysis.py) and is unchanged at -38.4. This
script records the superseded values and the exact 1.25 factor in the results
JSON for the E.2 reconciliation note.

Outputs:
  report_2x2_forest_results.json              (next to this script)
  ../figures/report_2x2_forest.png            (2x2 forest plot)

Run:
  conda run -n seu-sensitivity python \
    working_papers/seu_sensitivity_methodology/spikes/report_2x2_forest_spike.py
"""
from __future__ import annotations

import json
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))

# ---------------------------------------------------------------------------
# Cell specifications: committed alpha-draw locations + temperature grids.
# slope = global population-OLS Dalpha/DT; negative => alpha declines with T.
# superseded_median = the previously reported (ddof-inflated) headline value.
# ---------------------------------------------------------------------------
CELLS = [
    {
        "llm": "GPT-4o", "task": "Insurance", "ledger": "C9",
        "study": "temperature_study",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
        "superseded_median": -31.0,
    },
    {
        "llm": "GPT-4o", "task": "Ellsberg", "ledger": "C12",
        "study": "gpt4o_ellsberg_study",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
        "superseded_median": -38.4,  # correct formula was used; unchanged
    },
    {
        "llm": "Claude 3.5", "task": "Insurance", "ledger": "C10",
        "study": "claude_insurance_study",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
        "superseded_median": -3.6,
    },
    {
        "llm": "Claude 3.5", "task": "Ellsberg", "ledger": "C13",
        "study": "ellsberg_study",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
        "superseded_median": -18.8,
    },
]
DATA_BASE = os.path.join(PROJECT_ROOT, "reports", "applications")


def slope_posterior(cell: dict) -> dict:
    """Draw-wise population-OLS slope b^(s) = Cov(T, alpha^(s))/Var(T)."""
    temps = np.asarray(cell["temps"], dtype=float)
    tc = temps - temps.mean()
    w = tc / np.sum(tc**2)
    cols = []
    for k in cell["keys"]:
        path = os.path.join(DATA_BASE, cell["study"], "data",
                            f"alpha_draws_{k}.npz")
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
    rows = []
    for c in CELLS:
        post = slope_posterior(c)
        inflation = (c["superseded_median"] / post["median"]
                     if post["median"] != 0 else float("nan"))
        rows.append({
            "llm": c["llm"], "task": c["task"], "ledger": c["ledger"],
            "study": c["study"], "temps": c["temps"],
            **post,
            "superseded_median": c["superseded_median"],
            "superseded_over_canonical": round(inflation, 4),
        })

    results = {
        "spec": {
            "section": "7.5.5",
            "companion_table": "@tbl-2x2",
            "quantity": "global population-OLS slope Dalpha/DT "
                        "(alpha-units per unit T)",
            "estimator": "draw-wise b = Cov(T, alpha)/Var(T), population "
                         "moments; single canonical convention paper-wide",
            "ci_level": 0.90,
        },
        "cells": rows,
        "superseded_estimator_note": (
            "Earlier report-level summaries used np.cov(T, a)[0,1]/np.var(T) "
            "(ddof=1 covariance over ddof=0 variance), inflating each draw's "
            "slope -- median and CI endpoints alike -- by n/(n-1) = 1.25 for "
            "the 5-point grids; P(slope<0) is scale-invariant and unaffected. "
            "GPT-4o x Ellsberg (C12) used the correct formula and is "
            "unchanged. See claims_ledger C16 and Appendix E.2."
        ),
    }
    out_path = os.path.join(THIS_DIR, "report_2x2_forest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== 7.5.5: per-cell global-slope posteriors (canonical pop-OLS) ===\n")
    print(f"{'cell':22s} {'median':>8s}  {'90% CI':>18s}  {'P(<0)':>6s}  "
          f"{'superseded':>10s}  ledger")
    for r in rows:
        ci = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        print(f"{r['llm'] + ' x ' + r['task']:22s} {r['median']:8.1f}  "
              f"{ci:>18s}  {r['p_neg']:6.3f}  {r['superseded_median']:10.1f}  "
              f"{r['ledger']}")
    print(f"\nResults written to {out_path}")

    make_figure(rows)


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
                 "\nGPT-4o declines detected in both tasks; "
                 "Claude cells inconclusive",
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

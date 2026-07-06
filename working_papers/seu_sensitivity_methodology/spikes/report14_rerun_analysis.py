"""
Report-14 re-run analysis (methodological paper, gate 2).

Consumes the n>=100 matched-design recovery study
(`results/parameter_recovery/m1_matched_comparison_n100/`) and reports the
headline §6.4 numbers as **paired-iteration medians / bootstrap 90% CIs**, as
required by the plan ("Pre-drafting action items" #2; Definition-of-done gate 2).

Headline quantities (plan §6.4.1–6.4.2, claims-ledger rows C1–C3):
  C1  α RMSE reduction at matched choice count (B vs C)        -> expect ~15%
  C2  α RMSE change when adding only uncertain data (A vs B)   -> expect slight WORSENING
  C3  δ CI-width / δ RMSE reduction at matched count (B vs C)  -> expect ~2%, Wilcoxon-significant

Matched conditions (Report 14):
  A: m_0, M=25            B: m_0, M=50
  C: m_1, M=25 + N=25     D: m_1, M=50 + N=50
The central test is B vs C (same total choice count; only model + choice TYPE differ).
A vs B is the data-quantity control inside m_0.

Reporting conventions:
  - RMSE = sqrt(mean_i squared-error); its reduction CI comes from bootstrap
    over iterations (RMSE is an aggregate, not a per-iteration quantity).
  - CI-width reduction is also reported as a *paired-iteration* median of the
    per-iteration relative reduction, plus a bootstrap 90% CI -- the cleanest
    "paired-iteration median" the plan asks for.
  - Wilcoxon signed-rank on paired CI widths / squared errors gives the
    direction-of-effect significance.

Run (after the n=100 study finishes):
  python working_papers/seu_sensitivity_methodology/spikes/report14_rerun_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))

ROOT = os.path.join(
    PROJECT_ROOT, "results", "parameter_recovery", "m1_matched_comparison_n100"
)
CONDITIONS = ["A_m0_M25", "B_m0_M50", "C_m1_M25N25", "D_m1_M50N50"]
COND_LABELS = {
    "A_m0_M25": "A: m_0, M=25",
    "B_m0_M50": "B: m_0, M=50",
    "C_m1_M25N25": "C: m_1, M=25+N=25",
    "D_m1_M50N50": "D: m_1, M=50+N=50",
}
K, D_DIM = 3, 5
N_BOOT = 10000
BOOT_SEED = 20260617


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def load_all():
    it_dirs = sorted(
        glob.glob(os.path.join(ROOT, "iteration_*")),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    tp_list, sm_lists = [], {c: [] for c in CONDITIONS}
    for d in it_dirs:
        tp_path = os.path.join(d, "true_parameters.json")
        if not os.path.exists(tp_path):
            continue
        with open(tp_path) as f:
            tp = json.load(f)
        summaries, ok = {}, True
        for c in CONDITIONS:
            p = os.path.join(d, f"summary_{c}.csv")
            if not os.path.exists(p):
                ok = False
                break
            summaries[c] = pd.read_csv(p, index_col=0)
        if not ok:
            continue
        tp_list.append(tp)
        for c in CONDITIONS:
            sm_lists[c].append(summaries[c])
    return tp_list, sm_lists


# ----------------------------------------------------------------------------
# Per-iteration arrays
# ----------------------------------------------------------------------------
def alpha_arrays(true_params, sm):
    a_true = np.array([p["alpha"] for p in true_params])
    a_mean = np.array([s.loc["alpha", "Mean"] for s in sm])
    a_low = np.array([s.loc["alpha", "5%"] for s in sm])
    a_up = np.array([s.loc["alpha", "95%"] for s in sm])
    return a_true, a_mean, a_low, a_up


def delta_arrays(true_params, sm):
    """Per-iteration δ squared error (mean over components) and CI width."""
    n = len(sm)
    se = np.zeros(n)
    ciw = np.zeros(n)
    for i, s in enumerate(sm):
        se_k, ciw_k = [], []
        for k in range(K - 1):
            dt = true_params[i]["delta"][k]
            dm = s.loc[f"delta[{k+1}]", "Mean"]
            dl = s.loc[f"delta[{k+1}]", "5%"]
            du = s.loc[f"delta[{k+1}]", "95%"]
            se_k.append((dm - dt) ** 2)
            ciw_k.append(du - dl)
        se[i] = np.mean(se_k)
        ciw[i] = np.mean(ciw_k)
    return se, ciw


# ----------------------------------------------------------------------------
# Bootstrap helpers
# ----------------------------------------------------------------------------
def boot_indices(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def rmse_reduction(se_from, se_to, idx_matrix):
    """% RMSE reduction from->to (aggregate sqrt-mean) with bootstrap CI.

    reduction > 0 means `to` has lower RMSE (improvement).
    """
    rmse_from = np.sqrt(np.mean(se_from))
    rmse_to = np.sqrt(np.mean(se_to))
    point = (rmse_from - rmse_to) / rmse_from * 100.0
    boots = []
    for row in idx_matrix:
        rf = np.sqrt(np.mean(se_from[row]))
        rt = np.sqrt(np.mean(se_to[row]))
        boots.append((rf - rt) / rf * 100.0)
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "point_pct": float(point),
        "boot_median_pct": float(np.median(boots)),
        "ci90_pct": [float(lo), float(hi)],
        "rmse_from": float(rmse_from),
        "rmse_to": float(rmse_to),
    }


def paired_median_reduction(x_from, x_to, idx_matrix):
    """Median per-iteration RELATIVE reduction (x_from->x_to) + bootstrap CI.

    Per-iteration r_i = (x_from_i - x_to_i)/x_from_i; report median_i r_i.
    reduction > 0 means `to` is smaller (improvement).
    """
    r = (x_from - x_to) / x_from * 100.0
    point = float(np.median(r))
    boots = [float(np.median(r[row])) for row in idx_matrix]
    lo, hi = np.percentile(boots, [5, 95])
    return {
        "paired_median_pct": point,
        "boot_median_pct": float(np.median(boots)),
        "ci90_pct": [float(lo), float(hi)],
        "share_improved": float(np.mean(x_to < x_from)),
    }


def wilcoxon_pair(x_from, x_to):
    """Signed-rank on (from - to); positive statistic side = improvement."""
    diff = x_from - x_to
    nz = diff[diff != 0]
    if nz.size < 1:
        return {"W": float("nan"), "p": float("nan"), "n_nonzero": 0}
    w, p = stats.wilcoxon(nz)
    return {"W": float(w), "p": float(p), "n_nonzero": int(nz.size)}


# ----------------------------------------------------------------------------
# Aggregate per-condition table (mirrors Report 14)
# ----------------------------------------------------------------------------
def aggregate_table(true_params, summaries_by_cond):
    rows = []
    for c in CONDITIONS:
        sm = summaries_by_cond[c]
        a_true, a_mean, a_low, a_up = alpha_arrays(true_params, sm)
        b_rmse, b_ci, b_cov = [], [], []
        for k in range(K):
            for d_ix in range(D_DIM):
                bt = np.array([p["beta"][k][d_ix] for p in true_params])
                bm = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "Mean"] for s in sm])
                bl = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "5%"] for s in sm])
                bu = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "95%"] for s in sm])
                b_rmse.append(np.sqrt(np.mean((bm - bt) ** 2)))
                b_ci.append(np.mean(bu - bl))
                b_cov.append(np.mean((bt >= bl) & (bt <= bu)))
        d_se, d_ciw = delta_arrays(true_params, sm)
        d_cov = []
        for k in range(K - 1):
            dt = np.array([p["delta"][k] for p in true_params])
            dl = np.array([s.loc[f"delta[{k+1}]", "5%"] for s in sm])
            du = np.array([s.loc[f"delta[{k+1}]", "95%"] for s in sm])
            d_cov.append(np.mean((dt >= dl) & (dt <= du)))
        rows.append({
            "Condition": COND_LABELS[c],
            "n_iter": len(sm),
            "alpha_RMSE": float(np.sqrt(np.mean((a_mean - a_true) ** 2))),
            "alpha_CI": float(np.mean(a_up - a_low)),
            "alpha_cov": float(np.mean((a_true >= a_low) & (a_true <= a_up))),
            "beta_RMSE": float(np.mean(b_rmse)),
            "beta_CI": float(np.mean(b_ci)),
            "beta_cov": float(np.mean(b_cov)),
            "delta_RMSE": float(np.sqrt(np.mean(d_se))),
            "delta_CI": float(np.mean(d_ciw)),
            "delta_cov": float(np.mean(d_cov)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    true_params, summaries_by_cond = load_all()
    n_iter = len(true_params)
    print(f"Iterations completing all 4 conditions: {n_iter}")
    if n_iter < 100:
        print(f"WARNING: only {n_iter} complete iterations (<100); run may be "
              f"unfinished or some fits errored.")
    if n_iter == 0:
        sys.exit("No complete iterations found; aborting.")

    idx = boot_indices(n_iter, N_BOOT, BOOT_SEED)

    # Per-iteration arrays.
    aA = alpha_arrays(true_params, summaries_by_cond["A_m0_M25"])
    aB = alpha_arrays(true_params, summaries_by_cond["B_m0_M50"])
    aC = alpha_arrays(true_params, summaries_by_cond["C_m1_M25N25"])
    seA = (aA[1] - aA[0]) ** 2
    seB = (aB[1] - aB[0]) ** 2
    seC = (aC[1] - aC[0]) ** 2
    ciwA_a = aA[3] - aA[2]
    ciwB_a = aB[3] - aB[2]
    ciwC_a = aC[3] - aC[2]

    dseB, dciwB = delta_arrays(true_params, summaries_by_cond["B_m0_M50"])
    dseC, dciwC = delta_arrays(true_params, summaries_by_cond["C_m1_M25N25"])

    headline = {
        "n_iter": n_iter,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        # C1: alpha RMSE reduction B -> C (matched count). Expect ~15%.
        "C1_alpha_rmse_reduction_BtoC": rmse_reduction(seB, seC, idx),
        "C1b_alpha_ciw_paired_median_BtoC": paired_median_reduction(ciwB_a, ciwC_a, idx),
        "C1c_alpha_ciw_wilcoxon_BtoC": wilcoxon_pair(ciwB_a, ciwC_a),
        # C2: alpha RMSE change A -> B (data-quantity control). Expect slight worsening (<0).
        "C2_alpha_rmse_reduction_AtoB": rmse_reduction(seA, seB, idx),
        "C2b_alpha_ciw_paired_median_AtoB": paired_median_reduction(ciwA_a, ciwB_a, idx),
        # C3: delta CI-width and RMSE reduction B -> C. Expect ~2%, Wilcoxon-significant.
        "C3_delta_rmse_reduction_BtoC": rmse_reduction(dseB, dseC, idx),
        "C3b_delta_ciw_paired_median_BtoC": paired_median_reduction(dciwB, dciwC, idx),
        "C3c_delta_ciw_wilcoxon_BtoC": wilcoxon_pair(dciwB, dciwC),
        "C3d_delta_se_wilcoxon_BtoC": wilcoxon_pair(dseB, dseC),
    }

    table = aggregate_table(true_params, summaries_by_cond)

    # ---- print ----
    pd.set_option("display.width", 160)
    print("\n=== Aggregate per-condition recovery (n = %d) ===" % n_iter)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    def show(tag, d):
        print(f"\n{tag}")
        for k, v in d.items():
            print(f"    {k}: {v}")

    print("\n=== Headline contrasts (paired-iteration medians / bootstrap 90% CI) ===")
    c1 = headline["C1_alpha_rmse_reduction_BtoC"]
    print(f"\nC1  α RMSE reduction  B→C (matched count): "
          f"{c1['point_pct']:.1f}%  (90% CI [{c1['ci90_pct'][0]:.1f}, {c1['ci90_pct'][1]:.1f}])")
    c2 = headline["C2_alpha_rmse_reduction_AtoB"]
    print(f"C2  α RMSE change     A→B (more uncertain only): "
          f"{c2['point_pct']:.1f}%  (90% CI [{c2['ci90_pct'][0]:.1f}, {c2['ci90_pct'][1]:.1f}])  "
          f"[negative = worsening]")
    c3 = headline["C3_delta_rmse_reduction_BtoC"]
    print(f"C3  δ RMSE reduction  B→C: "
          f"{c3['point_pct']:.1f}%  (90% CI [{c3['ci90_pct'][0]:.1f}, {c3['ci90_pct'][1]:.1f}])")
    c3b = headline["C3b_delta_ciw_paired_median_BtoC"]
    print(f"C3b δ CI-width paired-median reduction B→C: "
          f"{c3b['paired_median_pct']:.1f}%  (90% CI [{c3b['ci90_pct'][0]:.1f}, {c3b['ci90_pct'][1]:.1f}])  "
          f"[narrower in {c3b['share_improved']*100:.0f}% of iters]")
    c3c = headline["C3c_delta_ciw_wilcoxon_BtoC"]
    print(f"C3c δ CI-width Wilcoxon (B vs C): W={c3c['W']:.1f}, p={c3c['p']:.4g}")

    out = {
        "headline": headline,
        "aggregate_table": table.to_dict(orient="records"),
        "results_dir": ROOT,
    }
    out_path = os.path.join(THIS_DIR, "report14_rerun_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")

    make_figure(seA, seB, seC, dciwB, dciwC, dseB, dseC, ciwB_a, ciwC_a, headline)


def make_figure(seA, seB, seC, dciwB, dciwC, dseB, dseC, ciwB_a, ciwC_a, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: alpha CI width B vs C (within-iteration).
    # Title reports the alpha CI-WIDTH contrast (C1b) -- the quantity the axes
    # show. Sign convention matches the paper (positive % = improvement =
    # narrower interval / smaller RMSE under C).
    ax = axes[0]
    lim = max(ciwB_a.max(), ciwC_a.max()) * 1.05
    ax.scatter(ciwB_a, ciwC_a, s=28, alpha=0.6, c="#1f77b4", edgecolor="white")
    ax.plot([0, lim], [0, lim], "r--", lw=1.3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("B (m_0, M=50)  α CI width")
    ax.set_ylabel("C (m_1, M=25+N=25)  α CI width")
    c1b = headline["C1b_alpha_ciw_paired_median_BtoC"]
    ax.set_title(f"α CI width: B→C {c1b['paired_median_pct']:+.1f}% "
                 f"[{c1b['ci90_pct'][0]:+.1f}, {c1b['ci90_pct'][1]:+.1f}]\n"
                 f"(positive = narrower under C; no detected gain)")
    ax.grid(True, alpha=0.3)

    # Panel 2: delta CI width B vs C. Title uses the same positive-=-
    # improvement convention as the text and Table 3 (+0.8% [0.6, 1.2]).
    ax = axes[1]
    lim = max(dciwB.max(), dciwC.max()) * 1.05
    ax.scatter(dciwB, dciwC, s=28, alpha=0.6, c="#2ca02c", edgecolor="white")
    ax.plot([0, lim], [0, lim], "r--", lw=1.3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("B  δ CI width")
    ax.set_ylabel("C  δ CI width")
    c3b = headline["C3b_delta_ciw_paired_median_BtoC"]
    ax.set_title(f"δ CI width: B→C {c3b['paired_median_pct']:+.1f}% "
                 f"[{c3b['ci90_pct'][0]:+.1f}, {c3b['ci90_pct'][1]:+.1f}]\n"
                 f"(positive = narrower under C; real but negligible)")
    ax.grid(True, alpha=0.3)

    # Panel 3: RMSE bars A/B/C for alpha and delta (relative to B).
    ax = axes[2]
    rmse_a = [np.sqrt(np.mean(seA)), np.sqrt(np.mean(seB)), np.sqrt(np.mean(seC))]
    rmse_d = [np.nan, np.sqrt(np.mean(dseB)), np.sqrt(np.mean(dseC))]
    xs = np.arange(3)
    ax.bar(xs - 0.18, np.array(rmse_a) / rmse_a[1], width=0.36, label="α (rel. B)",
           color="#1f77b4")
    ax.bar(xs + 0.18, np.array(rmse_d) / rmse_d[1], width=0.36, label="δ (rel. B)",
           color="#2ca02c")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(["A: M25", "B: M50", "C: M25+N25"])
    ax.set_ylabel("RMSE relative to B")
    ax.set_title("A→B→C RMSE (matched at B vs C)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(PAPER_DIR, "figures", "report14_rerun_matched.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()

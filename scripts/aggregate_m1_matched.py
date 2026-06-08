"""Aggregate matched-design recovery results into per-condition metrics."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "results/parameter_recovery/m1_matched_comparison"
CONDITIONS = ["A_m0_M25", "B_m0_M50", "C_m1_M25N25", "D_m1_M50N50"]
K = 3
D = 5


def load_iteration(it_dir):
    with open(os.path.join(it_dir, "true_parameters.json")) as f:
        tp = json.load(f)
    summaries = {}
    for cond in CONDITIONS:
        p = os.path.join(it_dir, f"summary_{cond}.csv")
        if os.path.exists(p):
            summaries[cond] = pd.read_csv(p, index_col=0)
    return tp, summaries


def metrics(true_vals, mean, low, up):
    err = mean - true_vals
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "ci_width": float(np.mean(up - low)),
        "coverage": float(np.mean((true_vals >= low) & (true_vals <= up))),
        "n": int(len(true_vals)),
    }


def aggregate():
    iter_dirs = sorted(glob.glob(os.path.join(ROOT, "iteration_*")),
                       key=lambda p: int(p.split("_")[-1]))
    all_tp = []
    all_sm = {c: [] for c in CONDITIONS}
    for d_ in iter_dirs:
        tp, sm = load_iteration(d_)
        # Only count iterations where all 4 conditions completed
        if not all(c in sm for c in CONDITIONS):
            continue
        all_tp.append(tp)
        for c in CONDITIONS:
            all_sm[c].append(sm[c])

    print(f"Iterations completing all 4 conditions: {len(all_tp)} / {len(iter_dirs)}")

    # Build per-condition metrics for alpha, beta (avg over K*D), delta (avg over K-1)
    rows = []
    for c in CONDITIONS:
        sm_list = all_sm[c]
        a_true = np.array([p["alpha"] for p in all_tp])
        a_mean = np.array([s.loc["alpha", "Mean"] for s in sm_list])
        a_low = np.array([s.loc["alpha", "5%"] for s in sm_list])
        a_up  = np.array([s.loc["alpha", "95%"] for s in sm_list])
        m_a = metrics(a_true, a_mean, a_low, a_up)

        rmses, ciws, covs, biases = [], [], [], []
        for k in range(K):
            for d_ix in range(D):
                bt = np.array([p["beta"][k][d_ix] for p in all_tp])
                bm = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "Mean"] for s in sm_list])
                bl = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "5%"] for s in sm_list])
                bu = np.array([s.loc[f"beta[{k+1},{d_ix+1}]", "95%"] for s in sm_list])
                mm = metrics(bt, bm, bl, bu)
                rmses.append(mm["rmse"]); ciws.append(mm["ci_width"])
                covs.append(mm["coverage"]); biases.append(mm["bias"])
        m_b = {"rmse": float(np.mean(rmses)), "ci_width": float(np.mean(ciws)),
               "coverage": float(np.mean(covs)), "bias": float(np.mean(biases))}

        d_rmse, d_ciw, d_cov, d_bias = [], [], [], []
        for k in range(K - 1):
            dt = np.array([p["delta"][k] for p in all_tp])
            dm = np.array([s.loc[f"delta[{k+1}]", "Mean"] for s in sm_list])
            dl = np.array([s.loc[f"delta[{k+1}]", "5%"] for s in sm_list])
            du = np.array([s.loc[f"delta[{k+1}]", "95%"] for s in sm_list])
            mm = metrics(dt, dm, dl, du)
            d_rmse.append(mm["rmse"]); d_ciw.append(mm["ci_width"])
            d_cov.append(mm["coverage"]); d_bias.append(mm["bias"])
        m_d = {"rmse": float(np.mean(d_rmse)), "ci_width": float(np.mean(d_ciw)),
               "coverage": float(np.mean(d_cov)), "bias": float(np.mean(d_bias))}

        rows.append({"condition": c, "n_iter": len(sm_list),
                     "alpha_rmse": m_a["rmse"], "alpha_ci": m_a["ci_width"], "alpha_cov": m_a["coverage"],
                     "beta_rmse": m_b["rmse"], "beta_ci": m_b["ci_width"], "beta_cov": m_b["coverage"],
                     "delta_rmse": m_d["rmse"], "delta_ci": m_d["ci_width"], "delta_cov": m_d["coverage"]})

    df = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "aggregate_metrics.csv")
    df.to_csv(out_path, index=False)

    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved to {out_path}")

    # Pairwise within-iteration deltas (B vs C, the key comparison)
    print("\n--- Within-iteration B (m_0 M=50) vs C (m_1 M=25+N=25) ---")
    bC, dC = [], []
    bB, dB = [], []
    for tp, smB, smC in zip(all_tp, all_sm["B_m0_M50"], all_sm["C_m1_M25N25"]):
        for k in range(K - 1):
            dt = tp["delta"][k]
            ciB = smB.loc[f"delta[{k+1}]", "95%"] - smB.loc[f"delta[{k+1}]", "5%"]
            ciC = smC.loc[f"delta[{k+1}]", "95%"] - smC.loc[f"delta[{k+1}]", "5%"]
            errB = smB.loc[f"delta[{k+1}]", "Mean"] - dt
            errC = smC.loc[f"delta[{k+1}]", "Mean"] - dt
            bC.append(ciC); bB.append(ciB)
            dC.append(errC ** 2); dB.append(errB ** 2)
    bB, bC, dB, dC = map(np.array, (bB, bC, dB, dC))
    print(f"delta CI width:  B median={np.median(bB):.3f}  C median={np.median(bC):.3f}  "
          f"diff median={np.median(bC - bB):+.3f}  (negative = C narrower)")
    print(f"delta squared error: B mean={dB.mean():.4f}  C mean={dC.mean():.4f}  "
          f"  (RMSE B={np.sqrt(dB.mean()):.3f}, C={np.sqrt(dC.mean()):.3f})")
    # Wilcoxon-style: fraction of iterations where C beats B on delta CI width
    n_pairs = len(bB)
    print(f"Fraction (per-component-iter): C narrower than B = {(bC < bB).mean():.2%} "
          f"of {n_pairs} (component,iteration) pairs")


if __name__ == "__main__":
    aggregate()

"""
Design-level identification diagnostics for the methodological paper.

Motivated by the 2026-07-05 review (overall themes A and B; detailed comments
1, 2, 8, 12): the paper's identification claims for alpha are conditional on
the expected-utility vector eta, the "irreducible nuisance" status of beta is a
design-specific rank condition rather than a structural fact, and the lottery
spanning condition of Proposition 5.2 should be verified rather than inferred
from the lottery count. This spike computes, from committed designs and
committed posterior draws only (no refits):

(1) FEATURE-MATRIX RANK per application condition: row rank of the R x D
    feature matrix w (R=30 items, D=32 PCA features). Full row rank means the
    linear belief map can represent arbitrary item-level logits, so the
    embedding features impose no cross-item constraint beyond the prior
    (review comment 1).

(2) BETA-MAP JACOBIAN RANK (generic, modulo the row-shift gauge): rank of
    J[r, (k,d)] = d eta_r / d beta_kd = psi_rk (v_k - eta_r) w_rd
    evaluated at random prior draws of (beta, delta). The row-shift gauge
    {1_K g^T : g in R^D} lies in the null space by construction, so the
    effective beta degrees of freedom are (K-1) x D. rank(J) < (K-1) x D
    means beta is not even locally identified from eta given v -- the
    design-specific content of "irreducible nuisance" (Prop 5.3 / B.4).
    Computed for the foundational design (K=3, D=5, R=15; np seed 42 per
    Appendix D.0) and each application design (K=3 or 4, D=32, R=30).

(3) LOTTERY SPANNING CHECK (Prop 5.2 / Appendix B.3): rank of the lottery
    difference matrix (pi_s - pi_1) for the S=15 committed m_1 lotteries
    (results/parameter_recovery/m1_matched_comparison/study_design.json).
    The proposition requires rank K-1 = 2; we report the rank and the
    nonzero singular values.

(4) PRIOR-PREDICTIVE ETA-GAP per application condition: within-menu range of
    eta (max - min) under prior draws of (beta, delta), summarising how much
    expected-utility curvature the design supplies for pinning alpha
    (review theme A: alpha and the spread of eta trade off).

(5) ALPHA CONTRACTION per application condition, from the committed alpha
    draws: sd of log alpha under the posterior versus the prior
    (sigma = 0.75 for both calibrated priors), plus the posterior/prior
    90%-interval width ratio on the alpha scale. Reported as
    posterior-to-prior contraction of the alpha marginal.

Outputs:
  report_design_diagnostics_results.json   (next to this script)

Run:
  conda run -n seu-sensitivity python \
    working_papers/seu_sensitivity_methodology/spikes/report_design_diagnostics_spike.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
sys.path.insert(0, PROJECT_ROOT)

DATA_BASE = os.path.join(PROJECT_ROOT, "reports", "applications")
SEED = 20260705
N_PRIOR_DRAWS_ETA = 500     # prior draws for the eta-gap diagnostic
N_JACOBIAN_POINTS = 20      # random prior points for the generic Jacobian rank

STUDIES = {
    "temperature_study": {
        "cell": "GPT-4o x Insurance", "K": 3,
        "alpha_prior": (3.0, 0.75),
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
    },
    "claude_insurance_study": {
        "cell": "Claude 3.5 x Insurance", "K": 3,
        "alpha_prior": (3.0, 0.75),
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
    },
    "gpt4o_ellsberg_study": {
        "cell": "GPT-4o x Ellsberg", "K": 4,
        "alpha_prior": (3.5, 0.75),
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
    },
    "ellsberg_study": {
        "cell": "Claude 3.5 x Ellsberg", "K": 4,
        "alpha_prior": (3.5, 0.75),
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
    },
}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def upsilon_from_delta(delta: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(delta)])


def eta_jacobian(beta: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """J[r, k*D+d] = psi_rk (v_k - eta_r) w_rd for eta_r = softmax(beta w_r)^T v."""
    K, D = beta.shape
    R = w.shape[0]
    psi = softmax(w @ beta.T)                # (R, K)
    eta = psi @ v                            # (R,)
    # (R, K): psi_rk * (v_k - eta_r)
    core = psi * (v[None, :] - eta[:, None])
    # J[r, k, d] = core[r, k] * w[r, d]
    J = core[:, :, None] * w[:, None, :]     # (R, K, D)
    return J.reshape(R, K * D)


def generic_jacobian_rank(w: np.ndarray, K: int, rng: np.random.Generator,
                          n_points: int = N_JACOBIAN_POINTS) -> dict:
    """Rank of the eta Jacobian and of the contrast-map (centered) Jacobian.

    Choice probabilities identify eta only up to additive constants, so the
    operative map for beta identification is the vector of eta CONTRASTS;
    its Jacobian is the row-centered J. Both ranks are reported.
    """
    D = w.shape[1]
    R = w.shape[0]
    ranks, c_ranks = [], []
    center = np.eye(R) - np.full((R, R), 1.0 / R)
    for _ in range(n_points):
        beta = rng.standard_normal((K, D))
        delta = rng.dirichlet(np.ones(K - 1))
        v = upsilon_from_delta(delta)
        J = eta_jacobian(beta, w, v)
        ranks.append(int(np.linalg.matrix_rank(J)))
        c_ranks.append(int(np.linalg.matrix_rank(center @ J)))
    return {
        "R": R, "K": K, "D": D,
        "beta_dim": K * D,
        "gauge_dim": D,
        "effective_beta_dof": (K - 1) * D,
        "jacobian_rank_max": max(ranks),
        "jacobian_rank_min": min(ranks),
        "contrast_jacobian_rank_max": max(c_ranks),
        "contrast_jacobian_rank_min": min(c_ranks),
        "n_points": n_points,
        "locally_identified_mod_gauge": max(c_ranks) >= (K - 1) * D,
    }


def prior_eta_gaps(w: np.ndarray, I: np.ndarray, K: int,
                   rng: np.random.Generator,
                   n_draws: int = N_PRIOR_DRAWS_ETA) -> dict:
    """Within-menu eta range (max - min) under prior draws of (beta, delta)."""
    menus = [np.flatnonzero(I[m]) for m in range(I.shape[0])]
    D = w.shape[1]
    per_draw_mean = np.empty(n_draws)
    all_gaps = []
    for s in range(n_draws):
        beta = rng.standard_normal((K, D))
        delta = rng.dirichlet(np.ones(K - 1))
        v = upsilon_from_delta(delta)
        eta = softmax(w @ beta.T) @ v        # (R,)
        gaps = np.array([eta[m].max() - eta[m].min() for m in menus])
        per_draw_mean[s] = gaps.mean()
        all_gaps.append(gaps)
    pooled = np.concatenate(all_gaps)
    return {
        "n_prior_draws": n_draws,
        "n_menus": len(menus),
        "pooled_gap_median": float(np.median(pooled)),
        "pooled_gap_q05": float(np.quantile(pooled, 0.05)),
        "pooled_gap_q95": float(np.quantile(pooled, 0.95)),
        "per_draw_mean_gap_median": float(np.median(per_draw_mean)),
        "per_draw_mean_gap_q05": float(np.quantile(per_draw_mean, 0.05)),
        "per_draw_mean_gap_q95": float(np.quantile(per_draw_mean, 0.95)),
    }


def alpha_contraction(study: str, spec: dict) -> list[dict]:
    mu, sigma = spec["alpha_prior"]
    prior_q05 = float(np.exp(mu - 1.6448536269514722 * sigma))
    prior_q95 = float(np.exp(mu + 1.6448536269514722 * sigma))
    prior_width = prior_q95 - prior_q05
    out = []
    for k in spec["keys"]:
        path = os.path.join(DATA_BASE, study, "data", f"alpha_draws_{k}.npz")
        with np.load(path) as z:
            a = np.asarray(z["alpha"], dtype=float)
        post_sd_log = float(np.std(np.log(a)))
        q05, q95 = float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95))
        out.append({
            "condition": k,
            "posterior_sd_log_alpha": post_sd_log,
            "prior_sd_log_alpha": sigma,
            "log_scale_contraction": post_sd_log / sigma,
            "posterior_90_width": q95 - q05,
            "prior_90_width": prior_width,
            "width_ratio_alpha_scale": (q95 - q05) / prior_width,
            "posterior_median": float(np.median(a)),
            "prior_median": float(np.exp(mu)),
        })
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict = {"spec": {
        "seed": SEED,
        "n_prior_draws_eta": N_PRIOR_DRAWS_ETA,
        "n_jacobian_points": N_JACOBIAN_POINTS,
        "review_items": ["theme A", "theme B", "comments 1, 2, 8, 12"],
    }}

    # ------------------------------------------------------------------
    # Foundational design (Appendix D.0): np seed 42, StudyDesign M25 K3 D5 R15
    # ------------------------------------------------------------------
    np.random.seed(42)
    from utils.study_design import StudyDesign
    study = StudyDesign(M=25, K=3, D=5, R=15,
                        min_alts_per_problem=2, max_alts_per_problem=5,
                        feature_dist="normal",
                        feature_params={"loc": 0, "scale": 1},
                        design_name="parameter_recovery")
    study.generate()
    w0 = np.asarray(study.w, dtype=float)
    I0 = np.asarray(study.I, dtype=int)
    found = {
        "design": "foundational m_0 (D.0): M25 K3 D5 R15, np seed 42",
        "w_shape": list(w0.shape),
        "w_row_rank": int(np.linalg.matrix_rank(w0)),
        "jacobian": generic_jacobian_rank(w0, 3, rng),
        "prior_eta_gaps": prior_eta_gaps(w0, I0, 3, rng),
    }
    results["foundational"] = found

    # ------------------------------------------------------------------
    # m_1 lottery spanning check (Prop 5.2 / B.3)
    # ------------------------------------------------------------------
    design_path = os.path.join(
        PROJECT_ROOT, "results", "parameter_recovery",
        "m1_matched_comparison", "study_design.json")
    with open(design_path) as f:
        d1 = json.load(f)
    lotteries = np.asarray(d1["x"], dtype=float)   # (S, K)
    diffs = lotteries[1:] - lotteries[0]           # (S-1, K)
    sv = np.linalg.svd(diffs, compute_uv=False)
    results["lottery_spanning"] = {
        "design": "m_1 matched design (D.4): committed study_design.json",
        "S": int(lotteries.shape[0]),
        "K": int(lotteries.shape[1]),
        "required_rank": int(lotteries.shape[1]) - 1,
        "difference_rank": int(np.linalg.matrix_rank(diffs)),
        "singular_values": [float(s) for s in sv],
        "spanning_condition_satisfied":
            int(np.linalg.matrix_rank(diffs)) == int(lotteries.shape[1]) - 1,
    }

    # ------------------------------------------------------------------
    # Application designs: per condition
    # ------------------------------------------------------------------
    app = {}
    for study_name, spec in STUDIES.items():
        K = spec["K"]
        conds = {}
        for key in spec["keys"]:
            sd_path = os.path.join(DATA_BASE, study_name, "data",
                                   f"stan_data_{key}.json")
            with open(sd_path) as f:
                sd = json.load(f)
            w = np.asarray(sd["w"], dtype=float)
            I = np.asarray(sd["I"], dtype=int)
            conds[key] = {
                "w_shape": list(w.shape),
                "w_row_rank": int(np.linalg.matrix_rank(w)),
                "full_row_rank": int(np.linalg.matrix_rank(w)) == w.shape[0],
                "jacobian": generic_jacobian_rank(w, K, rng),
                "prior_eta_gaps": prior_eta_gaps(w, I, K, rng),
            }
        app[study_name] = {
            "cell": spec["cell"], "K": K,
            "conditions": conds,
            "alpha_contraction": alpha_contraction(study_name, spec),
        }
    results["applications"] = app

    out_path = os.path.join(THIS_DIR, "report_design_diagnostics_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("=== Design-level identification diagnostics ===\n")
    fj = found["jacobian"]
    print(f"Foundational design (K3 D5 R15): w row rank {found['w_row_rank']}/5; "
          f"Jacobian rank {fj['jacobian_rank_max']} "
          f"(contrast {fj['contrast_jacobian_rank_max']}) vs effective beta dof "
          f"{fj['effective_beta_dof']} "
          f"(locally identified mod gauge: {fj['locally_identified_mod_gauge']})")
    fg = found["prior_eta_gaps"]
    print(f"  prior eta-gap pooled median {fg['pooled_gap_median']:.3f} "
          f"[{fg['pooled_gap_q05']:.3f}, {fg['pooled_gap_q95']:.3f}]")
    ls = results["lottery_spanning"]
    print(f"\nLottery spanning: rank {ls['difference_rank']} "
          f"(required {ls['required_rank']}); singular values "
          f"{[round(s, 3) for s in ls['singular_values']]}; "
          f"satisfied: {ls['spanning_condition_satisfied']}")
    for study_name, block in app.items():
        print(f"\n{block['cell']} (K={block['K']}):")
        for key, c in block["conditions"].items():
            j = c["jacobian"]
            g = c["prior_eta_gaps"]
            print(f"  {key}: w rank {c['w_row_rank']}/{c['w_shape'][0]} "
                  f"(full row rank: {c['full_row_rank']}); "
                  f"J rank {j['jacobian_rank_max']} "
                  f"(contrast {j['contrast_jacobian_rank_max']}) vs dof "
                  f"{j['effective_beta_dof']}; "
                  f"eta-gap med {g['pooled_gap_median']:.3f}")
        for row in block["alpha_contraction"]:
            print(f"    {row['condition']}: sd(log a) post/prior "
                  f"{row['posterior_sd_log_alpha']:.3f}/0.75 = "
                  f"{row['log_scale_contraction']:.2f}; "
                  f"90% width ratio (alpha scale) "
                  f"{row['width_ratio_alpha_scale']:.2f}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

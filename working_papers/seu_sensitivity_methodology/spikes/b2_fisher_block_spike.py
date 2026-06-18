"""
B.2 Fisher-block spike (methodological paper, Appendix B.2 / plan gate 1).

Question (plan "Pre-drafting action items" #1):
  In model m_0 (uncertain choices only), is the joint (beta, delta) posterior
  weakly identified along a *ridge*? Concretely:

    (a) Does the (beta, delta) Fisher information block have at least one
        near-zero eigenvalue at the design's sample sizes, at >= 3
        representative parameter draws -- with that smallest eigenvalue at
        least one order of magnitude below the bulk?  (after removing the
        EXACT beta row-shift gauge, which is a separate, known indeterminacy)

    (b) Is the near-flat direction approximately in the NEAR-KERNEL of the
        eta-Jacobian (i.e. is the ridge approximately eta-preserving)?  This is
        the claim that licenses the section 3.5 separation of alpha from
        (beta, delta): alpha is identified from eta, so an eta-preserving
        (beta, delta) indeterminacy leaves alpha untouched.

Outcome decides proposition-vs-theorem wording of sections 3.4 / 3.5 / B.2.

Model (m_0), exact parameterization (see models/m_0.stan):
  features w_r in R^D for each of R alternatives
  a_r = beta @ w_r in R^K   (logits);   psi_r = softmax(a_r)
  upsilon = cumsum([0, delta_1, ..., delta_{K-1}]);  upsilon_1 = 0, upsilon_K = 1
  eta_r = psi_r . upsilon
  choice in problem m:  p = softmax(alpha * eta over available alts)

Analytic gradients used here (verified against finite differences below):
  d eta_r / d beta_{k,d} = w_{r,d} * psi_{r,k} * (upsilon_k - eta_r)
  d eta_r / d delta_free  (K=3: free coord = delta_1 = upsilon_2) = psi_{r,2}

Exact beta row-shift gauge (handled separately, NOT the ridge):
  adding the same vector gamma in R^D to *every* row of beta shifts all K
  logits of every alternative by gamma . w_r, leaving softmax(a_r) -- and hence
  eta and the whole likelihood -- exactly invariant.  This contributes D exact
  zero eigenvalues.  We project them out before measuring the ridge.

Expected (per-design) Fisher block for the multinomial-logit choice model:
  I(theta) = sum_m  G_m^T (diag(p_m) - p_m p_m^T) G_m,
  with G_m = alpha * (eta-Jacobian rows for problem m's alternatives),
  p_m = softmax(alpha * eta_m).  theta ranges over (beta, delta_free).

Run:
  python working_papers/seu_sensitivity_methodology/spikes/b2_fisher_block_spike.py
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

from utils.study_design import StudyDesign  # noqa: E402

# ----------------------------------------------------------------------------
# Design constants (plan Appendix D.0; matched-design B condition is m_0, M=50).
# ----------------------------------------------------------------------------
K = 3
D = 5
R = 15
M = 50
MIN_ALTS = 2
MAX_ALTS = 5
DESIGN_SEED = 20260617  # fixed for reproducibility (Appendix E seed convention)


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / e.sum()


def build_design(seed: int = DESIGN_SEED):
    """Replicate the m_0 uncertain-choice design (features w, indicator I)."""
    np.random.seed(seed)
    design = StudyDesign(
        M=M, K=K, D=D, R=R,
        min_alts_per_problem=MIN_ALTS, max_alts_per_problem=MAX_ALTS,
        feature_dist="normal", feature_params={"loc": 0, "scale": 1},
        design_name="b2_spike",
    )
    design.generate()
    w = np.array([np.asarray(v, float) for v in design.w])  # (R, D)
    I = np.asarray(design.I, int)                            # (M, R)
    return w, I


def upsilon_from_delta(delta: np.ndarray) -> np.ndarray:
    """delta on simplex^{K-1} -> ordered utilities with upsilon_1=0, upsilon_K=1."""
    return np.concatenate([[0.0], np.cumsum(delta)])


def forward(w, beta, delta):
    """Per-alternative psi (R,K), eta (R,), and the ordered utilities."""
    ups = upsilon_from_delta(delta)
    psi = np.zeros((R, K))
    for r in range(R):
        psi[r] = softmax(beta @ w[r])
    eta = psi @ ups
    return psi, eta, ups


def eta_jacobian(w, psi, eta, ups):
    """
    d eta_r / d theta for theta = (beta flattened K*D, then K-2 free delta coords).
    For K=3 there is one free delta coord (delta_1 = upsilon_2); d eta_r/d delta_1
    = psi_{r,2}.  Returns J of shape (R, P) with P = K*D + (K-2).
    Column order for beta: row-major over (k, d): index = k*D + d.
    """
    P = K * D + (K - 2)
    J = np.zeros((R, P))
    # beta block
    for r in range(R):
        for k in range(K):
            for d in range(D):
                J[r, k * D + d] = w[r, d] * psi[r, k] * (ups[k] - eta[r])
    # delta block (K=3: single free coord, d eta_r/d delta_1 = psi_{r, 2})
    # general: free delta coords j=1..K-2 map to interior utilities upsilon_{j+1};
    # d eta_r / d upsilon_{j+1} = psi_{r, j+1}.  delta_j -> upsilon_{j+1} is the
    # identity on the free coordinate (the last simplex component absorbs the sum).
    for j in range(K - 2 + 1 - 1):  # = range(K-2)
        J[:, K * D + j] = psi[:, j + 1]
    if K == 3:
        J[:, K * D + 0] = psi[:, 1]  # explicit for the design case
    return J


def fisher_block(w, I, alpha, beta, delta):
    """Expected (per-design) (beta, delta_free) Fisher information block.

    Also returns the stacked per-(menu, alternative) eta-Jacobian and a
    within-menu-centred version of it.  A direction that lies in the near-kernel
    of the centred Jacobian changes no within-menu eta-contrast, hence preserves
    choice probabilities for *any* alpha -- the precise, alpha-independent
    notion that licenses the section 3.5 separation of alpha from (beta, delta).
    """
    psi, eta, ups = forward(w, beta, delta)
    Jeta = eta_jacobian(w, psi, eta, ups)  # (R, P) per distinct alternative
    P = Jeta.shape[1]
    F = np.zeros((P, P))
    stacked_rows = []          # per-(menu, alt) eta-Jacobian rows
    stacked_centred = []       # within-menu-centred rows
    for m in range(M):
        idx = np.where(I[m] == 1)[0]
        if idx.size < 2:
            continue
        eta_m = eta[idx]
        p = softmax(alpha * eta_m)
        G = alpha * Jeta[idx, :]                       # (N_m, P)
        Wm = np.diag(p) - np.outer(p, p)               # (N_m, N_m)
        F += G.T @ Wm @ G
        block = Jeta[idx, :]                           # (N_m, P)
        stacked_rows.append(block)
        stacked_centred.append(block - block.mean(axis=0, keepdims=True))
    Jstack = np.vstack(stacked_rows)
    Jcentred = np.vstack(stacked_centred)
    return F, Jeta, Jstack, Jcentred, psi, eta, ups


# ----------------------------------------------------------------------------
# Exact beta row-shift gauge subspace: gamma in R^D added to every beta row.
# Direction for coordinate d: beta_{k,d} += 1 for all k (delta unchanged).
# ----------------------------------------------------------------------------
def gauge_basis() -> np.ndarray:
    P = K * D + (K - 2)
    G = np.zeros((P, D))
    for d in range(D):
        for k in range(K):
            G[k * D + d, d] = 1.0
    # orthonormalize
    Q, _ = np.linalg.qr(G)
    return Q  # (P, D)


def project_out(F: np.ndarray, basis: np.ndarray):
    """Return F restricted to the orthogonal complement of `basis` columns,
    plus the orthonormal basis of that complement (P x (P-rank))."""
    P = F.shape[0]
    # Orthonormal basis for complement of span(basis)
    full = np.eye(P)
    proj = basis @ basis.T
    comp = full - proj
    # SVD to get an orthonormal basis of the complement
    U, s, _ = np.linalg.svd(comp)
    rank_comp = int(np.sum(s > 1e-9))
    Bc = U[:, :rank_comp]                 # (P, P-D)
    Fc = Bc.T @ F @ Bc
    return Fc, Bc


def eta_sensitivity(Jeta_full: np.ndarray, vec_full: np.ndarray) -> float:
    """RMS change in eta (over rows) per unit step along vec (unit norm)."""
    v = vec_full / (np.linalg.norm(vec_full) + 1e-300)
    return float(np.linalg.norm(Jeta_full @ v) / np.sqrt(Jeta_full.shape[0]))


def delta_schur(F: np.ndarray):
    """delta's marginal vs profiled (Schur-complement) Fisher information.

    delta is the last coordinate (index K*D).  Profiling out beta uses the
    pseudo-inverse of the beta block (which carries the exact gauge null space).
    Returns (F_delta_delta, F_delta_given_beta, ratio).  A small ratio means the
    (beta, delta) coupling severely weakens delta relative to what it would be
    if beta were known -- the precise 'weak identification of delta' quantity
    matched to Reports 4 and 14.
    """
    di = K * D
    F_dd = float(F[di, di])
    F_db = F[di, :di]
    F_bb = F[:di, :di]
    F_d_given_b = F_dd - float(F_db @ np.linalg.pinv(F_bb, rcond=1e-10) @ F_db)
    ratio = F_d_given_b / F_dd if F_dd > 0 else float("nan")
    return F_dd, F_d_given_b, ratio


def analyze_draw(name, w, I, alpha, beta, delta, gauge):
    F, Jeta, Jstack, Jcentred, psi, eta, ups = fisher_block(w, I, alpha, beta, delta)

    # delta's marginal vs profiled (Schur) Fisher info -- the weak-id quantity.
    F_dd, F_d_given_b, schur_ratio = delta_schur(F)

    # Verify the exact gauge really annihilates F and eta (sanity).
    gauge_eta = max(eta_sensitivity(Jeta, gauge[:, d]) for d in range(gauge.shape[1]))
    gauge_fisher = float(np.max(np.abs(F @ gauge)))

    # Remove exact gauge, then eigendecompose the residual (ridge) block.
    Fc, Bc = project_out(F, gauge)
    evals, evecs = np.linalg.eigh(Fc)         # ascending
    evals = np.clip(evals, 0, None)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    smallest = evals[0]
    second = evals[1]
    largest = evals[-1]
    gap_ratio_next = second / smallest if smallest > 0 else np.inf
    gap_ratio_max = largest / smallest if smallest > 0 else np.inf

    # Bottom eigenvector in full coordinates -> eta-sensitivity (near-kernel test).
    v_bottom_full = Bc @ evecs[:, 0]
    v_top_full = Bc @ evecs[:, -1]
    eta_sens_bottom = eta_sensitivity(Jstack, v_bottom_full)
    eta_sens_top = eta_sensitivity(Jstack, v_top_full)
    # median eta-sensitivity over the residual basis directions
    eta_sens_basis = np.median([eta_sensitivity(Jstack, Bc[:, j]) for j in range(Bc.shape[1])])

    # CHOICE-PROBABILITY-preserving test (alpha-independent): within-menu-centred
    # eta sensitivity.  Small for the ridge => it preserves within-menu contrasts
    # => preserves choice probs at any alpha => alpha (identified from those
    # contrasts) is untouched.  This is the precise section 3.5 diagnostic.
    cps_bottom = eta_sensitivity(Jcentred, v_bottom_full)
    cps_top = eta_sensitivity(Jcentred, v_top_full)
    cps_basis = np.median([eta_sensitivity(Jcentred, Bc[:, j]) for j in range(Bc.shape[1])])

    # delta-weight of the ridge direction (how much utility-shape it carries)
    delta_weight = float(v_bottom_full[K * D] ** 2 / (np.linalg.norm(v_bottom_full) ** 2))

    # Overlap between the Fisher bottom eigenvector and the eta-Jacobian's own
    # bottom right-singular vectors (restricted to the residual / gauge-fixed space).
    Jc = Jeta @ Bc                              # (R, P-D)
    _, sJ, VtJ = np.linalg.svd(Jc, full_matrices=False)
    # bottom-k eta-singular directions (k = number of near-flat Fisher modes ~ 1)
    v_bottom_c = evecs[:, 0]
    # cosine with the single smallest eta-right-singular vector
    cos_bottom = float(abs(VtJ[-1] @ v_bottom_c))

    return {
        "name": name,
        "alpha": float(alpha),
        "eta_range": [float(eta.min()), float(eta.max())],
        "P_full": int(F.shape[0]),
        "P_residual": int(Fc.shape[0]),
        "gauge_dim": int(gauge.shape[1]),
        "gauge_eta_sensitivity_max": gauge_eta,        # ~0 -> exact eta-preserving
        "gauge_fisher_action_max": gauge_fisher,       # ~0 -> exact Fisher null
        "eigenvalues_residual": [float(x) for x in evals],
        "smallest": float(smallest),
        "second": float(second),
        "largest": float(largest),
        "condition_number": float(gap_ratio_max),      # robust ill-conditioning
        "gap_ratio_next": float(gap_ratio_next),       # >=10 => 1 OOM gap to next
        "delta_marginal_fisher": F_dd,
        "delta_profiled_fisher": F_d_given_b,
        "delta_schur_ratio": schur_ratio,              # small => beta-coupling weakens delta
        "eta_sens_bottom": eta_sens_bottom,            # ridge raw-eta change
        "eta_sens_top": eta_sens_top,
        "eta_sens_basis_median": float(eta_sens_basis),
        "eta_sens_ratio_bottom_to_median": float(eta_sens_bottom / eta_sens_basis),
        "cps_bottom": cps_bottom,                      # ridge choice-prob change (key)
        "cps_top": cps_top,
        "cps_basis_median": float(cps_basis),
        "cps_ratio_bottom_to_median": float(cps_bottom / cps_basis),
        "ridge_delta_weight": delta_weight,
        "cos_bottom_vs_eta_nullvec": cos_bottom,       # ~1 => ridge ~ eta near-kernel
        "eta_singular_values": [float(x) for x in sJ],
    }


def finite_diff_check(w, beta, delta, eps=1e-6):
    """Verify the analytic eta-Jacobian against central finite differences."""
    psi, eta, ups = forward(w, beta, delta)
    J = eta_jacobian(w, psi, eta, ups)
    P = J.shape[1]
    Jfd = np.zeros_like(J)
    base = np.concatenate([beta.ravel(), delta[:K - 2]])
    for p in range(P):
        bp = base.copy(); bp[p] += eps
        bm = base.copy(); bm[p] -= eps

        def unpack(vec):
            b = vec[:K * D].reshape(K, D)
            d_free = vec[K * D:]
            d_last = 1.0 - np.sum(d_free)
            d = np.concatenate([d_free, [d_last]])
            return b, d

        bpb, bpd = unpack(bp)
        bmb, bmd = unpack(bm)
        _, ep, _ = forward(w, bpb, bpd)
        _, em, _ = forward(w, bmb, bmd)
        Jfd[:, p] = (ep - em) / (2 * eps)
    return float(np.max(np.abs(J - Jfd)))


def representative_draws(n=4, seed=2026):
    """Draw (alpha, beta, delta) from the m_1_sim priors used in Report 14."""
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        alpha = float(rng.lognormal(0.0, 1.0))
        beta = rng.normal(0.0, 1.0, size=(K, D))
        delta = rng.dirichlet(np.ones(K - 1))
        draws.append((f"draw{i+1}", alpha, beta, delta))
    return draws


def main():
    w, I = build_design()
    gauge = gauge_basis()

    # Jacobian correctness gate.
    _, a0, b0, d0 = representative_draws(1)[0]
    fd_err = finite_diff_check(w, b0, d0)
    print(f"[check] max |analytic - finite-diff| eta-Jacobian = {fd_err:.2e}")
    assert fd_err < 1e-6, "Analytic eta-Jacobian disagrees with finite differences."

    results = []
    print("\n=== (beta, delta) Fisher-block spike: m_0, K=3, D=5, M=50 ===")
    for name, alpha, beta, delta in representative_draws(4):
        res = analyze_draw(name, w, I, alpha, beta, delta, gauge)
        results.append(res)
        print(
            f"\n{name}: alpha={alpha:.3f}  eta in [{res['eta_range'][0]:.3f}, "
            f"{res['eta_range'][1]:.3f}]"
        )
        print(
            f"  gauge check: eta-sens(max)={res['gauge_eta_sensitivity_max']:.2e}, "
            f"Fisher action(max)={res['gauge_fisher_action_max']:.2e} (expect ~0)"
        )
        ev = res["eigenvalues_residual"]
        print(f"  residual eigenvalues (ascending): "
              f"{', '.join(f'{x:.3e}' for x in ev)}")
        print(f"  smallest={res['smallest']:.3e}  next={res['second']:.3e}  "
              f"largest={res['largest']:.3e}")
        print(f"  condition number = {res['condition_number']:.1f}x   "
              f"gap to next = {res['gap_ratio_next']:.1f}x")
        print(f"  delta Fisher: marginal={res['delta_marginal_fisher']:.3e}, "
              f"profiled={res['delta_profiled_fisher']:.3e}, "
              f"ratio={res['delta_schur_ratio']:.3f} (small => beta-coupling weakens delta)")
        print(f"  raw-eta sensitivity: ridge={res['eta_sens_bottom']:.3e}, "
              f"top={res['eta_sens_top']:.3e}, median={res['eta_sens_basis_median']:.3e} "
              f"(ratio {res['eta_sens_ratio_bottom_to_median']:.3f})")
        print(f"  choice-prob sensitivity: ridge={res['cps_bottom']:.3e}, "
              f"top={res['cps_top']:.3e}, median={res['cps_basis_median']:.3e} "
              f"(ratio {res['cps_ratio_bottom_to_median']:.3f} => ridge preserves choice probs)")
        print(f"  ridge delta-weight = {res['ridge_delta_weight']:.3f}   "
              f"cos(ridge, eta-null) = {res['cos_bottom_vs_eta_nullvec']:.3f}")

    # alpha sweep at a fixed (beta, delta) to show the moderate-alpha co-damping.
    print("\n=== alpha sweep at fixed (beta, delta) (draw1 params) ===")
    _, _, b1, d1 = representative_draws(1)[0]
    sweep = []
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        res = analyze_draw(f"alpha={alpha}", w, I, alpha, b1, d1, gauge)
        sweep.append(res)
        print(f"  alpha={alpha:5.1f}: cond={res['condition_number']:8.1f}x, "
              f"delta-schur-ratio={res['delta_schur_ratio']:.3f}, "
              f"cps-ratio={res['cps_ratio_bottom_to_median']:.3f}")

    out = {
        "design": {"K": K, "D": D, "R": R, "M": M, "design_seed": DESIGN_SEED},
        "fd_jacobian_max_err": fd_err,
        "draws": results,
        "alpha_sweep": sweep,
    }
    out_path = os.path.join(THIS_DIR, "b2_fisher_block_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")

    make_figure(results)


def make_figure(results):
    """Spectrum + delta-Schur-ratio figure for the figure manifest (Fig B.2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    for res in results:
        ev = np.array(res["eigenvalues_residual"])
        axL.semilogy(range(1, len(ev) + 1), ev[::-1], "o-", ms=4,
                     label=f"{res['name']} (\u03b1={res['alpha']:.2f})")
    axL.set_xlabel("eigenvalue index (descending)")
    axL.set_ylabel("Fisher eigenvalue (gauge-fixed)")
    axL.set_title("(\u03b2, \u03b4) Fisher spectrum: a low-curvature tail")
    axL.legend(fontsize=7)
    axL.grid(True, which="both", alpha=0.3)

    names = [r["name"] for r in results]
    ratios = [r["delta_schur_ratio"] for r in results]
    axR.bar(names, ratios, color="#1f77b4")
    axR.axhline(1.0, ls="--", c="gray", lw=1, label="no \u03b2-coupling cost")
    axR.set_ylabel("profiled / marginal \u03b4 Fisher")
    axR.set_title("\u03b4 information lost to \u03b2-coupling")
    axR.set_ylim(0, 1.05)
    axR.legend(fontsize=8)
    for i, v in enumerate(ratios):
        axR.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(PAPER_DIR, "figures", "b2_fisher_spike.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()

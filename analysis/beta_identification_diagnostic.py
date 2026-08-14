"""
Is beta's non-convergence benign or real?

BACKGROUND. At J=18 the belief maps for cells 6 and 7 reached R-hat 1.48 with
ESS 8, while every cell-level parameter stayed healthy. Two very different
explanations fit that observation and they have opposite consequences:

  BENIGN -- the disagreement lives in a direction the likelihood cannot see.
    softmax(v + c) = softmax(v), so beta[j] -> beta[j] + 1_K a^T changes nothing
    for any x. That is an exact D-dimensional invariance per cell, identified
    only by the prior. Chains are then free to sit anywhere along it, R-hat is
    meaningless there, and NOTHING the study reports is affected.

  REAL -- the disagreement lives in the identified part, i.e. the chains found
    genuinely different belief maps that fit the data differently. That would
    threaten RQ1-RQ3, because alpha is estimated jointly with beta.

THE DECOMPOSITION. Split each cell's beta into
    m[j][d]        = mean over k of beta[j][k,d]      NON-IDENTIFIED (D per cell)
    c[j][k,d]      = beta[j][k,d] - m[j][d]           IDENTIFIED
Under the std_normal prior, m[j][d] ~ N(0, 1/sqrt(K)) exactly, so if the
posterior sd of m matches 1/sqrt(K) = 0.577 at K=3 the invariance is confirmed
prior-dominated and the benign story is supported.

THE DECIDING TEST is not about beta at all. The likelihood sees beta only
through eta = dot(psi, upsilon), and the study reports alpha, not beta. So the
question that matters is whether the CHAINS AGREE ON eta AND alpha_cell even
where they disagree on beta. Between-chain agreement is used rather than R-hat
because it is what R-hat's numerator measures and it can be computed for derived
quantities without arviz, which is not installed.

Runs at J=6 by default -- about 900 s, against roughly 3 h at J=18. If the
pattern does not appear at J=6, that is itself informative and the run is
repeated at J=18 knowingly rather than by default.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.hierarchical_power import (
    build_pseudorep_design,
    interaction_component,
)
from utils.study_design_hierarchical import HierarchicalStudyDesign

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--factors", default="3,2")
parser.add_argument("--menus-per-cell", type=int, default=30)
parser.add_argument("--iter-warmup", type=int, default=1000)
parser.add_argument("--iter-sampling", type=int, default=2000)
parser.add_argument("--max-treedepth", type=int, default=12)
parser.add_argument("--dense-sd", type=float, default=0.10)
parser.add_argument("--label", default="beta_identification_j6")
parser.add_argument(
    "--from-csv-dir",
    default=None,
    help="Analyse chains ALREADY ON DISK instead of fitting. Dimensions are "
    "inferred from the column names. Accepts .csv or .csv.gz. Use this before "
    "buying a fresh J=18 fit -- preserved chains from an earlier run answer "
    "the same question for free.",
)
args = parser.parse_args()

OUT = Path("results/power") / args.label
OUT.mkdir(parents=True, exist_ok=True)


def load_from_csv(directory):
    """Read preserved chains, decompressing to a scratch dir if needed."""
    import glob
    import gzip
    import shutil
    import tempfile

    from cmdstanpy import from_csv

    paths = sorted(glob.glob(os.path.join(directory, "*.csv"))
                   + glob.glob(os.path.join(directory, "*.csv.gz")))
    if not paths:
        raise SystemExit(f"no chain CSVs under {directory}")
    scratch = tempfile.mkdtemp(prefix="beta_diag_")
    plain = []
    for path in paths:
        if path.endswith(".gz"):
            dest = os.path.join(scratch, os.path.basename(path)[:-3])
            with gzip.open(path, "rb") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            plain.append(dest)
        else:
            plain.append(path)
    print(f"reading {len(plain)} preserved chains from {directory}", flush=True)
    fit = from_csv(plain)
    return fit, scratch


if args.from_csv_dir:
    fit, _scratch = load_from_csv(args.from_csv_dir)
    elapsed = float("nan")
    names = list(fit.column_names)
    K = max(int(n.split(",")[1]) for n in names if n.startswith("beta["))
    D = max(int(n.split(",")[2].rstrip("]")) for n in names
            if n.startswith("beta["))
    J = max(int(n.split("[")[1].split(",")[0]) for n in names
            if n.startswith("beta["))
    print(f"inferred J={J} K={K} D={D}", flush=True)
    data = {"J": J, "K": K, "D": D, "M_total": 0}
else:
    fit = None
if fit is None:
    design = HierarchicalStudyDesign.from_factorial(
        factors=[int(v) for v in args.factors.split(",")],
        reference_indices=[0, 0], include_interactions=False,
        K=3, D=2, R=12, M_per_cell=args.menus_per_cell,
        menu_sizes=[2, 4, 6, 8],
        feature_dist="normal", feature_params={"loc": 0, "scale": 1},
        design_name=args.label,
    )
    design.generate()
    data = build_pseudorep_design(design, num_presentations=2)
    data["rho_copy"] = 0.9
    data["sigma_cell_sd"] = 0.0
    J, K, D = int(data["J"]), int(data["K"]), int(data["D"])
    X = np.asarray(data["X"], dtype=float)
    rng = np.random.default_rng(999)
    offsets = interaction_component(rng.normal(size=J), X)
    data["cell_offset"] = list(offsets * (args.dense_sd / offsets.std(ddof=1)))

    print(f"J={J} K={K} D={D}  M_total={data['M_total']}  "
          f"obs/cell={data['M_total'] // J}", flush=True)

    sim = CmdStanModel(stan_file="models/h_m01_size_pseudorep_sim.stan")
    inf = CmdStanModel(stan_file="models/h_m01_size.stan")
    draw = sim.sample(data=data, seed=12345, iter_sampling=1, iter_warmup=0,
                      chains=1, fixed_param=True,
                      adapt_engaged=False).draws_pd().iloc[0]

    SIM_ONLY = ("gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd",
                "beta_sd", "rho_copy", "n_menus", "menu_id",
                "num_presentations", "cell_offset", "gamma_size_mean",
                "gamma_size_sd", "menu_size", "mean_menu_size")
    inference_data = {k: v for k, v in data.items() if k not in SIM_ONLY}
    inference_data["y"] = [int(draw[f"y[{m + 1}]"])
                           for m in range(data["M_total"])]

    started = time.time()
    fit = inf.sample(
        data=inference_data, seed=54321,
        iter_sampling=args.iter_sampling, iter_warmup=args.iter_warmup,
        chains=4, adapt_delta=0.95, max_treedepth=args.max_treedepth,
        show_progress=False,
    )
    elapsed = time.time() - started
    print(f"fit took {elapsed:.0f} s", flush=True)

names = list(fit.column_names)
arr = fit.draws()                      # (draws, chains, params)
n_draws, n_chains, _ = arr.shape
idx = {n: i for i, n in enumerate(names)}


def grab(pattern_fn, shape):
    out = np.empty((n_draws, n_chains) + shape)
    for index in np.ndindex(*shape):
        out[(slice(None), slice(None)) + index] = arr[:, :, idx[pattern_fn(index)]]
    return out


beta = grab(lambda i: f"beta[{i[0]+1},{i[1]+1},{i[2]+1}]", (J, K, D))
alpha_cell = grab(lambda i: f"alpha_cell[{i[0]+1}]", (J,))

# The invariance split. beta is (draws, chains, J, K, D), so the mean over
# CONSEQUENCES k is axis 3 -- averaging over the wrong axis here would silently
# produce a meaningless decomposition rather than an error.
m = beta.mean(axis=3)                           # -> (draws, chains, J, D)
centered = beta - m[:, :, :, None, :]


def between_chain_disagreement(x):
    """Between-chain spread of the posterior mean, in within-chain SD units.

    This is what R-hat's numerator measures. Values below ~0.3 mean the chains
    agree; above ~1 they are describing different things.
    """
    chain_means = x.mean(axis=0)                       # (chains, ...)
    within = x.std(axis=0).mean(axis=0)                # (...)
    between = chain_means.std(axis=0, ddof=1)          # (...)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(within > 0, between / within, np.nan)
    return ratio


dis_beta = between_chain_disagreement(beta)             # (J,K,D)
dis_m = between_chain_disagreement(m)                   # (J,D)
dis_c = between_chain_disagreement(centered)            # (J,K,D)
dis_alpha = between_chain_disagreement(alpha_cell)      # (J,)

eta_names = [n for n in names if n.startswith("eta[")]
step = max(1, len(eta_names) // 300)
eta = np.stack([arr[:, :, idx[n]] for n in eta_names[::step]], axis=-1)
dis_eta = between_chain_disagreement(eta)

print("\n" + "=" * 74)
print("BETWEEN-CHAIN DISAGREEMENT (posterior mean spread / within-chain SD)")
print("  < 0.3 chains agree      > 1.0 chains describe different things")
print("=" * 74)
print(f"  {'quantity':<34}{'mean':>10}{'max':>10}{'worst cell':>14}")


def line(label, d, per_cell=None):
    flat = d[~np.isnan(d)]
    worst = ""
    if per_cell is not None:
        cell_scores = np.nanmax(per_cell.reshape(J, -1), axis=1)
        worst = f"j={int(np.argmax(cell_scores)) + 1} ({cell_scores.max():.2f})"
    print(f"  {label:<34}{flat.mean():>10.3f}{flat.max():>10.3f}{worst:>14}")


line("beta (raw)", dis_beta, dis_beta)
line("  non-identified part m", dis_m, dis_m)
line("  IDENTIFIED part (k-centered)", dis_c, dis_c)
line("eta (what the likelihood sees)", dis_eta)
line("alpha_cell (what we report)", dis_alpha, dis_alpha)

prior_sd = 1.0 / np.sqrt(K)
post_sd_m = m.reshape(-1, J, D).std(axis=0)
print(f"\nNON-IDENTIFIED DIRECTION vs ITS PRIOR")
print(f"  prior sd of m = 1/sqrt(K) = {prior_sd:.4f}")
print(f"  posterior sd of m: mean {post_sd_m.mean():.4f}  "
      f"min {post_sd_m.min():.4f}  max {post_sd_m.max():.4f}")
print(f"  ratio posterior/prior: {post_sd_m.mean() / prior_sd:.3f}")
print("  (a ratio near 1 confirms the likelihood says nothing about this")
print("   direction, so chain disagreement there is expected and harmless)")

summary = {
    "settings": vars(args),
    "seconds": elapsed,
    "J": J, "K": K, "D": D, "M_total": int(data["M_total"]),
    "disagreement": {
        "beta_raw": {"mean": float(np.nanmean(dis_beta)),
                     "max": float(np.nanmax(dis_beta))},
        "beta_non_identified": {"mean": float(np.nanmean(dis_m)),
                                "max": float(np.nanmax(dis_m))},
        "beta_identified": {"mean": float(np.nanmean(dis_c)),
                            "max": float(np.nanmax(dis_c))},
        "eta": {"mean": float(np.nanmean(dis_eta)),
                "max": float(np.nanmax(dis_eta))},
        "alpha_cell": {"mean": float(np.nanmean(dis_alpha)),
                       "max": float(np.nanmax(dis_alpha))},
    },
    "non_identified_prior_sd": float(prior_sd),
    "non_identified_posterior_sd_mean": float(post_sd_m.mean()),
    "posterior_over_prior_ratio": float(post_sd_m.mean() / prior_sd),
    "per_cell": {
        f"j{j+1}": {
            "beta_identified_max": float(np.nanmax(dis_c[j])),
            "beta_non_identified_max": float(np.nanmax(dis_m[j])),
            "alpha_cell": float(dis_alpha[j]),
        } for j in range(J)
    },
}
with open(OUT / "beta_identification.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {OUT / 'beta_identification.json'}")

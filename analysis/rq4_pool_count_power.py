"""
Regime (f): RQ4 ordering-agreement power as a function of POOL COUNT.

Why this exists
---------------
RQ4 asks whether a model's alpha ordering is stable across pools, via pairwise
ordering agreement over the C(6,2) = 15 model pairs.  Its power therefore depends
on the NUMBER OF POOLS -- and none of the five §8.5 regimes (a)-(e) covers pool
count.  Demoting the insurance pool takes the design from 3 pools to 2, which
collapses RQ4 to a single cross-pool comparison with no replication, and nothing
in Phase D as originally scoped would have revealed how much that costs.  E3
would have frozen a pool count that was never power-analysed.

Why it is cheap
---------------
The ordering statistic is a function of the per-pool ``gamma_model`` posteriors
only.  So the expensive part (what is the sampling spread of gamma-hat at the
real design?) is measured ONCE by a J=18 fit, and everything else is a numpy
Monte Carlo.  Brute-forcing this by fitting every pool in every iteration would
cost roughly 3 fits x ~3 h = ~9 h per iteration.

*** The SE must be the SAMPLING SE of gamma-hat around truth, not the posterior
SD.  Regime (a) showed those diverge under pseudo-replication: the posterior SD
is too small when presentations are correlated, so using it would overstate RQ4
power for exactly the reason regime (a) exists to catch. ``hierarchical_power``
reports the right quantity as ``gamma_sampling_se``. ***

Model
-----
Six models; ``gamma`` holds the 5 treatment-coded contrasts against the reference
model, so a pool's alpha ordering is the ordering of ``(0, gamma_1..gamma_5)``.

    true, pool p :  gamma_p = mu + tau * eps_p ,  eps_p ~ N(0, I)
    observed     :  gamma_hat_p = gamma_p + se * z_p ,  z_p ~ N(0, I)

``tau`` is the true between-pool instability: tau = 0 means the ordering is
genuinely identical in every pool (the RQ4 null of "disposition, not artifact"),
and large tau means orderings genuinely differ.

Statistic: the fraction of the 15 model pairs whose ordering is identical across
ALL pools.  Note this falls mechanically as pools are added even when tau = 0,
because noise gets more chances to flip one comparison -- which is precisely why
the decision rule has to be calibrated per pool count rather than compared to a
fixed number.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

N_MODELS = 6
PAIRS = list(itertools.combinations(range(N_MODELS), 2))


def agreement_fraction(gamma_hat: np.ndarray) -> np.ndarray:
    """
    Fraction of model pairs whose ordering is identical across all pools.

    ``gamma_hat`` has shape (n_sims, n_pools, N_MODELS) and already includes the
    reference model's structural zero.
    """
    n_sims, n_pools, _ = gamma_hat.shape
    agree = np.zeros((n_sims, len(PAIRS)), dtype=bool)
    for index, (i, j) in enumerate(PAIRS):
        signs = np.sign(gamma_hat[:, :, i] - gamma_hat[:, :, j])
        # Identical ordering in every pool.
        agree[:, index] = (signs == signs[:, [0]]).all(axis=1)
    return agree.mean(axis=1)


def simulate(
    n_pools: int,
    tau: float,
    se: np.ndarray,
    mu_sd: float,
    n_sims: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Return observed and truth-based agreement fractions."""
    n_contrasts = N_MODELS - 1

    mu = rng.normal(0.0, mu_sd, size=(n_sims, 1, n_contrasts))
    eps = rng.normal(0.0, 1.0, size=(n_sims, n_pools, n_contrasts))
    gamma = mu + tau * eps

    noise = rng.normal(0.0, 1.0, size=(n_sims, n_pools, n_contrasts)) * se
    gamma_hat = gamma + noise

    zeros = np.zeros((n_sims, n_pools, 1))
    return {
        "observed": agreement_fraction(np.concatenate([zeros, gamma_hat], axis=2)),
        "truth": agreement_fraction(np.concatenate([zeros, gamma], axis=2)),
    }


def variance_component_power(
    n_pools: int,
    tau: float,
    se: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> float:
    """
    Power of a VARIANCE-COMPONENT test of tau = 0, as an alternative estimand.

    The ordering statistic reduces every comparison to a sign, discarding
    magnitude entirely.  Estimating the between-pool spread directly keeps it.
    Under H0 the between-pool sum of squares, scaled by the known sampling
    variance, is chi-square with (n_pools - 1) * n_contrasts degrees of freedom,
    so the test is exact and needs no calibration.

    This is what RQ4 could be reframed as if the ordering-agreement version is
    underpowered.  It answers "do the pools differ at all", not "does this
    specific pair swap", so it is a genuinely different -- and weaker --
    scientific claim.  Reported so that trade is explicit rather than implied.
    """
    from scipy.stats import chi2

    n_contrasts = N_MODELS - 1
    df = (n_pools - 1) * n_contrasts
    critical = chi2.ppf(1.0 - alpha, df)

    mu = rng.normal(0.0, 1.0, size=(n_sims, 1, n_contrasts))  # location is nuisance
    eps = rng.normal(0.0, 1.0, size=(n_sims, n_pools, n_contrasts))
    noise = rng.normal(0.0, 1.0, size=(n_sims, n_pools, n_contrasts)) * se
    gamma_hat = mu + tau * eps + noise

    centred = gamma_hat - gamma_hat.mean(axis=1, keepdims=True)
    # Standardize each contrast by its own sampling sd before pooling.
    q = ((centred / se) ** 2).sum(axis=(1, 2))
    return float((q > critical).mean())


def contrast_se_floor(sigma_cell: float, cells_per_model: int = 3) -> float:
    """
    Irreducible SE of a model contrast, set by the cell-level random effect.

    Each model appears in ``cells_per_model`` cells (the prompt conditions), and
    each cell carries its own ``sigma_cell * z_j``.  A contrast differences two
    model means, so this component survives ANY number of menus -- more data
    shrinks the within-cell term and nothing else.
    """
    return sigma_cell * np.sqrt(2.0 / cells_per_model)


def _sigma_cell_sweep(args) -> int:
    """
    RQ4 viability as a function of sigma_cell, for BOTH candidate estimands.

    The J=18 measurement showed the contrast SE is floored by sigma_cell, whose
    value in simulation is only the PRIOR (half-normal 0.3 => mean 0.239). The
    real value is estimable from the E1 smoke run. Rather than wait, this turns
    the open question into a threshold: below which sigma_cell does RQ4 reach the
    target power, under each estimand?
    """
    rng = np.random.default_rng(args.seed)
    pool_counts = [int(x) for x in args.pool_counts.split(",")]
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.2394, 0.30, 0.40]
    tau = args.target_tau

    print("=" * 78)
    print("RQ4 viability vs sigma_cell  (the floor is set by sigma_cell alone)")
    print("=" * 78)
    print(f"  contrast SE floor = sigma_cell * sqrt(2/3)   [3 cells per model]")
    print(f"  evaluated at INFINITE menus, i.e. the BEST CASE at each sigma_cell")
    print(f"  true instability tau = {tau};  target power = {args.target_power}")
    print(f"\n  {'sigma_cell':>11}{'floor SE':>10}", end="")
    for p in pool_counts:
        print(f"{'order/' + str(p):>10}{'varcomp/' + str(p):>13}", end="")
    print()

    rows = []
    for sigma in sigmas:
        floor = contrast_se_floor(sigma)
        se = np.full(N_MODELS - 1, floor)
        line = f"  {sigma:>11.4f}{floor:>10.4f}"
        row = {"sigma_cell": sigma, "floor_se": floor, "by_pools": {}}
        for p in pool_counts:
            null = simulate(p, 0.0, se, args.mu_sd, args.n_sims, rng)
            cutoff = float(np.quantile(null["observed"], 0.05))
            sim = simulate(p, tau, se, args.mu_sd, args.n_sims, rng)
            order_power = float((sim["observed"] < cutoff).mean())
            vc_power = variance_component_power(p, tau, se, args.n_sims, rng)
            line += f"{order_power:>10.3f}{vc_power:>13.3f}"
            row["by_pools"][p] = {
                "ordering_power": order_power,
                "variance_component_power": vc_power,
            }
        rows.append(row)
        print(line)

    print("\n  ORDER = the pre-registered pairwise ordering-agreement statistic")
    print("  VARCOMP = a variance-component test of tau = 0 (uses magnitudes)")

    # Critical sigma_cell per estimand and pool count.
    print("\n" + "-" * 78)
    print(f"  Largest sigma_cell still reaching power {args.target_power} "
          f"at tau={tau} (infinite menus):")
    for p in pool_counts:
        for name, key in (("ordering", "ordering_power"),
                          ("varcomp ", "variance_component_power")):
            ok = [r["sigma_cell"] for r in rows if r["by_pools"][p][key] >= args.target_power]
            best = max(ok) if ok else None
            print(f"    {p} pools, {name}: "
                  + (f"sigma_cell <= {best}" if best is not None
                     else f"NEVER reached (max "
                          f"{max(r['by_pools'][p][key] for r in rows):.3f})"))

    out = Path("results/power/rq4_sigma_cell_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tau": tau,
                "target_power": args.target_power,
                "mu_sd": args.mu_sd,
                "n_sims": args.n_sims,
                "prior_mean_sigma_cell": 0.3 * float(np.sqrt(2 / np.pi)),
                "rows": rows,
                "note": "Infinite-menu best case. sigma_cell here is the PRIOR; "
                        "the real value must come from the E1 smoke run.",
                "provisional": True,
                "frozen_at": None,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="results/power/h_m01_size_j18/_measure/summary.json",
        help="J=18 power summary carrying gamma_sampling_se.",
    )
    parser.add_argument(
        "--se",
        type=float,
        default=None,
        help="Override the measured SE with a scalar (sensitivity checks only).",
    )
    parser.add_argument("--pool-counts", default="2,3,4")
    parser.add_argument("--taus", default="0.0,0.1,0.2,0.4")
    parser.add_argument("--mu-sd", type=float, default=0.5,
                        help="Prior sd of the common model contrasts (gamma_sd).")
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sigma-cell-sweep",
        action="store_true",
        help="Map sigma_cell -> irreducible contrast SE -> RQ4 power, and report "
        "the critical sigma_cell at which RQ4 becomes viable. Turns 'measure "
        "sigma_cell later' into a threshold E1 can test cheaply.",
    )
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--target-tau", type=float, default=0.4)
    args = parser.parse_args(argv)

    if args.sigma_cell_sweep:
        return _sigma_cell_sweep(args)

    # -- Resolve the SE, refusing to invent one -----------------------------
    se_source = "override"
    if args.se is not None:
        se = np.full(N_MODELS - 1, args.se)
    else:
        path = Path(args.summary)
        if not path.exists():
            raise SystemExit(
                f"No measured summary at {path}.\n"
                "Regime (f) needs the SAMPLING SE of gamma-hat from a J=18 fit; "
                "it will not fabricate one. Run the J=18 measurement first, or "
                "pass --se for an explicit sensitivity check."
            )
        summary = json.loads(path.read_text())
        measured = summary.get("gamma_sampling_se")
        if not measured:
            raise SystemExit(
                f"{path} has no gamma_sampling_se (needs >= 2 iterations)."
            )
        if int(summary.get("P", 0)) < N_MODELS - 1:
            raise SystemExit(
                f"{path} has P={summary.get('P')}, which cannot contain the "
                f"{N_MODELS - 1} model contrasts RQ4 needs. That summary came "
                "from a reduced design (factors [3,2] => P=2); regime (f) "
                "requires the real J=18 / P=7 structure."
            )
        # The first N_MODELS-1 columns are the model dummies; the remainder are
        # the prompt dummies.
        se = np.asarray(measured[: N_MODELS - 1], dtype=float)
        se_source = str(path)

    rng = np.random.default_rng(args.seed)
    pool_counts = [int(x) for x in args.pool_counts.split(",")]
    taus = [float(x) for x in args.taus.split(",")]

    print("=" * 78)
    print("Regime (f): RQ4 ordering-agreement power vs POOL COUNT")
    print("=" * 78)
    print(f"  gamma sampling SE : {np.round(se, 4).tolist()}  ({se_source})")
    print(f"  mu_sd {args.mu_sd}   n_sims {args.n_sims}   {len(PAIRS)} model pairs")

    results: List[Dict] = []
    for n_pools in pool_counts:
        print(f"\n  --- {n_pools} pools ---")
        null = simulate(n_pools, 0.0, se, args.mu_sd, args.n_sims, rng)
        # Decision rule calibrated PER POOL COUNT: flag instability when the
        # observed agreement falls below the 5th percentile of the stable case.
        cutoff = float(np.quantile(null["observed"], 0.05))
        print(f"    stable-case mean agreement {null['observed'].mean():.3f}; "
              f"5th pct cutoff {cutoff:.3f}")
        print(f"    {'tau':>6}{'true agree':>12}{'obs agree':>11}{'power':>9}")
        for tau in taus:
            sim = simulate(n_pools, tau, se, args.mu_sd, args.n_sims, rng)
            power = float((sim["observed"] < cutoff).mean())
            print(
                f"    {tau:>6.2f}{sim['truth'].mean():>12.3f}"
                f"{sim['observed'].mean():>11.3f}{power:>9.3f}"
            )
            results.append(
                {
                    "n_pools": n_pools,
                    "tau": tau,
                    "true_agreement": float(sim["truth"].mean()),
                    "observed_agreement": float(sim["observed"].mean()),
                    "power": power,
                    "cutoff": cutoff,
                }
            )

    out = Path("results/power/rq4_pool_count_power.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gamma_sampling_se": se.tolist(),
                "se_source": se_source,
                "mu_sd": args.mu_sd,
                "n_sims": args.n_sims,
                "cells": results,
                "provisional": True,
                "frozen_at": None,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {out}")
    print("  NOTE: tau=0 is the RQ4 null (ordering genuinely identical). Power "
          "is P(flagging instability) at each true tau.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

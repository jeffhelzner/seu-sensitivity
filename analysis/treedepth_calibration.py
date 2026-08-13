"""
Treedepth calibration: is 12 enough, and what does it cost?

WHY J=6 IS A FAIR PROXY. Max-treedepth severity across every Phase D log tracks
OBSERVATIONS PER CELL, which is what a hierarchical funnel predicts. J=6 with
menus/cell 30 and 2 presentations gives M_total 360 and 60 obs/cell -- the SAME
60 obs/cell as the J=18 menus/cell 30 cell that saturated on all four chains --
while costing about a tenth as much. The generating regime is regime (b)'s
(sigma_cell_sd = 0 with a spike installed as data), because that is the setting
that produced the worst saturation: with the exchangeable term switched off the
posterior for sigma_cell concentrates near zero, which is where the funnel is
tightest.

WHAT IT DECIDES, and note the second question does not depend on transporting
any ratio across J:
  1. the wall-clock multiplier of treedepth 12 over 10, and the mixing gain
     (ESS/second is the honest cost unit, not seconds/fit)
  2. whether 12 STILL saturates. If it does, the geometry is bad rather than
     the limit being low, no treedepth setting rescues it, and the J=18 run
     should not be bought at all -- it would need a reparameterization instead.

CAVEAT recorded deliberately: applying the RATIO measured here to J=18 is still
an extrapolation across J, and this project has been burned three times by
extrapolated costs. Treat the multiplier as indicative and the saturation
verdict as the load-bearing result.
"""

import json
import sys
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.hierarchical_power import (
    build_pseudorep_design,
    interaction_component,
    sparse_interaction_offsets,
)
from utils.study_design_hierarchical import HierarchicalStudyDesign

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--factors", default="3,2")
parser.add_argument("--menus-per-cell", type=int, default=30)
parser.add_argument("--treedepths", default="10,12")
parser.add_argument("--iter-warmup", type=int, default=1000)
parser.add_argument("--iter-sampling", type=int, default=2000)
parser.add_argument("--rho-copy", type=float, default=0.9)
parser.add_argument(
    "--spike-magnitude",
    type=float,
    default=None,
    help="Install a SPARSE cell offset of this size (regime (b) geometry).",
)
parser.add_argument(
    "--dense-sd",
    type=float,
    default=None,
    help="Install a DENSE cell offset with this across-cell SD, i.e. pin the "
    "true sigma_cell. Small values are the STRESS case: the non-centred "
    "funnel tightens as sigma_cell approaches 0, so a small pinned value is "
    "the worst realistic geometry rather than a typical one.",
)
parser.add_argument("--label", default="treedepth_calibration")
args = parser.parse_args()

TREEDEPTHS = [int(v) for v in args.treedepths.split(",")]
OUT = Path("results/power") / args.label
OUT.mkdir(parents=True, exist_ok=True)

if args.iter_sampling < 2000:
    print("NOTE: sampling draws are reduced. This run is a SATURATION verdict "
          "only -- its wall clock is NOT a valid basis for sizing anything.",
          flush=True)

design = HierarchicalStudyDesign.from_factorial(
    factors=[int(v) for v in args.factors.split(",")],
    reference_indices=[0, 0], include_interactions=False,
    K=3, D=2, R=12, M_per_cell=args.menus_per_cell, menu_sizes=[2, 4, 6, 8],
    feature_dist="normal", feature_params={"loc": 0, "scale": 1},
    design_name=args.label,
)
design.generate()
data = build_pseudorep_design(design, num_presentations=2)
data["rho_copy"] = args.rho_copy
J = int(data["J"])
X = np.asarray(data["X"], dtype=float)

if args.spike_magnitude is not None:
    data["sigma_cell_sd"] = 0.0
    data["cell_offset"] = list(
        sparse_interaction_offsets(J, J - 1, args.spike_magnitude)
    )
    truth = f"sparse spike {args.spike_magnitude}"
elif args.dense_sd is not None:
    data["sigma_cell_sd"] = 0.0
    rng = np.random.default_rng(999)
    offsets = interaction_component(rng.normal(size=J), X)
    offsets = offsets * (args.dense_sd / offsets.std(ddof=1))
    data["cell_offset"] = list(offsets)
    truth = f"dense sigma_cell pinned at {args.dense_sd}"
else:
    truth = f"sigma_cell ~ half-normal({data['sigma_cell_sd']})"

print(f"J={J}  M_total={data['M_total']}  obs/cell={data['M_total'] // J}  "
      f"truth: {truth}", flush=True)
print(f"warmup {args.iter_warmup}  sampling {args.iter_sampling}  "
      f"treedepths {TREEDEPTHS}", flush=True)

sim = CmdStanModel(stan_file="models/h_m01_size_pseudorep_sim.stan")
inf = CmdStanModel(stan_file="models/h_m01_size.stan")

sim_fit = sim.sample(data=data, seed=12345, iter_sampling=1, iter_warmup=0,
                     chains=1, fixed_param=True, adapt_engaged=False)
draw = sim_fit.draws_pd().iloc[0]

SIM_ONLY = ("gamma0_mean", "gamma0_sd", "gamma_sd", "sigma_cell_sd", "beta_sd",
            "rho_copy", "n_menus", "menu_id", "num_presentations",
            "cell_offset", "gamma_size_mean", "gamma_size_sd", "menu_size",
            "mean_menu_size")
inference_data = {k: v for k, v in data.items() if k not in SIM_ONLY}
inference_data["y"] = [int(draw[f"y[{m + 1}]"])
                       for m in range(data["M_total"])]

KEY = ["lp__", "gamma0", "gamma_size", "sigma_cell"] + [
    f"z_alpha[{j + 1}]" for j in range(J)
]


def ess_columns(summary):
    bulk = [c for c in summary.columns if "ESS_bulk" in c or c == "N_Eff"]
    tail = [c for c in summary.columns if "ESS_tail" in c]
    return (bulk[0] if bulk else None), (tail[0] if tail else None)


results = {}
for treedepth in TREEDEPTHS:
    print(f"\n=== max_treedepth = {treedepth} ===", flush=True)
    started = time.time()
    fit = inf.sample(
        data=inference_data,
        seed=54321,
        iter_sampling=args.iter_sampling,
        iter_warmup=args.iter_warmup,
        chains=4,
        adapt_delta=0.95,
        max_treedepth=treedepth,
        show_progress=False,
    )
    elapsed = time.time() - started

    mv = fit.method_variables()
    td = np.asarray(mv["treedepth__"])
    div = np.asarray(mv["divergent__"])
    saturated = float((td >= treedepth).mean())

    summary = fit.summary()
    bulk_col, tail_col = ess_columns(summary)
    present = [k for k in KEY if k in summary.index]
    bulk = {k: float(summary.loc[k, bulk_col]) for k in present} if bulk_col else {}
    tail = {k: float(summary.loc[k, tail_col]) for k in present} if tail_col else {}
    rhat_col = [c for c in summary.columns if "hat" in c.lower()]
    rhat = ({k: float(summary.loc[k, rhat_col[0]]) for k in present}
            if rhat_col else {})

    results[treedepth] = {
        "seconds": elapsed,
        "treedepth_saturated_share": saturated,
        "mean_treedepth": float(td.mean()),
        "max_treedepth_reached": int(td.max()),
        # The whole histogram, because it makes ONE run answer the
        # counterfactual: the share of draws at depth >= L is exactly what a
        # limit of L would have truncated. So a clean run at 12 also tells us
        # whether 10 would have bound, with no second fit.
        "would_saturate_at": {
            str(limit): float((td >= limit).mean()) for limit in range(6, 13)
        },
        "divergences": int(div.sum()),
        "ess_bulk": bulk,
        "ess_tail": tail,
        "rhat": rhat,
        "max_rhat": max(rhat.values()) if rhat else None,
        "min_ess_bulk": min(bulk.values()) if bulk else None,
        "posterior_mean": {
            k: float(fit.draws_pd()[k].mean()) for k in present if k != "lp__"
        },
    }
    print(f"  seconds {elapsed:.0f}  saturated {saturated:.3f}  "
          f"mean td {td.mean():.2f}  divergences {int(div.sum())}", flush=True)
    if bulk:
        print(f"  min ESS_bulk {min(bulk.values()):.0f}", flush=True)

with open(OUT / "calibration.json", "w") as fh:
    json.dump({"settings": vars(args), "results": results}, fh, indent=2)

for treedepth in TREEDEPTHS:
    r = results[treedepth]
    print(f"\n=== treedepth {treedepth}: what a LOWER limit would have cost ===")
    print(f"  max depth actually reached: {r['max_treedepth_reached']}")
    print(f"  {'limit':>7}{'share of draws truncated':>28}")
    for limit, share in sorted(r["would_saturate_at"].items(), key=lambda kv: int(kv[0])):
        print(f"  {limit:>7}{share:>28.3f}")
    if r["max_rhat"] is not None:
        print(f"  max R-hat {r['max_rhat']:.4f}   min ESS_bulk "
              f"{r['min_ess_bulk']:.0f}   divergences {r['divergences']}")

if len(TREEDEPTHS) < 2:
    print("\nSingle treedepth run: no ratio to report. The table above is the "
          "verdict -- the share truncated at each lower limit is what that "
          "limit would have done to this posterior.")
    raise SystemExit(0)

a, b = TREEDEPTHS[0], TREEDEPTHS[-1]
ra, rb = results[a], results[b]
print("\n" + "=" * 66)
print(f"{'metric':<28}{'td ' + str(a):>13}{'td ' + str(b):>13}{'ratio':>11}")
print("=" * 66)


def row(name, va, vb, fmt="{:.0f}"):
    ratio = (vb / va) if (va not in (None, 0) and vb is not None) else float("nan")
    print(f"{name:<28}{fmt.format(va):>13}{fmt.format(vb):>13}{ratio:>11.2f}")


row("wall clock (s)", ra["seconds"], rb["seconds"])
row("saturated share", ra["treedepth_saturated_share"],
    rb["treedepth_saturated_share"], "{:.3f}")
row("mean treedepth", ra["mean_treedepth"], rb["mean_treedepth"], "{:.2f}")
if ra["min_ess_bulk"]:
    row("min ESS_bulk", ra["min_ess_bulk"], rb["min_ess_bulk"])
    row("min ESS_bulk per 1000 s",
        ra["min_ess_bulk"] / (ra["seconds"] / 1000),
        rb["min_ess_bulk"] / (rb["seconds"] / 1000), "{:.1f}")
print(f"{'divergences':<28}{ra['divergences']:>13}{rb['divergences']:>13}")

print("\nposterior means (do the two agree? disagreement means treedepth 10 was")
print("not merely slow but WRONG):")
print(f"  {'param':<16}{'td ' + str(a):>12}{'td ' + str(b):>12}{'diff':>12}")
for k in sorted(ra["posterior_mean"]):
    va, vb = ra["posterior_mean"][k], rb["posterior_mean"][k]
    print(f"  {k:<16}{va:>12.4f}{vb:>12.4f}{vb - va:>12.4f}")
print(f"\nwrote {OUT / 'calibration.json'}")

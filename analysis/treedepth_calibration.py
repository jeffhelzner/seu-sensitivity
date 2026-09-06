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
    sparse_interaction_offsets,
)
from utils.cmdstan_artifacts import gzip_csv_files
from utils.study_design_hierarchical import HierarchicalStudyDesign

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--factors", default="3,2")
parser.add_argument("--menus-per-cell", type=int, default=30)
parser.add_argument("--treedepths", default="10,12")
parser.add_argument(
    "--models",
    default="models/h_m01_size.stan",
    help="Comma-separated inference models to compare ON THE SAME simulated "
    "data. Reparameterisations must agree on the POSTERIOR and differ only in "
    "sampling efficiency, so fitting them to one dataset is what makes the "
    "agreement check meaningful.",
)
parser.add_argument("--iter-warmup", type=int, default=1000)
parser.add_argument("--iter-sampling", type=int, default=2000)
parser.add_argument("--rho-copy", type=float, default=0.9)
parser.add_argument("--design-seed", type=int, default=24680)
parser.add_argument(
    "--from-data-json",
    default=None,
    help="Reuse an inference_data.json written by an earlier run. This permits "
    "cost-bounded model fits in separate launches on exactly the same dataset.",
)
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

if args.from_data_json:
    with open(args.from_data_json) as fh:
        inference_data = json.load(fh)
    J = int(inference_data["J"])
    truth = f"loaded from {args.from_data_json}"
else:
    np.random.seed(args.design_seed)
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

    sim = CmdStanModel(stan_file="models/h_m01_size_pseudorep_sim.stan")
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

with open(OUT / "inference_data.json", "w") as fh:
    json.dump(inference_data, fh)

print(f"J={J}  M_total={inference_data['M_total']}  "
      f"obs/cell={inference_data['M_total'] // J}  "
      f"truth: {truth}", flush=True)
print(f"warmup {args.iter_warmup}  sampling {args.iter_sampling}  "
      f"treedepths {TREEDEPTHS}", flush=True)

MODEL_PATHS = [p for p in args.models.split(",") if p.strip()]

# Which parameters to compute convergence diagnostics over.
#
# This started as an explicit list -- lp__, gamma0, gamma_size, sigma_cell and
# z_alpha -- and that list HID THE ACTUAL PROBLEM. At J=18 the worst mixing was
# in beta (ESS 8, R-hat 1.48 for cells 6 and 7) and in the shared delta
# (ESS 24), neither of which was being monitored; the only symptom that showed
# through was lp__ at ESS 45. A hand-picked list of "interesting" parameters
# silently excuses the ones you did not think to suspect, so monitor EVERYTHING
# except the per-observation arrays, which are large and derived.
DIAGNOSTIC_EXCLUDE_PREFIXES = (
    "log_lik", "y_pred", "eta", "alpha_obs", "log_alpha_obs",
    "T_obs", "T_rep", "ppc_",
)


def diagnostic_params(summary):
    return [
        name for name in summary.index
        if not name.startswith(DIAGNOSTIC_EXCLUDE_PREFIXES)
    ]


def ess_columns(summary):
    bulk = [c for c in summary.columns if "ESS_bulk" in c or c == "N_Eff"]
    tail = [c for c in summary.columns if "ESS_tail" in c]
    return (bulk[0] if bulk else None), (tail[0] if tail else None)


results = {}
for model_path in MODEL_PATHS:
    model_name = os.path.basename(model_path).replace(".stan", "")
    inf = CmdStanModel(stan_file=model_path)
    for treedepth in TREEDEPTHS:
        key = f"{model_name}@td{treedepth}"
        chain_dir = OUT / "chains" / key
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key} ===", flush=True)
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
            output_dir=str(chain_dir),
        )
        elapsed = time.time() - started

        # Preserve before diagnostics: if any summary code fails, the expensive
        # draws still survive. Writing directly to chain_dir also retains raw
        # completed CSVs if the process is interrupted during sampling.
        chain_files = gzip_csv_files(fit.runset.csv_files)
        print(f"  preserved {len(chain_files)} chains under {chain_dir}", flush=True)

        mv = fit.method_variables()
        td = np.asarray(mv["treedepth__"])
        div = np.asarray(mv["divergent__"])
        saturated = float((td >= treedepth).mean())

        summary = fit.summary()
        bulk_col, tail_col = ess_columns(summary)
        present = diagnostic_params(summary)
        bulk = ({k: float(summary.loc[k, bulk_col]) for k in present}
                if bulk_col else {})
        bulk = {k: v for k, v in bulk.items() if v == v}  # drop NaN (constants)
        tail = ({k: float(summary.loc[k, tail_col]) for k in present}
                if tail_col else {})
        rhat_col = [c for c in summary.columns if "hat" in c.lower()]
        rhat = ({k: float(summary.loc[k, rhat_col[0]]) for k in present}
                if rhat_col else {})
        rhat = {k: v for k, v in rhat.items() if v == v}

        results[key] = {
            "model": model_path,
            "chain_files": [str(path) for path in chain_files],
            "max_treedepth": treedepth,
            "seconds": elapsed,
            "treedepth_saturated_share": saturated,
            "mean_treedepth": float(td.mean()),
            "max_treedepth_reached": int(td.max()),
            # The whole histogram, because it makes ONE run answer the
            # counterfactual: the share of draws at depth >= L is exactly what
            # a limit of L would have truncated. So a clean run at 12 also
            # tells us whether 10 would have bound, with no second fit.
            "would_saturate_at": {
                str(limit): float((td >= limit).mean())
                for limit in range(6, 13)
            },
            "divergences": int(div.sum()),
            "ess_bulk": bulk,
            "ess_tail": tail,
            "rhat": rhat,
            "max_rhat": max(rhat.values()) if rhat else None,
            "min_ess_bulk": min(bulk.values()) if bulk else None,
            "worst_ess": sorted(bulk.items(), key=lambda kv: kv[1])[:12],
            "worst_rhat": sorted(rhat.items(), key=lambda kv: -kv[1])[:12],
            "posterior_mean": {
                k: float(fit.draws_pd()[k].mean())
                for k in present if k != "lp__" and k in bulk
            },
        }
        print(f"  seconds {elapsed:.0f}  saturated {saturated:.3f}  "
              f"mean td {td.mean():.2f}  divergences {int(div.sum())}",
              flush=True)
        if bulk:
            print(f"  min ESS_bulk {min(bulk.values()):.0f}  "
                  f"ESS/1000s {min(bulk.values()) / (elapsed / 1000):.1f}  "
                  f"max R-hat {max(rhat.values()):.4f}", flush=True)
            print("  worst mixing: " + ", ".join(
                f"{n}={v:.0f}" for n, v in
                sorted(bulk.items(), key=lambda kv: kv[1])[:5]), flush=True)

with open(OUT / "calibration.json", "w") as fh:
    json.dump({"settings": vars(args), "results": results}, fh, indent=2)

keys = list(results)
print("\n" + "=" * 78)
print(f"{'config':<34}{'secs':>7}{'meantd':>8}{'max':>5}{'>=10':>7}"
      f"{'minESS':>8}{'ESS/1ks':>9}{'div':>5}")
print("=" * 78)
for key in keys:
    r = results[key]
    print(f"{key:<34}{r['seconds']:>7.0f}{r['mean_treedepth']:>8.2f}"
          f"{r['max_treedepth_reached']:>5}"
          f"{r['would_saturate_at']['10']:>7.3f}"
          f"{(r['min_ess_bulk'] or 0):>8.0f}"
          f"{((r['min_ess_bulk'] or 0) / (r['seconds'] / 1000)):>9.1f}"
          f"{r['divergences']:>5}")

for key in keys:
    r = results[key]
    print(f"\n{key}: what a LOWER limit would have cost "
          f"(max depth reached {r['max_treedepth_reached']})")
    print(f"  {'limit':>7}{'share of draws truncated':>28}")
    for limit, share in sorted(r["would_saturate_at"].items(),
                               key=lambda kv: int(kv[0])):
        print(f"  {limit:>7}{share:>28.3f}")
    if r["max_rhat"] is not None:
        print(f"  max R-hat {r['max_rhat']:.4f}")

if len(keys) < 2:
    print("\nSingle config: nothing to compare. The table above is the verdict.")
    raise SystemExit(0)

# Posterior agreement. For a REPARAMETERISATION this is the correctness check,
# not a nicety: the two forms describe the same posterior, so a disagreement
# larger than Monte Carlo error means one of them is not sampling it.
first = keys[0]
print("\n" + "=" * 78)
print(f"posterior mean comparison vs {first}")
print("=" * 78)
print(f"  {'param':<16}" + "".join(f"{k.split('@')[0][-14:]:>16}" for k in keys)
      + f"{'max |diff|':>13}")
for param in sorted(results[first]["posterior_mean"]):
    vals = [results[k]["posterior_mean"].get(param) for k in keys]
    if any(v is None for v in vals):
        continue
    worst = max(abs(v - vals[0]) for v in vals)
    print(f"  {param:<16}" + "".join(f"{v:>16.4f}" for v in vals)
          + f"{worst:>13.4f}")
print("\nFor reparameterisations, differences beyond Monte Carlo error indicate a "
    "problem. Substantive variants such as a pinned utility scale change the "
    "posterior by design; use this table to quantify that change.")
print(f"\nwrote {OUT / 'calibration.json'}")

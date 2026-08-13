"""
Salvage the J=18 treedepth probe verdict from the chains that completed.

The run was killed after 9h01m with chain 1 still in warmup, so the script's own
summary never ran. Chains 2-4 finished and their CmdStan CSVs carry treedepth__
directly, which is the load-bearing quantity -- the share of draws at depth >= L
is exactly what a limit of L would have truncated. Writes summary.json next to
the preserved chain CSVs so the finding survives the disposable _tmp directory.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/power/treedepth_j18_probe")
paths = sorted(glob.glob(str(OUT / "chains" / "h_m01_size-*_[0-9].csv.gz")))

chains = {}
all_td = []
for path in paths:
    chain = os.path.basename(path).split("_")[-1].split(".")[0]
    df = pd.read_csv(path, comment="#", compression="gzip")
    td = df["treedepth__"].to_numpy()
    all_td.append(td)
    chains[chain] = {
        "draws": int(len(td)),
        "mean_treedepth": float(td.mean()),
        "max_treedepth_reached": int(td.max()),
        "divergences": int(df["divergent__"].sum()),
        "stepsize": float(df["stepsize__"].iloc[0]),
        "would_saturate_at": {
            str(limit): float((td >= limit).mean()) for limit in range(7, 13)
        },
    }

td = np.concatenate(all_td)
summary = {
    "label": "treedepth_j18_probe",
    "status": "PARTIAL -- killed after 9h01m; chain 1 never left warmup",
    "settings": {
        "factors": [6, 3], "J": 18, "menus_per_cell": 60, "M_total": 2160,
        "obs_per_cell": 120, "rho_copy": 0.9,
        "sigma_cell": "PINNED at 0.10 (stress case, and the RQ4-viable value)",
        "max_treedepth": 12, "iter_warmup": 1000, "iter_sampling": 200,
        "adapt_delta": 0.95, "chains_requested": 4, "chains_completed": 3,
    },
    "timing_is_not_a_sizing_basis": (
        "Sampling draws were cut to 200, so wall clock here cannot size "
        "anything. Recorded only to show the ORDER of the problem."
    ),
    "chain_wall_clock": {
        "2": "4h02m", "3": "5h29m", "4": "6h21m",
        "1": ">9h01m, still in warmup when killed",
    },
    "per_chain": chains,
    "pooled": {
        "draws": int(len(td)),
        "mean_treedepth": float(td.mean()),
        "implied_leapfrog_steps_per_draw": float(2 ** td.mean()),
        "would_saturate_at": {
            str(limit): float((td >= limit).mean()) for limit in range(7, 13)
        },
    },
    "verdict": {
        "treedepth_12_sufficient": True,
        "treedepth_10_was_binding": True,
        "share_truncated_at_10": float((td >= 10).mean()),
        "reading": (
            "12 is enough -- no draw reached it, max was 11. But 10 would have "
            "truncated 81% of draws, so the ab036f1 fix was necessary, not "
            "cosmetic, and every earlier J=18 power fit was badly truncated. "
            "The cost is the problem: ~925 leapfrog steps per draw, 4-6.4 h per "
            "chain for only 1000 warmup + 200 sampling, and one chain in four "
            "did not finish warmup at all. A full 2000-draw fit is therefore "
            "tens of hours per chain, against 6.3 h measured at treedepth 10."
        ),
        "caveat": (
            "sigma_cell was PINNED at 0.10 deliberately, as the stress case. "
            "At the 0.239 prior mean the geometry should be far kinder -- the "
            "J=6 calibration fitted 0.25 and never exceeded depth 7. Which "
            "world the real study is in depends on the true sigma_cell, which "
            "the E1 smoke can estimate cheaply. That was already the open "
            "question for RQ4; it now also sets the compute budget."
        ),
    },
}

with open(OUT / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(json.dumps(summary["pooled"], indent=2))
print(f"\nwrote {OUT / 'summary.json'}")

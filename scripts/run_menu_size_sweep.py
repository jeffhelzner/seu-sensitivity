#!/usr/bin/env python
"""
Run the §8.5(e) menu-size range sweep under an agreed wall-clock budget.

The analytic ranking (analysis/menu_size_range_analysis.py) already settles
gamma_size PRECISION exactly, since Var(gamma_size_hat) ~ 1/(M * Var(s)).  What
it cannot settle is (e)(iii): the position-stability subset retains menus WITHIN
size stratum, and retention falls as menu size grows, so the candidates that lose
on precision win on retention.  This sweep measures that trade directly.

Cost is predicted by scaling a MEASURED timing by mean menu size, because the
likelihood loops over alternatives rather than observations.  Following the J=18
result -- where linear-in-M_total was wrong by 1.75x -- the prediction is treated
as an estimate to be checked, and predicted-vs-actual is recorded per candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Repeating a size weights it: _allocate_menu_sizes spreads the LIST evenly.
CANDIDATES: Dict[str, str] = {
    "balanced_2468": "2,4,6,8",
    "extreme_weighted_2468": "2,2,4,6,8,8",
    "fallback_2346": "2,3,4,6",
    "narrow_234": "2,3,4",
}


def mean_size(spec: str) -> float:
    values = [int(v) for v in spec.split(",")]
    return sum(values) / len(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/h_m01_size_power_regime_a_config.json"
    )
    parser.add_argument(
        "--timing",
        default="results/power/h_m01_size_regime_a/menus30_rho0p9/timing.json",
        help="MEASURED timing to cost from. Defaults to the 6-iteration regime "
        "(a) cell rather than the noisier 2-iteration probe.",
    )
    parser.add_argument("--menus-per-cell", type=int, default=30)
    parser.add_argument("--rho-copy", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument("--output-root", default="results/power/menu_size_sweep")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    timing_path = Path(args.timing)
    if not timing_path.exists():
        raise SystemExit(
            f"No measured timing at {timing_path}. This sweep will not cost "
            "itself from an estimate."
        )
    timing = json.loads(timing_path.read_text())
    reference_s = float(timing["seconds_per_iteration"])

    with open(args.config) as fh:
        config = json.load(fh)
    reference_sizes = config["study_design_config"]["menu_sizes"]
    reference_mean = sum(reference_sizes) / len(reference_sizes)

    selected = [c.strip() for c in args.candidates.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown candidate(s) {unknown}; have {list(CANDIDATES)}")

    rows: List[Dict] = []
    total_seconds = 0.0
    for name in selected:
        spec = CANDIDATES[name]
        scale = mean_size(spec) / reference_mean
        per_iter = reference_s * scale
        cell_seconds = per_iter * args.iterations
        total_seconds += cell_seconds
        rows.append(
            {
                "candidate": name,
                "menu_sizes": spec,
                "mean_menu_size": mean_size(spec),
                "predicted_seconds_per_iteration": per_iter,
                "predicted_hours": cell_seconds / 3600.0,
            }
        )

    print("=" * 78)
    print(f"§8.5(e) menu-size sweep, costed from MEASURED {timing_path}")
    print(f"  reference {reference_s:.0f} s/iter at mean menu size "
          f"{reference_mean:.2f}")
    print("=" * 78)
    print(f"  {'candidate':<24}{'sizes':<14}{'mean':>6}{'s/iter':>9}{'hours':>8}")
    for row in rows:
        print(
            f"  {row['candidate']:<24}{row['menu_sizes']:<14}"
            f"{row['mean_menu_size']:>6.2f}"
            f"{row['predicted_seconds_per_iteration']:>9.0f}"
            f"{row['predicted_hours']:>8.2f}"
        )
    total_hours = total_seconds / 3600.0
    print("-" * 78)
    print(f"  TOTAL {total_hours:.2f} h ({args.iterations} iterations/candidate)"
          f"   budget {args.budget_hours:.2f} h")
    if total_hours > args.budget_hours:
        raise SystemExit(
            f"REFUSING to launch: predicted {total_hours:.2f} h exceeds the "
            f"agreed {args.budget_hours:.2f} h."
        )
    print("  within budget.\n")
    if args.dry_run:
        print("Dry run: nothing launched.")
        return 0

    root = Path(args.output_root)
    results = []
    for row in rows:
        out_dir = root / row["candidate"]
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            print(f"[skip] {out_dir} already complete")
            results.append(json.loads(summary_path.read_text()))
            continue
        print(f"\n[run] {row['candidate']} sizes={row['menu_sizes']}")
        started = time.time()
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "run_hierarchical_power.py"),
            "--config", args.config,
            "--menus-per-cell", str(args.menus_per_cell),
            "--rho-copy", str(args.rho_copy),
            "--menu-sizes", row["menu_sizes"],
            "--output-dir", str(out_dir),
            "--iterations", str(args.iterations),
        ]
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}); continuing")
            continue
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["candidate"] = row["candidate"]
            summary["predicted_seconds_per_iteration"] = row[
                "predicted_seconds_per_iteration"
            ]
            summary["prediction_error_ratio"] = (
                summary["seconds_per_iteration"]
                / row["predicted_seconds_per_iteration"]
            )
            results.append(summary)
            print(f"  actual {summary['seconds_per_iteration']:.0f} s/iter "
                  f"(ratio {summary['prediction_error_ratio']:.2f})")
        print(f"  wall clock {(time.time() - started) / 3600:.2f} h")

    root.mkdir(parents=True, exist_ok=True)
    out = root / "sweep_summary.json"
    out.write_text(
        json.dumps(
            {
                "candidates": results,
                "iterations_per_candidate": args.iterations,
                "menus_per_cell": args.menus_per_cell,
                "rho_copy": args.rho_copy,
                "predicted_total_hours": total_hours,
                "measured_sampling_hours": sum(
                    c.get("total_seconds", 0.0) for c in results
                ) / 3600.0,
                "provisional": True,
                "frozen_at": None,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

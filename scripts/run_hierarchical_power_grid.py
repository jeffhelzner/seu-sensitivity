#!/usr/bin/env python
"""
Run a GRID of Phase D power cells, under an agreed wall-clock budget.

The budget is enforced by the script, not by discipline.  Before running
anything it costs the whole grid from a MEASURED ``timing.json`` and refuses to
start if the prediction exceeds ``--budget-hours``.  Phase C's a-priori estimate
was wrong by 1.8x because the benchmark used different sampler settings than the
config; this script therefore also refuses to use a timing whose recorded
sampler settings differ from the ones the grid will run under.

Usage
-----
    # cost the grid without running it
    python scripts/run_hierarchical_power_grid.py --config <cfg> \
        --cells 15:0.9,30:0.9,45:0.9,60:0.9 --budget-hours 30 --dry-run

    # run it
    python scripts/run_hierarchical_power_grid.py --config <cfg> \
        --cells 15:0.9,30:0.9 --budget-hours 30 --iterations 10

Each cell is ``menus_per_cell:rho_copy``.  Completed cells are skipped, so an
interrupted grid resumes rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_cells(text: str) -> List[Tuple[int, float]]:
    cells: List[Tuple[int, float]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        menus, rho = chunk.split(":")
        cells.append((int(menus), float(rho)))
    return cells


def load_timing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"No measured timing at {path}.\n"
            "Run a single cell with --measure first. This script will not cost a "
            "grid from an estimate."
        )
    with open(path) as fh:
        return json.load(fh)


def predict_seconds(
    timing: Dict[str, Any], menus_per_cell: int, J: int, num_presentations: int
) -> float:
    """
    Predict per-iteration seconds for a cell, scaling linearly in M_total.

    Linear scaling is an APPROXIMATION: the likelihood is linear in the number of
    observations, but sampler geometry is not guaranteed to be. It is used only
    to size the grid up front; every cell writes its own measured timing, and the
    grid summary reports predicted-vs-actual so the assumption is checked rather
    than trusted.
    """
    reference_m = float(timing["M_total"])
    reference_s = float(timing["seconds_per_iteration"])
    m_total = menus_per_cell * J * num_presentations
    return reference_s * (m_total / reference_m)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument("--timing", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-sampler-mismatch",
        action="store_true",
        help="Cost the grid from a timing taken under DIFFERENT sampler settings. "
        "This is the Phase C failure mode; require an explicit opt-in.",
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        config = json.load(fh)

    cells = parse_cells(args.cells)
    J = 1
    for factor in config["study_design_config"]["factors"]:
        J *= factor
    num_presentations = config.get("num_presentations", 2)

    base_output = Path(config["output_dir"])
    timing_path = Path(
        args.timing or base_output / "_measure" / "timing.json"
    )
    timing = load_timing(timing_path)

    grid_sampler = {
        "iter_warmup": config.get("n_mcmc_warmup"),
        "iter_sampling": config.get("n_mcmc_samples", 2000),
        "chains": config.get("n_mcmc_chains", 4),
        "adapt_delta": config.get("adapt_delta", 0.95),
    }
    if timing.get("sampler") != grid_sampler and not args.allow_sampler_mismatch:
        raise SystemExit(
            "REFUSING to cost this grid.\n"
            f"  measured under : {timing.get('sampler')}\n"
            f"  grid will run  : {grid_sampler}\n"
            "A timing taken under different sampler settings is exactly the "
            "Phase C 1.8x error. Re-measure, or pass --allow-sampler-mismatch."
        )

    # -- Cost the grid --
    rows = []
    total_seconds = 0.0
    for menus, rho in cells:
        per_iter = predict_seconds(timing, menus, J, num_presentations)
        cell_seconds = per_iter * args.iterations
        total_seconds += cell_seconds
        rows.append(
            {
                "menus_per_cell": menus,
                "rho_copy": rho,
                "M_total": menus * J * num_presentations,
                "predicted_seconds_per_iteration": per_iter,
                "predicted_hours": cell_seconds / 3600.0,
            }
        )

    print("=" * 78)
    print(f"Phase D power grid -- costed from MEASURED timing {timing_path}")
    print(f"  reference: {timing['seconds_per_iteration']:.1f} s/iter at "
          f"M_total={timing['M_total']}  sampler={timing.get('sampler')}")
    print("=" * 78)
    print(f"  {'menus/cell':>11}{'rho':>7}{'M_total':>9}{'s/iter':>10}{'hours':>9}")
    for row in rows:
        print(
            f"  {row['menus_per_cell']:>11}{row['rho_copy']:>7}"
            f"{row['M_total']:>9}{row['predicted_seconds_per_iteration']:>10.0f}"
            f"{row['predicted_hours']:>9.2f}"
        )
    total_hours = total_seconds / 3600.0
    print("-" * 78)
    print(f"  {'TOTAL':>11}{'':>7}{'':>9}{'':>10}{total_hours:>9.2f} hours "
          f"({args.iterations} iterations/cell)")
    print(f"  budget: {args.budget_hours:.2f} hours")

    if total_hours > args.budget_hours:
        raise SystemExit(
            f"\nREFUSING to launch: predicted {total_hours:.2f} h exceeds the "
            f"agreed budget of {args.budget_hours:.2f} h.\n"
            "Reduce --iterations, drop cells, or raise the budget explicitly."
        )
    print("  within budget.\n")

    if args.dry_run:
        print("Dry run: nothing launched.")
        return 0

    # -- Run --
    results = []
    started_all = time.time()
    for row in rows:
        menus, rho = row["menus_per_cell"], row["rho_copy"]
        out_dir = base_output / f"menus{menus}_rho{str(rho).replace('.', 'p')}"
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            print(f"[skip] {out_dir} already complete")
            with open(summary_path) as fh:
                summary = json.load(fh)
            summary["predicted_seconds_per_iteration"] = row[
                "predicted_seconds_per_iteration"
            ]
            summary["prediction_error_ratio"] = (
                summary["seconds_per_iteration"]
                / row["predicted_seconds_per_iteration"]
            )
            results.append(summary)
            continue

        print(f"\n[run] menus/cell={menus} rho={rho} -> {out_dir}")
        started = time.time()
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "run_hierarchical_power.py"),
            "--config", args.config,
            "--menus-per-cell", str(menus),
            "--rho-copy", str(rho),
            "--output-dir", str(out_dir),
            "--iterations", str(args.iterations),
        ]
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0:
            print(f"  cell FAILED (exit {proc.returncode}); continuing")
            continue
        actual = time.time() - started

        measured_summary = out_dir / "summary.json"
        if measured_summary.exists():
            with open(measured_summary) as fh:
                summary = json.load(fh)
            summary["predicted_seconds_per_iteration"] = row[
                "predicted_seconds_per_iteration"
            ]
            summary["prediction_error_ratio"] = (
                summary["seconds_per_iteration"]
                / row["predicted_seconds_per_iteration"]
            )
            results.append(summary)
            print(
                f"  predicted {row['predicted_seconds_per_iteration']:.0f}s/iter, "
                f"actual {summary['seconds_per_iteration']:.0f}s/iter "
                f"(ratio {summary['prediction_error_ratio']:.2f})"
            )
        print(f"  cell wall clock {actual / 3600:.2f} h")

    grid_path = base_output / "grid_summary.json"
    os.makedirs(base_output, exist_ok=True)
    # Sum the cells' own measured sampling time rather than this process's wall
    # clock: a re-aggregation run skips every completed cell, so its wall clock
    # is ~0 and would silently overwrite the real figure.
    measured_hours = sum(
        c.get("total_seconds", 0.0) for c in results
    ) / 3600.0
    with open(grid_path, "w") as fh:
        json.dump(
            {
                "cells": results,
                "budget_hours": args.budget_hours,
                "predicted_total_hours": total_hours,
                "measured_sampling_hours": measured_hours,
                "wall_clock_this_process_hours": (time.time() - started_all) / 3600.0,
                "iterations_per_cell": args.iterations,
                "provisional": True,
                "frozen_at": None,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

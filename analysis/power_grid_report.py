"""Report a Phase D power grid, and answer regime (a)'s decisive question.

The question is NOT "is there a pseudo-replication effect" -- the one-cell
measurement already showed coverage collapsing while power read 1.0. It is
whether the displacement is small-sample bias (which more menus fix) or
systematic (which more menus make WORSE, by narrowing intervals around a
displaced centre). Under the copy mechanism each observation's MARGINAL
distribution is still exactly correct, so theory says the estimator should be
consistent with wrong standard errors -- i.e. bias should fall roughly as 1/N.
The menus ladder at fixed rho tests that directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load(grid_path: Path) -> List[Dict[str, Any]]:
    with open(grid_path) as fh:
        return json.load(fh)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        default="results/power/h_m01_size_regime_a/grid_summary.json",
    )
    args = parser.parse_args(argv)

    grid = load(Path(args.grid))
    cells = sorted(
        grid["cells"], key=lambda r: (r["rho_copy"], r["menus_per_cell"])
    )

    head = (
        f"{'menus':>6}{'rho':>6}{'M_tot':>7}{'bias':>9}{'rmse':>8}{'CI_w':>8}"
        f"{'cover':>7}{'pow':>6}{'ROPE':>6}{'typeS':>7}{'agree':>7}"
        f"{'s/iter':>8}{'pred':>6}"
    )
    print("=" * len(head))
    print("Phase D regime (a) -- pseudo-replication.  PROVISIONAL until E3.")
    print("=" * len(head))
    print(head)
    for r in cells:
        print(
            f"{r['menus_per_cell']:>6.0f}{r['rho_copy']:>6}{r['M_total']:>7}"
            f"{r['bias']:>9.4f}{r['rmse']:>8.4f}{r['mean_ci_width']:>8.4f}"
            f"{r['coverage']:>7.2f}{r['power_excludes_zero']:>6.2f}"
            f"{r['power_outside_rope']:>6.2f}{r['type_s_rate']:>7.2f}"
            f"{r['mean_agreement_rate']:>7.3f}"
            f"{r['seconds_per_iteration']:>8.0f}"
            f"{r.get('prediction_error_ratio', float('nan')):>6.2f}"
        )
    print(f"\n  nominal coverage {cells[0]['nominal_coverage']}")
    measured = grid.get("measured_sampling_hours")
    if measured:
        print(
            f"  predicted total {grid['predicted_total_hours']:.2f} h  |  "
            f"measured sampling {measured:.2f} h  "
            f"(ratio {measured / grid['predicted_total_hours']:.2f})"
        )
    else:
        print(f"  predicted total {grid['predicted_total_hours']:.2f} h")

    # -- rho -> effect curve at fixed menus ---------------------------------
    by_menus: Dict[int, List[Dict[str, Any]]] = {}
    for r in cells:
        by_menus.setdefault(int(r["menus_per_cell"]), []).append(r)

    print("\n" + "-" * len(head))
    print("rho -> effect, at fixed menus/cell (is there a safe rho?)")
    for menus, rows in sorted(by_menus.items()):
        if len(rows) < 2:
            continue
        print(f"  menus/cell {menus}:")
        for r in sorted(rows, key=lambda x: x["rho_copy"]):
            print(
                f"    rho {r['rho_copy']:<5} bias {r['bias']:+.4f}  "
                f"CI width {r['mean_ci_width']:.4f}  coverage {r['coverage']:.2f}"
            )

    # -- The decisive test: does bias shrink with N? ------------------------
    ladder = sorted(
        [r for r in cells if r["rho_copy"] == max(c["rho_copy"] for c in cells)],
        key=lambda r: r["menus_per_cell"],
    )
    print("\n" + "-" * len(head))
    print("DECISIVE TEST -- bias vs N at the highest rho")
    print("  If bias ~ 1/N it is small-sample and MORE MENUS FIX IT.")
    print("  If bias is flat it is systematic; more menus NARROW the interval")
    print("  around a displaced centre and coverage gets WORSE, so the answer")
    print("  is the plan's menu-level random effect, not a bigger num_problems.")
    if len(ladder) >= 2:
        print(
            f"\n  {'menus':>6}{'n_menus':>9}{'bias':>10}{'|bias|xN':>11}"
            f"{'CI_w':>9}{'cover':>7}"
        )
        for r in ladder:
            n_menus = r["n_menus"]
            print(
                f"  {r['menus_per_cell']:>6.0f}{n_menus:>9}{r['bias']:>10.4f}"
                f"{abs(r['bias']) * n_menus:>11.1f}"
                f"{r['mean_ci_width']:>9.4f}{r['coverage']:>7.2f}"
            )

        n = np.array([r["n_menus"] for r in ladder], dtype=float)
        b = np.array([abs(r["bias"]) for r in ladder], dtype=float)
        if (b > 0).all():
            slope = float(np.polyfit(np.log(n), np.log(b), 1)[0])
            print(f"\n  log|bias| ~ {slope:+.2f} * log(n_menus)")
            print("    slope near -1  => 1/N small-sample bias (more menus fix it)")
            print("    slope near  0  => systematic (more menus do NOT fix it)")
            verdict = (
                "SMALL-SAMPLE (more menus help)"
                if slope < -0.5
                else "SYSTEMATIC (more menus do NOT help)"
                if slope > -0.25
                else "AMBIGUOUS at this n -- needs more iterations"
            )
            print(f"\n  READING: {verdict}")
        widths = [r["mean_ci_width"] for r in ladder]
        covers = [r["coverage"] for r in ladder]
        if widths[-1] < widths[0] and covers[-1] <= covers[0]:
            print(
                "  NOTE: interval NARROWED while coverage did not improve -- the "
                "signature of a displaced centre, not of insufficient data."
            )
    print("\n  All values provisional; nothing here is frozen until E3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

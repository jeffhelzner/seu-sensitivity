"""
Regime (e), analytic component: the menu-size range decision (§8.5(e)).

Most of this question does not need simulation.  gamma_size is identified from
the spread of the centered menu-size covariate ``s``, and for a linear predictor
the sampling variance of its coefficient scales as

    Var(gamma_size_hat)  ~  1 / (M * Var(s))

so the candidate size sets can be ranked exactly, before any sampling.  The
simulation is then only needed for what the closed form cannot give: whether the
ranking survives the model's nonlinearity, and how the position-stability subset
(§8.5(e)(iii)) retains menus by size.

Phase C already delivered (e)(ii), the null calibration: with gamma_size pinned
at 0 the fit returns 0, so a detected slope is behavioural rather than
choice-set geometry.  What remains is the RANGE sweep and the retention question.

*** Also flags the odd-size problem, which is easy to miss. Presentations are
REVERSALS (§3.4), and reversal leaves the MIDDLE item of an odd-sized menu in
place. Every pre-registered size is currently even precisely so that "every item
strictly changes position" holds. Both fallback sets contain size 3. ***
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

#: Candidate size sets. Weights are the share of menus at each size; None means
#: balanced (equal shares), which is the current pre-registered allocation.
CANDIDATES: Dict[str, Dict] = {
    "balanced_2468": {"sizes": [2, 4, 6, 8], "weights": None},
    "extreme_weighted_2468": {"sizes": [2, 4, 6, 8], "weights": [1 / 3, 1 / 6, 1 / 6, 1 / 3]},
    "fallback_2346": {"sizes": [2, 3, 4, 6], "weights": None},
    "narrow_234": {"sizes": [2, 3, 4], "weights": None},
}

REFERENCE = "balanced_2468"


def predictor_variance(sizes: Sequence[int], weights: Sequence[float] | None) -> float:
    """Var(s) under the given allocation; s is the centered menu size."""
    p = (
        np.full(len(sizes), 1.0 / len(sizes))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    p = p / p.sum()
    x = np.asarray(sizes, dtype=float)
    mean = float((p * x).sum())
    return float((p * (x - mean) ** 2).sum())


def mean_menu_size(sizes: Sequence[int], weights: Sequence[float] | None) -> float:
    p = (
        np.full(len(sizes), 1.0 / len(sizes))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    p = p / p.sum()
    return float((p * np.asarray(sizes, dtype=float)).sum())


def odd_size_share(sizes: Sequence[int], weights: Sequence[float] | None) -> float:
    """Share of menus whose reversal leaves an item fixed (odd sizes)."""
    p = (
        np.full(len(sizes), 1.0 / len(sizes))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    p = p / p.sum()
    return float(sum(pi for pi, s in zip(p, sizes) if s % 2 == 1))


def main() -> int:
    ref_var = predictor_variance(**CANDIDATES[REFERENCE])

    rows: List[Dict] = []
    print("=" * 84)
    print("Regime (e) analytic: menu-size range. PROVISIONAL until E3.")
    print("=" * 84)
    print(f"  SE(gamma_size) ~ 1/sqrt(M * Var(s)); ratios are relative to "
          f"{REFERENCE}")
    print(
        f"\n  {'candidate':<24}{'sizes':<14}{'Var(s)':>9}{'SE ratio':>10}"
        f"{'mean size':>11}{'odd share':>11}{'assess cost':>12}"
    )
    for name, spec in CANDIDATES.items():
        var = predictor_variance(**spec)
        se_ratio = float(np.sqrt(ref_var / var))
        mean_size = mean_menu_size(**spec)
        odd = odd_size_share(**spec)
        # §12: assessment payload scales with the mean number of items shown, so
        # a smaller mean menu is cheaper per choice call.
        cost_ratio = mean_size / mean_menu_size(**CANDIDATES[REFERENCE])
        rows.append(
            {
                "candidate": name,
                "sizes": spec["sizes"],
                "weights": spec["weights"],
                "var_s": var,
                "se_ratio_vs_reference": se_ratio,
                "mean_menu_size": mean_size,
                "odd_size_share": odd,
                "relative_assessment_payload": cost_ratio,
            }
        )
        print(
            f"  {name:<24}{str(spec['sizes']):<14}{var:>9.3f}{se_ratio:>10.3f}"
            f"{mean_size:>11.2f}{odd:>11.2f}{cost_ratio:>12.2f}"
        )

    print("\n  SE ratio > 1 means WIDER intervals on gamma_size than the "
          "current design.")

    print("\n" + "-" * 84)
    print("  Menus needed to match the reference's gamma_size precision:")
    for row in rows:
        factor = row["se_ratio_vs_reference"] ** 2
        print(f"    {row['candidate']:<24}{factor:>7.2f}x the menus"
              + ("   (reference)" if row["candidate"] == REFERENCE else ""))

    print("\n" + "-" * 84)
    print("  ODD-SIZE / POSITION-FLIP PROBLEM (§3.4, §8.8)")
    print("  Presentations are REVERSALS, and reversal fixes the middle item of")
    print("  an odd-sized menu, so that item cannot be probed for position bias.")
    for row in rows:
        if row["odd_size_share"] == 0:
            continue
        fixed = [
            (s, 1.0 / s) for s in row["sizes"] if s % 2 == 1
        ]
        detail = ", ".join(f"size {s}: {frac:.0%} of items fixed" for s, frac in fixed)
        print(f"    {row['candidate']:<24}{row['odd_size_share']:.0%} of menus "
              f"are odd  ({detail})")
    print("    balanced_2468 and extreme_weighted_2468: 0% -- every item moves.")

    out = Path("results/power/menu_size_range_analytic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "reference": REFERENCE,
                "rows": rows,
                "note": "Analytic ranking only. Simulation still needed for the "
                        "model's nonlinearity and for (e)(iii) retention by size.",
                "provisional": True,
                "frozen_at": None,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

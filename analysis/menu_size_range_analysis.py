"""
Regime (e), analytic component: the menu-size range decision (§8.5(e)).

gamma_size is identified from the spread of the centered menu-size covariate
``s``, and for a LINEAR predictor the sampling variance of its coefficient scales
as

    Var(gamma_size_hat)  ~  1 / (M * Var(s))

*** THIS CLOSED FORM IS NOT RELIABLE HERE, AND THE SIMULATION PROVED IT. ***
Measured against the sweep (2026-08-10), it is accurate for a REALLOCATION of the
same sizes (extreme-weighted: predicted 0.889, observed 0.905) but badly
overstates the penalty for a REDUCED RANGE:

    fallback {2,3,4,6}   predicted 1.512x SE, observed 1.031x
    narrow   {2,3,4}     predicted 2.739x SE, observed 1.694x

The reason is that ``h_m01_size`` is a softmax choice model, not a linear one:
per-observation INFORMATION also depends on menu size, because with fewer
alternatives the choice probability is better determined and each observation
says more about alpha.  Shrinking the range loses predictor variance but gains
information density, and the two partly cancel.  The formula below is therefore
kept as a DIAGNOSTIC and an upper bound on the penalty -- not as a ranking that
can retire a candidate without simulating it.

What the closed form still gives correctly is the ordering of Var(s), the mean
menu size (which drives assessment payload, §12), and the odd-size accounting.

Phase C already delivered (e)(ii), the null calibration: with gamma_size pinned
at 0 the fit returns 0, so a detected slope is behavioural rather than
choice-set geometry.

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


def _report_sweep(sweep_root: Path, rows: List[Dict]) -> None:
    """
    Merge the simulated results with the analytic ranking.

    The two halves answer different halves of §8.5(e): the closed form gives
    gamma_size PRECISION exactly, and only simulation gives (e)(iii) RETENTION,
    which moves the opposite way -- a smaller maximum menu size raises the
    stable-menu share while widening the slope.  Printed together so the trade
    is visible in one place rather than inferred across two artefacts.
    """
    analytic = {r["candidate"]: r for r in rows}
    found = []
    for name in analytic:
        path = sweep_root / name / "summary.json"
        if path.exists():
            found.append((name, json.loads(path.read_text())))

    if not found:
        return

    print("\n" + "=" * 84)
    print(f"SIMULATED ({len(found)}/{len(analytic)} candidates complete)")
    print("=" * 84)
    print(
        f"  {'candidate':<24}{'CI width':>10}{'pred SE':>9}{'cover':>7}"
        f"{'retain':>8}{'worst sz':>10}{'s/iter':>9}{'ratio':>7}"
    )
    for name, s in found:
        ret = s.get("retention") or {}
        print(
            f"  {name:<24}{s['mean_ci_width']:>10.4f}"
            f"{analytic[name]['se_ratio_vs_reference']:>9.3f}"
            f"{s['coverage']:>7.2f}"
            f"{ret.get('balanced_subset_retention', float('nan')):>8.3f}"
            f"{ret.get('worst_size_stable_share', float('nan')):>10.3f}"
            f"{s['seconds_per_iteration']:>9.0f}"
            f"{s.get('prediction_error_ratio', float('nan')):>7.2f}"
        )

    reference = next((s for n, s in found if n == REFERENCE), None)
    if reference is not None:
        print("\n  Observed CI width vs the analytic prediction "
              "(both relative to the reference):")
        for name, s in found:
            observed = s["mean_ci_width"] / reference["mean_ci_width"]
            predicted = analytic[name]["se_ratio_vs_reference"]
            print(f"    {name:<24}observed {observed:>6.3f}   "
                  f"predicted {predicted:>6.3f}")

    print("\n  retain = size-stratified stability subset as a share of all menus")
    print("  worst sz = stable share at the LARGEST size, which caps the subset")

    print("\n  per-size stable share:")
    for name, s in found:
        ret = (s.get("retention") or {}).get("per_size_stable_share") or {}
        detail = "  ".join(f"{k}:{v:.3f}" for k, v in sorted(ret.items(), key=lambda kv: int(kv[0])))
        print(f"    {name:<24}{detail}")


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
    print("  *** UPPER BOUND ONLY -- the closed form overstates the penalty for")
    print("  *** reduced-range sets. Measured: fallback 1.06x (not 2.29x),")
    print("  *** narrow 2.87x (not 7.50x). See the SIMULATED section below.")
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
                "note": "Var(s), mean size and odd-size accounting are exact. "
                        "The SE ratio is an UPPER BOUND on the penalty: the "
                        "linear-model formula overstates it for reduced-range "
                        "sets because a softmax's per-observation information "
                        "also rises as menus shrink. See menu_size_sweep.",
                "provisional": True,
                "frozen_at": None,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {out}")

    _report_sweep(Path("results/power/menu_size_sweep"), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

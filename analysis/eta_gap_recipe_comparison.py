"""
Phase D0 -- resolve the size-correlated eta-gap decision against REAL embeddings.

Offline. Reads the ``embeddings_reduced.npz`` artefacts written by the ``embed``
phase and makes NO API calls, so it can be re-run freely (the embed phase itself
is *not* cached and would re-bill).

Background
----------
A stratum recipe's top-two quality gap is size-invariant iff the recipe has >= 2
contenders.  With a single contender the runner-up comes from the filler pool, so
the gap is measured against max-of-(n-1) filler draws, which grows with n.  The
synthetic study (2026-08-03) showed the drift grows with embedding noise, and that
a two-contender variant reduces absolute cross-size drift in all six pool x noise
cells -- but compresses stratum contrast, badly for insurance.  Real embedding
noise decides it; this script measures it.

Decision rule -- FIXED BEFORE LOOKING AT THE NUMBERS
----------------------------------------------------
Adopt variant D globally iff, in all three pools:
  1. worst cross-size absolute gap difference is LOWER under D than under A, and
  2. the design's stratum ordering survives with adjacent separation >= 0.05
     in pool-standardized units.

On the ordering: the recipes are built so that the ``strong`` stratum has the
WIDEST gap (one clear winner over filler), ``ambiguous`` the NARROWEST (two
near-tied strong contenders -- this is where alpha is identified), and ``weak``
sits in between (a middling winner over filler).  The intended ordering of mean
gaps is therefore strong > weak > ambiguous.

If D fixes drift but collapses insurance's strata, the fallback is PER-POOL
recipes: insurance keeps A (it has no merit lattice, only labels, so its
within-tier spread is small), the two lattice pools take D.

Nothing here freezes anything.  Outputs are provisional until E3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study import item_validation, pools as pools_module
from applications.seu_sensitivity_study import problem_generation as pg
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig

POOL_IDS = ("insurance", "venture", "hiring")

#: Intended ordering of mean gaps, widest first.
STRATUM_ORDER = ("strong", "weak", "ambiguous")

#: Minimum adjacent separation, in pool-standardized units, for the design to
#: still distinguish its strata.  Provisional; re-picked at E3.
MIN_ADJACENT_SEPARATION = 0.05

RECIPE_VARIANTS: Dict[str, Sequence[pg.StratumRecipe]] = {
    # Variant A -- current problem_generation.DEFAULT_RECIPES.
    "A_current": pg.DEFAULT_RECIPES,
    # Variant D -- two contenders in every recipe.
    "D_two_contenders": (
        pg.StratumRecipe(
            stratum="strong", contenders=("strong", "ambiguous"), filler_label="weak"
        ),
        pg.StratumRecipe(
            stratum="ambiguous", contenders=("strong", "strong"), filler_label="weak"
        ),
        pg.StratumRecipe(
            stratum="weak", contenders=("ambiguous", "ambiguous"), filler_label="weak"
        ),
    ),
}


def _load_reduced(results_dir: Path, pool_id: str) -> Dict[str, np.ndarray]:
    path = results_dir / "pools" / pool_id / "embeddings_reduced.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No embeddings for pool {pool_id!r} at {path}. "
            "Run: python -m applications.seu_sensitivity_study run "
            "--phases design,embed"
        )
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _mean(values: Sequence[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if array.size else None


def _axis_summary(axis_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten the nested quality-axis report into the few numbers that matter."""
    chosen = axis_report.get("axis")
    pc1 = axis_report.get("pc1", {})
    lda = axis_report.get("lda", {})
    if chosen == "lda" and lda:
        auc = lda.get("loo_auc_strong_vs_weak")
        rho = lda.get("loo_spearman_vs_merit")
    else:
        auc = pc1.get("auc_strong_vs_weak")
        rho = pc1.get("spearman_vs_merit")
    return {
        "axis": chosen,
        "status": axis_report.get("status"),
        "passed": axis_report.get("passed"),
        "auc_strong_vs_weak": auc,
        "spearman_vs_merit": rho,
        "pc1_auc": pc1.get("auc_strong_vs_weak"),
        "lda_loo_auc": lda.get("loo_auc_strong_vs_weak"),
        "n_items": axis_report.get("n_items"),
        "merit_source": axis_report.get("merit_source"),
    }


def _evaluate_variant(
    pool: Mapping[str, Any],
    quality: Mapping[str, float],
    recipes: Sequence[pg.StratumRecipe],
    config: SEUSensitivityStudyConfig,
) -> Dict[str, Any]:
    """Generate a problem set under *recipes* and summarize its eta gaps."""
    problem_set = pg.generate_problem_set(
        pool,
        problems_per_family=config.problems_for(pool["pool_id"]),
        seed=config.seed,
        menu_sizes=config.menu_sizes,
        num_presentations=config.num_presentations,
        presentation_mode=config.presentation_mode,
        recipes=recipes,
    )
    gaps = item_validation._menu_gaps(problem_set, quality)

    sizes = sorted({g["menu_size"] for g in gaps})
    by_size = {s: _mean([g["gap"] for g in gaps if g["menu_size"] == s]) for s in sizes}
    by_stratum = {
        st: _mean([g["gap"] for g in gaps if g["stratum"] == st])
        for st in STRATUM_ORDER
    }

    # Worst absolute cross-size difference (the quantity the gate blocks on).
    size_diffs: Dict[str, float] = {}
    for i, left in enumerate(sizes):
        for right in sizes[i + 1 :]:
            size_diffs[f"{left}_vs_{right}"] = abs(by_size[left] - by_size[right])
    worst_size_diff = max(size_diffs.values(), default=0.0)

    # Adjacent stratum separations, in the intended (widest-first) order.
    separations: Dict[str, float] = {}
    for left, right in zip(STRATUM_ORDER, STRATUM_ORDER[1:]):
        if by_stratum[left] is not None and by_stratum[right] is not None:
            separations[f"{left}_minus_{right}"] = by_stratum[left] - by_stratum[right]
    min_separation = min(separations.values(), default=float("nan"))
    ordering_holds = all(v > 0 for v in separations.values())

    return {
        "n_menus": len(gaps),
        "overall_mean_gap": _mean([g["gap"] for g in gaps]),
        "by_size": {str(k): v for k, v in by_size.items()},
        "by_stratum": by_stratum,
        "cross_size_diffs": size_diffs,
        "worst_cross_size_diff": worst_size_diff,
        "stratum_separations": separations,
        "min_adjacent_separation": min_separation,
        "ordering_holds": ordering_holds,
        "contrast_ok": bool(
            ordering_holds and min_separation >= MIN_ADJACENT_SEPARATION
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    config = SEUSensitivityStudyConfig()
    results_dir = Path(args.results_dir or config.results_dir)
    thresholds = item_validation.load_gate_thresholds()

    report: Dict[str, Any] = {
        "decision_rule": {
            "min_adjacent_separation": MIN_ADJACENT_SEPARATION,
            "stratum_order_widest_first": list(STRATUM_ORDER),
            "fixed_before_measurement": True,
        },
        "provisional": True,
        "frozen_at": None,
        "pools": {},
    }

    for pool_id in POOL_IDS:
        pool = pools_module.load_pool(pool_id)
        embeddings = _load_reduced(results_dir, pool_id)
        axis_report, quality = item_validation._check_quality_axis(
            pool, embeddings, thresholds
        )
        if quality is None:
            report["pools"][pool_id] = {
                "error": "quality axis unavailable (R4 failed)",
                "quality_axis": _axis_summary(axis_report),
                "axis_detail": axis_report,
            }
            continue

        report["pools"][pool_id] = {
            "quality_axis": _axis_summary(axis_report),
            "variants": {
                name: _evaluate_variant(pool, quality, recipes, config)
                for name, recipes in RECIPE_VARIANTS.items()
            },
        }

    # -- Cross-pool spread per variant (the other arm the gate blocks on) --
    report["cross_pool"] = {}
    for name in RECIPE_VARIANTS:
        means = {
            pid: data["variants"][name]["overall_mean_gap"]
            for pid, data in report["pools"].items()
            if "variants" in data
        }
        worst = 0.0
        ids = sorted(means)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                worst = max(worst, abs(means[left] - means[right]))
        report["cross_pool"][name] = {
            "overall_mean_gap": means,
            "worst_cross_pool_diff": worst,
        }

    _apply_decision(report)
    _print_report(report, thresholds)

    out = Path(args.output) if args.output else results_dir / "eta_gap_recipe_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out}")
    return 0


def _apply_decision(report: Dict[str, Any]) -> None:
    pools_with_data = {
        pid: d for pid, d in report["pools"].items() if "variants" in d
    }
    missing = [pid for pid in POOL_IDS if pid not in pools_with_data]

    drift_improved: Dict[str, bool] = {}
    contrast_ok: Dict[str, bool] = {}
    for pid, data in pools_with_data.items():
        a = data["variants"]["A_current"]
        d = data["variants"]["D_two_contenders"]
        drift_improved[pid] = d["worst_cross_size_diff"] < a["worst_cross_size_diff"]
        contrast_ok[pid] = d["contrast_ok"]

    # A pool with no usable quality axis is NOT evidence for D -- it is missing
    # evidence.  Concluding "adopt globally" from the pools that happened to
    # work would silently exclude the pool the decision was most in doubt for.
    if missing:
        verdict = "INCOMPLETE"
        adopt_globally = None
    elif all(drift_improved.values()) and all(contrast_ok.values()):
        verdict = "ADOPT_D_GLOBALLY"
        adopt_globally = True
    else:
        verdict = "PER_POOL"
        adopt_globally = False

    per_pool = {
        pid: (
            "D_two_contenders"
            if (drift_improved[pid] and contrast_ok[pid])
            else "A_current"
        )
        for pid in pools_with_data
    }
    for pid in missing:
        per_pool[pid] = "UNDECIDED (no usable quality axis)"

    report["decision"] = {
        "verdict": verdict,
        "pools_without_quality_axis": missing,
        "drift_improved_under_D": drift_improved,
        "contrast_ok_under_D": contrast_ok,
        "adopt_D_globally": adopt_globally,
        "per_pool_recommendation": per_pool,
        "note": "Provisional. Phase D proposes; E3 freezes.",
    }


def _fmt(value: Any, width: int = 8) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.3f}"


def _print_report(report: Dict[str, Any], thresholds: Any) -> None:
    print("=" * 78)
    print("Phase D0 -- eta-gap recipe comparison against REAL embeddings")
    print("=" * 78)
    print(
        f"Gate threshold (provisional): cross-size <= "
        f"{thresholds.eta_gap_max_cross_size_diff}, cross-pool <= "
        f"{thresholds.eta_gap_max_cross_pool_diff}"
    )
    print(f"Decision rule: min adjacent separation >= {MIN_ADJACENT_SEPARATION}, "
          f"ordering {' > '.join(STRATUM_ORDER)}")

    for pid, data in report["pools"].items():
        axis = data.get("quality_axis", {})
        axis_line = (
            f"\n[{pid}]  quality axis: chosen={axis.get('axis')} "
            f"status={axis.get('status')} passed={axis.get('passed')} "
            f"auc={_fmt(axis.get('auc_strong_vs_weak'), 6)} "
            f"rho={_fmt(axis.get('spearman_vs_merit'), 6)} "
            f"(pc1_auc={_fmt(axis.get('pc1_auc'), 6)}, "
            f"lda_loo_auc={_fmt(axis.get('lda_loo_auc'), 6)})"
        )
        print(axis_line)
        if "variants" not in data:
            print(f"  *** R4 FAILED -- {data.get('error')}. "
                  f"No eta-gap comparison is possible for this pool. ***")
            continue
        header = (
            f"  {'variant':<18}{'worstΔsize':>11}{'strong':>9}{'weak':>9}"
            f"{'ambig':>9}{'minSep':>9}{'contrast':>10}"
        )
        print(header)
        for name, v in data["variants"].items():
            st = v["by_stratum"]
            print(
                f"  {name:<18}{_fmt(v['worst_cross_size_diff'], 11)}"
                f"{_fmt(st.get('strong'), 9)}{_fmt(st.get('weak'), 9)}"
                f"{_fmt(st.get('ambiguous'), 9)}"
                f"{_fmt(v['min_adjacent_separation'], 9)}"
                f"{('OK' if v['contrast_ok'] else 'COLLAPSE'):>10}"
            )
        print(f"  by menu size (mean gap):")
        for name, v in data["variants"].items():
            sizes = "  ".join(
                f"{s}:{_fmt(g, 6)}" for s, g in sorted(v["by_size"].items(), key=lambda kv: int(kv[0]))
            )
            print(f"    {name:<18}{sizes}")

    print("\n" + "-" * 78)
    print("Cross-pool spread of overall mean gap:")
    for name, cp in report["cross_pool"].items():
        print(f"  {name:<18}worst diff {_fmt(cp['worst_cross_pool_diff'], 7)}")

    dec = report["decision"]
    print("\n" + "-" * 78)
    print("DECISION (provisional until E3)")
    print(f"  VERDICT                : {dec['verdict']}")
    if dec["pools_without_quality_axis"]:
        print(f"  no quality axis        : {dec['pools_without_quality_axis']}")
    print(f"  drift improved under D : {dec['drift_improved_under_D']}")
    print(f"  contrast ok under D    : {dec['contrast_ok_under_D']}")
    print(f"  per-pool recommendation: {dec['per_pool_recommendation']}")


if __name__ == "__main__":
    raise SystemExit(main())

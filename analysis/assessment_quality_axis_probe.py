"""
Phase D0 probe -- can the DECIDERS see the quality signal the embeddings cannot?

Offline; reads persisted assessment artefacts.  No API calls.

Context
-------
`insurance_quality_axis_diagnostic.py` established that the authored insurance
fraud labels are not recoverable from the item text by any embedding- or
lexical-based estimator (observed TF-IDF LOO AUC sits exactly on its permutation
null).  The construct is relational -- do the documented facts CONTRADICT or
CORROBORATE the claim -- which embeddings do not encode.

But the eta-gap is a design control on how hard a menu is FOR THE DECIDER.  The
relevant question is therefore not whether an embedding separates the labels but
whether the models' own elicited beliefs do.  This script measures that, and
reports it beside the embedding axis so the two are directly comparable.

A pass here would motivate proposing (at E3, not before) that the eta-gap axis be
belief-based, with the embedding axis retained as a reported diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study import pools as pools_module
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig

LABEL_ORDINAL = {"weak": 0.0, "ambiguous": 1.0, "strong": 2.0}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
        / (pos.size * neg.size)
    )


def _perm_null_auc(
    scores: np.ndarray, is_strong: np.ndarray, n: int, seed: int = 0
) -> Dict[str, float]:
    """
    Permutation null for the AUC.

    Unlike the leave-one-out estimators in the sibling diagnostic, a raw AUC on
    a fixed score vector is unbiased under the null (it centres on 0.5).  The
    null is computed anyway so the two scripts' numbers are read the same way
    and the p-value is exact rather than asymptotic at n=21.
    """
    rng = np.random.default_rng(seed)
    observed = _auc(scores[is_strong], scores[~is_strong])
    null = np.array(
        [
            _auc(scores[perm], scores[~perm])
            for perm in (rng.permutation(is_strong) for _ in range(n))
        ]
    )
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_two_sided": float((np.abs(null - 0.5) >= abs(observed - 0.5)).mean()),
    }


def probe(pool_id: str, results_dir: Path, n_permutations: int) -> Dict[str, Any]:
    from scipy.stats import spearmanr

    pool = pools_module.load_pool(pool_id)
    by_id = {i["id"]: i for i in pool["items"]}
    consequences = pool.get("consequences")

    assess_dir = results_dir / "pools" / pool_id / "assessments"
    files = sorted(assess_dir.glob("*.json")) if assess_dir.exists() else []
    if not files:
        raise FileNotFoundError(f"No assessment artefacts under {assess_dir}")

    out: Dict[str, Any] = {
        "pool_id": pool_id,
        "consequences": consequences,
        "models": {},
        "n_models": len(files),
    }

    per_model_scores: Dict[str, Dict[str, float]] = {}
    for path in files:
        payload = json.loads(path.read_text())
        model = payload["model_name"]
        scores: Dict[str, float] = {}
        for rec in payload["assessments"]:
            if not rec.get("parse_ok") or not rec.get("probabilities"):
                continue
            p = np.asarray(rec["probabilities"], dtype=float)
            # Expected position on the ordered consequence scale.  The scale runs
            # from "neither investigator recommends investigation" upward, so a
            # higher expectation == a stronger fraud signal, matching the label
            # rubric's direction.
            scores[rec["item_id"]] = float((np.arange(p.size) * p).sum())
        per_model_scores[model] = scores

        ids = [i for i in scores if i in by_id]
        labels = [by_id[i]["quality_label"] for i in ids]
        vals = np.array([scores[i] for i in ids])
        strong = np.array([l == "strong" for l in labels])
        weak = np.array([l == "weak" for l in labels])
        sub = strong | weak
        stats = _perm_null_auc(vals[sub], strong[sub], n_permutations)
        out["models"][model] = {
            "n_items": len(ids),
            "auc_strong_vs_weak": stats["observed"],
            "null_mean": stats["null_mean"],
            "p_two_sided": stats["p_two_sided"],
            "spearman_vs_ordinal": float(
                spearmanr(vals, [LABEL_ORDINAL[l] for l in labels]).statistic
            ),
            "mean_by_label": {
                l: round(float(vals[[x == l for x in labels]].mean()), 4)
                for l in ("strong", "ambiguous", "weak")
                if any(x == l for x in labels)
            },
        }

    # -- Across-model mean belief (the quantity R3 already uses per pool) -----
    common = set.intersection(*(set(s) for s in per_model_scores.values()))
    ids = sorted(i for i in common if i in by_id)
    mean_scores = np.array(
        [np.mean([per_model_scores[m][i] for m in per_model_scores]) for i in ids]
    )
    labels = [by_id[i]["quality_label"] for i in ids]
    strong = np.array([l == "strong" for l in labels])
    weak = np.array([l == "weak" for l in labels])
    sub = strong | weak
    stats = _perm_null_auc(mean_scores[sub], strong[sub], n_permutations)
    out["across_model_mean"] = {
        "n_items": len(ids),
        "auc_strong_vs_weak": stats["observed"],
        "null_mean": stats["null_mean"],
        "null_sd": stats["null_sd"],
        "p_two_sided": stats["p_two_sided"],
        "spearman_vs_ordinal": float(
            spearmanr(mean_scores, [LABEL_ORDINAL[l] for l in labels]).statistic
        ),
        "mean_by_label": {
            l: round(float(mean_scores[[x == l for x in labels]].mean()), 4)
            for l in ("strong", "ambiguous", "weak")
        },
    }
    return out


def _fmt(v: Any, w: int = 6) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'n/a':>{w}}"
    return f"{v:>{w}.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=str, default="insurance")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args(argv)

    config = SEUSensitivityStudyConfig()
    results_dir = Path(args.results_dir or config.results_dir)

    reports = {}
    for pool_id in [p.strip() for p in args.pools.split(",")]:
        rep = probe(pool_id, results_dir, args.permutations)
        reports[pool_id] = rep
        print("=" * 78)
        print(f"[{pool_id}] BELIEF-BASED quality axis  "
              f"({rep['n_models']} model(s) assessed)")
        if rep.get("consequences"):
            print("  consequence scale (index 0 -> last):")
            for k, c in enumerate(rep["consequences"]):
                label = c if isinstance(c, str) else c.get("label", str(c))
                print(f"    [{k}] {label}")
        print("-" * 78)
        print(f"  {'model':<34}{'AUC':>8}{'p':>8}{'rho':>8}  mean by label")
        for model, m in rep["models"].items():
            print(
                f"  {model:<34}{_fmt(m['auc_strong_vs_weak'])}"
                f"{_fmt(m['p_two_sided'])}{_fmt(m['spearman_vs_ordinal'])}"
                f"  {m['mean_by_label']}"
            )
        a = rep["across_model_mean"]
        print(
            f"  {'ACROSS-MODEL MEAN':<34}{_fmt(a['auc_strong_vs_weak'])}"
            f"{_fmt(a['p_two_sided'])}{_fmt(a['spearman_vs_ordinal'])}"
            f"  {a['mean_by_label']}"
        )
        print(f"\n  Compare EMBEDDING axis for this pool: PC1 auc 0.546, "
              f"LDA LOO auc 0.472 (both at null).")
        print()

    out = results_dir / "assessment_quality_axis_probe.json"
    out.write_text(json.dumps(reports, indent=2, default=float))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

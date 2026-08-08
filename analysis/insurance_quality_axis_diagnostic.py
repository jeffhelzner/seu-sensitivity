"""
Phase D0 follow-up -- why does the insurance quality axis fail R4?

Offline; reads the persisted embeddings and pool text. No API calls.

The question this answers: is the authored fraud-signal label ABSENT from the
claim text, or is it PRESENT but invisible to the gate's particular estimator
(PC1, or LDA restricted to the first ``lda_dim``=8 principal components)?

Those two diagnoses have opposite consequences.  "Absent" means the insurance
design cannot support a quality-stratified menu design at all.  "Present but
missed" means the gate's estimator window is too narrow and the pool is fine.

Discriminating evidence collected here:
  1. Per-PC association between each principal component and the label ordinal,
     across ALL components -- shows WHERE in the spectrum any signal lives.
  2. LDA LOO AUC as a function of how many PCs the estimator is given.
  3. A lexical (TF-IDF) LOO classifier -- if bag-of-words separates the labels,
     the signal is demonstrably in the text and the embedding is the problem.
  4. Text-length confound check, and worked examples per label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study import item_validation, pools as pools_module
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig

LABEL_ORDINAL = {"weak": 0.0, "ambiguous": 1.0, "strong": 2.0}


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC; 0.5 is chance."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _loo_auc_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    """Leave-one-out AUC of a regularized logistic classifier."""
    from sklearn.linear_model import LogisticRegression

    scores = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        if len(np.unique(y[mask])) < 2:
            return float("nan")
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X[mask], y[mask])
        scores[i] = model.decision_function(X[i : i + 1])[0]
    return _auc(scores[y == 1], scores[y == 0])


def diagnose(pool_id: str, results_dir: Path, thresholds: Any) -> Dict[str, Any]:
    pool = pools_module.load_pool(pool_id)
    reduced = _load_npz(results_dir / "pools" / pool_id / "embeddings_reduced.npz")
    raw = _load_npz(results_dir / "pools" / pool_id / "embeddings_raw.npz")

    items = [i for i in pool["items"] if i["id"] in reduced]
    ids = [i["id"] for i in items]
    labels = [i["quality_label"] for i in items]
    ordinal = np.array([LABEL_ORDINAL[l] for l in labels])
    texts = [i["text"] for i in items]

    Xr = np.vstack([reduced[i] for i in ids])
    Xraw = np.vstack([raw[i] for i in ids])
    strong = np.array([l == "strong" for l in labels])
    weak = np.array([l == "weak" for l in labels])
    subset = strong | weak

    out: Dict[str, Any] = {
        "pool_id": pool_id,
        "n_items": len(ids),
        "n_pcs": int(Xr.shape[1]),
        "label_counts": {l: labels.count(l) for l in ("strong", "ambiguous", "weak")},
    }

    # -- 1. Per-PC association with the label ordinal -------------------------
    from scipy.stats import spearmanr

    per_pc = []
    for k in range(Xr.shape[1]):
        rho = float(spearmanr(Xr[:, k], ordinal).statistic)
        auc_k = _auc(Xr[strong, k], Xr[weak, k])
        per_pc.append(
            {"pc": k + 1, "spearman_vs_ordinal": rho, "auc_strong_vs_weak": auc_k}
        )
    out["per_pc"] = per_pc
    ranked = sorted(per_pc, key=lambda d: -abs(d["auc_strong_vs_weak"] - 0.5))
    out["most_informative_pcs"] = ranked[:6]

    # -- 2. LDA LOO AUC vs number of PCs the estimator sees -------------------
    sweep = []
    for dim in (2, 4, 8, 12, 16, 20, min(24, Xr.shape[1]), Xr.shape[1]):
        dim = int(min(dim, Xr.shape[1]))
        scores = item_validation._loo_lda_scores(
            Xr[subset][:, :dim], strong[subset], thresholds.lda_shrinkage
        )
        auc_d = (
            float("nan")
            if scores is None
            else _auc(scores[strong[subset]], scores[~strong[subset]])
        )
        sweep.append({"n_pcs": dim, "loo_auc": auc_d})
    # de-duplicate while preserving order
    seen = set()
    out["lda_dim_sweep"] = [
        s for s in sweep if not (s["n_pcs"] in seen or seen.add(s["n_pcs"]))
    ]

    # -- 3. Logistic on raw embeddings (does ANY linear direction work?) ------
    out["logistic_loo_auc_raw"] = _loo_auc_logistic(
        Xraw[subset], strong[subset].astype(int)
    )

    # -- 4. Lexical baseline: is the signal in the WORDS at all? --------------
    from sklearn.feature_extraction.text import TfidfVectorizer

    sub_texts = [t for t, m in zip(texts, subset) if m]
    sub_y = strong[subset].astype(int)
    lex_scores = np.zeros(len(sub_y), dtype=float)
    for i in range(len(sub_y)):
        mask = np.ones(len(sub_y), dtype=bool)
        mask[i] = False
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
        Xtr = vec.fit_transform([sub_texts[j] for j in range(len(sub_y)) if mask[j]])
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(Xtr, sub_y[mask])
        lex_scores[i] = model.decision_function(vec.transform([sub_texts[i]]))[0]
    out["tfidf_loo_auc"] = _auc(lex_scores[sub_y == 1], lex_scores[sub_y == 0])

    # -- 5. Length confound ---------------------------------------------------
    lengths = np.array([len(t) for t in texts], dtype=float)
    out["length_by_label"] = {
        l: round(float(lengths[[x == l for x in labels]].mean()), 1)
        for l in ("strong", "ambiguous", "weak")
    }
    out["length_spearman_vs_ordinal"] = float(spearmanr(lengths, ordinal).statistic)
    out["pc1_spearman_vs_length"] = float(spearmanr(Xr[:, 0], lengths).statistic)

    out["_examples"] = {
        l: [t for t, lab in zip(texts, labels) if lab == l][:2]
        for l in ("strong", "weak")
    }
    return out


def _tfidf_loo_auc(texts: List[str], y: np.ndarray) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    scores = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        if len(np.unique(y[mask])) < 2:
            return float("nan")
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
        Xtr = vec.fit_transform([texts[j] for j in range(len(y)) if mask[j]])
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(Xtr, y[mask])
        scores[i] = model.decision_function(vec.transform([texts[i]]))[0]
    return _auc(scores[y == 1], scores[y == 0])


def permutation_null(
    pool_id: str, results_dir: Path, n_permutations: int, seed: int = 0
) -> Dict[str, Any]:
    """
    Null distribution of the LOO AUC under label permutation.

    Leave-one-out on a small sample is NEGATIVELY biased when there is no
    signal: removing a point shifts the training centroid away from it, so the
    held-out point tends to score on the wrong side.  An observed AUC well
    below 0.5 is therefore NOT by itself evidence of inverted labels -- it has
    to be compared against this null, not against 0.5.
    """
    pool = pools_module.load_pool(pool_id)
    reduced = _load_npz(results_dir / "pools" / pool_id / "embeddings_reduced.npz")
    items = [i for i in pool["items"] if i["id"] in reduced]
    labels = [i["quality_label"] for i in items]
    texts = [i["text"] for i in items]

    strong = np.array([l == "strong" for l in labels])
    weak = np.array([l == "weak" for l in labels])
    subset = strong | weak
    sub_texts = [t for t, m in zip(texts, subset) if m]
    y = strong[subset].astype(int)

    observed = _tfidf_loo_auc(sub_texts, y)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        null.append(_tfidf_loo_auc(sub_texts, rng.permutation(y)))
    null_arr = np.array([v for v in null if not np.isnan(v)], dtype=float)

    # Two-sided empirical p against the permutation null.
    centre = float(null_arr.mean())
    p_two_sided = float(
        (np.abs(null_arr - centre) >= abs(observed - centre)).mean()
    )
    return {
        "observed_tfidf_loo_auc": observed,
        "null_mean": centre,
        "null_sd": float(null_arr.std(ddof=1)),
        "null_q05": float(np.quantile(null_arr, 0.05)),
        "null_q95": float(np.quantile(null_arr, 0.95)),
        "n_permutations": int(null_arr.size),
        "p_two_sided": p_two_sided,
    }


def _fmt(v: Any, w: int = 6) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'n/a':>{w}}"
    return f"{v:>{w}.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=str, default="insurance,venture,hiring")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Label-permutation null for the TF-IDF LOO AUC (insurance only).",
    )
    args = parser.parse_args(argv)

    config = SEUSensitivityStudyConfig()
    results_dir = Path(args.results_dir or config.results_dir)
    thresholds = item_validation.load_gate_thresholds()

    reports = {}
    for pool_id in args.pools.split(","):
        rep = diagnose(pool_id.strip(), results_dir, thresholds)
        reports[pool_id.strip()] = rep

        print("=" * 78)
        print(f"[{rep['pool_id']}]  n={rep['n_items']}  PCs={rep['n_pcs']}  "
              f"labels={rep['label_counts']}")
        print("-" * 78)
        print("  LDA LOO AUC vs number of PCs given to the estimator:")
        line = "    " + "  ".join(
            f"{s['n_pcs']:>3}PC:{_fmt(s['loo_auc'], 5)}" for s in rep["lda_dim_sweep"]
        )
        print(line)
        print(f"  logistic LOO AUC on RAW 1536-d embedding : "
              f"{_fmt(rep['logistic_loo_auc_raw'], 6)}")
        print(f"  TF-IDF lexical LOO AUC (signal in words?) : "
              f"{_fmt(rep['tfidf_loo_auc'], 6)}")
        print("  most label-informative PCs (|AUC-0.5| ranked):")
        for d in rep["most_informative_pcs"]:
            print(f"    PC{d['pc']:<3} auc={_fmt(d['auc_strong_vs_weak'], 6)} "
                  f"rho={_fmt(d['spearman_vs_ordinal'], 6)}")
        print(f"  mean text length by label : {rep['length_by_label']}")
        print(f"  length vs label ordinal   : "
              f"{_fmt(rep['length_spearman_vs_ordinal'], 6)}")
        print(f"  PC1 vs text length        : "
              f"{_fmt(rep['pc1_spearman_vs_length'], 6)}")

        if args.show_text and rep["pool_id"] == "insurance":
            for label, examples in rep["_examples"].items():
                for t in examples:
                    print(f"\n  --- {label.upper()} example ---\n  {t[:420]}")
        print()

    if args.permutations:
        for pool_id in ("insurance", "venture"):
            if pool_id not in reports:
                continue
            null = permutation_null(pool_id, results_dir, args.permutations)
            reports[pool_id]["permutation_null"] = null
            print("=" * 78)
            print(f"[{pool_id}] TF-IDF LOO AUC vs label-permutation null "
                  f"({null['n_permutations']} permutations)")
            print(f"  observed        : {_fmt(null['observed_tfidf_loo_auc'])}")
            print(f"  null mean (sd)  : {_fmt(null['null_mean'])} "
                  f"({_fmt(null['null_sd'])})")
            print(f"  null 5%-95%     : {_fmt(null['null_q05'])} .. "
                  f"{_fmt(null['null_q95'])}")
            print(f"  p (two-sided)   : {_fmt(null['p_two_sided'])}")
            print()

    out = results_dir / "insurance_axis_diagnostic.json"
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "_examples"}
        for k, v in reports.items()
    }
    out.write_text(json.dumps(serializable, indent=2, default=float))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

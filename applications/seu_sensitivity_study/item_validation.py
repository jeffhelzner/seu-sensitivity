"""
The pre-choice validation gate (§5 R3, §6.3 R4, §3.4/§9 item 8).

:func:`run_gate` is the single thing standing between the design artefacts and
the 13,680 choice API calls.  ``study_runner._require_gate`` refuses to start
choice collection unless this module's report carries ``passed: True``, so
every check here is a spend gate, not a diagnostic printout.

Four checks, each answering a question the study cannot answer after the money
is spent:

**R4 -- is PC1 actually quality?** (§6.3)  The principal embedding axis is used
as the item-quality proxy that the eta-gap equalization is defined on.  That
must be verified, not assumed: PC1 has to separate the authored
strong/ambiguous/weak labels.  When it does not, the plan's own fallback is a
**label-supervised LDA axis**, which this module computes and uses instead,
recording which axis the rest of the gate ran on.  A failure here is a failure
of the *proxy*, so it changes the axis rather than immediately failing the run;
the run fails only if neither axis separates the labels out of sample.

**§6.3 -- are pool difficulties comparable, across pools and across menu
sizes?**  The RQ5 difference-in-differences is only as clean as the pool
comparison behind it, and the RQ6 size slope is only interpretable if choice-set
geometry does not drift with size.  The size arm is the subtle one: by order
statistics the expected spacing between the best and next-best of N draws
shrinks as N grows, so a size-correlated gap shift can be manufactured by the
draw count alone.  ``problem_generation`` already holds the contender count
size-invariant to prevent that; this check *measures* whether it worked.

**R3 -- do the item-text embeddings carry the belief signal at all?** (§5)
Item-text embeddings are the sole channel into ``h_m01``.  The parsed K outcome
probabilities are regressed (as K-1 additive log-ratio responses) on the
embedding, cross-validated.  Reported per pool -- which blocks -- and per
model x pool, which is the §9.2 differential-mis-measurement diagnostic and
reports only, because §5 states the pass rule per pool.

**Assessment parse health.**  §6.4 makes NA rate a reported quantity, but no
choice NA exists yet at gate time.  The parse-failure rate on the *assessment*
responses is the analogous quantity that does exist, and a model that cannot
emit a parseable probability line will not suddenly start at the choice step.

Thresholds
----------
Every threshold is **provisional** until build phase E3 freezes the
pre-registration (§13).  They live in ``configs/gate_thresholds.json``, are
overridable per run through ``config.gate_thresholds``, and are echoed into
every report next to ``provisional: true`` -- so a report produced under
guessed thresholds can never be mistaken for one produced under frozen ones.

Estimator note (why ridge on a few PCs)
---------------------------------------
The insurance anchor has 30 items and ``reduce_embeddings`` clamps PCA to
``min(target_dim, n_items)``, so an unregularized regression of belief on the
full embedding would have p = n and a meaningless cross-validated R-squared.
The R3 regression therefore runs on the leading ``r3_dim`` principal components
with ridge and exact leave-one-out cross-validation.  This changes only the
validity *check*; the ``w`` matrix handed to Stan is untouched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import schemas

logger = logging.getLogger(__name__)

__all__ = [
    "GateThresholds",
    "load_gate_thresholds",
    "run_gate",
    "GATE_STATUS",
]

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_THRESHOLD_FILE = _CONFIG_DIR / "gate_thresholds.json"

#: Terminal statuses a gate report can carry.
GATE_STATUS: Tuple[str, ...] = (
    "passed",
    "failed",
    "awaiting_embeddings",
    "awaiting_assessments",
)

#: Label -> ordinal, used when a pool carries no authored ``merit_total``
#: (the insurance anchor's text is frozen, so it has labels but no grid).
_LABEL_ORDINAL = {"weak": 0.0, "ambiguous": 1.0, "strong": 2.0}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateThresholds:
    """
    Gate cut-offs.  **All values are provisional until build phase E3.**

    The defaults below are deliberately marked rather than silently adopted: a
    threshold invented during implementation and later mistaken for a
    pre-registered one would turn the whole gate into post-hoc tuning.
    """

    #: R4 (§6.3): PC1 must separate strong from weak this well to be used as
    #: the quality proxy; otherwise the LDA fallback is substituted.
    pc1_auc_strong_vs_weak: float = 0.75
    #: R4: rank correlation of the axis against authored merit.
    pc1_spearman_vs_merit: float = 0.35

    #: R3 (§5): cross-validated R^2 of belief on embedding, per pool. BLOCKS.
    r0_pool: float = 0.30
    #: R3: same quantity per model x pool. Reported, does not block (§5 states
    #: the pass rule per pool and calls this matrix a diagnostic).
    r0_model_pool: float = 0.20

    #: §6.3: max difference in mean top-two eta gap between this pool and any
    #: already-gated sibling pool. BLOCKS.
    #:
    #: In **pool-standardized units** -- ``_menu_gaps`` already divides by the
    #: spread of item quality scores, so a gap of 0.25 means the same thing in
    #: every pool and at every menu size.  An earlier version blocked on
    #: Cohen's d instead, which divides a second time by the *within-size*
    #: spread of gaps.  That is inverted for this purpose: it rewards designs
    #: whose menus vary wildly within a size and punishes tightly controlled
    #: ones.  A design whose size-2 menus all sit at 0.85 and whose size-8
    #: menus all sit at 0.80 is close to ideal for RQ6, yet its within-size
    #: spreads are near zero, so its d diverges and the gate would block it.
    eta_gap_max_cross_pool_diff: float = 0.25
    #: §3.4/§9 item 8: same, between menu-size strata within this pool. BLOCKS.
    eta_gap_max_cross_size_diff: float = 0.25

    #: §6.4 analogue available before any choice is collected. BLOCKS.
    max_assessment_parse_failure: float = 0.05

    # -- Estimator settings (not pass/fail cut-offs) --
    r3_dim: int = 8
    r3_ridge_alpha: float = 1.0
    lda_dim: int = 8
    lda_shrinkage: float = 1e-3
    #: Below this many observations a sub-check reports ``insufficient_data``
    #: instead of a number, rather than blocking on noise.
    min_items_for_check: int = 12

    #: False only once E3 freezes the pre-registration.
    provisional: bool = True
    frozen_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_gate_thresholds(
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    path: Optional[Path] = None,
) -> GateThresholds:
    """
    Resolve thresholds: packaged JSON, then per-run overrides.

    Unknown keys are ignored with a warning rather than raising, so a stale key
    in a run config cannot block a gate that is otherwise ready to run -- but
    it is never silent, because a typo'd threshold key would otherwise look
    like it had been applied.
    """
    known = {field.name for field in fields(GateThresholds)}
    values: Dict[str, Any] = {}

    source = path or _THRESHOLD_FILE
    if source.exists():
        payload = json.loads(source.read_text())
        values.update({k: v for k, v in payload.items() if k in known})

    for key, value in (overrides or {}).items():
        if key in known:
            values[key] = value
        else:
            logger.warning("Ignoring unknown gate threshold %r", key)

    return GateThresholds(**values)


# ---------------------------------------------------------------------------
# Small statistics helpers (kept local; scipy is not a study dependency)
# ---------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        ranked[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return ranked


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(np.asarray(a, dtype=float)), _ranks(np.asarray(b, dtype=float))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of a positive > score of a negative)."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranked = _ranks(combined)
    rank_sum = ranked[: len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def _ridge_loo_r2(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """
    Exact leave-one-out cross-validated R^2 for ridge regression.

    Uses the hat-matrix shortcut, so LOO costs one fit rather than n.  Both
    inputs are centred and the intercept dropped, which is what keeps the
    intercept out of the penalty.
    """
    n = X.shape[0]
    if n <= X.shape[1] + 1:
        return float("nan")

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    scale = Xc.std(axis=0)
    scale[scale == 0] = 1.0
    Xc = Xc / scale

    gram = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    try:
        beta_hat = np.linalg.solve(gram, Xc.T @ yc)
        hat_diag = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(gram), Xc)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the n check
        return float("nan")

    residual = yc - Xc @ beta_hat
    denominator = np.clip(1.0 - hat_diag, 1e-8, None)
    loo_residual = residual / denominator

    ss_total = float((yc**2).sum())
    if ss_total <= 0:
        return float("nan")
    return float(1.0 - (loo_residual**2).sum() / ss_total)


def _alr(probabilities: Sequence[float], eps: float = 1e-3) -> np.ndarray:
    """K probabilities -> K-1 additive log-ratios against the last category."""
    p = np.clip(np.asarray(probabilities, dtype=float), eps, None)
    p = p / p.sum()
    return np.log(p[:-1] / p[-1])


# ---------------------------------------------------------------------------
# R4: is PC1 the quality axis?
# ---------------------------------------------------------------------------


def _lda_axis(
    X: np.ndarray, positive: np.ndarray, shrinkage: float
) -> Optional[np.ndarray]:
    """Binary Fisher discriminant with shrinkage; None when unidentifiable."""
    if positive.sum() < 2 or (~positive).sum() < 2:
        return None
    mean_pos = X[positive].mean(axis=0)
    mean_neg = X[~positive].mean(axis=0)
    centred = np.vstack([X[positive] - mean_pos, X[~positive] - mean_neg])
    within = centred.T @ centred / max(len(centred) - 2, 1)
    within = within + shrinkage * np.trace(within) / X.shape[1] * np.eye(X.shape[1])
    try:
        return np.linalg.solve(within, mean_pos - mean_neg)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None


def _loo_lda_scores(
    X: np.ndarray, positive: np.ndarray, shrinkage: float
) -> Optional[np.ndarray]:
    """
    Score each item from an LDA axis fitted without that item.

    An in-sample LDA axis separates almost any labelling, so validating the
    fallback against its own training fit would make R4 vacuous exactly where
    it is being relied on.
    """
    scores = np.empty(len(X))
    for index in range(len(X)):
        keep = np.ones(len(X), dtype=bool)
        keep[index] = False
        axis = _lda_axis(X[keep], positive[keep], shrinkage)
        if axis is None:
            return None
        scores[index] = float(X[index] @ axis)
    return scores


def _check_quality_axis(
    pool: Mapping[str, Any],
    embeddings: Mapping[str, np.ndarray],
    thresholds: GateThresholds,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    item_ids = [item["id"] for item in pool["items"] if item["id"] in embeddings]
    by_id = {item["id"]: item for item in pool["items"]}
    if len(item_ids) < thresholds.min_items_for_check:
        return (
            {
                "status": "insufficient_data",
                "passed": False,
                "n_items": len(item_ids),
                "note": "fewer embedded items than min_items_for_check",
            },
            None,
        )

    X = np.vstack([np.asarray(embeddings[i], dtype=float) for i in item_ids])
    labels = [by_id[i]["quality_label"] for i in item_ids]
    merit = np.array(
        [
            float(by_id[i].get("attributes", {}).get("merit_total", np.nan))
            for i in item_ids
        ]
    )
    merit_source = "merit_total"
    if np.isnan(merit).any():
        merit = np.array([_LABEL_ORDINAL[label] for label in labels])
        merit_source = "label_ordinal"

    strong = np.array([label == "strong" for label in labels])
    weak = np.array([label == "weak" for label in labels])

    def evaluate(scores: np.ndarray) -> Dict[str, float]:
        # Orient so that strong scores high; sign of a PCA component is
        # arbitrary, so an unoriented axis would produce an AUC near 0 for a
        # perfectly good quality axis.
        oriented = scores
        if strong.any() and weak.any() and scores[strong].mean() < scores[weak].mean():
            oriented = -scores
        return {
            "auc_strong_vs_weak": _auc(oriented[strong], oriented[weak]),
            "spearman_vs_merit": _spearman(oriented, merit),
            "_scores": oriented,
        }

    pc1 = evaluate(X[:, 0])
    pc1_ok = (
        pc1["auc_strong_vs_weak"] >= thresholds.pc1_auc_strong_vs_weak
        and abs(pc1["spearman_vs_merit"]) >= thresholds.pc1_spearman_vs_merit
    )

    report: Dict[str, Any] = {
        "n_items": len(item_ids),
        "merit_source": merit_source,
        "label_counts": {
            label: int(sum(1 for l in labels if l == label))
            for label in schemas.QUALITY_LABELS
        },
        "pc1": {
            "auc_strong_vs_weak": round(pc1["auc_strong_vs_weak"], 4),
            "spearman_vs_merit": round(pc1["spearman_vs_merit"], 4),
            "meets_threshold": bool(pc1_ok),
        },
    }

    if pc1_ok:
        report.update({"axis": "pc1", "status": "ok", "passed": True})
        return report, dict(zip(item_ids, pc1["_scores"].tolist()))

    # Fallback (§6.3): the label-supervised direction, scored out of sample.
    subset = strong | weak
    lda_dim = min(thresholds.lda_dim, X.shape[1])
    scores = _loo_lda_scores(X[subset][:, :lda_dim], strong[subset], thresholds.lda_shrinkage)
    if scores is None:
        report.update(
            {
                "axis": "pc1",
                "status": "failed",
                "passed": False,
                "note": "PC1 failed to separate the labels and the LDA fallback is "
                "not identifiable; the quality proxy is unusable (§6.3 R4)",
            }
        )
        return report, None

    lda_eval = {
        "auc_strong_vs_weak": _auc(
            scores[strong[subset]], scores[~strong[subset]]
        ),
        "spearman_vs_merit": _spearman(scores, merit[subset]),
    }
    lda_ok = lda_eval["auc_strong_vs_weak"] >= thresholds.pc1_auc_strong_vs_weak

    # Score every item from the full-sample axis once the out-of-sample check
    # has decided whether the axis is trustworthy at all.
    axis = _lda_axis(X[subset][:, :lda_dim], strong[subset], thresholds.lda_shrinkage)
    full_scores = X[:, :lda_dim] @ axis if axis is not None else None

    report.update(
        {
            "axis": "lda",
            "status": "ok" if lda_ok else "failed",
            "passed": bool(lda_ok),
            "lda": {
                "dim": int(lda_dim),
                "loo_auc_strong_vs_weak": round(lda_eval["auc_strong_vs_weak"], 4),
                "loo_spearman_vs_merit": round(lda_eval["spearman_vs_merit"], 4),
                "meets_threshold": bool(lda_ok),
            },
            "note": "PC1 did not separate the authored labels; the plan's "
            "label-supervised fallback is in use as the quality proxy (§6.3 R4)",
        }
    )
    if not lda_ok or full_scores is None:
        return report, None
    return report, dict(zip(item_ids, np.asarray(full_scores).tolist()))


# ---------------------------------------------------------------------------
# §6.3: eta-gap equalization
# ---------------------------------------------------------------------------


def _menu_gaps(
    problem_set: Mapping[str, Any], quality: Mapping[str, float]
) -> List[Dict[str, Any]]:
    """Standardized best-vs-next-best quality gap for every menu."""
    scores = np.array(list(quality.values()), dtype=float)
    spread = scores.std(ddof=1) if len(scores) > 1 else 1.0
    spread = spread if spread > 0 else 1.0

    gaps: List[Dict[str, Any]] = []
    for problem in problem_set["problems"]:
        values = sorted(
            (quality[i] for i in problem["item_ids"] if i in quality), reverse=True
        )
        if len(values) < 2:
            continue
        gaps.append(
            {
                "menu_size": int(problem["menu_size"]),
                "family": problem["family"],
                "stratum": problem["difficulty_stratum"],
                "gap": float((values[0] - values[1]) / spread),
            }
        )
    return gaps


def _summarize(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4) if array.size > 1 else 0.0,
    }


def _sibling_gap_summaries(
    pool_id: str, config: Any
) -> Dict[str, Dict[str, Any]]:
    """Overall eta-gap summaries from other pools already gated in this run."""
    results_dir = getattr(config, "results_dir", None)
    if not results_dir:
        return {}
    siblings: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(results_dir).glob("pools/*/gate_report.json")):
        if path.parent.name == pool_id:
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            continue
        overall = report.get("checks", {}).get("eta_gap", {}).get("overall")
        if overall and overall.get("n"):
            siblings[path.parent.name] = overall
    return siblings


def _check_eta_gap(
    problem_set: Mapping[str, Any],
    quality: Mapping[str, float],
    thresholds: GateThresholds,
    config: Any,
) -> Dict[str, Any]:
    gaps = _menu_gaps(problem_set, quality)
    if len(gaps) < thresholds.min_items_for_check:
        return {
            "status": "insufficient_data",
            "passed": False,
            "n_menus": len(gaps),
        }

    values = np.array([g["gap"] for g in gaps])
    sizes = sorted({g["menu_size"] for g in gaps})
    by_size = {
        size: np.array([g["gap"] for g in gaps if g["menu_size"] == size])
        for size in sizes
    }

    # -- Across menu sizes (§3.4, §9 item 8) --
    size_diffs: Dict[str, float] = {}
    size_ds: Dict[str, float] = {}
    for i, left in enumerate(sizes):
        for right in sizes[i + 1 :]:
            key = f"{left}_vs_{right}"
            size_diffs[key] = round(
                abs(float(by_size[left].mean()) - float(by_size[right].mean())), 4
            )
            d = _cohens_d(by_size[left], by_size[right])
            if not np.isnan(d):
                size_ds[key] = round(abs(d), 4)
    worst_size = max(size_diffs.values(), default=0.0)
    size_ok = worst_size <= thresholds.eta_gap_max_cross_size_diff

    # -- Across pools (§6.3) --
    siblings = _sibling_gap_summaries(problem_set["pool_id"], config)
    pool_diffs: Dict[str, float] = {}
    for name, summary in siblings.items():
        pool_diffs[name] = round(
            abs(float(values.mean()) - float(summary["mean"])), 4
        )
    worst_pool = max(pool_diffs.values(), default=0.0)
    cross_pool_ok = worst_pool <= thresholds.eta_gap_max_cross_pool_diff

    report: Dict[str, Any] = {
        "overall": _summarize(values),
        "by_size": {str(size): _summarize(by_size[size]) for size in sizes},
        "by_family_size": {
            f"{family}/{size}": _summarize(
                [g["gap"] for g in gaps if g["family"] == family and g["menu_size"] == size]
            )
            for family in sorted({g["family"] for g in gaps})
            for size in sizes
        },
        "by_stratum": {
            stratum: _summarize([g["gap"] for g in gaps if g["stratum"] == stratum])
            for stratum in sorted({g["stratum"] for g in gaps})
        },
        "cross_size": {
            "pairwise_abs_diff": size_diffs,
            "worst": round(worst_size, 4),
            "threshold": thresholds.eta_gap_max_cross_size_diff,
            "passed": bool(size_ok),
            "pairwise_abs_cohens_d": size_ds,
            "units": "pool-standardized quality-score units; Cohen's d is "
            "reported alongside as a diagnostic and is not the pass rule",
        },
    }

    if not siblings:
        # Pools are gated one at a time, so the first pool through has nothing
        # to compare against.  Deferring is honest; failing would make the
        # first pool unpassable, and silently passing would drop the check.
        report["cross_pool"] = {
            "status": "deferred",
            "pools_compared": [],
            "note": "no sibling pool has been gated yet; re-run the validate phase "
            "across all pools once each has a gate report to resolve the §6.3 "
            "cross-pool comparison",
        }
        report["passed"] = bool(size_ok)
        report["status"] = "ok" if size_ok else "failed"
        return report

    report["cross_pool"] = {
        "status": "evaluated",
        "pools_compared": sorted(siblings),
        "pairwise_abs_diff": pool_diffs,
        "worst": round(worst_pool, 4),
        "threshold": thresholds.eta_gap_max_cross_pool_diff,
        "passed": bool(cross_pool_ok),
    }
    report["passed"] = bool(size_ok and cross_pool_ok)
    report["status"] = "ok" if report["passed"] else "failed"
    return report


# ---------------------------------------------------------------------------
# R3: predictive validity
# ---------------------------------------------------------------------------


def _belief_matrix(
    assessment_set: Mapping[str, Any], embeddings: Mapping[str, np.ndarray]
) -> Tuple[List[str], Optional[np.ndarray]]:
    ids: List[str] = []
    rows: List[np.ndarray] = []
    for record in assessment_set["assessments"]:
        if not record.get("parse_ok") or record.get("probabilities") is None:
            continue
        if record["item_id"] not in embeddings:
            continue
        ids.append(record["item_id"])
        rows.append(_alr(record["probabilities"]))
    if not rows:
        return [], None
    return ids, np.vstack(rows)


def _r2_for(
    ids: Sequence[str],
    Y: np.ndarray,
    embeddings: Mapping[str, np.ndarray],
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    if len(ids) < thresholds.min_items_for_check:
        return {"status": "insufficient_data", "n": len(ids), "r2": None}
    dim = min(thresholds.r3_dim, len(next(iter(embeddings.values()))))
    X = np.vstack([np.asarray(embeddings[i], dtype=float)[:dim] for i in ids])
    per_response = [
        _ridge_loo_r2(X, Y[:, column], thresholds.r3_ridge_alpha)
        for column in range(Y.shape[1])
    ]
    finite = [value for value in per_response if not np.isnan(value)]
    return {
        "status": "ok" if finite else "insufficient_data",
        "n": len(ids),
        "dim": int(dim),
        "r2_by_response": [
            None if np.isnan(v) else round(float(v), 4) for v in per_response
        ],
        "r2": round(float(np.mean(finite)), 4) if finite else None,
    }


def _check_predictive_validity(
    problem_set: Mapping[str, Any],
    embeddings: Mapping[str, np.ndarray],
    assessments: Mapping[str, Mapping[str, Any]],
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    if not assessments:
        return {
            "status": "awaiting_assessments",
            "passed": False,
            "note": "R3 regresses parsed belief on the item embedding, so the "
            "assess phase must run before the gate (§5)",
        }

    by_model: Dict[str, Any] = {}
    stacked: Dict[str, List[np.ndarray]] = {}
    for slug, assessment_set in sorted(assessments.items()):
        ids, Y = _belief_matrix(assessment_set, embeddings)
        if Y is None:
            by_model[slug] = {"status": "no_parsed_assessments", "n": 0, "r2": None}
            continue
        by_model[slug] = _r2_for(ids, Y, embeddings, thresholds)
        for item_id, row in zip(ids, Y):
            stacked.setdefault(item_id, []).append(row)

    # The per-pool quantity is the regression on the across-model *mean* belief.
    # Stacking model x item rows instead would put the same item in both the
    # training and held-out folds through its other models' rows, inflating the
    # very number the gate blocks on.
    pooled_ids = sorted(stacked)
    pooled: Dict[str, Any]
    if pooled_ids:
        Y = np.vstack([np.mean(stacked[i], axis=0) for i in pooled_ids])
        pooled = _r2_for(pooled_ids, Y, embeddings, thresholds)
    else:
        pooled = {"status": "no_parsed_assessments", "n": 0, "r2": None}

    # By menu size (§6.3): size-correlated mis-measurement of eta is the
    # artefact the RQ6 slope must not be confounded with.
    items_by_size: Dict[int, set] = {}
    for problem in problem_set["problems"]:
        items_by_size.setdefault(int(problem["menu_size"]), set()).update(
            problem["item_ids"]
        )
    by_size: Dict[str, Any] = {}
    for size in sorted(items_by_size):
        subset = [i for i in pooled_ids if i in items_by_size[size]]
        if subset:
            Y = np.vstack([np.mean(stacked[i], axis=0) for i in subset])
            by_size[str(size)] = _r2_for(subset, Y, embeddings, thresholds)

    model_flags = {
        slug: bool(result.get("r2") is not None and result["r2"] >= thresholds.r0_model_pool)
        for slug, result in by_model.items()
    }
    passed = pooled.get("r2") is not None and pooled["r2"] >= thresholds.r0_pool

    return {
        "status": "ok" if passed else "failed",
        "passed": bool(passed),
        "pooled": pooled,
        "threshold_pool": thresholds.r0_pool,
        "by_model": by_model,
        "threshold_model_pool": thresholds.r0_model_pool,
        "model_meets_threshold": model_flags,
        "by_menu_size": by_size,
        "note": "pooled R^2 blocks; the per-model x pool matrix is the §9.2 "
        "differential-mis-measurement diagnostic and reports only (§5)",
    }


# ---------------------------------------------------------------------------
# Assessment parse health (§6.4 analogue available pre-choice)
# ---------------------------------------------------------------------------


def _check_assessment_parse(
    assessments: Mapping[str, Mapping[str, Any]], thresholds: GateThresholds
) -> Dict[str, Any]:
    if not assessments:
        return {"status": "awaiting_assessments", "passed": False}

    rates: Dict[str, float] = {}
    for slug, assessment_set in sorted(assessments.items()):
        records = assessment_set["assessments"]
        failures = sum(1 for r in records if not r.get("parse_ok"))
        rates[slug] = round(failures / len(records), 4) if records else 1.0

    worst = max(rates.values(), default=1.0)
    passed = worst <= thresholds.max_assessment_parse_failure
    return {
        "status": "ok" if passed else "failed",
        "passed": bool(passed),
        "failure_rate_by_model": rates,
        "worst": worst,
        "threshold": thresholds.max_assessment_parse_failure,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gate(
    pool: Mapping[str, Any],
    problem_set: Mapping[str, Any],
    reduced_embeddings: Mapping[str, np.ndarray],
    assessments: Mapping[str, Mapping[str, Any]],
    config: Any,
) -> Dict[str, Any]:
    """
    Run every pre-choice check for one pool and return its gate report.

    ``study_runner`` writes the return value to ``gate_report.json`` and blocks
    choice collection unless ``report["passed"]`` is True.  Checks that cannot
    be evaluated yet report their own status and count as *not passed*, so a
    missing input can never be mistaken for a cleared gate.
    """
    thresholds = load_gate_thresholds(getattr(config, "gate_thresholds", None))
    pool_id = pool["pool_id"]

    report: Dict[str, Any] = {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": pool_id,
        "n_items": len(pool["items"]),
        "n_menus": len(problem_set["problems"]),
        "design_seed": problem_set.get("design_seed"),
        "thresholds": thresholds.to_dict(),
        "provisional_thresholds": thresholds.provisional,
        "checks": {},
    }

    if not reduced_embeddings:
        report.update(
            {
                "status": "awaiting_embeddings",
                "passed": False,
                "note": "run the embed phase for this pool first",
            }
        )
        return report

    axis_report, quality = _check_quality_axis(pool, reduced_embeddings, thresholds)
    report["checks"]["quality_axis"] = axis_report
    report["quality_axis"] = axis_report.get("axis")

    if quality is None:
        report["checks"]["eta_gap"] = {
            "status": "skipped",
            "passed": False,
            "note": "no usable quality axis, so the §6.3 equalization has nothing "
            "to equalize on",
        }
    else:
        report["checks"]["eta_gap"] = _check_eta_gap(
            problem_set, quality, thresholds, config
        )

    report["checks"]["predictive_validity"] = _check_predictive_validity(
        problem_set, reduced_embeddings, assessments, thresholds
    )
    report["checks"]["assessment_parse"] = _check_assessment_parse(
        assessments, thresholds
    )

    failures = [
        name for name, check in report["checks"].items() if not check.get("passed")
    ]
    report["failed_checks"] = failures
    report["passed"] = not failures

    if report["passed"]:
        report["status"] = "passed"
    elif not assessments:
        report["status"] = "awaiting_assessments"
    else:
        report["status"] = "failed"

    logger.info(
        "Gate %s: status=%s axis=%s failed=%s",
        pool_id,
        report["status"],
        report.get("quality_axis"),
        failures,
    )
    return report

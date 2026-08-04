"""
Tests for the pre-choice validation gate.

The gate is a spend gate: ``study_runner`` refuses to start ~13,680 choice API
calls unless it reports ``passed``.  So these tests care as much about the
*failure* paths -- missing inputs, an unusable quality axis, a size-correlated
eta gap, weak predictive validity -- as about the happy one.  A gate that
cannot fail is not a gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from applications.seu_sensitivity_study import item_validation, problem_generation, schemas
from applications.seu_sensitivity_study.item_validation import GateThresholds, run_gate

LABEL_SCORE = {"weak": 0.0, "ambiguous": 1.0, "strong": 2.0}


def make_pool(n_per_label=(6, 6, 14), pool_id="test"):
    items = []
    counter = 0
    for label, count in zip(schemas.QUALITY_LABELS, n_per_label):
        for _ in range(count):
            counter += 1
            items.append(
                {
                    "id": f"T{counter:03d}",
                    "family": "main",
                    "text": f"item {counter} ({label})",
                    "quality_label": label,
                    "matched_key": None,
                    "attributes": {"merit_total": LABEL_SCORE[label] * 5 + counter % 3},
                }
            )
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": pool_id,
        "framing": "positive",
        "consequences": ["bad", "mid", "good"],
        "families": {"main": {"description": "test"}},
        "items": items,
    }


def make_embeddings(pool, *, quality_signal=1.0, dim=12, seed=0):
    """Embeddings whose first axis carries quality with tunable strength."""
    rng = np.random.default_rng(seed)
    out = {}
    for item in pool["items"]:
        base = np.zeros(dim)
        base[0] = quality_signal * item["attributes"]["merit_total"]
        out[item["id"]] = base + rng.normal(scale=0.3, size=dim)
    return out


def make_assessments(pool, *, slugs=("a", "b"), signal=1.0, parse_ok_share=1.0, seed=1):
    rng = np.random.default_rng(seed)
    sets = {}
    for slug in slugs:
        records = []
        for index, item in enumerate(pool["items"]):
            quality = item["attributes"]["merit_total"] / 10.0
            logits = np.array([1.0 - signal * quality, 0.5, signal * quality])
            probabilities = np.exp(logits + rng.normal(scale=0.05, size=3))
            probabilities /= probabilities.sum()
            ok = (index / len(pool["items"])) < parse_ok_share
            records.append(
                {
                    "item_id": item["id"],
                    "text": "assessment",
                    "probabilities": [float(p) for p in probabilities] if ok else None,
                    "parse_ok": bool(ok),
                }
            )
        sets[slug] = {
            "schema_version": schemas.SCHEMA_VERSION,
            "pool_id": pool["pool_id"],
            "model_name": slug,
            "instruction": schemas.ASSESSMENT_INSTRUCTION,
            "assessments": records,
        }
    return sets


class Config:
    """Minimal stand-in carrying only what run_gate reads."""

    def __init__(self, results_dir=None, gate_thresholds=None):
        self.results_dir = str(results_dir) if results_dir else None
        self.gate_thresholds = gate_thresholds or {}


@pytest.fixture
def pool():
    return make_pool()


@pytest.fixture
def problem_set(pool):
    return problem_generation.generate_problem_set(
        pool, problems_per_family=32, seed=3
    )


class TestThresholds:
    def test_packaged_defaults_are_flagged_provisional(self):
        thresholds = item_validation.load_gate_thresholds()
        assert thresholds.provisional is True
        assert thresholds.frozen_at is None

    def test_overrides_apply(self):
        thresholds = item_validation.load_gate_thresholds({"r0_pool": 0.9})
        assert thresholds.r0_pool == 0.9

    def test_unknown_override_is_warned_not_silently_dropped(self, caplog):
        with caplog.at_level("WARNING"):
            item_validation.load_gate_thresholds({"r0_poool": 0.9})
        assert "r0_poool" in caplog.text

    def test_report_echoes_the_thresholds_it_ran_under(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(),
        )
        assert report["provisional_thresholds"] is True
        assert report["thresholds"]["r0_pool"] == GateThresholds().r0_pool


class TestGatePasses:
    def test_clean_inputs_pass(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(),
        )
        assert report["passed"] is True
        assert report["status"] == "passed"
        assert report["quality_axis"] == "pc1"

    def test_report_is_json_serializable(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(),
        )
        json.loads(json.dumps(report))


class TestMissingInputs:
    def test_no_embeddings_blocks(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings={},
            assessments={},
            config=Config(),
        )
        assert report["passed"] is False
        assert report["status"] == "awaiting_embeddings"

    def test_no_assessments_blocks(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments={},
            config=Config(),
        )
        assert report["passed"] is False
        assert report["status"] == "awaiting_assessments"
        assert "predictive_validity" in report["failed_checks"]


class TestQualityAxis:
    def test_lda_fallback_when_pc1_is_not_quality(self, pool, problem_set):
        """§6.3: PC1 failing selects the label-supervised axis, not a run failure."""
        rng = np.random.default_rng(5)
        embeddings = {}
        for item in pool["items"]:
            vector = rng.normal(scale=3.0, size=12)  # dominant nuisance variance
            vector[5] = item["attributes"]["merit_total"] * 2.0  # quality on a minor axis
            embeddings[item["id"]] = vector

        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=embeddings,
            assessments=make_assessments(pool),
            config=Config(),
        )
        axis = report["checks"]["quality_axis"]
        assert axis["pc1"]["meets_threshold"] is False
        assert axis["axis"] == "lda"

    def test_pure_noise_fails_both_axes(self, pool, problem_set):
        rng = np.random.default_rng(9)
        embeddings = {
            item["id"]: rng.normal(size=12) for item in pool["items"]
        }
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=embeddings,
            assessments=make_assessments(pool),
            config=Config(),
        )
        assert report["passed"] is False
        assert "quality_axis" in report["failed_checks"]
        # With no usable proxy there is nothing to equalize on.
        assert report["checks"]["eta_gap"]["status"] == "skipped"

    def test_axis_sign_is_oriented_towards_strong(self, pool, problem_set):
        """A PCA component's sign is arbitrary; an unoriented axis would score ~0 AUC."""
        flipped = {
            item_id: -vector for item_id, vector in make_embeddings(pool).items()
        }
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=flipped,
            assessments=make_assessments(pool),
            config=Config(),
        )
        assert report["checks"]["quality_axis"]["pc1"]["auc_strong_vs_weak"] > 0.5


class TestEtaGap:
    def test_gaps_are_reported_by_size_and_stratum(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(),
        )
        check = report["checks"]["eta_gap"]
        assert set(check["by_size"]) == {str(s) for s in schemas.MENU_SIZES}
        assert set(check["by_stratum"]) == set(schemas.QUALITY_LABELS)

    def test_size_correlated_gap_is_caught(self, pool, problem_set):
        """The §9-item-8 artefact: geometry drifting with menu size."""
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(gate_thresholds={"eta_gap_max_cross_size_diff": 0.0}),
        )
        assert report["checks"]["eta_gap"]["cross_size"]["passed"] is False
        assert report["passed"] is False

    def test_cross_size_blocks_on_absolute_difference_not_cohens_d(
        self, pool, problem_set
    ):
        """
        Gaps are already pool-standardized, so re-standardizing by the
        within-size spread would punish tightly controlled designs -- the
        opposite of what §6.3 asks for.
        """
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(),
        )
        cross = report["checks"]["eta_gap"]["cross_size"]
        assert set(cross["pairwise_abs_diff"]) == set(cross["pairwise_abs_cohens_d"])
        assert cross["worst"] == max(cross["pairwise_abs_diff"].values())

    def test_cross_pool_defers_until_a_sibling_exists(self, pool, problem_set, tmp_path):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(results_dir=tmp_path),
        )
        assert report["checks"]["eta_gap"]["cross_pool"]["status"] == "deferred"
        assert report["passed"] is True  # deferral must not make pool one unpassable

    def test_cross_pool_evaluates_against_a_sibling_report(
        self, pool, problem_set, tmp_path
    ):
        sibling = tmp_path / "pools" / "other" / "gate_report.json"
        sibling.parent.mkdir(parents=True)
        sibling.write_text(
            json.dumps(
                {
                    "pool_id": "other",
                    "checks": {
                        "eta_gap": {"overall": {"n": 32, "mean": 9.0, "sd": 0.5}}
                    },
                }
            )
        )
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(results_dir=tmp_path),
        )
        cross = report["checks"]["eta_gap"]["cross_pool"]
        assert cross["status"] == "evaluated"
        assert cross["pools_compared"] == ["other"]
        assert cross["passed"] is False  # a wildly different sibling must fail
        assert report["passed"] is False


class TestPredictiveValidity:
    def test_reported_per_pool_per_model_and_per_size(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool, slugs=("a", "b", "c")),
            config=Config(),
        )
        check = report["checks"]["predictive_validity"]
        assert check["pooled"]["r2"] is not None
        assert set(check["by_model"]) == {"a", "b", "c"}
        assert set(check["by_menu_size"]) == {str(s) for s in schemas.MENU_SIZES}

    def test_pool_level_r2_blocks(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(gate_thresholds={"r0_pool": 0.999}),
        )
        assert report["checks"]["predictive_validity"]["passed"] is False
        assert report["passed"] is False

    def test_model_level_r2_reports_but_does_not_block(self, pool, problem_set):
        """§5 states the pass rule per pool; the model x pool matrix is a diagnostic."""
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool),
            config=Config(gate_thresholds={"r0_model_pool": 0.999}),
        )
        check = report["checks"]["predictive_validity"]
        assert not any(check["model_meets_threshold"].values())
        assert check["passed"] is True
        assert report["passed"] is True

    def test_noise_embeddings_fail_the_pool_check(self, pool, problem_set):
        rng = np.random.default_rng(11)
        embeddings = {item["id"]: rng.normal(size=12) for item in pool["items"]}
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=embeddings,
            assessments=make_assessments(pool),
            config=Config(),
        )
        assert report["checks"]["predictive_validity"]["passed"] is False

    def test_loo_r2_of_pure_noise_is_not_inflated(self):
        """The LOO shortcut must not hand back an in-sample R^2."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 8))
        y = rng.normal(size=60)
        assert item_validation._ridge_loo_r2(X, y, 1.0) < 0.2


class TestAssessmentParse:
    def test_parse_failures_block(self, pool, problem_set):
        report = run_gate(
            pool=pool,
            problem_set=problem_set,
            reduced_embeddings=make_embeddings(pool),
            assessments=make_assessments(pool, parse_ok_share=0.7),
            config=Config(),
        )
        check = report["checks"]["assessment_parse"]
        assert check["passed"] is False
        assert check["worst"] > GateThresholds().max_assessment_parse_failure
        assert report["passed"] is False

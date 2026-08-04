"""
Tests for embedding reduction and Stan-data assembly (§5, §6.1 steps 4-5).

The first test class is the important one: ``y`` must be the chosen item's rank
among the menu's *sorted pool indices*, not its position in the presentation
order.  Getting this wrong scrambles every observation whose menu was not
already sorted, which after reversal counterbalancing is most of them, and
nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.seu_sensitivity_study import data_preparation as dp
from applications.seu_sensitivity_study import schemas


ITEM_IDS = ["T001", "T002", "T003", "T004"]


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return {item_id: rng.normal(size=3) for item_id in ITEM_IDS}


def _problem_set(problems):
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "testpool",
        "design_seed": 1,
        "menu_sizes": [2, 4],
        "num_presentations": 2,
        "problems": problems,
    }


def _problem(pid, items, stratum="ambiguous"):
    return {
        "id": pid,
        "family": "startup",
        "item_ids": list(items),
        "menu_size": len(items),
        "difficulty_stratum": stratum,
        "presentations": [
            {"presentation_id": 1, "order": list(items)},
            {"presentation_id": 2, "order": list(reversed(items))},
        ],
    }


def _choice_set(cell_id, records):
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "cell_id": cell_id,
        "pool_id": "testpool",
        "model_name": "gpt-4o",
        "prompt_condition": "neutral",
        "answer_format_version": schemas.ANSWER_TOKEN_VERSION,
        "choices": records,
    }


def _record(pid, presentation_id, order, position, menu_size, stratum="ambiguous"):
    resolved = position is not None
    return {
        "problem_id": pid,
        "presentation_id": presentation_id,
        "menu_size": menu_size,
        "difficulty_stratum": stratum,
        "family": "startup",
        "chosen_position": position,
        "chosen_item_id": order[position - 1] if resolved else None,
        "resolution_path": "answer_token" if resolved else "unresolved",
        "raw_response": f"ANSWER: {position}" if resolved else "I can't choose.",
    }


class TestActiveSetIndexing:
    def test_y_is_rank_within_sorted_active_set(self, embeddings):
        """
        Menu [T003, T001] shown in that order; the model picks position 1
        (= T003).  Sorted pool indices are [T001, T003], so T003 has rank 2.
        A naive pass-through would emit y = 1.
        """
        order = ["T003", "T001"]
        problems = [_problem("P1", order), _problem("P2", ITEM_IDS)]
        records = [
            _record("P1", 1, order, 1, 2),
            _record("P1", 2, list(reversed(order)), 2, 2),
            _record("P2", 1, ITEM_IDS, 1, 4),
            _record("P2", 2, list(reversed(ITEM_IDS)), 1, 4),
        ]
        stan_data, _ = dp.build_stan_data(
            pool={"pool_id": "testpool"},
            problem_set=_problem_set(problems),
            choice_sets={"c1": _choice_set("c1", records)},
            reduced_embeddings=embeddings,
            design_matrix=np.zeros((1, 1)),
            cell_ids=["c1"],
            K=3,
        )
        # Both P1 observations chose T003 (position 1 then position 2 after
        # reversal), so both must map to the same y.
        assert stan_data["y"][0] == 2
        assert stan_data["y"][1] == 2

    def test_reversed_presentations_of_the_same_item_agree(self, embeddings):
        """Position differs across presentations; the resolved y must not."""
        order = ["T004", "T002", "T001"]
        problems = [_problem("P1", order), _problem("P2", ITEM_IDS)]
        records = [
            _record("P1", 1, order, 2, 3),  # T002
            _record("P1", 2, list(reversed(order)), 2, 3),  # also T002
            _record("P2", 1, ITEM_IDS, 1, 4),
            _record("P2", 2, list(reversed(ITEM_IDS)), 1, 4),
        ]
        stan_data, _ = dp.build_stan_data(
            pool={"pool_id": "testpool"},
            problem_set=_problem_set(problems),
            choice_sets={"c1": _choice_set("c1", records)},
            reduced_embeddings=embeddings,
            design_matrix=np.zeros((1, 1)),
            cell_ids=["c1"],
            K=3,
        )
        assert stan_data["y"][0] == stan_data["y"][1]

    def test_indicator_row_marks_exactly_the_menu(self, embeddings):
        order = ["T003", "T001"]
        problems = [_problem("P1", order), _problem("P2", ITEM_IDS)]
        records = [
            _record("P1", 1, order, 1, 2),
            _record("P1", 2, list(reversed(order)), 1, 2),
            _record("P2", 1, ITEM_IDS, 1, 4),
            _record("P2", 2, list(reversed(ITEM_IDS)), 1, 4),
        ]
        stan_data, _ = dp.build_stan_data(
            pool={"pool_id": "testpool"},
            problem_set=_problem_set(problems),
            choice_sets={"c1": _choice_set("c1", records)},
            reduced_embeddings=embeddings,
            design_matrix=np.zeros((1, 1)),
            cell_ids=["c1"],
            K=3,
        )
        assert stan_data["I"][0] == [1, 0, 1, 0]  # T001, T003
        assert stan_data["y"][0] <= sum(stan_data["I"][0])


class TestStanDataAssembly:
    def _build(self, embeddings, **kwargs):
        problems = [_problem("P1", ITEM_IDS[:2]), _problem("P2", ITEM_IDS)]
        records = [
            _record("P1", 1, ITEM_IDS[:2], 1, 2),
            _record("P1", 2, list(reversed(ITEM_IDS[:2])), 1, 2),
            _record("P2", 1, ITEM_IDS, 3, 4),
            _record("P2", 2, list(reversed(ITEM_IDS)), 2, 4),
        ]
        return dp.build_stan_data(
            pool={"pool_id": "testpool"},
            problem_set=_problem_set(problems),
            choice_sets={"c1": _choice_set("c1", records)},
            reduced_embeddings=embeddings,
            design_matrix=np.zeros((1, 2)),
            cell_ids=["c1"],
            K=3,
            **kwargs,
        )

    def test_payload_validates(self, embeddings):
        stan_data, _ = self._build(embeddings)
        assert schemas.validate_stan_data(stan_data) == []

    def test_dimensions(self, embeddings):
        stan_data, _ = self._build(embeddings)
        assert stan_data["R"] == 4
        assert stan_data["D"] == 3
        assert stan_data["J"] == 1
        assert stan_data["P"] == 2
        assert stan_data["M_total"] == 4
        assert stan_data["M_per_cell"] == [4]

    def test_menu_size_covariate_is_centered(self, embeddings):
        stan_data, report = self._build(embeddings, include_menu_size=True)
        assert schemas.validate_stan_data(stan_data, model="h_m01_size") == []
        assert sum(stan_data["s"]) == pytest.approx(0.0)
        assert report["mean_menu_size"] == pytest.approx(3.0)

    def test_size_covariate_absent_by_default(self, embeddings):
        stan_data, _ = self._build(embeddings)
        assert "s" not in stan_data

    def test_missing_cell_raises(self, embeddings):
        problems = [_problem("P1", ITEM_IDS[:2])]
        with pytest.raises(KeyError, match="No choice set supplied"):
            dp.build_stan_data(
                pool={"pool_id": "testpool"},
                problem_set=_problem_set(problems),
                choice_sets={},
                reduced_embeddings=embeddings,
                design_matrix=np.zeros((1, 1)),
                cell_ids=["c1"],
                K=3,
            )

    def test_all_na_cell_raises(self, embeddings):
        problems = [_problem("P1", ITEM_IDS[:2])]
        records = [
            _record("P1", 1, ITEM_IDS[:2], None, 2),
            _record("P1", 2, list(reversed(ITEM_IDS[:2])), None, 2),
        ]
        with pytest.raises(ValueError, match="no resolved observations"):
            dp.build_stan_data(
                pool={"pool_id": "testpool"},
                problem_set=_problem_set(problems),
                choice_sets={"c1": _choice_set("c1", records)},
                reduced_embeddings=embeddings,
                design_matrix=np.zeros((1, 1)),
                cell_ids=["c1"],
                K=3,
            )


class TestNAFiltering:
    def test_log_is_stratified(self):
        records = [
            _record("P1", 1, ITEM_IDS[:2], 1, 2, "strong"),
            _record("P1", 2, ITEM_IDS[:2], None, 2, "strong"),
            _record("P2", 1, ITEM_IDS, None, 4, "ambiguous"),
        ]
        resolved, log = dp.filter_resolved_choices(_choice_set("c1", records))
        assert len(resolved) == 1
        assert log["na_count"] == 2
        assert log["na_rate"] == pytest.approx(2 / 3)
        assert log["na_by_stratum"] == {"ambiguous": 1, "strong": 1}
        assert log["na_by_menu_size"] == {"2": 1, "4": 1}
        assert log["resolution_paths"]["unresolved"] == 2

    def test_raw_responses_are_retained_for_audit(self):
        records = [_record("P1", 1, ITEM_IDS[:2], None, 2)]
        _, log = dp.filter_resolved_choices(_choice_set("c1", records))
        assert log["removed_observations"][0]["raw_response"]


class TestReduceEmbeddings:
    def test_dimension_clamped_to_item_count(self):
        rng = np.random.default_rng(1)
        raw = {f"T{i:03d}": rng.normal(size=64) for i in range(10)}
        reduced, info = dp.reduce_embeddings(raw, target_dim=32, seed=0)
        assert info["effective_dim"] <= 10
        assert all(vec.shape == (info["effective_dim"],) for vec in reduced.values())

    def test_reports_explained_variance(self):
        rng = np.random.default_rng(2)
        raw = {f"T{i:03d}": rng.normal(size=16) for i in range(40)}
        _, info = dp.reduce_embeddings(raw, target_dim=8, seed=0)
        assert 0.0 < info["explained_variance_ratio"] <= 1.0
        assert info["effective_dim"] == 8

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            dp.reduce_embeddings({})

    def test_is_deterministic(self):
        rng = np.random.default_rng(3)
        raw = {f"T{i:03d}": rng.normal(size=16) for i in range(20)}
        first, _ = dp.reduce_embeddings(raw, target_dim=4, seed=7)
        second, _ = dp.reduce_embeddings(raw, target_dim=4, seed=7)
        for key in first:
            assert np.allclose(first[key], second[key])

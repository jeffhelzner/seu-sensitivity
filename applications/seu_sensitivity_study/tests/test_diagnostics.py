"""
Tests for collection diagnostics (§3.1, §6.4, §8.8).

The stability-subset tests encode the v0.5 correction: an absolute zero-flip
subset is size-confounded, so the R6 robustness subset must keep a balanced
size margin or the RQ6 slope has nothing left to be estimated from.
"""

from __future__ import annotations

import pytest

from applications.seu_sensitivity_study import diagnostics
from applications.seu_sensitivity_study import schemas


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


def _obs(pid, presentation_id, chosen, menu_size=2, stratum="ambiguous", path=None):
    resolved = chosen is not None
    return {
        "problem_id": pid,
        "presentation_id": presentation_id,
        "menu_size": menu_size,
        "difficulty_stratum": stratum,
        "family": "startup",
        "chosen_position": 1 if resolved else None,
        "chosen_item_id": chosen,
        "resolution_path": path or ("answer_token" if resolved else "unresolved"),
        "raw_response": "",
    }


def _menu(pid, first, second, menu_size=2, stratum="ambiguous"):
    return [
        _obs(pid, 1, first, menu_size, stratum),
        _obs(pid, 2, second, menu_size, stratum),
    ]


class TestNATable:
    def test_rows_are_per_cell_stratum_and_size(self):
        records = _menu("P1", "A", "A", 2, "strong") + _menu("P2", "B", None, 4, "weak")
        rows = diagnostics.na_table({"c1": _choice_set("c1", records)})
        keys = {(row["difficulty_stratum"], row["menu_size"]) for row in rows}
        assert keys == {("strong", 2), ("weak", 4)}

    def test_na_rate_and_paths_are_reported(self):
        records = _menu("P1", "A", None, 2, "strong")
        rows = diagnostics.na_table({"c1": _choice_set("c1", records)})
        row = rows[0]
        assert row["total"] == 2
        assert row["na_count"] == 1
        assert row["na_rate"] == pytest.approx(0.5)
        assert row["unresolved"] == 1
        assert row["answer_token"] == 1

    def test_fallback_share_is_tracked(self):
        """A rising fallback share with menu size is the parser-degradation signature."""
        records = [
            _obs("P1", 1, "A", 8, path="fallback_parse"),
            _obs("P1", 2, "A", 8, path="answer_token"),
        ]
        rows = diagnostics.na_table({"c1": _choice_set("c1", records)})
        assert rows[0]["fallback_share"] == pytest.approx(0.5)


class TestMenuStability:
    def test_same_item_twice_is_stable(self):
        cs = _choice_set("c1", _menu("P1", "A", "A"))
        assert diagnostics.menu_stability(cs)["P1"]["stable"] is True

    def test_different_items_is_a_flip(self):
        cs = _choice_set("c1", _menu("P1", "A", "B"))
        assert diagnostics.menu_stability(cs)["P1"]["stable"] is False

    def test_single_resolved_observation_is_untested_not_stable(self):
        """Counting it as stable would inflate the subset with untested menus."""
        cs = _choice_set("c1", _menu("P1", "A", None))
        assert diagnostics.menu_stability(cs)["P1"]["stable"] is None

    def test_flip_summary_breaks_down_by_size(self):
        records = _menu("P1", "A", "B", 2) + _menu("P2", "C", "C", 8)
        summary = diagnostics.position_flip_summary(_choice_set("c1", records))
        assert summary["flip_rate"] == pytest.approx(0.5)
        sizes = {row["menu_size"]: row["flip_rate"] for row in summary["by_menu_size"]}
        assert sizes == {2: 1.0, 8: 0.0}

    def test_untested_menus_excluded_from_flip_rate(self):
        records = _menu("P1", "A", "A") + _menu("P2", "B", None)
        summary = diagnostics.position_flip_summary(_choice_set("c1", records))
        assert summary["menus_comparable"] == 1
        assert summary["flip_rate"] == 0.0


class TestStabilitySubset:
    def _pool_of_menus(self):
        """
        Six stable size-2 menus, two stable size-8 menus, one flipped size-8.

        Mirrors the real asymmetry: P(flip) rises with menu size, so an
        unbalanced subset would be dominated by the small menus.
        """
        records = []
        for index in range(6):
            records += _menu(f"S{index}", "A", "A", 2)
        for index in range(2):
            records += _menu(f"L{index}", "B", "B", 8)
        records += _menu("L9", "C", "D", 8)
        return {"c1": _choice_set("c1", records)}

    def test_flipped_menus_are_excluded(self):
        subset, report = diagnostics.size_balanced_stability_subset(self._pool_of_menus())
        assert "L9" not in subset
        assert report["menus_flipped_somewhere"] == 1

    def test_unbalanced_subset_is_dominated_by_small_menus(self):
        _, report = diagnostics.size_balanced_stability_subset(
            self._pool_of_menus(), balance=False
        )
        assert report["retention_before_balance"] == {2: 6, 8: 2}

    def test_balancing_restores_the_size_margin(self):
        """Without this, gamma_size has almost no predictor variance left."""
        subset, report = diagnostics.size_balanced_stability_subset(
            self._pool_of_menus(), balance=True
        )
        assert report["retention_after_balance"] == {2: 2, 8: 2}
        assert len(subset) == 4

    def test_menu_flipped_in_any_cell_is_excluded_everywhere(self):
        """The subset must be a property of the design, not of one arm."""
        stable = _choice_set("c1", _menu("P1", "A", "A", 2) + _menu("P2", "B", "B", 8))
        flipped = _choice_set("c2", _menu("P1", "A", "B", 2) + _menu("P2", "B", "B", 8))
        subset, report = diagnostics.size_balanced_stability_subset(
            {"c1": stable, "c2": flipped}
        )
        assert "P1" not in subset
        assert report["menus_flipped_somewhere"] == 1

    def test_selection_is_deterministic(self):
        menus = self._pool_of_menus()
        first, _ = diagnostics.size_balanced_stability_subset(menus, seed=5)
        second, _ = diagnostics.size_balanced_stability_subset(menus, seed=5)
        assert first == second

    def test_empty_subset_warns_rather_than_crashing(self, caplog):
        records = _menu("P1", "A", "B", 2)
        with caplog.at_level("WARNING"):
            subset, report = diagnostics.size_balanced_stability_subset(
                {"c1": _choice_set("c1", records)}
            )
        assert subset == []
        assert report["menus_stable_everywhere"] == 0
        assert "cannot be computed" in caplog.text

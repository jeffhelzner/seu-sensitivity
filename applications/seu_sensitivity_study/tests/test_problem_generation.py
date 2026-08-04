"""
Tests for menu design generation (study plan §6.1 step 1, §6.3, §3.1).

The two properties worth locking are the ones the study depends on and that a
naive implementation would get wrong: menu sizes must be balanced (the RQ6
slope lives on that spread), and the top-two quality gap must not shrink as
menus grow (otherwise order statistics manufacture a size-alpha slope).
"""

from __future__ import annotations

from collections import Counter

import pytest

from applications.seu_sensitivity_study import problem_generation as pg
from applications.seu_sensitivity_study import schemas


class TestAllocation:
    def test_allocation_is_balanced(self):
        plan = pg.allocate_menu_plan(120, [2, 4, 6, 8], ["strong", "ambiguous", "weak"])
        counts = Counter(plan)
        assert len(plan) == 120
        assert set(counts.values()) == {10}

    def test_remainder_spread_within_one(self):
        plan = pg.allocate_menu_plan(11, [2, 4], ["strong", "ambiguous"])
        counts = Counter(plan)
        assert len(plan) == 11
        assert max(counts.values()) - min(counts.values()) <= 1

    @pytest.mark.parametrize("total", [7, 8, 11, 17, 23, 47, 100])
    def test_size_margin_is_exactly_floor_or_ceil(self, total):
        """
        Regression: a flat round-robin over the size x stratum grid let the
        remainder cluster in the first sizes, skewing the size margin by up to
        one menu per stratum while each individual cell still looked balanced.
        The size margin is what identifies the RQ6 slope, so it is the margin
        that must be exact.
        """
        sizes = [2, 4, 6, 8]
        plan = pg.allocate_menu_plan(total, sizes, ["strong", "ambiguous", "weak"])
        assert len(plan) == total
        by_size = Counter(size for size, _ in plan)
        low, remainder = divmod(total, len(sizes))
        allowed = {low, low + 1} if remainder else {low}
        assert set(by_size.values()) <= allowed

    def test_zero_problems_yields_empty_plan(self):
        assert pg.allocate_menu_plan(0, [2, 4], ["strong"]) == []

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            pg.allocate_menu_plan(10, [], ["strong"])


class TestRecipes:
    def test_contender_count_is_size_invariant(self):
        """The mechanism that keeps the top-two gap stable across menu sizes."""
        recipe = pg.StratumRecipe(
            stratum="ambiguous", contenders=("strong", "strong"), filler_label="weak"
        )
        for size in (2, 4, 6, 8):
            labels = recipe.compose(size)
            assert len(labels) == size
            assert labels[:2] == ["strong", "strong"]
            assert set(labels[2:]) <= {"weak"}

    def test_menu_too_small_for_recipe_raises(self):
        recipe = pg.StratumRecipe(
            stratum="ambiguous", contenders=("strong", "strong"), filler_label="weak"
        )
        with pytest.raises(ValueError, match="at least 2"):
            recipe.compose(1)


class TestGeneration:
    def test_generated_design_validates(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        assert schemas.validate_problem_set(design, pool=single_family_pool) == []

    def test_menu_sizes_are_balanced(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        counts = Counter(p["menu_size"] for p in design["problems"])
        assert sorted(counts) == [2, 4, 6, 8]
        assert set(counts.values()) == {12}

    def test_strata_are_balanced(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        counts = Counter(p["difficulty_stratum"] for p in design["problems"])
        assert set(counts.values()) == {16}

    def test_top_tier_composition_is_size_invariant(self, single_family_pool):
        """
        Guards the order-statistics failure mode (§6.3): the number of top-tier
        contenders in an ambiguous menu must not vary with menu size.
        """
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        labels = {item["id"]: item["quality_label"] for item in single_family_pool["items"]}
        by_size: dict[int, set[int]] = {}
        for problem in design["problems"]:
            if problem["difficulty_stratum"] != "ambiguous":
                continue
            n_strong = sum(1 for i in problem["item_ids"] if labels[i] == "strong")
            by_size.setdefault(problem["menu_size"], set()).add(n_strong)
        assert by_size, "expected some ambiguous menus"
        assert all(counts == {2} for counts in by_size.values())

    def test_no_repeated_item_within_a_menu(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        for problem in design["problems"]:
            assert len(set(problem["item_ids"])) == problem["menu_size"]

    def test_generation_is_deterministic(self, single_family_pool):
        first = pg.generate_problem_set(
            single_family_pool, problems_per_family=24, seed=7
        )
        second = pg.generate_problem_set(
            single_family_pool, problems_per_family=24, seed=7
        )
        assert first == second

    @pytest.mark.parametrize("count", [13, 25, 47])
    def test_awkward_counts_still_validate(self, single_family_pool, count):
        """The design must stay schema-valid when the count is not a multiple."""
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=count, seed=3
        )
        assert len(design["problems"]) == count
        assert schemas.validate_problem_set(design, pool=single_family_pool) == []

    def test_different_seeds_differ(self, single_family_pool):
        first = pg.generate_problem_set(
            single_family_pool, problems_per_family=24, seed=7
        )
        second = pg.generate_problem_set(
            single_family_pool, problems_per_family=24, seed=8
        )
        assert first["problems"] != second["problems"]

    def test_insufficient_items_raises_actionable_error(self, single_family_pool):
        pool = dict(single_family_pool)
        pool["items"] = [i for i in pool["items"] if i["quality_label"] != "weak"][:9]
        with pytest.raises(ValueError, match="Author more"):
            pg.generate_problem_set(pool, problems_per_family=8, seed=1)


class TestPresentations:
    def test_reverse_mode_flips_every_position(self, single_family_pool):
        """
        Every pre-registered menu size is even, so under reversal no item keeps
        its position -- the strongest 2-point probe of position bias available.
        """
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=48, seed=42
        )
        for problem in design["problems"]:
            first, second = (p["order"] for p in problem["presentations"])
            assert second == list(reversed(first))
            assert all(a != b for a, b in zip(first, second))

    def test_presentation_count_matches_frozen_value(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool, problems_per_family=24, seed=42
        )
        for problem in design["problems"]:
            assert len(problem["presentations"]) == schemas.NUM_PRESENTATIONS

    def test_random_mode_still_yields_distinct_orders(self, single_family_pool):
        design = pg.generate_problem_set(
            single_family_pool,
            problems_per_family=24,
            seed=42,
            presentation_mode="random",
        )
        for problem in design["problems"]:
            orders = [tuple(p["order"]) for p in problem["presentations"]]
            assert len(set(orders)) == len(orders)

    def test_unknown_presentation_mode_raises(self, single_family_pool):
        with pytest.raises(ValueError, match="presentation_mode"):
            pg.generate_problem_set(
                single_family_pool,
                problems_per_family=8,
                seed=1,
                presentation_mode="spiral",
            )


class TestFamilies:
    def test_families_are_separate_strata(self, two_family_pool):
        """Matched items must never mix into the ordinary comparator menus."""
        design = pg.generate_problem_set(
            two_family_pool,
            problems_per_family={"startup": 24, "procurement": 12},
            seed=42,
        )
        by_family = Counter(p["family"] for p in design["problems"])
        assert by_family == {"startup": 24, "procurement": 12}

        families = {i["id"]: i["family"] for i in two_family_pool["items"]}
        for problem in design["problems"]:
            assert {families[i] for i in problem["item_ids"]} == {problem["family"]}

    def test_family_can_be_excluded(self, two_family_pool):
        design = pg.generate_problem_set(
            two_family_pool,
            problems_per_family=24,
            seed=42,
            families=["startup"],
        )
        assert {p["family"] for p in design["problems"]} == {"startup"}

    def test_unknown_family_raises(self, two_family_pool):
        with pytest.raises(ValueError, match="no family"):
            pg.generate_problem_set(
                two_family_pool, problems_per_family=8, seed=1, families=["ghost"]
            )

"""
Tests for the frozen artefact schemas (study plan v0.5).

These lock the invariants that would otherwise fail silently and far
downstream: a counterbalancing bug, a prompt-scope leak, a Stan index that is
off by one, or a menu-size design that has quietly lost its balance.
"""

from __future__ import annotations

import copy

import pytest

from applications.seu_sensitivity_study import schemas
from applications.seu_sensitivity_study.schemas import SchemaError


# ---------------------------------------------------------------------------
# Item pool
# ---------------------------------------------------------------------------


class TestItemPool:
    def test_valid_pool_has_no_errors(self, single_family_pool):
        assert schemas.validate_item_pool(single_family_pool) == []

    def test_duplicate_item_id_rejected(self, single_family_pool):
        pool = copy.deepcopy(single_family_pool)
        pool["items"][1]["id"] = pool["items"][0]["id"]
        errors = schemas.validate_item_pool(pool)
        assert any("duplicate item id" in e for e in errors)

    def test_undeclared_family_rejected(self, single_family_pool):
        pool = copy.deepcopy(single_family_pool)
        pool["items"][0]["family"] = "ghost"
        errors = schemas.validate_item_pool(pool)
        assert any("not declared in pool.families" in e for e in errors)

    def test_bad_quality_label_rejected(self, single_family_pool):
        pool = copy.deepcopy(single_family_pool)
        pool["items"][0]["quality_label"] = "excellent"
        errors = schemas.validate_item_pool(pool)
        assert any("quality_label" in e for e in errors)

    def test_family_smaller_than_largest_menu_rejected(self, single_family_pool):
        """A family is drawn on as its own stratum, so >= 8 binds per family."""
        pool = copy.deepcopy(single_family_pool)
        pool["items"] = pool["items"][:5]
        errors = schemas.validate_item_pool(pool)
        assert any("menus of size" in e for e in errors)

    def test_matched_key_collision_within_family_rejected(self, two_family_pool):
        pool = copy.deepcopy(two_family_pool)
        procurement = [i for i in pool["items"] if i["family"] == "procurement"]
        procurement[1]["matched_key"] = procurement[0]["matched_key"]
        errors = schemas.validate_item_pool(pool)
        assert any("already used by" in e for e in errors)

    def test_schema_version_mismatch_rejected(self, single_family_pool):
        pool = copy.deepcopy(single_family_pool)
        pool["schema_version"] = "0.9"
        errors = schemas.validate_item_pool(pool)
        assert any("schema_version" in e for e in errors)


# ---------------------------------------------------------------------------
# Problem set
# ---------------------------------------------------------------------------


def _problem(pid: str, items: list[str], stratum: str = "ambiguous") -> dict:
    return {
        "id": pid,
        "family": "startup",
        "item_ids": items,
        "menu_size": len(items),
        "difficulty_stratum": stratum,
        "presentations": [
            {"presentation_id": 1, "order": list(items)},
            {"presentation_id": 2, "order": list(reversed(items))},
        ],
    }


def _problem_set(problems: list[dict]) -> dict:
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "testpool",
        "design_seed": 42,
        "menu_sizes": [2, 4],
        "num_presentations": 2,
        "problems": problems,
    }


class TestProblemSet:
    def test_balanced_design_passes(self):
        problem_set = _problem_set(
            [
                _problem("P0001", ["a", "b"]),
                _problem("P0002", ["c", "d"]),
                _problem("P0003", ["a", "b", "c", "d"]),
                _problem("P0004", ["e", "f", "g", "h"]),
            ]
        )
        assert schemas.validate_problem_set(problem_set) == []

    def test_unbalanced_menu_sizes_rejected(self):
        problem_set = _problem_set(
            [
                _problem("P0001", ["a", "b"]),
                _problem("P0002", ["c", "d"]),
                _problem("P0003", ["e", "f"]),
                _problem("P0004", ["a", "b", "c", "d"]),
            ]
        )
        errors = schemas.validate_problem_set(problem_set)
        assert any("not balanced" in e for e in errors)

    def test_missing_size_stratum_rejected(self):
        problem_set = _problem_set(
            [_problem("P0001", ["a", "b"]), _problem("P0002", ["c", "d"])]
        )
        errors = schemas.validate_problem_set(problem_set)
        assert any("no menus at size" in e for e in errors)

    def test_menu_size_must_match_item_count(self):
        problem_set = _problem_set(
            [_problem("P0001", ["a", "b"]), _problem("P0002", ["a", "b", "c", "d"])]
        )
        problem_set["problems"][0]["menu_size"] = 4
        errors = schemas.validate_problem_set(problem_set)
        assert any("menu_size 4 != len(item_ids) 2" in e for e in errors)

    def test_identical_presentations_rejected(self):
        """An identical repeat makes the position-flip statistic vacuous."""
        problem = _problem("P0001", ["a", "b"])
        problem["presentations"][1]["order"] = list(problem["item_ids"])
        problem_set = _problem_set([problem, _problem("P0002", ["a", "b", "c", "d"])])
        errors = schemas.validate_problem_set(problem_set)
        assert any("pairwise distinct" in e for e in errors)

    def test_presentation_must_be_a_permutation(self):
        problem = _problem("P0001", ["a", "b"])
        problem["presentations"][1]["order"] = ["a", "z"]
        problem_set = _problem_set([problem, _problem("P0002", ["a", "b", "c", "d"])])
        errors = schemas.validate_problem_set(problem_set)
        assert any("not a permutation" in e for e in errors)

    def test_wrong_presentation_count_rejected(self):
        problem_set = _problem_set(
            [_problem("P0001", ["a", "b"]), _problem("P0002", ["a", "b", "c", "d"])]
        )
        problem_set["num_presentations"] = 3
        errors = schemas.validate_problem_set(problem_set)
        assert any("frozen" in e for e in errors)

    def test_menu_mixing_families_rejected(self, two_family_pool):
        startup = [i["id"] for i in two_family_pool["items"] if i["family"] == "startup"]
        procurement = [
            i["id"] for i in two_family_pool["items"] if i["family"] == "procurement"
        ]
        mixed = _problem("P0001", [startup[0], procurement[0]])
        clean = _problem("P0002", startup[:4])
        problem_set = _problem_set([mixed, clean])
        problem_set["pool_id"] = "venture"
        errors = schemas.validate_problem_set(problem_set, pool=two_family_pool)
        assert any("mixes families" in e for e in errors)


# ---------------------------------------------------------------------------
# Assessment set
# ---------------------------------------------------------------------------


def _assessment_set(pool: dict, *, instruction: str = "neutral") -> dict:
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": pool["pool_id"],
        "model_name": "gpt-4o",
        "instruction": instruction,
        "assessments": [
            {
                "item_id": item["id"],
                "text": "An assessment.",
                "probabilities": [0.2, 0.5, 0.3],
                "parse_ok": True,
                "raw_response": "...",
            }
            for item in pool["items"]
        ],
    }


class TestAssessmentSet:
    def test_valid_set_passes(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        assert schemas.validate_assessment_set(payload, pool=single_family_pool) == []

    def test_non_neutral_instruction_rejected(self, single_family_pool):
        """B2: the prompt manipulation must not reach the assessment step."""
        payload = _assessment_set(single_family_pool, instruction="seu_maximizing")
        errors = schemas.validate_assessment_set(payload, pool=single_family_pool)
        assert any("neutral instruction only" in e for e in errors)

    def test_probabilities_must_sum_to_one(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        payload["assessments"][0]["probabilities"] = [0.2, 0.2, 0.2]
        errors = schemas.validate_assessment_set(payload, pool=single_family_pool)
        assert any("is not 1" in e for e in errors)

    def test_null_probabilities_require_parse_ok_false(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        payload["assessments"][0]["probabilities"] = None
        errors = schemas.validate_assessment_set(payload, pool=single_family_pool)
        assert any("parse_ok is true but probabilities are null" in e for e in errors)

    def test_unparseable_assessment_is_allowed(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        payload["assessments"][0]["probabilities"] = None
        payload["assessments"][0]["parse_ok"] = False
        assert schemas.validate_assessment_set(payload, pool=single_family_pool) == []

    def test_missing_item_detected(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        payload["assessments"].pop()
        errors = schemas.validate_assessment_set(payload, pool=single_family_pool)
        assert any("no assessment for item" in e for e in errors)

    def test_wrong_k_detected(self, single_family_pool):
        payload = _assessment_set(single_family_pool)
        payload["assessments"][0]["probabilities"] = [0.5, 0.5]
        errors = schemas.validate_assessment_set(payload, pool=single_family_pool)
        assert any("!= K=3" in e for e in errors)


# ---------------------------------------------------------------------------
# Choice set
# ---------------------------------------------------------------------------


def _choice_set(problem_set: dict) -> dict:
    choices = []
    for problem in problem_set["problems"]:
        for presentation in problem["presentations"]:
            order = presentation["order"]
            choices.append(
                {
                    "problem_id": problem["id"],
                    "presentation_id": presentation["presentation_id"],
                    "menu_size": problem["menu_size"],
                    "difficulty_stratum": problem["difficulty_stratum"],
                    "chosen_position": 1,
                    "chosen_item_id": order[0],
                    "resolution_path": "answer_token",
                    "raw_response": "ANSWER: 1",
                }
            )
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "cell_id": "gpt_4o_neutral_testpool",
        "pool_id": problem_set["pool_id"],
        "model_name": "gpt-4o",
        "prompt_condition": "neutral",
        "answer_format_version": schemas.ANSWER_TOKEN_VERSION,
        "choices": choices,
    }


@pytest.fixture
def small_design():
    return _problem_set(
        [_problem("P0001", ["a", "b"]), _problem("P0002", ["a", "b", "c", "d"])]
    )


class TestChoiceSet:
    def test_valid_choice_set_passes(self, small_design):
        payload = _choice_set(small_design)
        assert schemas.validate_choice_set(payload, problem_set=small_design) == []

    def test_chosen_item_must_match_presentation_order(self, small_design):
        """This is the check that catches a counterbalancing bug."""
        payload = _choice_set(small_design)
        # Presentation 1 of P0001 has order [a, b], so position 1 is "a".
        assert payload["choices"][0]["chosen_item_id"] == "a"
        payload["choices"][0]["chosen_item_id"] = "b"
        errors = schemas.validate_choice_set(payload, problem_set=small_design)
        assert any("counterbalancing bug" in e for e in errors)

    def test_reversed_presentation_resolves_to_the_other_item(self, small_design):
        """Position 1 of the reversed presentation is a *different* item."""
        payload = _choice_set(small_design)
        first, second = payload["choices"][0], payload["choices"][1]
        assert first["presentation_id"] == 1 and second["presentation_id"] == 2
        assert first["chosen_position"] == second["chosen_position"] == 1
        assert first["chosen_item_id"] != second["chosen_item_id"]
        assert schemas.validate_choice_set(payload, problem_set=small_design) == []

    def test_na_requires_unresolved_path(self, small_design):
        payload = _choice_set(small_design)
        payload["choices"][0]["chosen_position"] = None
        payload["choices"][0]["chosen_item_id"] = None
        errors = schemas.validate_choice_set(payload, problem_set=small_design)
        assert any("inconsistent with resolution_path" in e for e in errors)

    def test_consistent_na_passes(self, small_design):
        payload = _choice_set(small_design)
        payload["choices"][0].update(
            chosen_position=None, chosen_item_id=None, resolution_path="unresolved"
        )
        assert schemas.validate_choice_set(payload, problem_set=small_design) == []

    def test_position_out_of_range_rejected(self, small_design):
        payload = _choice_set(small_design)
        payload["choices"][0]["chosen_position"] = 9
        errors = schemas.validate_choice_set(payload, problem_set=small_design)
        assert any("outside [1, 2]" in e for e in errors)

    def test_missing_observation_detected(self, small_design):
        payload = _choice_set(small_design)
        payload["choices"].pop()
        errors = schemas.validate_choice_set(payload, problem_set=small_design)
        assert any("design observation(s) absent" in e for e in errors)

    def test_wrong_answer_format_version_rejected(self, small_design):
        payload = _choice_set(small_design)
        payload["answer_format_version"] = "legacy-freeform"
        errors = schemas.validate_choice_set(payload, problem_set=small_design)
        assert any("answer_format_version" in e for e in errors)


# ---------------------------------------------------------------------------
# Stan data
# ---------------------------------------------------------------------------


def _stan_data(**overrides) -> dict:
    data = {
        "J": 2,
        "K": 3,
        "D": 2,
        "R": 3,
        "P": 1,
        "w": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "M_total": 4,
        "cell": [1, 1, 2, 2],
        "I": [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        "y": [1, 2, 1, 3],
        "X": [[0.0], [1.0]],
        "M_per_cell": [2, 2],
    }
    data.update(overrides)
    return data


class TestStanData:
    def test_valid_payload_passes(self):
        assert schemas.validate_stan_data(_stan_data()) == []

    def test_y_must_index_within_active_set(self):
        """y is 1-indexed within the active set, not within the full pool."""
        data = _stan_data(y=[3, 2, 1, 3])
        errors = schemas.validate_stan_data(data)
        assert any("outside [1, 2]" in e for e in errors)

    def test_menu_needs_two_alternatives(self):
        data = _stan_data(I=[[1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        errors = schemas.validate_stan_data(data)
        assert any("need >= 2" in e for e in errors)

    def test_m_per_cell_must_match_realised_counts(self):
        data = _stan_data(M_per_cell=[3, 1])
        errors = schemas.validate_stan_data(data)
        assert any("disagrees with the realised" in e for e in errors)

    def test_design_matrix_shape_checked(self):
        data = _stan_data(X=[[0.0, 1.0], [1.0, 0.0]])
        errors = schemas.validate_stan_data(data)
        assert any("length P=1" in e for e in errors)

    def test_size_variant_requires_s(self):
        errors = schemas.validate_stan_data(_stan_data(), model="h_m01_size")
        assert any("missing required key" in e for e in errors)

    def test_size_variant_accepts_s(self):
        data = _stan_data(s=[-1.0, -1.0, 1.0, 1.0])
        assert schemas.validate_stan_data(data, model="h_m01_size") == []


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def _manifest(**overrides) -> dict:
    manifest = {
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "seu-20260802-01",
        "started_at": "2026-08-02T12:00:00Z",
        "answer_format_version": schemas.ANSWER_TOKEN_VERSION,
        "menu_sizes": [2, 4, 6, 8],
        "num_presentations": 2,
        "design_seed": 42,
        "embedding_model": "text-embedding-3-small",
        "pca_target_dim": 32,
        "prompt_hashes": {"venture:neutral": "abc123"},
        "models": [
            {
                "model_name": "o3-mini",
                "endpoint_id": "o3-mini-2025-01-31",
                "provider": "openai",
                "accessed_at": "2026-08-02",
                "request_params": {"reasoning_effort": "medium", "max_tokens": 64},
            }
        ],
    }
    manifest.update(overrides)
    return manifest


class TestRunManifest:
    def test_valid_manifest_passes(self):
        assert schemas.validate_run_manifest(_manifest()) == []

    def test_empty_request_params_rejected(self):
        """For the reasoning tier these settings are part of the treatment."""
        manifest = _manifest()
        manifest["models"][0]["request_params"] = {}
        errors = schemas.validate_run_manifest(manifest)
        assert any("must pin the full request parameters" in e for e in errors)

    def test_substitution_requires_reason(self):
        manifest = _manifest()
        manifest["models"][0]["substituted_for"] = "gpt-4o"
        errors = schemas.validate_run_manifest(manifest)
        assert any("substitution_reason" in e for e in errors)

    def test_presentation_count_pinned(self):
        errors = schemas.validate_run_manifest(_manifest(num_presentations=3))
        assert any("frozen" in e for e in errors)


class TestCheck:
    def test_check_raises_with_all_problems(self):
        with pytest.raises(SchemaError) as excinfo:
            schemas.check(["first problem", "second problem"], context="widget")
        message = str(excinfo.value)
        assert "widget" in message
        assert "first problem" in message
        assert "second problem" in message

    def test_check_is_quiet_when_clean(self):
        schemas.check([], context="widget")

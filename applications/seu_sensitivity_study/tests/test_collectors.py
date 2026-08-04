"""
Tests for the assessment and choice collectors (§6.1, §3.2 B2, §6.4).

All API traffic is mocked; these lock the pipeline's correctness properties,
above all that presentations are actually iterated and that the recorded
``chosen_item_id`` is resolved through the presentation order.
"""

from __future__ import annotations

import json

import pytest

from applications.seu_sensitivity_study import problem_generation as pg
from applications.seu_sensitivity_study import schemas
from applications.seu_sensitivity_study.assessment_collection import AssessmentCollector
from applications.seu_sensitivity_study.choice_collection import ChoiceCollector
from applications.seu_sensitivity_study.config import CellSpec


@pytest.fixture
def design(single_family_pool):
    return pg.generate_problem_set(
        single_family_pool, problems_per_family=24, seed=11
    )


@pytest.fixture
def cell():
    return CellSpec(
        cell_id="gpt_4o_neutral_testpool",
        model_name="gpt-4o",
        provider="openai",
        prompt_condition="neutral",
        pool_id="testpool",
        temperature=0.0,
    )


@pytest.fixture
def assessments(single_family_pool):
    return {
        item["id"]: f"Assessment of {item['id']}."
        for item in single_family_pool["items"]
    }


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------


def _assessment_responder(prompt, system_prompt):
    return "A short assessment.\nPROBABILITIES: 0.2, 0.5, 0.3"


class TestAssessmentCollector:
    def test_produces_a_valid_assessment_set(
        self, single_family_pool, prompt_set, mock_client_factory
    ):
        client = mock_client_factory(_assessment_responder)
        collector = AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=client,
            model_name="gpt-4o",
        )
        payload = collector.collect()
        assert schemas.validate_assessment_set(payload, pool=single_family_pool) == []
        assert len(payload["assessments"]) == len(single_family_pool["items"])

    def test_instruction_is_always_neutral(
        self, single_family_pool, prompt_set, mock_client_factory
    ):
        """B2: the collector has no way to express a treatment prompt."""
        collector = AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=mock_client_factory(_assessment_responder),
            model_name="gpt-4o",
        )
        assert collector.collect()["instruction"] == schemas.ASSESSMENT_INSTRUCTION

    def test_one_call_per_item(
        self, single_family_pool, prompt_set, mock_client_factory
    ):
        client = mock_client_factory(_assessment_responder)
        AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=client,
            model_name="gpt-4o",
        ).collect()
        assert len(client.calls) == len(single_family_pool["items"])

    def test_unparseable_line_is_recorded_not_raised(
        self, single_family_pool, prompt_set, mock_client_factory
    ):
        client = mock_client_factory(lambda p, s: "I cannot give numbers.")
        payload = AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=client,
            model_name="gpt-4o",
        ).collect()
        assert all(record["parse_ok"] is False for record in payload["assessments"])
        assert all(record["probabilities"] is None for record in payload["assessments"])
        assert schemas.validate_assessment_set(payload, pool=single_family_pool) == []

    def test_probability_line_can_be_stripped_from_choice_context(
        self, single_family_pool, prompt_set, mock_client_factory
    ):
        payload = AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=mock_client_factory(_assessment_responder),
            model_name="gpt-4o",
            keep_probability_line=False,
        ).collect()
        record = payload["assessments"][0]
        assert "PROBABILITIES" not in record["text"]
        assert record["probabilities"] is not None  # still parsed for the §5 check

    def test_resumes_from_checkpoint(
        self, single_family_pool, prompt_set, mock_client_factory, tmp_path
    ):
        checkpoint = tmp_path / "assessments.partial.json"
        first = mock_client_factory(_assessment_responder)
        AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=first,
            model_name="gpt-4o",
        ).collect(checkpoint_path=checkpoint, checkpoint_every=1)

        second = mock_client_factory(_assessment_responder)
        payload = AssessmentCollector(
            pool=single_family_pool,
            prompt_sets={"startup": prompt_set},
            llm_client=second,
            model_name="gpt-4o",
        ).collect(checkpoint_path=checkpoint)

        assert second.calls == []  # nothing re-spent
        assert len(payload["assessments"]) == len(single_family_pool["items"])


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------


class TestChoiceCollector:
    def _collector(self, cell, design, prompt_set, assessments, client):
        return ChoiceCollector(
            cell=cell,
            problem_set=design,
            prompt_sets={"startup": prompt_set},
            assessments=assessments,
            llm_client=client,
        )

    def test_produces_a_valid_choice_set(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        client = mock_client_factory(default="ANSWER: 1")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        assert schemas.validate_choice_set(payload, problem_set=design) == []

    def test_one_observation_per_menu_and_presentation(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        """The old implementation dropped presentations entirely."""
        client = mock_client_factory(default="ANSWER: 1")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        expected = len(design["problems"]) * schemas.NUM_PRESENTATIONS
        assert len(payload["choices"]) == expected
        assert len(client.calls) == expected

    def test_chosen_item_resolves_through_the_presentation_order(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        client = mock_client_factory(default="ANSWER: 1")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        orders = {
            (p["id"], pres["presentation_id"]): pres["order"]
            for p in design["problems"]
            for pres in p["presentations"]
        }
        for record in payload["choices"]:
            key = (record["problem_id"], record["presentation_id"])
            assert record["chosen_item_id"] == orders[key][0]

    def test_reversed_presentation_yields_a_different_item(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        """Position 1 means different items across the two presentations."""
        client = mock_client_factory(default="ANSWER: 1")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        by_problem: dict[str, set] = {}
        for record in payload["choices"]:
            by_problem.setdefault(record["problem_id"], set()).add(
                record["chosen_item_id"]
            )
        assert all(len(items) == 2 for items in by_problem.values())

    def test_assessments_are_inserted_in_presentation_order(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        captured = {}

        def responder(prompt, system_prompt):
            captured.setdefault("prompt", prompt)
            return "ANSWER: 1"

        client = mock_client_factory(responder)
        self._collector(cell, design, prompt_set, assessments, client).collect()

        first_problem = design["problems"][0]
        order = first_problem["presentations"][0]["order"]
        prompt = captured["prompt"]
        positions = [prompt.index(assessments[item_id]) for item_id in order]
        assert positions == sorted(positions)

    def test_prompt_condition_selects_the_instruction(
        self, design, prompt_set, assessments, mock_client_factory
    ):
        seen = {}
        for condition in ("neutral", "seu_maximizing", "deliberative"):
            cell = CellSpec(
                cell_id=f"gpt_4o_{condition}_testpool",
                model_name="gpt-4o",
                provider="openai",
                prompt_condition=condition,
                pool_id="testpool",
            )
            client = mock_client_factory(default="ANSWER: 1")
            self._collector(cell, design, prompt_set, assessments, client).collect()
            seen[condition] = client.calls[0]["prompt"]
        assert len(set(seen.values())) == 3
        assert prompt_set.choice_instructions["deliberative"].strip() in seen["deliberative"]

    def test_refusal_is_recorded_as_na(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        client = mock_client_factory(default="I can't choose between these.")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        assert all(record["chosen_position"] is None for record in payload["choices"])
        assert all(
            record["resolution_path"] == "unresolved" for record in payload["choices"]
        )
        assert schemas.validate_choice_set(payload, problem_set=design) == []

    def test_resolution_path_is_recorded(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        client = mock_client_factory(default="1")
        payload = self._collector(cell, design, prompt_set, assessments, client).collect()
        assert {r["resolution_path"] for r in payload["choices"]} == {"fallback_parse"}

    def test_missing_assessment_fails_before_spending(
        self, cell, design, prompt_set, assessments, mock_client_factory
    ):
        partial = dict(assessments)
        partial.pop(next(iter(partial)))
        client = mock_client_factory(default="ANSWER: 1")
        with pytest.raises(ValueError, match="no assessment for"):
            self._collector(cell, design, prompt_set, partial, client).collect()
        assert client.calls == []

    def test_pool_mismatch_is_rejected(
        self, design, prompt_set, assessments, mock_client_factory
    ):
        other = CellSpec(
            cell_id="x",
            model_name="gpt-4o",
            provider="openai",
            prompt_condition="neutral",
            pool_id="venture",
        )
        with pytest.raises(ValueError, match="belongs to"):
            self._collector(other, design, prompt_set, assessments, mock_client_factory())

    def test_resumes_from_checkpoint(
        self, cell, design, prompt_set, assessments, mock_client_factory, tmp_path
    ):
        checkpoint = tmp_path / "choices.partial.json"
        first = mock_client_factory(default="ANSWER: 1")
        self._collector(cell, design, prompt_set, assessments, first).collect(
            checkpoint_path=checkpoint, checkpoint_every=1
        )
        assert json.loads(checkpoint.read_text())["choices"]

        second = mock_client_factory(default="ANSWER: 1")
        payload = self._collector(cell, design, prompt_set, assessments, second).collect(
            checkpoint_path=checkpoint
        )
        assert second.calls == []
        assert schemas.validate_choice_set(payload, problem_set=design) == []

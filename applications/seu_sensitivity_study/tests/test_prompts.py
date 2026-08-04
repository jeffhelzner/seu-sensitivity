"""
Tests for prompt loading, per-family overrides, and the prompt-scope guards
(§3.2, §3.3, §6.5).
"""

from __future__ import annotations

import pytest
import yaml

from applications.seu_sensitivity_study import pools, prompts


class TestRealPromptFiles:
    @pytest.mark.parametrize("pool_id", ["insurance", "venture", "hiring"])
    def test_every_pool_resolves_all_its_families(self, pool_id):
        resolved = prompts.load_prompt_sets(pool_id)
        assert set(resolved) == set(pools.get_pool_spec(pool_id).families)

    @pytest.mark.parametrize("pool_id", ["insurance", "venture", "hiring"])
    def test_all_three_conditions_present(self, pool_id):
        for prompt_set in prompts.load_prompt_sets(pool_id).values():
            assert set(prompt_set.choice_instructions) == set(prompts.PROMPT_CONDITIONS)

    @pytest.mark.parametrize("pool_id", ["insurance", "venture", "hiring"])
    def test_conditions_share_everything_except_the_instruction(self, pool_id):
        """The RQ2 contrast must be the instruction and nothing else (§3.2)."""
        for prompt_set in prompts.load_prompt_sets(pool_id).values():
            rendered = {
                condition: prompt_set.render_choice(condition, ["a", "b"])
                for condition in prompts.PROMPT_CONDITIONS
            }
            assert len(set(rendered.values())) == 3
            for condition, text in rendered.items():
                instruction = prompt_set.choice_instructions[condition].strip()
                assert instruction in text
                # Removing the instruction leaves an identical shell.
                assert text.replace(instruction, "<I>") == rendered["neutral"].replace(
                    prompt_set.choice_instructions["neutral"].strip(), "<I>"
                )

    def test_procurement_family_carries_its_own_task_label(self):
        """
        The matched pair must vary the task label without varying item content,
        so the procurement family cannot inherit 'which venture to fund' (§3.3).
        """
        resolved = prompts.load_prompt_sets("venture")
        startup = resolved["startup"]
        procurement = resolved["procurement"]
        assert "vendor" in procurement.choice_instructions["neutral"].lower()
        assert "venture" in startup.choice_instructions["neutral"].lower()
        assert procurement.choice_system != startup.choice_system

    def test_hiring_matched_family_inherits_the_hiring_label(self):
        """The matched hiring half must be presented AS hiring."""
        resolved = prompts.load_prompt_sets("hiring")
        assert (
            resolved["matched"].choice_instructions
            == resolved["candidates"].choice_instructions
        )

    @pytest.mark.parametrize("pool_id", ["insurance", "venture", "hiring"])
    def test_choice_prompt_requests_the_answer_token(self, pool_id):
        for prompt_set in prompts.load_prompt_sets(pool_id).values():
            assert "ANSWER:" in prompt_set.choice_user

    @pytest.mark.parametrize("pool_id", ["insurance", "venture", "hiring"])
    def test_assessment_prompt_requests_the_structured_line(self, pool_id):
        for prompt_set in prompts.load_prompt_sets(pool_id).values():
            assert "{probability_format}" in prompt_set.assessment_user


class TestRendering:
    def test_assessment_interpolates_consequences_and_format(self, prompt_set):
        text = prompt_set.render_assessment("An item.", ["a", "b", "c"])
        assert "An item." in text
        assert "1. a" in text and "3. c" in text
        assert "PROBABILITIES: <p1>, <p2>, <p3>" in text

    def test_choice_numbers_assessments_in_the_given_order(self, prompt_set):
        text = prompt_set.render_choice("neutral", ["first", "second", "third"])
        assert "1. first" in text
        assert "2. second" in text
        assert "3. third" in text
        assert "{" not in text.replace("{}", "")

    def test_choice_reports_the_correct_n_max(self, prompt_set):
        text = prompt_set.render_choice("neutral", ["a"] * 8)
        assert "between 1 and 8" in text or "of 8 items" in text

    def test_unknown_condition_raises(self, prompt_set):
        with pytest.raises(KeyError, match="not defined"):
            prompt_set.render_choice("chain_of_thought", ["a", "b"])

    def test_fingerprint_is_stable_and_sensitive(self, prompt_set):
        import dataclasses

        first = prompt_set.fingerprint()
        assert first == prompt_set.fingerprint()
        changed = dataclasses.replace(prompt_set, choice_system="Different role.")
        assert changed.fingerprint()["choice_system"] != first["choice_system"]


class TestScopeGuards:
    """A future prompt edit must fail loudly, not silently change what RQ2 measures."""

    def _write(self, tmp_path, deliberative_text):
        payload = {
            "schema_version": "1.0",
            "pool_id": "venture",
            "assessment": {
                "system_prompt": "sys",
                "user_prompt": "{item_text}{consequence_lines}{probability_format}",
            },
            "choice": {
                "system_prompt": "sys",
                "user_prompt": "{instruction}{assessments_list}{n_max}",
                "instructions": {
                    "neutral": "Choose one.",
                    "seu_maximizing": "Maximize subjective expected return.",
                    "deliberative": deliberative_text,
                },
            },
        }
        path = tmp_path / "prompts.yaml"
        path.write_text(yaml.safe_dump(payload))
        return path

    def test_clean_deliberative_arm_loads(self, tmp_path):
        path = self._write(tmp_path, "Think carefully and reason step by step.")
        resolved = prompts.load_prompt_sets("venture", path=path)
        assert "startup" in resolved

    @pytest.mark.parametrize(
        "text",
        [
            "Think about the consequences and their likelihoods.",
            "Consider the expected value of each option.",
            "Weigh the probability of each outcome.",
        ],
    )
    def test_seu_vocabulary_in_deliberative_arm_is_rejected(self, tmp_path, text):
        path = self._write(tmp_path, text)
        with pytest.raises(ValueError, match="SEU vocabulary"):
            prompts.load_prompt_sets("venture", path=path)

    def test_missing_condition_is_rejected(self, tmp_path):
        payload = yaml.safe_load(self._write(tmp_path, "Think carefully.").read_text())
        del payload["choice"]["instructions"]["seu_maximizing"]
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="missing condition"):
            prompts.load_prompt_sets("venture", path=path)

    def test_wrong_pool_id_is_rejected(self, tmp_path):
        payload = yaml.safe_load(self._write(tmp_path, "Think carefully.").read_text())
        payload["pool_id"] = "hiring"
        path = tmp_path / "mismatch.yaml"
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="declares pool_id"):
            prompts.load_prompt_sets("venture", path=path)

    def test_override_for_undeclared_family_is_rejected(self, tmp_path):
        payload = yaml.safe_load(self._write(tmp_path, "Think carefully.").read_text())
        payload["families"] = {"ghost": {"choice": {"system_prompt": "x"}}}
        path = tmp_path / "ghost.yaml"
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="not declared for pool"):
            prompts.load_prompt_sets("venture", path=path)


class TestManifestHashes:
    def test_hashes_are_namespaced_by_pool_and_family(self, prompt_set):
        flat = prompts.prompt_hashes({"venture": {"startup": prompt_set}})
        assert "venture/startup/choice_user" in flat
        assert all(len(digest) == 16 for digest in flat.values())

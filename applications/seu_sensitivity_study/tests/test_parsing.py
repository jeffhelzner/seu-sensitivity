"""
Tests for response parsing (§6.4, §5).

The menu-size cases are the point: the hardened answer token must resolve
identically at size 2 and size 8, and stray in-range integers in a vignette
must not be able to masquerade as a choice.
"""

from __future__ import annotations

import pytest

from applications.seu_sensitivity_study import parsing


class TestAnswerToken:
    def test_plain_token(self):
        assert parsing.parse_choice_response("ANSWER: 3", 4) == (3, "answer_token")

    def test_token_after_reasoning(self):
        response = "Item 2 looks strongest on balance.\n\nANSWER: 2"
        assert parsing.parse_choice_response(response, 4) == (2, "answer_token")

    def test_case_and_separator_tolerant(self):
        for text in ("answer: 1", "Answer - 1", "ANSWER=1", "ANSWER :  1"):
            assert parsing.parse_choice_response(text, 4) == (1, "answer_token")

    def test_repeated_agreeing_tokens_accepted(self):
        assert parsing.parse_choice_response("ANSWER: 2\nANSWER: 2", 4) == (
            2,
            "answer_token",
        )

    def test_disagreeing_tokens_are_unresolved(self):
        """
        A weaker heuristic must not be allowed to break the tie: that would
        manufacture data where the honest reading is 'no usable answer'.
        """
        assert parsing.parse_choice_response("ANSWER: 1\nANSWER: 3", 4) == (
            None,
            "unresolved",
        )

    def test_out_of_range_token_is_unresolved(self):
        assert parsing.parse_choice_response("ANSWER: 7", 4) == (None, "unresolved")

    @pytest.mark.parametrize("n_max", [2, 4, 6, 8])
    def test_resolution_is_menu_size_independent(self, n_max):
        assert parsing.parse_choice_response(f"ANSWER: {n_max}", n_max) == (
            n_max,
            "answer_token",
        )


class TestFallbackAndRefusal:
    def test_bare_number_uses_the_fallback_path(self):
        position, path = parsing.parse_choice_response("2", 4)
        assert (position, path) == (2, "fallback_parse")

    def test_refusal_is_unresolved(self):
        response = "I'm not able to rank these candidates."
        assert parsing.parse_choice_response(response, 4) == (None, "unresolved")

    def test_empty_response_is_unresolved(self):
        assert parsing.parse_choice_response("", 4) == (None, "unresolved")
        assert parsing.parse_choice_response(None, 4) == (None, "unresolved")

    def test_answer_token_survives_distractor_integers_at_size_8(self):
        """
        The failure mode the hardening exists for: at size 8 a stray "5" or "8"
        from vignette text is in range and could be mistaken for a choice.
        """
        response = (
            "Candidate 5 has 8 years of experience and scored 7 on the work "
            "sample, but candidate 3 is stronger overall.\n\nANSWER: 3"
        )
        assert parsing.parse_choice_response(response, 8) == (3, "answer_token")

    def test_distractors_without_a_token_do_not_silently_resolve(self):
        response = "This profile lists 8 years of experience and a score of 7."
        position, path = parsing.parse_choice_response(response, 8)
        assert path in {"fallback_parse", "unresolved"}
        if path == "fallback_parse":
            # If the legacy heuristic does fire, the path records that it was
            # the weaker parser -- which is exactly what §6.4 needs to audit.
            assert position is not None


class TestProbabilities:
    def test_simple_line(self):
        result = parsing.parse_probabilities("PROBABILITIES: 0.2, 0.5, 0.3", 3)
        assert result == pytest.approx([0.2, 0.5, 0.3])

    def test_line_after_prose(self):
        response = "This venture is promising.\nPROBABILITIES: 0.1, 0.3, 0.6"
        assert parsing.parse_probabilities(response, 3) == pytest.approx([0.1, 0.3, 0.6])

    def test_percentages_are_converted(self):
        result = parsing.parse_probabilities("PROBABILITIES: 20%, 50%, 30%", 3)
        assert result == pytest.approx([0.2, 0.5, 0.3])

    def test_near_miss_is_renormalized(self):
        """Values are a check outcome, never a Stan input, so a slip is salvaged."""
        result = parsing.parse_probabilities("PROBABILITIES: 0.2, 0.5, 0.4", 3)
        assert sum(result) == pytest.approx(1.0)
        assert result[1] > result[0]

    def test_unmarked_percentages_are_converted(self):
        result = parsing.parse_probabilities("PROBABILITIES: 20, 50, 30", 3)
        assert result == pytest.approx([0.2, 0.5, 0.3])

    def test_boundary_sum_is_accepted_despite_float_error(self):
        """0.2 + 0.5 + 0.4 is 1.1000000000000001 in binary floating point."""
        assert parsing.parse_probabilities("PROBABILITIES: 0.2, 0.5, 0.4", 3) is not None

    def test_far_miss_is_rejected(self):
        assert parsing.parse_probabilities("PROBABILITIES: 0.1, 0.1, 0.1", 3) is None

    def test_wrong_length_rejected(self):
        assert parsing.parse_probabilities("PROBABILITIES: 0.5, 0.5", 3) is None

    def test_missing_line_rejected(self):
        assert parsing.parse_probabilities("It seems moderately likely.", 3) is None

    def test_negative_values_rejected(self):
        assert parsing.parse_probabilities("PROBABILITIES: -0.2, 0.9, 0.3", 3) is None

    def test_last_labelled_line_wins(self):
        """Models sometimes echo the format before committing to an answer."""
        response = (
            "Format: PROBABILITIES: <p1>, <p2>, <p3>\n"
            "PROBABILITIES: 0.25, 0.25, 0.50"
        )
        assert parsing.parse_probabilities(response, 3) == pytest.approx(
            [0.25, 0.25, 0.50]
        )

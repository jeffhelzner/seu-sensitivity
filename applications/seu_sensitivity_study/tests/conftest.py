"""
Shared fixtures for the SEU sensitivity study test suite.

Everything here is offline: no API keys, no network, no Stan compilation.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pytest

from applications.seu_sensitivity_study import schemas
from applications.seu_sensitivity_study.prompts import PromptSet


class MockLLMClient:
    """
    Stand-in for an LLM client that records every call.

    Matches the ``generate`` signature of the real clients so the collectors
    exercise their true code path, including the keyword-only arguments.
    """

    def __init__(
        self,
        responder: Optional[Callable[[str, Optional[str]], str]] = None,
        default: str = "ANSWER: 1",
    ):
        self.responder = responder
        self.default = default
        self.calls: List[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.responder is not None:
            return self.responder(prompt, system_prompt)
        return self.default

    def get_usage_summary(self) -> dict:
        """Mirrors the real clients' accounting hook."""
        return {
            "model": "mock",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "calls": len(self.calls),
        }


@pytest.fixture
def mock_client_factory():
    return MockLLMClient


@pytest.fixture
def prompt_set() -> PromptSet:
    """A minimal, valid prompt set usable without touching the YAML files."""
    return PromptSet(
        pool_id="testpool",
        family="startup",
        assessment_system="You are an analyst.",
        assessment_user=(
            "Assess this item.\n\n{item_text}\n\nOutcomes:\n{consequence_lines}\n\n"
            "End with:\n{probability_format}\n"
        ),
        choice_system="You are an analyst choosing one item.",
        choice_instructions={
            "neutral": "Choose which item to advance.",
            "seu_maximizing": (
                "Choose the item that maximizes subjective expected value given "
                "your assessed likelihoods."
            ),
            "deliberative": "Think carefully and reason step by step before choosing.",
        },
        choice_user=(
            "Below are assessments of {n_max} items.\n\n{assessments_list}\n\n"
            "{instruction}\n\nEnd with:\nANSWER: n\n"
        ),
    )


def _make_items(prefix: str, family: str, counts: dict[str, int]) -> list[dict]:
    items: list[dict] = []
    index = 0
    for label, count in counts.items():
        for _ in range(count):
            index += 1
            items.append(
                {
                    "id": f"{prefix}{index:03d}",
                    "family": family,
                    "text": (
                        f"{label.capitalize()} {family} item {index}: a templated "
                        f"vignette varying only the decision-relevant signals."
                    ),
                    "quality_label": label,
                    "attributes": {"signal_strength": label},
                    "matched_key": None,
                }
            )
    return items


@pytest.fixture
def label_counts() -> dict[str, int]:
    """Enough of each tier to satisfy every default recipe at menu size 8."""
    return {"strong": 10, "ambiguous": 10, "weak": 15}


@pytest.fixture
def single_family_pool(label_counts) -> dict:
    """A minimal single-family pool that passes validation."""
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "testpool",
        "framing": "positive",
        "consequences": ["loss", "break_even", "high_return"],
        "families": {"startup": {"description": "test family"}},
        "items": _make_items("T", "startup", label_counts),
    }


@pytest.fixture
def two_family_pool(label_counts) -> dict:
    """A venture-shaped pool: a primary family plus a matched family."""
    startup = _make_items("S", "startup", label_counts)
    procurement = _make_items("P", "procurement", label_counts)
    for offset, item in enumerate(procurement):
        item["matched_key"] = f"merit-{offset:03d}"
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "venture",
        "framing": "positive",
        "consequences": ["loss", "break_even", "high_return"],
        "families": {
            "startup": {"description": "startup prospects"},
            "procurement": {"description": "matched procurement twins"},
        },
        "items": startup + procurement,
    }


@pytest.fixture
def hiring_like_pool(label_counts) -> dict:
    """A hiring pool whose items carry the same matched keys as `two_family_pool`."""
    items = _make_items("H", "candidates", label_counts)
    for offset, item in enumerate(items):
        item["matched_key"] = f"merit-{offset:03d}"
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "hiring",
        "framing": "positive",
        "consequences": ["underperforms", "meets", "exceeds"],
        "families": {"candidates": {"description": "sterilized candidate profiles"}},
        "items": items,
    }

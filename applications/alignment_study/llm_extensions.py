"""
LLM client extensions for reasoning models.

Extends the base OpenAIClient and AnthropicClient to support:
- OpenAI reasoning models (o3-mini) with reasoning_effort parameter
- Anthropic extended thinking (Claude 3.7) with thinking blocks
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from applications.alignment_study.config import CellSpec

from applications.temperature_study.llm_client import (
    OpenAIClient,
    AnthropicClient,
    LLMClient,
    OPENAI_PRICING,
    ANTHROPIC_PRICING,
)

logger = logging.getLogger(__name__)


# Extended pricing tables for models not in the base module
EXTENDED_OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "o3-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

EXTENDED_ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}


class OpenAIReasoningClient(OpenAIClient):
    """
    Client for OpenAI reasoning models (o1, o3-mini, etc.).

    These models:
    - Do NOT accept a temperature parameter
    - Accept reasoning_effort ("low", "medium", "high") instead
    - May produce longer responses due to internal chain-of-thought
    """

    def __init__(
        self,
        model: str = "o3-mini",
        reasoning_effort: str = "medium",
        **kwargs: Any,
    ):
        # Remove temperature from kwargs if present
        kwargs.pop("default_temperature", None)
        super().__init__(model=model, default_temperature=0.0, **kwargs)
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,  # Ignored for reasoning models
        max_tokens: int = 256,
    ) -> str:
        """Generate response. Ignores temperature; uses reasoning_effort."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                if resp.usage:
                    self.total_input_tokens += resp.usage.prompt_tokens
                    self.total_output_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )

    def get_estimated_cost(self) -> float:
        pricing = EXTENDED_OPENAI_PRICING.get(
            self.model, OPENAI_PRICING.get(self.model, {"input": 0, "output": 0})
        )
        return (
            (self.total_input_tokens / 1_000_000) * pricing["input"]
            + (self.total_output_tokens / 1_000_000) * pricing["output"]
        )


class AnthropicThinkingClient(AnthropicClient):
    """
    Client for Anthropic models with extended thinking.

    These models:
    - Require temperature=1.0 when extended thinking is enabled
    - Accept budget_tokens to control thinking length
    - Return both thinking and text blocks in the response
    """

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        budget_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.budget_tokens = budget_tokens

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,  # Forced to 1.0
        max_tokens: int = 256,
    ) -> str:
        """
        Generate response with extended thinking.

        Temperature is forced to 1.0 (Anthropic requirement for thinking).
        The thinking block is discarded; only the text response is returned.
        """
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens + self.budget_tokens,
            temperature=1.0,
            thinking={"type": "enabled", "budget_tokens": self.budget_tokens},
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                if resp.usage:
                    self.total_input_tokens += resp.usage.input_tokens
                    self.total_output_tokens += resp.usage.output_tokens
                # Extract text block (skip thinking blocks)
                for block in resp.content:
                    if block.type == "text":
                        return block.text.strip()
                # Fallback: return first block's text
                return resp.content[0].text.strip()
            except Exception as e:
                last_error = e
                logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )

    def get_estimated_cost(self) -> float:
        pricing = EXTENDED_ANTHROPIC_PRICING.get(
            self.model, ANTHROPIC_PRICING.get(self.model, {"input": 0, "output": 0})
        )
        return (
            (self.total_input_tokens / 1_000_000) * pricing["input"]
            + (self.total_output_tokens / 1_000_000) * pricing["output"]
        )


# Reasoning model identifiers (models that use reasoning_effort instead of temperature)
REASONING_MODELS = {"o3-mini", "o1", "o1-mini", "o1-preview", "o3"}


def create_alignment_llm_client(
    cell_spec: "CellSpec",
    **kwargs: Any,
) -> LLMClient:
    """
    Factory that creates the appropriate LLM client for a cell.

    Routes to:
    - OpenAIReasoningClient for o3-mini, o1, etc.
    - AnthropicThinkingClient for Claude 3.7 with extended_thinking=True
    - OpenAIClient for standard OpenAI models
    - AnthropicClient for standard Anthropic models
    """
    provider_kwargs = cell_spec.provider_kwargs or {}

    if cell_spec.provider == "openai":
        if cell_spec.model_name in REASONING_MODELS:
            return OpenAIReasoningClient(
                model=cell_spec.model_name,
                reasoning_effort=provider_kwargs.get("reasoning_effort", "medium"),
                max_retries=kwargs.get("max_retries", 3),
                retry_delay=kwargs.get("retry_delay", 2.0),
            )
        else:
            return OpenAIClient(
                model=cell_spec.model_name,
                default_temperature=cell_spec.temperature,
                max_retries=kwargs.get("max_retries", 3),
                retry_delay=kwargs.get("retry_delay", 2.0),
            )
    elif cell_spec.provider == "anthropic":
        if provider_kwargs.get("extended_thinking", False):
            return AnthropicThinkingClient(
                model=cell_spec.model_name,
                budget_tokens=provider_kwargs.get("budget_tokens", 4096),
                max_retries=kwargs.get("max_retries", 3),
                retry_delay=kwargs.get("retry_delay", 2.0),
            )
        else:
            return AnthropicClient(
                model=cell_spec.model_name,
                default_temperature=cell_spec.temperature,
                max_retries=kwargs.get("max_retries", 3),
                retry_delay=kwargs.get("retry_delay", 2.0),
            )
    else:
        raise ValueError(f"Unknown provider: {cell_spec.provider}")

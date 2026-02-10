"""
LLM Client Module for the Temperature Study.

Provides OpenAI (and Anthropic) chat + embedding interfaces with:
- System prompt support (required by deliberation & choice prompts)
- Per-call temperature override (core to the study design)
- NA-safe choice parsing: returns None on failure, never defaults to a position
- Retry logic with exponential back-off
- Token usage / cost tracking
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pricing tables (per 1M tokens) — single source of truth
# ──────────────────────────────────────────────────────────────────────
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

EMBEDDING_PRICING: Dict[str, float] = {
    # cost per 1M tokens
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


# ──────────────────────────────────────────────────────────────────────
# Base client
# ──────────────────────────────────────────────────────────────────────
class LLMClient:
    """Abstract base for LLM chat clients."""

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ---------- core interface ----------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        """Return a text completion.  Subclasses must override."""
        raise NotImplementedError

    # ---------- choice helpers ----------

    def make_choice(
        self,
        prompt: str,
        num_alternatives: int,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 64,
    ) -> Optional[int]:
        """
        Send *prompt* and parse the response as a 1-indexed choice.

        Returns:
            The chosen position (1-indexed) or **None** if parsing fails.
            Never silently defaults to any position.
        """
        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = parse_choice(response, num_alternatives)
        if parsed is None:
            logger.warning(
                "Could not parse valid choice from response: %r", response
            )
        return parsed

    # ---------- cost tracking ----------

    def get_estimated_cost(self) -> float:  # pragma: no cover
        return 0.0

    def reset_usage(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.get_estimated_cost(),
        }


# ──────────────────────────────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────────────────────────────
class OpenAIClient(LLMClient):
    """OpenAI chat client with retry logic and cost tracking."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        default_temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        import openai  # lazy — only fail when actually used

        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
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
                    temperature=temperature,
                    max_tokens=max_tokens,
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
        pricing = OPENAI_PRICING.get(self.model, {"input": 0, "output": 0})
        return (
            (self.total_input_tokens / 1_000_000) * pricing["input"]
            + (self.total_output_tokens / 1_000_000) * pricing["output"]
        )


# ──────────────────────────────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────────────────────────────
class AnthropicClient(LLMClient):
    """Anthropic Claude chat client with retry logic and cost tracking."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        default_temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
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
        pricing = ANTHROPIC_PRICING.get(self.model, {"input": 0, "output": 0})
        return (
            (self.total_input_tokens / 1_000_000) * pricing["input"]
            + (self.total_output_tokens / 1_000_000) * pricing["output"]
        )


# ──────────────────────────────────────────────────────────────────────
# Embedding client (OpenAI only)
# ──────────────────────────────────────────────────────────────────────
class EmbeddingClient:
    """OpenAI embedding client with batching and cost tracking."""

    MAX_BATCH = 100  # OpenAI limit per request

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_tokens: int = 0

        import openai

        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts, batching if necessary.

        Returns:
            List of embedding vectors (one per input text).
        """
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batch = texts[start : start + self.MAX_BATCH]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(
                    model=self.model, input=texts
                )
                if resp.usage:
                    self.total_tokens += resp.usage.total_tokens
                # Sort by index to guarantee ordering
                sorted_data = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in sorted_data]
            except Exception as e:
                last_error = e
                logger.warning(
                    "Embedding call failed (attempt %d): %s", attempt + 1, e
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} attempts: {last_error}"
        )

    def get_estimated_cost(self) -> float:
        per_million = EMBEDDING_PRICING.get(self.model, 0.0)
        return (self.total_tokens / 1_000_000) * per_million

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.get_estimated_cost(),
        }


# ──────────────────────────────────────────────────────────────────────
# Choice parsing (NA-safe)
# ──────────────────────────────────────────────────────────────────────
def parse_choice(response: str, num_alternatives: int) -> Optional[int]:
    """
    Parse a 1-indexed choice from an LLM response.

    Returns:
        The chosen position (1-indexed) if unambiguously parseable,
        or **None** if parsing fails or is ambiguous.
        Never defaults to any position.
    """
    text = response.strip()

    # ── 1. Check for "Claim N" mentions ──
    claim_matches = re.findall(r"[Cc]laim\s*(\d+)", text)
    if claim_matches:
        # Keep only in-range values
        valid = {int(m) for m in claim_matches if 1 <= int(m) <= num_alternatives}
        if len(valid) == 1:
            return valid.pop()
        # Multiple distinct valid claim references → ambiguous
        return None

    # ── 2. Bare number (entire response is just a digit) ──
    match = re.match(r"^(\d+)\s*$", text)
    if match:
        value = int(match.group(1))
        if 1 <= value <= num_alternatives:
            return value

    # ── 3. Single number anywhere in response ──
    all_nums = re.findall(r"\d+", text)
    if len(all_nums) == 1:
        value = int(all_nums[0])
        if 1 <= value <= num_alternatives:
            return value

    # If none of the strict patterns matched, do NOT fall back.
    return None


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────
def create_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Factory function to create a chat LLM client.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.
        model: Model name (uses provider default if not given).
        **kwargs: Forwarded to the client constructor.
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o", **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-3-sonnet-20240229", **kwargs
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'openai' or 'anthropic'."
        )

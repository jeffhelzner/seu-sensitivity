"""
Resilient client layer (build plan A8; study plan §12, §6.5).

Wraps a provider client with the two things a 10,800-call collection run needs
and the inherited clients do not have: an on-disk response cache and
rate-limit-aware backoff.

The cache is what makes the run restartable **and** makes a mid-run bug fix
affordable: re-running a phase after a downstream change costs nothing for
calls already made.  It is keyed on everything that could change a response --
model, system prompt, user prompt, temperature, token budget, and the pinned
request parameters -- so a prompt edit correctly misses the cache rather than
silently serving a stale answer under a new prompt.

Retries live here rather than in the provider clients so there is exactly one
retry policy.  The wrapped client is therefore constructed with its own
retrying disabled; nesting the two would multiply attempts and turn a rate
limit into a much longer stall than intended.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CellSpec
from .llm_extensions import create_seu_sensitivity_llm_client

logger = logging.getLogger(__name__)

__all__ = ["ResilientClient", "build_client", "is_rate_limit_error"]


#: Substrings that mark a provider error as retryable rather than fatal.
_RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "overloaded",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
)


def is_rate_limit_error(error: BaseException) -> bool:
    """Whether *error* looks transient and worth retrying."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


class ResilientClient:
    """
    Caching, retrying facade over a provider client.

    Exposes the same ``generate`` signature as the underlying clients, so the
    collectors are unaware of it.
    """

    def __init__(
        self,
        inner: Any,
        *,
        model_name: str,
        request_params: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Any = time.sleep,
    ):
        self.inner = inner
        self.model_name = model_name
        self.request_params = dict(request_params or {})
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep

        self.cache_hits = 0
        self.cache_misses = 0
        self.retries = 0

    # -- Public API --

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        key = self._cache_key(prompt, system_prompt, temperature, max_tokens)
        cached = self._read_cache(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        response = self._call_with_retries(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._write_cache(key, response, prompt=prompt, system_prompt=system_prompt)
        return response

    def get_usage_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if hasattr(self.inner, "get_usage_summary"):
            summary.update(self.inner.get_usage_summary())
        summary.update(
            {
                "model": self.model_name,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "retries": self.retries,
            }
        )
        return summary

    # -- Internals --

    def _call_with_retries(self, prompt: str, **kwargs: Any) -> str:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return self.inner.generate(prompt, **kwargs)
            except Exception as error:  # noqa: BLE001 - provider SDKs vary
                last_error = error
                if not is_rate_limit_error(error) or attempt == self.max_retries - 1:
                    raise
                self.retries += 1
                delay = self._backoff_delay(attempt, error)
                logger.warning(
                    "Retryable error from %s (attempt %d/%d), sleeping %.1fs: %s",
                    self.model_name,
                    attempt + 1,
                    self.max_retries,
                    delay,
                    error,
                )
                self._sleep(delay)

        raise RuntimeError(
            f"{self.model_name}: exhausted {self.max_retries} attempts"
        ) from last_error

    def _backoff_delay(self, attempt: int, error: BaseException) -> float:
        """Exponential backoff with jitter, honouring Retry-After when offered."""
        retry_after = getattr(error, "retry_after", None)
        if retry_after is None:
            headers = getattr(error, "headers", None) or {}
            try:
                retry_after = headers.get("retry-after")
            except AttributeError:
                retry_after = None
        if retry_after is not None:
            try:
                return min(float(retry_after), self.max_delay)
            except (TypeError, ValueError):
                pass

        delay = min(self.base_delay * (2**attempt), self.max_delay)
        # Full jitter: without it, concurrent workers that hit the same rate
        # limit would retry in lockstep and trip it again together.
        return random.uniform(0.0, delay)

    def _cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "request_params": self.request_params,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        slug = self.model_name.replace("/", "_")
        # Two-character shard keeps directories from growing to ~10k entries.
        return self.cache_dir / slug / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with open(path) as handle:
                return json.load(handle)["response"]
        except (OSError, json.JSONDecodeError, KeyError) as error:
            logger.warning("Ignoring unreadable cache entry %s: %s", path, error)
            return None

    def _write_cache(
        self, key: str, response: str, *, prompt: str, system_prompt: Optional[str]
    ) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.model_name,
            "request_params": self.request_params,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as handle:
            json.dump(record, handle)
        tmp.replace(path)


def build_client(
    cell: CellSpec,
    *,
    cache_dir: Optional[Path] = None,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> ResilientClient:
    """
    Build the wrapped client for a cell.

    The inner provider client is created with ``max_retries=1`` so this wrapper
    owns the single retry policy (see module docstring).
    """
    inner = create_seu_sensitivity_llm_client(cell, max_retries=1, retry_delay=retry_delay)
    return ResilientClient(
        inner,
        model_name=cell.model_name,
        request_params=cell.request_params,
        cache_dir=cache_dir,
        max_retries=max_retries,
        base_delay=retry_delay,
    )

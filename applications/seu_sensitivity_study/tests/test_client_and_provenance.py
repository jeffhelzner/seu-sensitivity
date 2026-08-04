"""
Tests for the resilient client layer and the provenance manifest
(build plan A8/A10; study plan §6.5, §3.1).
"""

from __future__ import annotations

import json

import pytest

from applications.seu_sensitivity_study import provenance, schemas
from applications.seu_sensitivity_study.client import ResilientClient, is_rate_limit_error
from applications.seu_sensitivity_study.config import (
    REFERENCE_MODEL,
    CellSpec,
    SEUSensitivityStudyConfig,
)


class FlakyClient:
    """Fails with *errors* in sequence, then succeeds."""

    def __init__(self, errors, response="ok"):
        self.errors = list(errors)
        self.response = response
        self.calls = 0

    def generate(self, prompt, *, system_prompt=None, temperature=None, max_tokens=256):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


class RateLimitError(Exception):
    status_code = 429


class FatalError(Exception):
    status_code = 400


class TestRetryClassification:
    @pytest.mark.parametrize(
        "error",
        [
            RateLimitError("rate limit exceeded"),
            Exception("429 Too Many Requests"),
            Exception("Server overloaded, try again"),
            Exception("Connection reset by peer"),
        ],
    )
    def test_transient_errors_are_retryable(self, error):
        assert is_rate_limit_error(error)

    @pytest.mark.parametrize(
        "error",
        [FatalError("invalid request"), ValueError("malformed prompt")],
    )
    def test_fatal_errors_are_not(self, error):
        assert not is_rate_limit_error(error)


class TestResilientClient:
    def test_retries_transient_errors(self):
        inner = FlakyClient([RateLimitError("rate limit"), RateLimitError("rate limit")])
        client = ResilientClient(
            inner, model_name="gpt-4o", max_retries=5, sleep=lambda _: None
        )
        assert client.generate("hi") == "ok"
        assert inner.calls == 3
        assert client.retries == 2

    def test_fatal_errors_are_not_retried(self):
        inner = FlakyClient([FatalError("bad request")])
        client = ResilientClient(inner, model_name="gpt-4o", sleep=lambda _: None)
        with pytest.raises(FatalError):
            client.generate("hi")
        assert inner.calls == 1

    def test_exhausted_retries_raise(self):
        inner = FlakyClient([RateLimitError("rate limit")] * 10)
        client = ResilientClient(
            inner, model_name="gpt-4o", max_retries=3, sleep=lambda _: None
        )
        with pytest.raises(RateLimitError):
            client.generate("hi")
        assert inner.calls == 3

    def test_backoff_honours_retry_after(self):
        error = RateLimitError("rate limit")
        error.retry_after = 7
        client = ResilientClient(FlakyClient([]), model_name="gpt-4o", max_delay=60)
        assert client._backoff_delay(0, error) == 7.0

    def test_backoff_is_bounded_and_jittered(self):
        client = ResilientClient(
            FlakyClient([]), model_name="gpt-4o", base_delay=2.0, max_delay=10.0
        )
        delays = [client._backoff_delay(5, Exception("x")) for _ in range(20)]
        assert all(0.0 <= delay <= 10.0 for delay in delays)
        assert len(set(delays)) > 1  # jitter, not lockstep


class TestCaching:
    def test_second_call_is_served_from_cache(self, tmp_path):
        inner = FlakyClient([], response="cached me")
        client = ResilientClient(inner, model_name="gpt-4o", cache_dir=tmp_path)
        assert client.generate("prompt", system_prompt="sys") == "cached me"
        assert client.generate("prompt", system_prompt="sys") == "cached me"
        assert inner.calls == 1
        assert client.cache_hits == 1
        assert client.cache_misses == 1

    def test_cache_survives_a_new_client_instance(self, tmp_path):
        first = ResilientClient(
            FlakyClient([], response="v1"), model_name="gpt-4o", cache_dir=tmp_path
        )
        first.generate("prompt")
        second = ResilientClient(
            FlakyClient([RateLimitError("would fail")]),
            model_name="gpt-4o",
            cache_dir=tmp_path,
            sleep=lambda _: None,
        )
        assert second.generate("prompt") == "v1"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prompt": "different"},
            {"system_prompt": "other role"},
            {"temperature": 1.0},
            {"max_tokens": 128},
        ],
    )
    def test_any_request_difference_misses_the_cache(self, tmp_path, kwargs):
        """A prompt edit must miss, not silently serve a stale answer."""
        inner = FlakyClient([], response="x")
        client = ResilientClient(inner, model_name="gpt-4o", cache_dir=tmp_path)
        base = {
            "prompt": "prompt",
            "system_prompt": "sys",
            "temperature": 0.0,
            "max_tokens": 64,
        }
        client.generate(**base)
        client.generate(**{**base, **kwargs})
        assert inner.calls == 2

    def test_request_params_are_part_of_the_key(self, tmp_path):
        """reasoning_effort / budget_tokens are part of the treatment (§3.1)."""
        low = ResilientClient(
            FlakyClient([], response="low"),
            model_name="o3-mini",
            request_params={"reasoning_effort": "low"},
            cache_dir=tmp_path,
        )
        high_inner = FlakyClient([], response="high")
        high = ResilientClient(
            high_inner,
            model_name="o3-mini",
            request_params={"reasoning_effort": "high"},
            cache_dir=tmp_path,
        )
        low.generate("prompt")
        assert high.generate("prompt") == "high"
        assert high_inner.calls == 1

    def test_corrupt_cache_entry_is_ignored(self, tmp_path):
        inner = FlakyClient([], response="fresh")
        client = ResilientClient(inner, model_name="gpt-4o", cache_dir=tmp_path)
        client.generate("prompt")
        for path in tmp_path.rglob("*.json"):
            path.write_text("{not json")
        assert client.generate("prompt") == "fresh"
        assert inner.calls == 2

    def test_no_cache_dir_means_no_caching(self):
        inner = FlakyClient([], response="x")
        client = ResilientClient(inner, model_name="gpt-4o", cache_dir=None)
        client.generate("prompt")
        client.generate("prompt")
        assert inner.calls == 2


class TestProvenance:
    def test_manifest_validates(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        assert schemas.validate_run_manifest(manifest) == []

    def test_one_entry_per_model_with_pinned_params(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        assert len(manifest["models"]) == 6
        by_name = {entry["model_name"]: entry for entry in manifest["models"]}
        assert by_name["o3-mini"]["request_params"]["reasoning_effort"] == "medium"
        assert by_name["claude-3-7-sonnet-20250219"]["request_params"]["budget_tokens"] == 4096

    def test_temperature_none_is_recorded_explicitly(self):
        """An absent key would read as an oversight rather than a fact."""
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        by_name = {entry["model_name"]: entry for entry in manifest["models"]}
        assert "temperature" in by_name["o3-mini"]["request_params"]
        assert by_name["o3-mini"]["request_params"]["temperature"] is None

    def test_undated_endpoints_are_flagged(self, caplog):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        with caplog.at_level("WARNING"):
            manifest = provenance.build_run_manifest(config)
        assert any(entry["endpoint_is_dated"] is False for entry in manifest["models"])
        assert "dated endpoint" in caplog.text

    def test_dated_endpoints_are_recorded(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(
            config, endpoint_ids={"gpt-4o": "gpt-4o-2024-11-20"}
        )
        by_name = {entry["model_name"]: entry for entry in manifest["models"]}
        assert by_name["gpt-4o"]["endpoint_id"] == "gpt-4o-2024-11-20"
        assert by_name["gpt-4o"]["endpoint_is_dated"] is True

    def test_substitution_requires_a_reason(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        provenance.record_substitution(
            manifest,
            "gpt-4o",
            substituted_for="gpt-4o-deprecated",
            reason="Endpoint retired; nearest same-tier same-vendor successor.",
        )
        assert schemas.validate_run_manifest(manifest) == []

    def test_unknown_model_substitution_raises(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        with pytest.raises(KeyError):
            provenance.record_substitution(
                manifest, "gpt-5", substituted_for="gpt-4o", reason="x"
            )

    def test_reference_levels_are_pinned(self):
        config = SEUSensitivityStudyConfig(pool_ids=["venture"])
        manifest = provenance.build_run_manifest(config)
        assert manifest["reference_model"] == REFERENCE_MODEL

    def test_toolchain_versions_recorded(self):
        versions = provenance.toolchain_versions()
        assert versions["python"]
        assert "numpy" in versions and "cmdstan" in versions

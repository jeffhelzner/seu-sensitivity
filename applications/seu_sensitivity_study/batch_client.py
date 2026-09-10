"""Restartable provider Batch API clients for choice collection."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .config import CellSpec
from .llm_extensions import pricing_for


@dataclass(frozen=True)
class BatchPrompt:
    custom_id: str
    prompt: str
    system_prompt: Optional[str]
    temperature: Optional[float]
    max_tokens: int


@dataclass(frozen=True)
class BatchOutcome:
    responses: Dict[str, str]
    usage: Dict[str, Any]
    failed_custom_ids: tuple[str, ...] = ()
    duplicate_custom_ids: tuple[str, ...] = ()
    result_records: tuple[Dict[str, Any], ...] = ()
    provider_errors: tuple[Dict[str, Any], ...] = ()


class BatchPending(RuntimeError):
    """Retained for callers that prefer an exception-based pending signal."""


class BatchResultError(RuntimeError):
    """A provider batch or one of its requests failed terminally."""


class ProviderBatchClient:
    """Submit once, then retrieve on a later invocation using durable state."""

    def __init__(
        self,
        cell: CellSpec,
        *,
        sdk_client: Any = None,
        submission_reserver: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        self.cell = cell
        self.provider = cell.provider
        self.model = cell.endpoint
        self.request_params = dict(cell.request_params or {})
        self.reasoning_reserve = cell.reasoning_token_reserve
        self._sdk_client = sdk_client or self._new_sdk_client()
        self._submission_reserver = submission_reserver
        self.last_usage: Dict[str, Any] = {}

    def process(
        self, requests: Sequence[BatchPrompt], *, state_path: Path
    ) -> Optional[BatchOutcome]:
        with _single_writer(state_path):
            request_hash = self.request_hash(requests)
            if not state_path.exists():
                state = {
                    "schema_version": 2,
                    "provider": self.provider,
                    "model": self.model,
                    "request_hash": request_hash,
                    "request_count": len(requests),
                    "status": "reserving",
                    "submission_id": uuid.uuid4().hex,
                }
                _write_json(state_path, state)
                self._start_submission(requests, state_path, state)
                return None

            state = json.loads(state_path.read_text())
            if state.get("request_hash") != request_hash:
                raise BatchResultError(
                    f"Batch request changed after submission for {state_path}; "
                    "refusing to attach existing results"
                )
            if state.get("request_count") != len(requests):
                raise BatchResultError(
                    f"Batch request count does not match durable state for {state_path}"
                )
            if not state.get("batch_id"):
                if state.get("status") == "reserving":
                    self._start_submission(requests, state_path, state)
                    return None
                batch_id = self._reconcile_submission(state)
                state.update(
                    batch_id=batch_id,
                    status="submitted",
                    reconciled_from_provider=True,
                )
                _write_json(state_path, state)

            outcome, status = self._retrieve(state["batch_id"])
            state["status"] = status
            if outcome is not None:
                self.last_usage = outcome.usage
                state["usage"] = outcome.usage
                state["responses"] = outcome.responses
                state["failed_custom_ids"] = list(outcome.failed_custom_ids)
                state["duplicate_custom_ids"] = list(outcome.duplicate_custom_ids)
                state["result_records"] = list(outcome.result_records)
                state["provider_errors"] = list(outcome.provider_errors)
                _write_json(state_path, state)

                if outcome.duplicate_custom_ids:
                    raise BatchResultError(
                        f"Batch {state['batch_id']} has duplicate result ID(s): "
                        f"{list(outcome.duplicate_custom_ids)}"
                    )
                if outcome.failed_custom_ids:
                    raise BatchResultError(
                        f"Batch {state['batch_id']} has failed request(s): "
                        f"{list(outcome.failed_custom_ids)}"
                    )
                if outcome.provider_errors or status != "completed":
                    raise BatchResultError(
                        f"Batch {state['batch_id']} ended with status {status}; "
                        f"provider_errors={list(outcome.provider_errors)}"
                    )
                expected = {request.custom_id for request in requests}
                received = set(outcome.responses)
                if received != expected:
                    missing = sorted(expected - received)
                    unexpected = sorted(received - expected)
                    raise BatchResultError(
                        f"Batch {state['batch_id']} result IDs do not match requests; "
                        f"missing={missing}, unexpected={unexpected}"
                    )
            else:
                _write_json(state_path, state)
            return outcome

    def _start_submission(
        self,
        requests: Sequence[BatchPrompt],
        state_path: Path,
        state: Dict[str, Any],
    ) -> None:
        if self._submission_reserver is not None:
            self._submission_reserver(state)
        state["status"] = "submitting"
        _write_json(state_path, state)
        try:
            batch_id = self._submit(
                requests,
                state_path,
                state["request_hash"],
                state["submission_id"],
            )
        except Exception as error:
            state["status"] = "submission_ambiguous"
            state["submission_error"] = f"{type(error).__name__}: {error}"
            _write_json(state_path, state)
            raise
        state.update(batch_id=batch_id, status="submitted")
        _write_json(state_path, state)

    def request_hash(self, requests: Sequence[BatchPrompt]) -> str:
        """Hash the exact provider request bodies that would be transmitted."""
        encoded = json.dumps(
            self.render_requests(requests), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def render_requests(
        self, requests: Sequence[BatchPrompt]
    ) -> List[Dict[str, Any]]:
        """Render the canonical provider request bodies without network access."""
        return [
            self._openai_request(request)
            if self.provider == "openai"
            else self._anthropic_request(request)
            for request in requests
        ]

    def recover_usage(self, state_path: Path) -> Dict[str, Any]:
        """Recover usage after results were checkpointed but not yet ledgered."""
        if not state_path.exists():
            return {}
        state = json.loads(state_path.read_text())
        usage = dict(state.get("usage") or {})
        self.last_usage = usage
        return usage

    def recover_partial_responses(
        self, state_path: Path, *, expected_request_hash: str
    ) -> Dict[str, str]:
        """Return only successful, unambiguous responses from durable state."""
        if not state_path.exists():
            return {}
        state = json.loads(state_path.read_text())
        if state.get("request_hash") != expected_request_hash:
            raise BatchResultError(
                f"Batch request identity does not match partial state {state_path}"
            )
        blocked = set(state.get("failed_custom_ids") or ()) | set(
            state.get("duplicate_custom_ids") or ()
        )
        return {
            str(custom_id): str(response)
            for custom_id, response in (state.get("responses") or {}).items()
            if custom_id not in blocked
        }

    def attach_ambiguous_batch(
        self, *, state_path: Path, batch_id: str, operator_note: str
    ) -> Dict[str, Any]:
        """Attach an operator-identified Anthropic batch without submitting."""
        if self.provider != "anthropic":
            raise BatchResultError(
                "Manual attachment is restricted to Anthropic ambiguous submissions"
            )
        if not batch_id.strip() or not operator_note.strip():
            raise ValueError("batch_id and operator_note must be nonempty")
        with _single_writer(state_path):
            if not state_path.exists():
                raise BatchResultError(f"Batch state does not exist: {state_path}")
            state = json.loads(state_path.read_text())
            if state.get("provider") != self.provider or state.get("model") != self.model:
                raise BatchResultError("Batch state provider or model does not match client")
            if state.get("status") != "submission_ambiguous" or state.get("batch_id"):
                raise BatchResultError(
                    "Manual attachment requires an ambiguous state without a batch ID"
                )
            remote = self._sdk_client.messages.batches.retrieve(batch_id)
            remote_total = _anthropic_request_count(
                getattr(remote, "request_counts", None)
            )
            if remote_total is None:
                raise BatchResultError(
                    "Anthropic batch exposes no complete request count; "
                    "manual attachment cannot verify identity"
                )
            if remote_total != int(state["request_count"]):
                raise BatchResultError(
                    "Anthropic batch request count does not match durable state"
                )
            state.update(
                batch_id=batch_id,
                status="submitted",
                reconciled_from_provider=True,
                reconciliation={
                    "method": "operator_attached_anthropic_batch",
                    "operator_note": operator_note,
                    "attached_at": datetime.now(timezone.utc).isoformat(),
                    "remote_processing_status": str(remote.processing_status),
                    "remote_request_count": remote_total,
                },
            )
            _write_json(state_path, state)
            return state

    def _new_sdk_client(self) -> Any:
        if self.provider == "openai":
            import openai

            return openai.OpenAI()
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic()
        raise ValueError(f"Unknown provider: {self.provider}")

    def _submit(
        self,
        requests: Sequence[BatchPrompt],
        state_path: Path,
        request_hash: str,
        submission_id: str,
    ) -> str:
        if self.provider == "openai":
            lines = [json.dumps(self._openai_request(request)) for request in requests]
            payload = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
            payload.name = f"{state_path.stem}.jsonl"
            uploaded = self._sdk_client.files.create(file=payload, purpose="batch")
            batch = self._sdk_client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "cell_id": self.cell.cell_id,
                    "request_hash": request_hash,
                    "submission_id": submission_id,
                },
            )
            return batch.id

        batch = self._sdk_client.messages.batches.create(
            requests=[self._anthropic_request(request) for request in requests]
        )
        return batch.id

    def _reconcile_submission(self, state: Mapping[str, Any]) -> str:
        if self.provider != "openai":
            raise BatchResultError(
                "Provider-side reconciliation is unavailable for Anthropic batches "
                "because they do not carry request-bound metadata"
            )

        candidates = []
        for batch in self._sdk_client.batches.list():
            metadata = getattr(batch, "metadata", None) or {}
            if (
                metadata.get("cell_id") == self.cell.cell_id
                and metadata.get("request_hash") == state.get("request_hash")
                and metadata.get("submission_id") == state.get("submission_id")
            ):
                candidates.append(str(batch.id))
        if len(candidates) != 1:
            raise BatchResultError(
                "Provider-side reconciliation requires exactly one OpenAI batch "
                f"matching cell and request hash; found {len(candidates)}"
            )
        return candidates[0]

    def _retrieve(self, batch_id: str) -> tuple[Optional[BatchOutcome], str]:
        if self.provider == "openai":
            return self._retrieve_openai(batch_id)
        return self._retrieve_anthropic(batch_id)

    def _retrieve_openai(
        self, batch_id: str
    ) -> tuple[Optional[BatchOutcome], str]:
        batch = self._sdk_client.batches.retrieve(batch_id)
        status = str(batch.status)
        if status in {"validating", "in_progress", "finalizing", "cancelling"}:
            return None, status
        if status not in {"completed", "failed", "expired", "cancelled"}:
            raise BatchResultError(f"OpenAI batch {batch_id} has unknown status {status}")

        responses: Dict[str, str] = {}
        usages = []
        failures = []
        duplicates = []
        records = []
        batch_errors = getattr(batch, "errors", None)
        provider_errors = [
            _openai_error_record(error)
            for error in (_object_value(batch_errors, "data", []) or [])
        ]
        file_ids = [
            file_id
            for file_id in (
                getattr(batch, "output_file_id", None),
                getattr(batch, "error_file_id", None),
            )
            if file_id
        ]
        for file_id in dict.fromkeys(file_ids):
            content = self._sdk_client.files.content(file_id).text
            records.extend(json.loads(line) for line in content.splitlines() if line)
        if not records and not provider_errors:
            provider_errors.append(
                {"status": status, "message": "Batch supplied no output or error evidence"}
            )
        for result in records:
            custom_id = result.get("custom_id")
            if not custom_id:
                provider_errors.append(result)
                continue
            if custom_id in responses or custom_id in failures:
                duplicates.append(custom_id)
                continue
            response = result.get("response")
            if result.get("error") or not response or response.get("status_code") != 200:
                failures.append(custom_id)
                continue
            body = response["body"]
            responses[custom_id] = body["choices"][0]["message"]["content"].strip()
            usages.append(_openai_usage(body.get("usage") or {}))
        batch_usage = getattr(batch, "usage", None)
        if batch_usage is not None:
            usages = [_openai_batch_usage(batch_usage)]
        usage_summary = self._usage_summary(usages)
        request_counts = getattr(batch, "request_counts", None)
        if request_counts is not None:
            remote_total = _object_value(request_counts, "total", None)
            if remote_total is None:
                usage_summary["calls"] = None
                provider_errors.append(
                    {
                        "status": status,
                        "message": "Batch request_counts omitted total",
                    }
                )
            else:
                usage_summary["calls"] = int(remote_total)
        return BatchOutcome(
            responses,
            usage_summary,
            tuple(sorted(set(failures))),
            tuple(sorted(set(duplicates))),
            tuple(records),
            tuple(provider_errors),
        ), status

    def _retrieve_anthropic(
        self, batch_id: str
    ) -> tuple[Optional[BatchOutcome], str]:
        batch = self._sdk_client.messages.batches.retrieve(batch_id)
        status = str(batch.processing_status)
        if status != "ended":
            return None, status

        responses: Dict[str, str] = {}
        usages = []
        failures = []
        duplicates = []
        records = []
        for item in self._sdk_client.messages.batches.results(batch_id):
            result = item.result
            custom_id = item.custom_id
            result_type = str(result.type)
            records.append({"custom_id": custom_id, "result_type": result_type})
            if custom_id in responses or custom_id in failures:
                duplicates.append(custom_id)
                continue
            if result_type != "succeeded":
                failures.append(custom_id)
                continue
            message = result.message
            text = next(
                (block.text for block in message.content if block.type == "text"), ""
            )
            responses[custom_id] = text.strip()
            usages.append(_anthropic_usage(message.usage))
        return BatchOutcome(
            responses,
            self._usage_summary(usages),
            tuple(sorted(set(failures))),
            tuple(sorted(set(duplicates))),
            tuple(records),
        ), status

    def _openai_request(self, request: BatchPrompt) -> Dict[str, Any]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        body: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.request_params.get("reasoning_effort"):
            body.update(
                max_completion_tokens=request.max_tokens + (self.reasoning_reserve or 2048),
                reasoning_effort=self.request_params["reasoning_effort"],
            )
        else:
            body["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                body["temperature"] = request.temperature
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _anthropic_request(self, request: BatchPrompt) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            params["system"] = request.system_prompt
        if self.request_params.get("extended_thinking"):
            budget = int(self.request_params.get("budget_tokens", 4096))
            params.update(
                max_tokens=request.max_tokens + budget,
                temperature=1.0,
                thinking={"type": "enabled", "budget_tokens": budget},
            )
        elif request.temperature is not None:
            params["temperature"] = request.temperature
        return {"custom_id": request.custom_id, "params": params}

    def _usage_summary(self, usages: Sequence[Mapping[str, int]]) -> Dict[str, Any]:
        totals = {
            key: sum(usage.get(key, 0) for usage in usages)
            for key in {
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            }
        }
        pricing = pricing_for(self.model, self.provider)
        totals.update(
            model=self.cell.model_name,
            provider=self.provider,
            batch_discount=0.5,
            calls=len(usages),
            estimated_cost_usd=0.5
            * (
                totals["input_tokens"] * pricing["input"]
                + totals["output_tokens"] * pricing["output"]
            )
            / 1_000_000,
        )
        return totals


def _openai_usage(usage: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "cached_input_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "reasoning_tokens": int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        ),
    }


def _openai_batch_usage(usage: Any) -> Dict[str, int]:
    input_details = _object_value(usage, "input_tokens_details", {}) or {}
    output_details = _object_value(usage, "output_tokens_details", {}) or {}
    return {
        "input_tokens": int(_object_value(usage, "input_tokens", 0) or 0),
        "output_tokens": int(_object_value(usage, "output_tokens", 0) or 0),
        "cached_input_tokens": int(_object_value(input_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(
            _object_value(output_details, "reasoning_tokens", 0) or 0
        ),
    }


def _object_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _anthropic_request_count(request_counts: Any) -> Optional[int]:
    if request_counts is None:
        return None
    total = _object_value(request_counts, "total", None)
    if total is not None:
        return int(total)
    statuses = ("processing", "succeeded", "errored", "canceled", "expired")
    counts = [_object_value(request_counts, status, None) for status in statuses]
    if any(count is None for count in counts):
        return None
    return sum(int(count) for count in counts)


def _openai_error_record(error: Any) -> Dict[str, Any]:
    if isinstance(error, Mapping):
        return dict(error)
    if hasattr(error, "model_dump"):
        return dict(error.model_dump())
    return {
        key: getattr(error, key)
        for key in ("code", "line", "message", "param")
        if getattr(error, key, None) is not None
    }


def _anthropic_usage(usage: Any) -> Dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_input_tokens": int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


@contextmanager
def _single_writer(state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchResultError(
                f"Batch state is already being modified: {state_path}"
            ) from error
        owner = json.dumps({"hostname": socket.gethostname(), "pid": os.getpid()})
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, owner.encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)

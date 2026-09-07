"""Restartable provider Batch API clients for choice collection."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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


class BatchPending(RuntimeError):
    """Retained for callers that prefer an exception-based pending signal."""


class BatchResultError(RuntimeError):
    """A provider batch or one of its requests failed terminally."""


class ProviderBatchClient:
    """Submit once, then retrieve on a later invocation using durable state."""

    def __init__(self, cell: CellSpec, *, sdk_client: Any = None):
        self.cell = cell
        self.provider = cell.provider
        self.model = cell.endpoint
        self.request_params = dict(cell.request_params or {})
        self.reasoning_reserve = cell.reasoning_token_reserve
        self._sdk_client = sdk_client or self._new_sdk_client()
        self.last_usage: Dict[str, Any] = {}

    def process(
        self, requests: Sequence[BatchPrompt], *, state_path: Path
    ) -> Optional[BatchOutcome]:
        request_hash = _request_hash(self.cell, requests)
        if not state_path.exists():
            batch_id = self._submit(requests, state_path)
            _write_json(
                state_path,
                {
                    "schema_version": 1,
                    "provider": self.provider,
                    "model": self.model,
                    "batch_id": batch_id,
                    "request_hash": request_hash,
                    "request_count": len(requests),
                    "status": "submitted",
                },
            )
            return None

        state = json.loads(state_path.read_text())
        if state.get("request_hash") != request_hash:
            raise BatchResultError(
                f"Batch request changed after submission for {state_path}; "
                "refusing to attach existing results"
            )
        outcome, status = self._retrieve(state["batch_id"])
        state["status"] = status
        if outcome is not None:
            expected = {request.custom_id for request in requests}
            received = set(outcome.responses)
            if received != expected:
                missing = sorted(expected - received)
                unexpected = sorted(received - expected)
                raise BatchResultError(
                    f"Batch {state['batch_id']} result IDs do not match requests; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            self.last_usage = outcome.usage
            state["usage"] = outcome.usage
        _write_json(state_path, state)
        return outcome

    def recover_usage(self, state_path: Path) -> Dict[str, Any]:
        """Recover usage after results were checkpointed but not yet ledgered."""
        if not state_path.exists():
            return {}
        state = json.loads(state_path.read_text())
        usage = dict(state.get("usage") or {})
        self.last_usage = usage
        return usage

    def _new_sdk_client(self) -> Any:
        if self.provider == "openai":
            import openai

            return openai.OpenAI()
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic()
        raise ValueError(f"Unknown provider: {self.provider}")

    def _submit(self, requests: Sequence[BatchPrompt], state_path: Path) -> str:
        if self.provider == "openai":
            lines = [json.dumps(self._openai_request(request)) for request in requests]
            payload = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
            payload.name = f"{state_path.stem}.jsonl"
            uploaded = self._sdk_client.files.create(file=payload, purpose="batch")
            batch = self._sdk_client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"cell_id": self.cell.cell_id},
            )
            return batch.id

        batch = self._sdk_client.messages.batches.create(
            requests=[self._anthropic_request(request) for request in requests]
        )
        return batch.id

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
        if status != "completed":
            raise BatchResultError(f"OpenAI batch {batch_id} ended with status {status}")
        if not batch.output_file_id:
            raise BatchResultError(f"OpenAI batch {batch_id} has no output file")

        content = self._sdk_client.files.content(batch.output_file_id).text
        responses: Dict[str, str] = {}
        usages = []
        failures = []
        for line in content.splitlines():
            result = json.loads(line)
            response = result.get("response")
            if result.get("error") or not response or response.get("status_code") != 200:
                failures.append(result["custom_id"])
                continue
            body = response["body"]
            responses[result["custom_id"]] = body["choices"][0]["message"]["content"].strip()
            usages.append(_openai_usage(body.get("usage") or {}))
        if failures:
            raise BatchResultError(
                f"OpenAI batch {batch_id} has failed request(s): {sorted(failures)}"
            )
        return BatchOutcome(responses, self._usage_summary(usages)), status

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
        for item in self._sdk_client.messages.batches.results(batch_id):
            result = item.result
            if str(result.type) != "succeeded":
                failures.append(item.custom_id)
                continue
            message = result.message
            text = next(
                (block.text for block in message.content if block.type == "text"), ""
            )
            responses[item.custom_id] = text.strip()
            usages.append(_anthropic_usage(message.usage))
        if failures:
            raise BatchResultError(
                f"Anthropic batch {batch_id} has failed request(s): {sorted(failures)}"
            )
        return BatchOutcome(responses, self._usage_summary(usages)), status

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


def _request_hash(cell: CellSpec, requests: Sequence[BatchPrompt]) -> str:
    payload = {
        "provider": cell.provider,
        "model": cell.endpoint,
        "request_params": cell.request_params,
        "requests": [request.__dict__ for request in requests],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)

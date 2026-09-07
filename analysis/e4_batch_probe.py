"""Run or resume the paid E4 Batch API contract probe.

This submits exactly two short choice requests to each provider's cheapest
configured production endpoint. Re-running retrieves existing batches by their
durable IDs; it never resubmits a provider whose state file already exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from applications.seu_sensitivity_study.batch_client import (  # noqa: E402
    BatchPrompt,
    ProviderBatchClient,
)
from applications.seu_sensitivity_study.config import (  # noqa: E402
    CellSpec,
    SEUSensitivityStudyConfig,
    get_model_spec,
)
from applications.seu_sensitivity_study.study_runner import (  # noqa: E402
    SEUSensitivityStudyRunner,
)

OUTPUT_DIR = (
    ROOT / "applications" / "seu_sensitivity_study" / "results" / "e4_batch_probe"
)
REQUESTS = (
    BatchPrompt(
        custom_id="probe-choice-1",
        prompt="Choose one option. Option 1: A. Option 2: B. Reply exactly ANSWER: 1",
        system_prompt="Return only the requested answer token.",
        temperature=0.0,
        max_tokens=64,
    ),
    BatchPrompt(
        custom_id="probe-choice-2",
        prompt="Choose one option. Option 1: C. Option 2: D. Reply exactly ANSWER: 2",
        system_prompt="Return only the requested answer token.",
        temperature=0.0,
        max_tokens=64,
    ),
)
MODELS = ("gpt-4o-mini", "claude-haiku-4-5")


def _cell(model_name: str) -> CellSpec:
    model = get_model_spec(model_name)
    return CellSpec(
        cell_id=f"e4_batch_probe_{model.slug}",
        model_name=model.name,
        provider=model.provider,
        prompt_condition="neutral",
        pool_id="e4_probe",
        request_params=dict(model.request_params),
        temperature=model.temperature,
        endpoint_id=model.endpoint_id,
        reasoning_token_reserve=model.reasoning_token_reserve,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def main() -> int:
    config = SEUSensitivityStudyConfig(
        results_dir=str(OUTPUT_DIR),
        cache_dir=str(OUTPUT_DIR / "_cache"),
    )
    runner = SEUSensitivityStudyRunner(config)
    statuses: dict[str, Any] = {}

    for model_name in MODELS:
        cell = _cell(model_name)
        state_path = OUTPUT_DIR / "batch_state" / f"{cell.provider}.json"
        result_path = OUTPUT_DIR / "results" / f"{cell.provider}.json"
        if result_path.exists():
            statuses[cell.provider] = "complete"
            continue

        client = ProviderBatchClient(cell)
        outcome = client.process(REQUESTS, state_path=state_path)
        if outcome is None:
            state = json.loads(state_path.read_text())
            statuses[cell.provider] = {
                "status": state["status"],
                "batch_id": state["batch_id"],
            }
            continue

        result = {
            "provider": cell.provider,
            "model": cell.model_name,
            "endpoint": cell.endpoint,
            "responses": outcome.responses,
            "usage": outcome.usage,
        }
        runner._append_usage_event(
            {
                "phase": "e4_batch_probe",
                "collection_mode": "batch",
                "pool_id": "e4_probe",
                "cell_id": cell.cell_id,
                "model": cell.model_name,
                "provider": cell.provider,
                "artifact": str(result_path),
                "records": len(outcome.responses),
                "usage": outcome.usage,
            }
        )
        _write_json(result_path, result)
        statuses[cell.provider] = "complete"

    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0 if all(status == "complete" for status in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

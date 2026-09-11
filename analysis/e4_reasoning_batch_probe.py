"""Run or resume the authorized two-request reasoning-arm Batch probes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from applications.seu_sensitivity_study import prompts as prompts_module  # noqa: E402
from applications.seu_sensitivity_study.batch_client import ProviderBatchClient  # noqa: E402
from applications.seu_sensitivity_study.choice_collection import ChoiceCollector  # noqa: E402
from applications.seu_sensitivity_study.config import (  # noqa: E402
    SEUSensitivityStudyConfig,
)
from applications.seu_sensitivity_study.parsing import parse_choice_response  # noqa: E402
from applications.seu_sensitivity_study.study_runner import (  # noqa: E402
    SEUSensitivityStudyRunner,
)

OUTPUT_DIR = (
    ROOT
    / "applications"
    / "seu_sensitivity_study"
    / "results"
    / "e4_reasoning_batch_probe"
)
MODELS = ("o3-mini", "claude-sonnet-4-5-thinking")
POOL_ID = "venture"
PROMPT_CONDITION = "neutral"
BUDGET_CEILING_USD = 1.0
RESERVATION_PER_ARM_USD = 0.5


def _probe_inputs(config: SEUSensitivityStudyConfig, model_name: str):
    runner = SEUSensitivityStudyRunner(config)
    cell = next(
        cell
        for cell in config.cells
        if cell.pool_id == POOL_ID
        and cell.model_name == model_name
        and cell.prompt_condition == PROMPT_CONDITION
    )
    collector = ChoiceCollector(
        cell=cell,
        problem_set=runner._load_problem_set(POOL_ID),
        prompt_sets=prompts_module.load_prompt_sets(POOL_ID),
        assessments=runner._load_assessments(POOL_ID, model_name),
        llm_client=None,
        max_tokens=config.max_choice_tokens,
    )
    jobs, requests = collector.batch_jobs_and_requests()
    long_problem_id = next(
        problem["id"] for problem, _ in jobs if problem["menu_size"] == 8
    )
    selected = [
        (problem, presentation, request)
        for (problem, presentation), request in zip(jobs, requests)
        if problem["id"] == long_problem_id
    ]
    if len(selected) != config.num_presentations:
        raise RuntimeError("Reasoning probe requires both frozen presentations")
    return cell, selected


def _reserve_probe(state: Mapping[str, Any], cell_id: str) -> None:
    ledger_path = OUTPUT_DIR / "budget_reservations.jsonl"
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        records = []
        if ledger_path.exists():
            for line_number, line in enumerate(ledger_path.read_text().splitlines(), 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid probe budget ledger line {line_number}"
                    ) from error
        matching = [
            record
            for record in records
            if record.get("submission_id") == state["submission_id"]
        ]
        if matching:
            return
        if any(record.get("cell_id") == cell_id for record in records):
            raise RuntimeError(f"Probe budget already reserved for {cell_id}")
        reserved = sum(float(record["reservation_usd"]) for record in records)
        if reserved + RESERVATION_PER_ARM_USD > BUDGET_CEILING_USD:
            raise RuntimeError("Reasoning probe budget ceiling would be exceeded")
        record = {
            "submission_id": state["submission_id"],
            "cell_id": cell_id,
            "request_hash": state["request_hash"],
            "request_count": state["request_count"],
            "reservation_usd": RESERVATION_PER_ARM_USD,
            "budget_ceiling_usd": BUDGET_CEILING_USD,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        ledger = os.open(
            ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        try:
            os.write(ledger, line.encode("utf-8"))
            os.fsync(ledger)
        finally:
            os.close(ledger)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _record_usage(
    runner: SEUSensitivityStudyRunner,
    cell,
    result_path: Path,
    result: Mapping[str, Any],
) -> None:
    runner._append_usage_event(
        {
            "phase": "e4_reasoning_batch_probe",
            "collection_mode": "batch",
            "pool_id": POOL_ID,
            "cell_id": cell.cell_id,
            "model": cell.model_name,
            "provider": cell.provider,
            "artifact": str(result_path),
            "records": len(result["responses"]),
            "usage": result["usage"],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit or retrieve the authorized paid probes; default is dry-run",
    )
    args = parser.parse_args()

    source_results = ROOT / "applications" / "seu_sensitivity_study" / "results"
    config = SEUSensitivityStudyConfig(
        pool_ids=[POOL_ID],
        results_dir=str(source_results),
        cache_dir=str(source_results / "_cache"),
        collection_mode="batch",
    )
    usage_runner = SEUSensitivityStudyRunner(
        SEUSensitivityStudyConfig(
            results_dir=str(OUTPUT_DIR),
            cache_dir=str(OUTPUT_DIR / "_cache"),
        )
    )
    statuses = {}
    for model_name in MODELS:
        cell, selected = _probe_inputs(config, model_name)
        requests = [request for _, _, request in selected]
        offline_client = ProviderBatchClient(cell, sdk_client=object())
        request_hash = offline_client.request_hash(requests)
        plan = {
            "cell_id": cell.cell_id,
            "provider": cell.provider,
            "model": cell.model_name,
            "endpoint": cell.endpoint,
            "problem_id": selected[0][0]["id"],
            "menu_size": selected[0][0]["menu_size"],
            "presentation_ids": [item[1]["presentation_id"] for item in selected],
            "request_count": len(requests),
            "request_hash": request_hash,
            "reservation_usd": RESERVATION_PER_ARM_USD,
        }
        if not args.execute:
            statuses[model_name] = {"status": "dry_run", **plan}
            continue

        state_path = OUTPUT_DIR / "batch_state" / f"{cell.model_name}.json"
        result_path = OUTPUT_DIR / "results" / f"{cell.model_name}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            _validate_cached_result(plan, selected, result, json.loads(state_path.read_text()))
            _record_usage(
                usage_runner, cell, result_path, result
            )
            statuses[model_name] = "complete"
            continue
        client = ProviderBatchClient(
            cell,
            submission_reserver=lambda state, cell_id=cell.cell_id: _reserve_probe(
                state, cell_id
            ),
        )
        outcome = client.process(requests, state_path=state_path)
        if outcome is None:
            state = json.loads(state_path.read_text())
            statuses[model_name] = {
                "status": state["status"],
                "batch_id": state["batch_id"],
                **plan,
            }
            continue

        parsed = []
        for problem, presentation, request in selected:
            response = outcome.responses[request.custom_id]
            position, resolution_path = parse_choice_response(
                response, problem["menu_size"]
            )
            if position is None:
                raise RuntimeError(
                    f"Reasoning probe returned no visible answer for {request.custom_id}"
                )
            parsed.append(
                {
                    "custom_id": request.custom_id,
                    "presentation_id": presentation["presentation_id"],
                    "chosen_position": position,
                    "chosen_item_id": presentation["order"][position - 1],
                    "resolution_path": resolution_path,
                    "response": response,
                }
            )
        result = {**plan, "responses": parsed, "usage": outcome.usage}
        _record_usage(usage_runner, cell, result_path, result)
        _write_json(result_path, result)
        statuses[model_name] = "complete"

    print(json.dumps(statuses, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return 0 if all(status == "complete" for status in statuses.values()) else 2


def _validate_cached_result(plan, selected, result, state) -> None:
    if any(result.get(key) != value for key, value in plan.items()):
        raise RuntimeError("Cached probe request identity does not match current plan")
    if (
        state.get("request_hash") != plan["request_hash"]
        or state.get("request_count") != plan["request_count"]
        or state.get("provider") != plan["provider"]
        or state.get("model") != plan["endpoint"]
        or state.get("status") != "completed"
        or not state.get("batch_id")
        or not state.get("submission_id")
        or any(state.get(key) for key in ("failed_custom_ids", "duplicate_custom_ids", "provider_errors"))
        or state.get("usage") != result.get("usage")
    ):
        raise RuntimeError("Cached probe state or usage does not match completed request")
    expected_ids = {request.custom_id for _, _, request in selected}
    rows = result.get("responses", [])
    if (
        len(rows) != len(expected_ids)
        or {row.get("custom_id") for row in rows} != expected_ids
        or set(state.get("responses", {})) != expected_ids
    ):
        raise RuntimeError("Cached probe response IDs do not match")
    indexed = {row["custom_id"]: row for row in rows}
    for problem, presentation, request in selected:
        text = state["responses"][request.custom_id]
        position, resolution = parse_choice_response(text, problem["menu_size"])
        row = indexed[request.custom_id]
        if position is None or row != {
            "custom_id": request.custom_id,
            "presentation_id": presentation["presentation_id"],
            "chosen_position": position,
            "chosen_item_id": presentation["order"][position - 1],
            "resolution_path": resolution,
            "response": text,
        }:
            raise RuntimeError("Cached probe presentation mapping does not match")


if __name__ == "__main__":
    raise SystemExit(main())
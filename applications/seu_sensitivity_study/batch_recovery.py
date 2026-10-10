"""Explicit offline preparation of the single approved interrupted-attempt replacement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .batch_client import BatchResultError, _single_writer, _write_json


PRIOR_INTENT = "2b2fec7efa3e4968b196bf4ece08efd2"
CELL_ID = "gpt_4o_seu_maximizing_venture"
REQUEST_HASH = "b0d4403aeed60325b82d467071f7320e205e0b08e7ee3c176f63efbaf643ccb1"
PRIOR_WAVE = "production-20261010-wave01"
RESERVATION_USD = 2.2945375
CEILING_USD = 125.07
CAMPAIGN_ROOT = Path(__file__).resolve().parent / "results/production_20261010"
PRIOR_LEDGER_SHA256 = "260c894a42c3847467d5789d6a449fe3e671c66a09e0422115d51e62e004f163"
PRIOR_STATE_SHA256 = "fc58c2da47dc8430f15dd7195accf32f6eeed9a3e2b743cfad447c62277e46e5"
PRIOR_RESERVATION_SHA256 = "55a4589bacb0e5733c109e5abb48e73cf89c43f6f8cee78ab5ac957b6298a663"
PRIOR_MANIFEST_SHA256 = "3ad87744afdb297a5823c24f87629aca02d25556c57192d283c882f72e80ae60"


def json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_authorization(authorization: Any) -> dict[str, Any]:
    fixed = {
        "schema_version": 1,
        "cell_id": CELL_ID,
        "prior_submission_id": PRIOR_INTENT,
        "request_hash": REQUEST_HASH,
        "prior_wave_id": PRIOR_WAVE,
        "request_count": 280,
        "provider": "openai",
        "model": "gpt-4o",
        "reservation_usd": RESERVATION_USD,
        "budget_ceiling_usd": CEILING_USD,
        "campaign_root": str(CAMPAIGN_ROOT.resolve()),
        "prior_ledger_sha256": PRIOR_LEDGER_SHA256,
        "prior_state_sha256": PRIOR_STATE_SHA256,
        "prior_reservation_sha256": PRIOR_RESERVATION_SHA256,
        "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
    }
    required = set(fixed) | {
        "submission_id", "wave_id", "campaign_root", "operator_note",
        "prior_state_sha256", "prior_reservation_sha256",
        "prior_manifest_sha256", "reconciliation_sha256",
    }
    if not isinstance(authorization, dict) or set(authorization) != required:
        raise RuntimeError("Invalid replacement authorization fields")
    if any(type(authorization[key]) is not type(value) or authorization[key] != value
           for key, value in fixed.items()):
        raise RuntimeError("Replacement authorization does not match approved prior intent/cell/hash")
    for key in required - set(fixed):
        if not isinstance(authorization[key], str) or not authorization[key].strip():
            raise RuntimeError(f"Invalid replacement authorization {key}")
    for key in ("prior_state_sha256", "prior_reservation_sha256", "prior_manifest_sha256", "reconciliation_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", authorization[key]):
            raise RuntimeError(f"Invalid replacement authorization {key}")
    if not re.fullmatch(r"[0-9a-f]{32}", authorization["submission_id"]) or authorization["submission_id"] == PRIOR_INTENT:
        raise RuntimeError("Replacement requires one fresh submission ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", authorization["wave_id"]) or authorization["wave_id"] == PRIOR_WAVE:
        raise RuntimeError("Replacement requires a new named wave")
    root = Path(authorization["campaign_root"])
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise RuntimeError("Replacement campaign root must be an absolute canonical path")
    return authorization


def validate_replacement_record(record: Mapping[str, Any], prior: Mapping[str, Any]) -> None:
    authorization = validate_authorization(record.get("replacement_authorization"))
    if json_hash(prior) != authorization["prior_reservation_sha256"]:
        raise RuntimeError("Replacement prior reservation digest mismatch")
    for key in ("cell_id", "request_hash", "request_count", "provider", "model", "reservation_usd"):
        if prior.get(key) != authorization[key] or record.get(key) != authorization[key]:
            raise RuntimeError(f"Replacement reservation {key} mismatch")
    if (prior.get("submission_id") != PRIOR_INTENT or prior.get("wave_id") != PRIOR_WAVE
            or record.get("submission_id") != authorization["submission_id"]
            or record.get("wave_id") != authorization["wave_id"]
            or record.get("budget_ceiling_usd") != CEILING_USD):
        raise RuntimeError("Replacement reservation intent/wave mismatch")


def attempt_directory(root: Path, authorization: Mapping[str, Any]) -> Path:
    return root / "_checkpoints" / "venture" / "batches" / "attempts" / authorization["submission_id"]


def cell_state_path(root: Path, cell: Any) -> Path:
    return root / "_checkpoints" / cell.pool_id / "batches" / f"{cell.cell_id}.json"


def resolve_active_state(root: Path, cell: Any) -> Path:
    """Resolve active intent while the caller holds the canonical cell-state lock."""
    from .study_runner import SEUSensitivityStudyRunner

    records = SEUSensitivityStudyRunner._load_batch_budget_records(root / "batch_budget_reservations.jsonl")
    replacement = next((record for record in records
                        if record["cell_id"] == cell.cell_id and "replacement_authorization" in record), None)
    if replacement is not None:
        return validate_prepared_attempt(root, replacement)
    return cell_state_path(root, cell)


@contextmanager
def active_state_writer(state_path: Path, cell: Any, *, has_reserver: bool):
    """Serialize managed attempts with offline transitions, including direct clients."""
    path = state_path.resolve()
    checkpoints = next((parent for parent in path.parents if parent.name == "_checkpoints"), None)
    if checkpoints is None:
        with _single_writer(path):
            yield
        return
    root = checkpoints.parent
    original = cell_state_path(root, cell)
    with _single_writer(original):
        active = resolve_active_state(root, cell)
        if path != active.resolve():
            raise BatchResultError("Batch attempt is superseded or inactive; refusing state access")
        if active != original and not has_reserver:
            raise BatchResultError("Replacement processing requires an authorized submission reserver")
        with _single_writer(active) if active != original else nullcontext():
            yield


def validate_prepared_attempt(root: Path, record: Mapping[str, Any]) -> Path:
    authorization = validate_authorization(record.get("replacement_authorization"))
    if str(root.resolve()) != authorization["campaign_root"]:
        raise RuntimeError("Replacement campaign root mismatch")
    directory = attempt_directory(root, authorization)
    prior_path = root / "_checkpoints/venture/batches" / f"{CELL_ID}.json"
    bindings = {
        "prior_ledger.jsonl": "prior_ledger_sha256",
        "prior_state.json": "prior_state_sha256",
        "prior_manifest.json": "prior_manifest_sha256",
        "reconciliation.json": "reconciliation_sha256",
    }
    try:
        if file_hash(prior_path) != authorization["prior_state_sha256"]:
            raise RuntimeError("Replacement prior state changed; reconcile before proceeding")
        for name, key in bindings.items():
            if file_hash(directory / name) != authorization[key]:
                raise RuntimeError(f"Replacement archive binding mismatch: {name}")
        if json.loads((directory / "authorization.json").read_text()) != authorization:
            raise RuntimeError("Replacement archived authorization mismatch")
        evidence = json.loads((directory / "reconciliation.json").read_text())
        _validate_evidence_pages(evidence, directory / "evidence")
        prior = json.loads((directory / "prior_reservation.json").read_text())
        validate_replacement_record(record, prior)
        archive = json.loads((directory / "requests.json").read_text())
        manifest = json.loads((directory / "prior_manifest.json").read_text())
        if (json_hash(archive.get("requests")) != REQUEST_HASH
                or archive.get("request_hash") != REQUEST_HASH
                or len(archive["requests"]) != 280
                or manifest.get("generated_hashes", {}).get(f"requests/{CELL_ID}.json") != json_hash(archive)):
            raise RuntimeError("Replacement request archive mismatch")
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text())
        for key in ("submission_id", "request_hash", "request_count", "provider", "model", "cell_id", "wave_id"):
            if state.get(key) != authorization[key]:
                raise RuntimeError(f"Replacement state {key} mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Missing or invalid replacement archive/state") from error
    return state_path


def prepare_replacement_offline(runner: Any, *, reconciliation_path: Path) -> dict[str, Any]:
    """Consume explicit authorization locally; never construct a provider client or submit."""
    authorization = validate_authorization(runner.config.batch_replacement_authorization)
    root = runner.results_dir.resolve()
    if (authorization["campaign_root"] != str(root)
            or runner.config.collection_mode != "batch"
            or runner.config.batch_wave_id != authorization["wave_id"]
            or CELL_ID not in runner.config.batch_wave_cell_ids
            or runner.config.batch_choice_budget_usd != CEILING_USD):
        raise RuntimeError("Replacement config/root/wave/ceiling mismatch")
    state_path = root / "_checkpoints/venture/batches" / f"{CELL_ID}.json"
    ledger_path = root / "batch_budget_reservations.jsonl"
    with _single_writer(state_path), _single_writer(ledger_path):
        records = runner._load_batch_budget_records(ledger_path)
        prior_records = [record for record in records if record["cell_id"] == CELL_ID]
        if len(prior_records) != 1 or any(record["submission_id"] == authorization["submission_id"] for record in records):
            raise RuntimeError("Replacement authorization already consumed or prior reservation missing")
        prior = prior_records[0]
        if file_hash(ledger_path) != authorization["prior_ledger_sha256"]:
            raise RuntimeError("Replacement requires the exact original campaign ledger snapshot")
        if (any(record["wave_id"] == authorization["wave_id"] for record in records)
                or (root / "production_stages" / authorization["wave_id"]).exists()):
            raise RuntimeError("Replacement requires an unused new wave")
        record = {
            **prior,
            "submission_id": authorization["submission_id"],
            "wave_id": authorization["wave_id"],
            "budget_ceiling_usd": CEILING_USD,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "replacement_authorization": authorization,
        }
        validate_replacement_record(record, prior)
        state = json.loads(state_path.read_text())
        if (file_hash(state_path) != authorization["prior_state_sha256"]
                or state.get("status") != "submission_ambiguous"
                or any(state.get(key) for key in ("batch_id", "responses", "result_records", "usage"))):
            raise RuntimeError("Replacement requires unchanged ambiguous state without results or batch ID")
        for key in ("provider", "model", "request_count", "request_hash"):
            if state.get(key) != authorization[key]:
                raise RuntimeError(f"Replacement prior state {key} mismatch")
        if state.get("submission_id") != PRIOR_INTENT:
            raise RuntimeError("Replacement prior state intent mismatch")
        for path in (root / f"pools/venture/choices/{CELL_ID}.json",
                     root / f"_checkpoints/venture/choices_{CELL_ID}.json"):
            if path.exists():
                raise RuntimeError("Replacement refused: cell has choice results/checkpoint")
        stage = root / "production_stages" / PRIOR_WAVE
        manifest_path = stage / "preflight_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (file_hash(manifest_path) != authorization["prior_manifest_sha256"]
                or manifest.get("aggregate_hash") != json_hash({key: value for key, value in manifest.items() if key != "aggregate_hash"})
                or manifest.get("wave_id") != PRIOR_WAVE
                or manifest.get("request_hashes", {}).get(CELL_ID) != REQUEST_HASH):
            raise RuntimeError("Replacement prior stage provenance mismatch")
        archive_path = stage / "requests" / f"{CELL_ID}.json"
        archive = json.loads(archive_path.read_text())
        if (manifest.get("generated_hashes", {}).get(f"requests/{CELL_ID}.json") != json_hash(archive)
                or archive.get("request_hash") != REQUEST_HASH
                or json_hash(archive.get("requests")) != REQUEST_HASH
                or len(archive["requests"]) != 280):
            raise RuntimeError("Replacement original request hash/count mismatch")
        cell = next(cell for cell in runner.config.cells if cell.cell_id == CELL_ID)
        liability = runner._batch_request_liability(archive, cell)
        if liability != prior.get("liability") or liability["reservation_usd"] != RESERVATION_USD:
            raise RuntimeError("Replacement prior liability/reservation mismatch")
        record["liability"] = liability
        if math.fsum(item["reservation_usd"] for item in records) + RESERVATION_USD > CEILING_USD:
            raise RuntimeError("Replacement budget exceeded")
        evidence = json.loads(reconciliation_path.read_text())
        if not isinstance(evidence, dict):
            raise RuntimeError("Replacement reconciliation must be a JSON object")
        if file_hash(reconciliation_path) != authorization["reconciliation_sha256"]:
            raise RuntimeError("Replacement reconciliation digest mismatch")
        expected = {
            "schema_version": 1, "campaign_root": str(root), "provider": "openai",
            "cell_id": CELL_ID, "submission_id": PRIOR_INTENT, "request_hash": REQUEST_HASH,
            "request_count": 280, "prior_state_sha256": authorization["prior_state_sha256"],
            "prior_reservation_sha256": authorization["prior_reservation_sha256"],
            "read_only": True, "pagination_complete": True, "matching_batch_ids": [],
            "conflicts": [],
        }
        if any(type(evidence.get(key)) is not type(value) or evidence.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Replacement reconciliation is incomplete or conflicting; attach/retrieve matches")
        try:
            checked_at = datetime.fromisoformat(evidence["checked_at"])
            age = (datetime.now(timezone.utc) - checked_at).total_seconds()
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Replacement reconciliation timestamp invalid") from error
        if not 0 <= age <= 3600:
            raise RuntimeError("Replacement reconciliation must be fresh (at most one hour old)")
        evidence_sources = _validate_evidence_pages(evidence, reconciliation_path.parent)
        directory = attempt_directory(root, authorization)
        directory.mkdir(parents=True, exist_ok=False)
        for name, source in (("prior_ledger.jsonl", ledger_path), ("prior_state.json", state_path), ("prior_manifest.json", manifest_path),
                             ("requests.json", archive_path), ("reconciliation.json", reconciliation_path)):
            _write_bytes_exclusive(directory / name, source.read_bytes())
        for relative, source in evidence_sources:
            _write_bytes_exclusive(directory / "evidence" / relative, source.read_bytes())
        _write_json(directory / "prior_reservation.json", prior)
        _write_json(directory / "authorization.json", authorization)
        _write_json(directory / "state.json", {
            "schema_version": 2, "status": "reserving",
            **{key: authorization[key] for key in ("submission_id", "request_hash", "request_count", "provider", "model", "cell_id", "wave_id")},
        })
        validate_prepared_attempt(root, record)
        for path in directory.rglob("*"):
            if path.is_file() and path.name != "state.json":
                path.chmod(0o444)
        directories = [path for path in directory.rglob("*") if path.is_dir()]
        directories.extend([directory, directory.parent, directory.parent.parent])
        for path in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(ledger_path, os.O_APPEND | os.O_WRONLY)
        try:
            payload = (json.dumps(record, sort_keys=True) + "\n").encode()
            if os.write(descriptor, payload) != len(payload):
                raise RuntimeError("Short replacement ledger write; stop and reconcile")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record


def _validate_evidence_pages(evidence: Mapping[str, Any], base: Path) -> list[tuple[str, Path]]:
    files = evidence.get("evidence_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Replacement requires hashed raw reconciliation evidence")
    sources = []
    after = None
    seen_ids = set()
    more = True
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Invalid raw reconciliation evidence binding")
        source = (base / relative).resolve()
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not source.is_relative_to(base.resolve()) or file_hash(source) != digest:
            raise RuntimeError("Replacement raw reconciliation evidence mismatch")
        page = json.loads(source.read_text())
        if (not isinstance(page, dict) or page.get("method") != "GET"
                or page.get("path") != "/v1/batches" or page.get("request_after") != after or not more):
            raise RuntimeError("Invalid reconciliation GET pagination chain")
        response = page.get("response")
        if (not isinstance(response, dict) or not isinstance(response.get("data"), list)
                or type(response.get("has_more")) is not bool):
            raise RuntimeError("Incomplete reconciliation page")
        for batch in response["data"]:
            if not isinstance(batch, dict) or not isinstance(batch.get("id"), str) or not batch["id"] or batch["id"] in seen_ids:
                raise RuntimeError("Invalid or duplicate reconciliation batch")
            seen_ids.add(batch["id"])
            metadata = batch.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise RuntimeError("Invalid reconciliation metadata")
            if (metadata.get("submission_id") == PRIOR_INTENT
                    or metadata.get("cell_id") == CELL_ID):
                raise RuntimeError("Reconciliation contains a matching or conflicting batch; attach/retrieve, do not replace")
        more = response["has_more"]
        if more:
            if not response["data"] or response.get("last_id") != response["data"][-1]["id"]:
                raise RuntimeError("Missing reconciliation pagination cursor")
            after = response["last_id"]
        sources.append((relative, source))
    if more:
        raise RuntimeError("Incomplete reconciliation pagination")
    return sources


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
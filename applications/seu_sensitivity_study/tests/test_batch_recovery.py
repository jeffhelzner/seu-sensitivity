"""Offline recovery tests. All transitions use isolated synthetic campaign roots."""

import copy
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from applications.seu_sensitivity_study import batch_recovery as recovery
from applications.seu_sensitivity_study import study_runner
from applications.seu_sensitivity_study.batch_client import (
    BatchPrompt, BatchResultError, ProviderBatchClient, _single_writer, _write_json,
)
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/SDK construction forbidden in recovery tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(ProviderBatchClient, "_new_sdk_client", forbidden)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    config = SEUSensitivityStudyConfig(
        pool_ids=["venture", "hiring"], results_dir=str(tmp_path),
        collection_mode="batch", batch_choice_budget_usd=125.07,
        batch_wave_id="recovery-test-wave", batch_wave_cell_ids=[recovery.CELL_ID],
    )
    runner = study_runner.SEUSensitivityStudyRunner(config)
    cell = next(cell for cell in config.cells if cell.cell_id == recovery.CELL_ID)
    client = ProviderBatchClient(cell, sdk_client=object())
    requests = [BatchPrompt(f"request-{index:05d}", "x" * (3226 + (230 if index == 0 else 0)),
                            "s", 0.0, 64) for index in range(280)]
    archive = {"requests": client.render_requests(requests), "request_hash": client.request_hash(requests)}
    monkeypatch.setattr(recovery, "REQUEST_HASH", archive["request_hash"])
    liability = runner._batch_request_liability(archive, cell)
    assert liability["reservation_usd"] == recovery.RESERVATION_USD
    prior = {
        "submission_id": recovery.PRIOR_INTENT, "cell_id": recovery.CELL_ID,
        "request_hash": recovery.REQUEST_HASH, "request_count": 280,
        "wave_id": recovery.PRIOR_WAVE, "provider": "openai", "model": "gpt-4o",
        "reservation_usd": recovery.RESERVATION_USD, "liability": liability,
        "budget_ceiling_usd": 122.78, "recorded_at": "2026-10-10T00:00:00+00:00",
    }
    completed = {**prior, "submission_id": "completed-intent", "cell_id": "gpt_4o_neutral_venture",
                 "request_hash": "completed-hash", "reservation_usd": 2.2516875}
    completed.pop("liability")
    ledger = tmp_path / "batch_budget_reservations.jsonl"
    ledger.write_text("".join(json.dumps(record) + "\n" for record in (completed, prior)))
    state_path = tmp_path / "_checkpoints/venture/batches" / f"{recovery.CELL_ID}.json"
    _write_json(state_path, {
        "schema_version": 2, "provider": "openai", "model": "gpt-4o",
        "request_count": 280, "request_hash": recovery.REQUEST_HASH,
        "status": "submission_ambiguous", "submission_id": recovery.PRIOR_INTENT,
        "submission_error": "APIConnectionError: Connection error.",
    })
    _write_json(state_path.with_name("gpt_4o_neutral_venture.json"), {
        **{key: completed[key] for key in ("submission_id", "provider", "model", "request_hash", "request_count")},
        "status": "completed", "batch_id": "completed-batch",
        "usage": {"estimated_cost_usd": 0.31856, "usage_complete": True},
    })
    stage = tmp_path / "production_stages" / recovery.PRIOR_WAVE
    _write_json(stage / "requests" / f"{recovery.CELL_ID}.json", archive)
    manifest = {"wave_id": recovery.PRIOR_WAVE, "request_hashes": {recovery.CELL_ID: recovery.REQUEST_HASH},
                "generated_hashes": {f"requests/{recovery.CELL_ID}.json": recovery.json_hash(archive)}}
    manifest["aggregate_hash"] = recovery.json_hash(manifest)
    _write_json(stage / "preflight_manifest.json", manifest)
    monkeypatch.setattr(recovery, "CAMPAIGN_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "PRIOR_LEDGER_SHA256", recovery.file_hash(ledger))
    monkeypatch.setattr(recovery, "PRIOR_STATE_SHA256", recovery.file_hash(state_path))
    monkeypatch.setattr(recovery, "PRIOR_RESERVATION_SHA256", recovery.json_hash(prior))
    monkeypatch.setattr(recovery, "PRIOR_MANIFEST_SHA256", recovery.file_hash(stage / "preflight_manifest.json"))
    evidence_path = tmp_path / "reconciliation/check.json"
    page_path = evidence_path.parent / "pages/0001.json"
    _write_json(page_path, {"method": "GET", "path": "/v1/batches", "request_after": None,
                            "response": {"data": [], "has_more": False}})
    evidence = {
        "schema_version": 1, "campaign_root": str(tmp_path.resolve()), "provider": "openai",
        "cell_id": recovery.CELL_ID, "submission_id": recovery.PRIOR_INTENT,
        "request_hash": recovery.REQUEST_HASH, "request_count": 280,
        "prior_state_sha256": recovery.file_hash(state_path),
        "prior_reservation_sha256": recovery.json_hash(prior),
        "read_only": True, "pagination_complete": True, "matching_batch_ids": [], "conflicts": [],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "evidence_files": {"pages/0001.json": recovery.file_hash(page_path)},
    }
    _write_json(evidence_path, evidence)
    authorization = {
        "schema_version": 1, "cell_id": recovery.CELL_ID,
        "prior_submission_id": recovery.PRIOR_INTENT, "prior_wave_id": recovery.PRIOR_WAVE,
        "request_hash": recovery.REQUEST_HASH, "request_count": 280,
        "provider": "openai", "model": "gpt-4o", "reservation_usd": recovery.RESERVATION_USD,
        "budget_ceiling_usd": 125.07, "submission_id": "a" * 32, "wave_id": config.batch_wave_id,
        "campaign_root": str(tmp_path.resolve()), "operator_note": "Synthetic offline test authorization",
        "prior_ledger_sha256": recovery.file_hash(ledger),
        "prior_state_sha256": recovery.file_hash(state_path),
        "prior_reservation_sha256": recovery.json_hash(prior),
        "prior_manifest_sha256": recovery.file_hash(stage / "preflight_manifest.json"),
        "reconciliation_sha256": recovery.file_hash(evidence_path),
    }
    config.batch_replacement_authorization = authorization
    return SimpleNamespace(runner=runner, cell=cell, requests=requests, archive=archive,
                           prior=prior, ledger=ledger, state_path=state_path, authorization=authorization,
                           evidence_path=evidence_path, page_path=page_path, stage=stage)


def prepare(campaign):
    return recovery.prepare_replacement_offline(campaign.runner, reconciliation_path=campaign.evidence_path)


def test_stale_original_path_is_rejected_before_reconciliation(campaign, monkeypatch):
    selected = Event()
    resume = Event()
    calls = []
    client = ProviderBatchClient(campaign.cell, sdk_client=object())
    monkeypatch.setattr(client, "_reconcile_submission", lambda state: calls.append("reconcile") or "late-original")
    monkeypatch.setattr(client, "_retrieve", lambda batch_id: (None, "in_progress"))

    def stale_collector():
        path = campaign.runner._batch_state_path(campaign.cell)
        selected.set()
        assert resume.wait(5)
        return client.process(campaign.requests, state_path=path)

    before_state = campaign.state_path.read_bytes()
    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(stale_collector)
        try:
            assert selected.wait(5)
            prepare(campaign)
            replacement_path = campaign.runner._batch_state_path(campaign.cell)
            before_replacement = replacement_path.read_bytes()
            before_ledger = campaign.ledger.read_bytes()
        finally:
            resume.set()
        with pytest.raises(BatchResultError, match="superseded"):
            stale.result(timeout=5)
    assert calls == []
    assert campaign.state_path.read_bytes() == before_state
    assert replacement_path.read_bytes() == before_replacement
    assert campaign.ledger.read_bytes() == before_ledger


def update_evidence(campaign, **changes):
    evidence = json.loads(campaign.evidence_path.read_text())
    evidence.update(changes)
    _write_json(campaign.evidence_path, evidence)
    campaign.authorization["reconciliation_sha256"] = recovery.file_hash(campaign.evidence_path)


def test_original_processing_blocks_transition_and_routing(campaign, monkeypatch):
    entered = Event()
    resume = Event()
    client = ProviderBatchClient(campaign.cell, sdk_client=object())

    def reconcile(state):
        entered.set()
        assert resume.wait(5)
        return "late-original"

    monkeypatch.setattr(client, "_reconcile_submission", reconcile)
    monkeypatch.setattr(client, "_retrieve", lambda batch_id: (None, "in_progress"))
    before = campaign.ledger.read_bytes(), campaign.state_path.read_bytes()
    with ThreadPoolExecutor(max_workers=1) as executor:
        processing = executor.submit(client.process, campaign.requests, state_path=campaign.state_path)
        try:
            assert entered.wait(5)
            with pytest.raises(BatchResultError, match="already being modified"):
                prepare(campaign)
            with pytest.raises(BatchResultError, match="already being modified"):
                campaign.runner._batch_state_path(campaign.cell)
            assert (campaign.ledger.read_bytes(), campaign.state_path.read_bytes()) == before
            assert not recovery.attempt_directory(campaign.runner.results_dir, campaign.authorization).exists()
        finally:
            resume.set()
        assert processing.result(timeout=5) is None
    with pytest.raises(RuntimeError, match="unchanged ambiguous state"):
        prepare(campaign)
    assert campaign.ledger.read_bytes() == before[0]


@pytest.mark.parametrize("phase", ["reservation", "submission", "retrieval"])
def test_replacement_processing_serializes_both_attempts(campaign, monkeypatch, phase):
    stale_path = campaign.runner._batch_state_path(campaign.cell)
    prepare(campaign)
    path = campaign.runner._batch_state_path(campaign.cell)
    entered = Event()
    resume = Event()
    submitted = []
    reconciled = []
    monkeypatch.setattr(campaign.runner, "_assert_production_preflight", lambda *args: campaign.archive)

    def pause(current_phase):
        if current_phase == phase:
            entered.set()
            assert resume.wait(5)

    def reserve(state):
        campaign.runner._reserve_batch_budget(state, campaign.cell)
        pause("reservation")

    def submit(*args):
        pause("submission")
        submitted.append(args[-1])
        return "replacement-batch"

    def retrieve(batch_id):
        pause("retrieval")
        return None, "in_progress"

    client = ProviderBatchClient(campaign.cell, sdk_client=object(), submission_reserver=reserve)
    contender = ProviderBatchClient(campaign.cell, sdk_client=object(), submission_reserver=reserve)
    monkeypatch.setattr(client, "_submit", submit)
    monkeypatch.setattr(client, "_retrieve", retrieve)
    monkeypatch.setattr(contender, "_reconcile_submission", lambda state: reconciled.append(state) or "late-original")
    monkeypatch.setattr(contender, "_submit", lambda *args: pytest.fail("Competing submission"))
    if phase == "retrieval":
        assert client.process(campaign.requests, state_path=path) is None
    original_bytes = stale_path.read_bytes()
    ledger_bytes = campaign.ledger.read_bytes()
    with ThreadPoolExecutor(max_workers=1) as executor:
        processing = executor.submit(client.process, campaign.requests, state_path=path)
        try:
            assert entered.wait(5)
            replacement_bytes = path.read_bytes()
            for contender_path in (stale_path, path):
                with pytest.raises(BatchResultError, match="already being modified"):
                    contender.process(campaign.requests, state_path=contender_path)
            with pytest.raises(BatchResultError, match="already being modified"):
                prepare(campaign)
            with pytest.raises(BatchResultError, match="already being modified"):
                campaign.runner._batch_state_path(campaign.cell)
            assert path.read_bytes() == replacement_bytes
            assert stale_path.read_bytes() == original_bytes
            assert campaign.ledger.read_bytes() == ledger_bytes
        finally:
            resume.set()
        assert processing.result(timeout=5) is None
    with pytest.raises(BatchResultError, match="superseded"):
        contender.process(campaign.requests, state_path=stale_path)
    assert reconciled == []
    assert submitted == [campaign.authorization["submission_id"]]
    assert stale_path.read_bytes() == original_bytes
    assert campaign.ledger.read_bytes() == ledger_bytes
    assert campaign.runner._batch_state_path(campaign.cell) == path


def test_direct_replacement_without_reserver_fails_closed(campaign):
    prepare(campaign)
    path = campaign.runner._batch_state_path(campaign.cell)
    before = campaign.state_path.read_bytes(), path.read_bytes(), campaign.ledger.read_bytes()
    client = ProviderBatchClient(campaign.cell, sdk_client=object())
    with pytest.raises(BatchResultError, match="authorized submission reserver"):
        client.process(campaign.requests, state_path=path)
    assert (campaign.state_path.read_bytes(), path.read_bytes(), campaign.ledger.read_bytes()) == before


def test_approved_scope_and_disabled_defaults():
    assert recovery.PRIOR_INTENT == "2b2fec7efa3e4968b196bf4ece08efd2"
    assert recovery.REQUEST_HASH == "b0d4403aeed60325b82d467071f7320e205e0b08e7ee3c176f63efbaf643ccb1"
    config = SEUSensitivityStudyConfig.from_yaml(str(Path(study_runner.__file__).parent / "configs/preregistered.yaml"))
    assert config.batch_choice_budget_usd == 125.07
    assert config.batch_wave_id is None and config.batch_wave_cell_ids == []
    assert config.batch_replacement_authorization is None


def test_transition_keeps_both_reservations_original_bytes_and_status(campaign):
    before_ledger = campaign.ledger.read_bytes()
    before_state = campaign.state_path.read_bytes()
    record = prepare(campaign)
    assert campaign.ledger.read_bytes().startswith(before_ledger)
    assert campaign.state_path.read_bytes() == before_state
    records = campaign.runner._load_batch_budget_records(campaign.ledger)
    assert len(records) == 3 and records[-1] == record
    new_path = campaign.runner._batch_state_path(campaign.cell)
    assert new_path != campaign.state_path
    assert json.loads(new_path.read_text())["status"] == "reserving"
    assert (new_path.parent / "prior_state.json").read_bytes() == before_state
    assert not (new_path.parent / "prior_state.json").stat().st_mode & 0o222
    campaign.runner._write_batch_budget_report()
    report = json.loads((campaign.runner.results_dir / "batch_budget_report.json").read_text())
    assert report["reserved_total_usd"] == pytest.approx(6.8407625)
    assert report["known_usage_estimated_total_usd"] == pytest.approx(0.31856)
    assert [item["status"] for item in report["attempts"]] == ["completed", "submission_ambiguous", "reserving"]
    assert report["unresolved_submission_ids"] == [recovery.PRIOR_INTENT, campaign.authorization["submission_id"]]


@pytest.mark.parametrize("key,value", [
    ("cell_id", "gpt_4o_neutral_venture"), ("prior_submission_id", "wrong"),
    ("request_hash", "0" * 64), ("submission_id", recovery.PRIOR_INTENT),
    ("submission_id", "../escape"), ("wave_id", recovery.PRIOR_WAVE),
    ("prior_wave_id", "wrong"), ("request_count", 279), ("provider", "anthropic"),
    ("model", "gpt-4o-mini"), ("reservation_usd", 1.0), ("budget_ceiling_usd", 126.0),
    ("schema_version", True), ("operator_note", ""), ("campaign_root", "/wrong-root"),
    ("prior_state_sha256", "0" * 64), ("prior_reservation_sha256", "0" * 64),
    ("prior_manifest_sha256", "0" * 64), ("reconciliation_sha256", "0" * 64),
])
def test_wrong_authorization_is_rejected_without_mutation(campaign, key, value):
    before = campaign.ledger.read_bytes(), campaign.state_path.read_bytes()
    campaign.authorization[key] = value
    with pytest.raises(RuntimeError):
        prepare(campaign)
    assert (campaign.ledger.read_bytes(), campaign.state_path.read_bytes()) == before


def test_absent_authorization_and_config_roundtrip(campaign):
    campaign.runner.config.__post_init__()
    assert campaign.runner.config.to_dict()["batch_replacement_authorization"] == campaign.authorization
    campaign.runner.config.batch_replacement_authorization = None
    with pytest.raises(RuntimeError, match="authorization"):
        prepare(campaign)


def test_one_use_and_further_replacement_refused(campaign):
    prepare(campaign)
    before = campaign.ledger.read_bytes()
    with pytest.raises(RuntimeError, match="already consumed"):
        prepare(campaign)
    campaign.authorization = copy.deepcopy(campaign.authorization)
    campaign.authorization["submission_id"] = "b" * 32
    campaign.runner.config.batch_replacement_authorization = campaign.authorization
    with pytest.raises(RuntimeError, match="already consumed"):
        prepare(campaign)
    assert campaign.ledger.read_bytes() == before


@pytest.mark.parametrize("lock_name", ["ledger", "state_path"])
def test_transition_lock_contention(campaign, lock_name):
    before = campaign.ledger.read_bytes()
    with _single_writer(getattr(campaign, lock_name)):
        with pytest.raises(BatchResultError, match="already being modified"):
            prepare(campaign)
    assert campaign.ledger.read_bytes() == before
    assert not recovery.attempt_directory(campaign.runner.results_dir, campaign.authorization).exists()


def test_existing_archive_after_interrupted_transition_is_not_reset(campaign):
    directory = recovery.attempt_directory(campaign.runner.results_dir, campaign.authorization)
    directory.mkdir(parents=True)
    (directory / "evidence.txt").write_text("preserve")
    before = campaign.ledger.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(campaign)
    assert campaign.ledger.read_bytes() == before
    assert (directory / "evidence.txt").read_text() == "preserve"


@pytest.mark.parametrize("status", ["completed", "submitted", "reserving", "failed"])
def test_non_ambiguous_state_refused_even_if_digest_reauthorized(campaign, status, monkeypatch):
    state = json.loads(campaign.state_path.read_text())
    state["status"] = status
    _write_json(campaign.state_path, state)
    campaign.authorization["prior_state_sha256"] = recovery.file_hash(campaign.state_path)
    monkeypatch.setattr(recovery, "PRIOR_STATE_SHA256", campaign.authorization["prior_state_sha256"])
    with pytest.raises(RuntimeError, match="ambiguous state"):
        prepare(campaign)


@pytest.mark.parametrize("artifact", ["pools/venture/choices/", "_checkpoints/venture/choices_"])
def test_existing_choice_artifacts_refuse_transition(campaign, artifact):
    path = campaign.runner.results_dir / f"{artifact}{recovery.CELL_ID}.json"
    _write_json(path, {"choices": []})
    with pytest.raises(RuntimeError, match="choice results/checkpoint"):
        prepare(campaign)


@pytest.mark.parametrize("changes", [
    {"read_only": False}, {"pagination_complete": False}, {"matching_batch_ids": ["batch-found"]},
    {"conflicts": ["uncertain"]}, {"request_hash": "wrong"}, {"submission_id": "wrong"},
    {"checked_at": "not-a-date"}, {"checked_at": "2026-01-01T00:00:00"},
    {"checked_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()},
    {"checked_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
    {"evidence_files": {}},
])
def test_reconciliation_requirements(campaign, changes):
    before = campaign.ledger.read_bytes()
    update_evidence(campaign, **changes)
    with pytest.raises(RuntimeError):
        prepare(campaign)
    assert campaign.ledger.read_bytes() == before


@pytest.mark.parametrize("change", ["match", "conflict", "incomplete", "post", "cursor", "digest"])
def test_raw_page_evidence_is_checked(campaign, change):
    page = json.loads(campaign.page_path.read_text())
    if change in {"match", "conflict"}:
        metadata = {"submission_id": recovery.PRIOR_INTENT} if change == "match" else {"cell_id": recovery.CELL_ID}
        page["response"]["data"] = [{"id": "remote-batch", "metadata": metadata}]
    elif change == "incomplete":
        page["response"].update(data=[{"id": "remote-batch"}], has_more=True, last_id="remote-batch")
    elif change == "post":
        page["method"] = "POST"
    elif change == "cursor":
        page["request_after"] = "missing-page"
    _write_json(campaign.page_path, page)
    update_evidence(campaign, evidence_files={"pages/0001.json": "0" * 64 if change == "digest" else recovery.file_hash(campaign.page_path)})
    with pytest.raises(RuntimeError):
        prepare(campaign)


def test_budget_and_preflight_count_both_attempts(campaign, monkeypatch):
    with pytest.raises(RuntimeError, match="Conflicting Batch budget reservation"):
        campaign.runner._batch_budget_evidence([campaign.cell], {recovery.CELL_ID: campaign.archive})
    prepare(campaign)
    evidence = campaign.runner._batch_budget_evidence([campaign.cell], {recovery.CELL_ID: campaign.archive})
    assert evidence["reserved_total_usd"] == pytest.approx(6.8407625)
    assert evidence["additional_reservation_usd"] == 0
    assert len(evidence["retained_reservations"]) == 3
    assert evidence["wave_reservation_usd"] == recovery.RESERVATION_USD
    remaining = [cell for cell in campaign.runner.config.cells if cell.cell_id not in {recovery.CELL_ID, "gpt_4o_neutral_venture"}]
    full_original = 122.7729167015873
    per_cell = (full_original - 4.546225) / 34
    original_liability = campaign.runner._batch_request_liability
    monkeypatch.setattr(campaign.runner, "_batch_request_liability", lambda archive, cell:
                        original_liability(archive, cell) if cell == campaign.cell else {"reservation_usd": per_cell})
    archives = {cell.cell_id: campaign.archive for cell in remaining + [campaign.cell]}
    evidence = campaign.runner._batch_budget_evidence(remaining + [campaign.cell], archives)
    assert evidence["reserved_total_usd"] + evidence["additional_reservation_usd"] == pytest.approx(125.06745420158731)
    monkeypatch.setattr(campaign.runner, "_batch_request_liability", lambda archive, cell:
                        original_liability(archive, cell) if cell == campaign.cell else {"reservation_usd": per_cell + 1})
    with pytest.raises(RuntimeError, match="budget exceeded"):
        campaign.runner._batch_budget_evidence(remaining + [campaign.cell], archives)


def test_transition_over_budget(campaign, monkeypatch):
    other = {**campaign.prior, "submission_id": "other", "cell_id": "other", "reservation_usd": 120.0}
    with campaign.ledger.open("a") as handle:
        handle.write(json.dumps(other) + "\n")
    before = campaign.ledger.read_bytes()
    monkeypatch.setattr(recovery, "PRIOR_LEDGER_SHA256", recovery.file_hash(campaign.ledger))
    campaign.authorization["prior_ledger_sha256"] = recovery.file_hash(campaign.ledger)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        prepare(campaign)
    assert campaign.ledger.read_bytes() == before


@pytest.mark.parametrize("payload", ["null\n", "[]\n", "true\n", "{", "{}\n"])
def test_malformed_ledger(campaign, payload):
    campaign.ledger.write_text(payload)
    with pytest.raises(RuntimeError, match="Invalid or truncated"):
        prepare(campaign)
    assert campaign.ledger.read_text() == payload


@pytest.mark.parametrize("mutation", ["duplicate", "third", "wrong_hash", "wrong_prior", "amount", "missing_auth", "no_newline"])
def test_malformed_replacement_lineage(campaign, mutation):
    prepare(campaign)
    records = campaign.runner._load_batch_budget_records(campaign.ledger)
    if mutation == "duplicate":
        records.append(records[-1])
    elif mutation == "third":
        records.append({**records[-1], "submission_id": "b" * 32})
    elif mutation == "wrong_hash":
        records[-1]["request_hash"] = "wrong"
    elif mutation == "wrong_prior":
        records[-2]["submission_id"] = "wrong"
    elif mutation == "amount":
        records[-1]["reservation_usd"] = 0.01
    elif mutation == "missing_auth":
        records[-1].pop("replacement_authorization")
    payload = "".join(json.dumps(record) + "\n" for record in records)
    campaign.ledger.write_text(payload.rstrip() if mutation == "no_newline" else payload)
    with pytest.raises(RuntimeError):
        campaign.runner._load_batch_budget_records(campaign.ledger)


@pytest.mark.parametrize("name", ["prior_ledger.jsonl", "prior_state.json", "prior_manifest.json", "prior_reservation.json", "requests.json", "reconciliation.json", "authorization.json", "evidence/pages/0001.json", "state.json"])
def test_attempt_archive_tamper_or_missing_state_blocks_routing(campaign, name):
    prepare(campaign)
    directory = recovery.attempt_directory(campaign.runner.results_dir, campaign.authorization)
    path = directory / name
    path.chmod(0o644)
    path.write_text("{}")
    with pytest.raises(RuntimeError):
        campaign.runner._batch_state_path(campaign.cell)
    client = ProviderBatchClient(campaign.cell, sdk_client=object(), submission_reserver=lambda state: None)
    with pytest.raises(RuntimeError):
        client.process(campaign.requests, state_path=directory / "state.json")


def test_resume_ambiguity_never_creates_remote_and_old_attempt_stays_visible(campaign, monkeypatch):
    prepare(campaign)
    calls = []
    sdk = SimpleNamespace(batches=SimpleNamespace(list=lambda: []))
    client = ProviderBatchClient(campaign.cell, sdk_client=sdk,
                                 submission_reserver=lambda state: campaign.runner._reserve_batch_budget(state, campaign.cell))
    monkeypatch.setattr(campaign.runner, "_assert_production_preflight", lambda *args: campaign.archive)

    def failed_submit(*args):
        calls.append("submit")
        raise RuntimeError("synthetic lost upload")

    monkeypatch.setattr(client, "_submit", failed_submit)
    with pytest.raises(BatchResultError, match="superseded"):
        client.process(campaign.requests, state_path=campaign.state_path)
    assert not calls
    path = campaign.runner._batch_state_path(campaign.cell)
    with pytest.raises(RuntimeError, match="lost upload"):
        client.process(campaign.requests, state_path=path)
    with pytest.raises(BatchResultError, match="found 0"):
        client.process(campaign.requests, state_path=path)
    assert calls == ["submit"]
    campaign.runner._write_batch_budget_report()
    report = json.loads((campaign.runner.results_dir / "batch_budget_report.json").read_text())
    assert [item["status"] for item in report["attempts"]][1:] == ["submission_ambiguous", "submission_ambiguous"]


def test_reservation_is_idempotent_bound_and_preflight_guard_still_required(campaign, monkeypatch):
    prepare(campaign)
    path = campaign.runner._batch_state_path(campaign.cell)
    state = json.loads(path.read_text())
    before = campaign.ledger.read_bytes()
    with pytest.raises(RuntimeError, match="preflight has not been staged"):
        campaign.runner._reserve_batch_budget(state, campaign.cell)
    monkeypatch.setattr(campaign.runner, "_assert_production_preflight", lambda *args: campaign.archive)
    campaign.runner._reserve_batch_budget(state, campaign.cell)
    campaign.runner._reserve_batch_budget(state, campaign.cell)
    with _single_writer(campaign.ledger):
        with pytest.raises(RuntimeError, match="already being modified"):
            campaign.runner._reserve_batch_budget(state, campaign.cell)
    with pytest.raises(RuntimeError, match="already reserved"):
        campaign.runner._reserve_batch_budget({**state, "submission_id": "c" * 32}, campaign.cell)
    campaign.runner.config.batch_replacement_authorization = None
    with pytest.raises(RuntimeError, match="explicit config"):
        campaign.runner._reserve_batch_budget(state, campaign.cell)
    assert campaign.ledger.read_bytes() == before


def test_completed_cell_cannot_be_preflighted_or_reserved(campaign, monkeypatch):
    cell = next(cell for cell in campaign.runner.config.cells if cell.cell_id == "gpt_4o_neutral_venture")
    campaign.runner.config.batch_wave_cell_ids.append(cell.cell_id)
    with pytest.raises(RuntimeError, match="Completed cell"):
        campaign.runner._batch_budget_evidence([cell], {cell.cell_id: campaign.archive})
    monkeypatch.setattr(campaign.runner, "_assert_production_preflight", lambda *args: campaign.archive)
    with pytest.raises(RuntimeError, match="Completed cell"):
        campaign.runner._reserve_batch_budget({"submission_id": "new", "request_hash": "new", "request_count": 280}, cell)


def test_report_rejects_status_from_wrong_request(campaign):
    prepare(campaign)
    state = json.loads(campaign.state_path.read_text())
    state["request_hash"] = "wrong"
    _write_json(campaign.state_path, state)
    with pytest.raises(RuntimeError, match="mismatch|changed"):
        campaign.runner._write_batch_budget_report()


def test_recovery_source_is_in_production_binding():
    root = Path(study_runner.__file__).resolve().parents[2]
    assert Path(recovery.__file__) in study_runner._production_source_paths(root, "h_m01_size_assessment_anchored")


def test_replacement_successful_submit_then_resume_only_retrieves(campaign, monkeypatch):
    prepare(campaign)
    before = campaign.ledger.read_bytes()
    submitted = []
    retrieved = []
    client = ProviderBatchClient(campaign.cell, sdk_client=object(),
                                 submission_reserver=lambda state: campaign.runner._reserve_batch_budget(state, campaign.cell))
    monkeypatch.setattr(campaign.runner, "_assert_production_preflight", lambda *args: campaign.archive)
    monkeypatch.setattr(client, "_submit", lambda *args: submitted.append(args[-1]) or "new-batch")
    monkeypatch.setattr(client, "_retrieve", lambda batch_id: (retrieved.append(batch_id), "in_progress"))
    path = campaign.runner._batch_state_path(campaign.cell)
    assert client.process(campaign.requests, state_path=path) is None
    assert client.process(campaign.requests, state_path=path) is None
    assert client.process(campaign.requests, state_path=path) is None
    assert submitted == [campaign.authorization["submission_id"]]
    assert retrieved == ["new-batch", "new-batch"]
    assert campaign.ledger.read_bytes() == before
    assert json.loads(campaign.state_path.read_text())["status"] == "submission_ambiguous"


def test_removing_completed_reservation_invalidates_replacement_lineage(campaign):
    prepare(campaign)
    lines = campaign.ledger.read_text().splitlines(keepends=True)
    campaign.ledger.write_text("".join(lines[1:]))
    with pytest.raises(RuntimeError, match="prefix"):
        campaign.runner._load_batch_budget_records(campaign.ledger)


def test_completed_state_without_batch_id_never_submits(campaign, monkeypatch):
    state = json.loads(campaign.state_path.read_text())
    state["status"] = "completed"
    _write_json(campaign.state_path, state)
    client = ProviderBatchClient(campaign.cell, sdk_client=object())
    monkeypatch.setattr(client, "_submit", lambda *args: pytest.fail("Completed cell was submitted"))
    with pytest.raises(BatchResultError, match="Completed Batch state"):
        client.process(campaign.requests, state_path=campaign.state_path)


def test_paginated_raw_evidence(campaign):
    first = {"method": "GET", "path": "/v1/batches", "request_after": None,
             "response": {"data": [{"id": "older-batch", "metadata": {"cell_id": "unrelated"}}],
                          "has_more": True, "last_id": "older-batch"}}
    second = {"method": "GET", "path": "/v1/batches", "request_after": "older-batch",
              "response": {"data": [], "has_more": False}}
    _write_json(campaign.page_path, first)
    second_path = campaign.page_path.with_name("0002.json")
    _write_json(second_path, second)
    update_evidence(campaign, evidence_files={"pages/0001.json": recovery.file_hash(campaign.page_path),
                                              "pages/0002.json": recovery.file_hash(second_path)})
    prepare(campaign)
    campaign.runner._batch_state_path(campaign.cell)


def test_attempt_state_missing_never_gets_recreated(campaign):
    prepare(campaign)
    path = campaign.runner._batch_state_path(campaign.cell)
    path.unlink()
    before = campaign.ledger.read_bytes()
    with pytest.raises(RuntimeError, match="Missing or invalid"):
        campaign.runner._batch_state_path(campaign.cell)
    client = ProviderBatchClient(campaign.cell, sdk_client=object(), submission_reserver=lambda state: None)
    with pytest.raises(RuntimeError, match="Missing or invalid"):
        client.process(campaign.requests, state_path=path)
    assert not path.exists() and campaign.ledger.read_bytes() == before
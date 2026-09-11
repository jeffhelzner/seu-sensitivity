import json
import copy

import pytest

from analysis import e4_reasoning_batch_probe as probe
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig
from applications.seu_sensitivity_study.batch_client import ProviderBatchClient


@pytest.mark.parametrize("model_name", probe.MODELS)
def test_cached_live_probe_is_identity_bound(probe_config, model_name):
    cell, selected = probe._probe_inputs(probe_config, model_name)
    result = json.loads((probe.OUTPUT_DIR / "results" / f"{model_name}.json").read_text())
    state = json.loads((probe.OUTPUT_DIR / "batch_state" / f"{model_name}.json").read_text())
    plan = {key: value for key, value in result.items() if key not in {"responses", "usage"}}
    plan["request_hash"] = ProviderBatchClient(cell, sdk_client=object()).request_hash(
        [request for _, _, request in selected]
    )
    probe._validate_cached_result(plan, selected, result, state)
    with pytest.raises(RuntimeError, match="identity"):
        probe._validate_cached_result({**plan, "request_hash": "changed"}, selected, result, state)
    changed = copy.deepcopy(result)
    changed["responses"][0]["chosen_item_id"] = "wrong"
    with pytest.raises(RuntimeError, match="mapping"):
        probe._validate_cached_result(plan, selected, changed, state)
    with pytest.raises(RuntimeError, match="state or usage"):
        probe._validate_cached_result(plan, selected, result, {**state, "batch_id": None})


@pytest.fixture
def probe_config():
    source_results = (
        probe.ROOT / "applications" / "seu_sensitivity_study" / "results"
    )
    return SEUSensitivityStudyConfig(
        pool_ids=[probe.POOL_ID],
        results_dir=str(source_results),
        cache_dir=str(source_results / "_cache"),
        collection_mode="batch",
    )


@pytest.mark.parametrize("model_name", probe.MODELS)
def test_probe_inputs_select_both_presentations_of_frozen_size_eight_problem(
    probe_config, model_name
):
    cell, selected = probe._probe_inputs(probe_config, model_name)

    assert cell.model_name == model_name
    assert len(selected) == 2
    assert {problem["id"] for problem, _, _ in selected} == {"VEN0093"}
    assert {problem["menu_size"] for problem, _, _ in selected} == {8}
    assert [presentation["presentation_id"] for _, presentation, _ in selected] == [
        1,
        2,
    ]
    assert len({request.custom_id for _, _, request in selected}) == 2


def test_probe_reservations_are_idempotent_and_enforce_total_ceiling(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "OUTPUT_DIR", tmp_path)
    first = {
        "submission_id": "submission-1",
        "request_hash": "hash-1",
        "request_count": 2,
    }
    second = {
        "submission_id": "submission-2",
        "request_hash": "hash-2",
        "request_count": 2,
    }
    third = {
        "submission_id": "submission-3",
        "request_hash": "hash-3",
        "request_count": 2,
    }

    probe._reserve_probe(first, "cell-1")
    probe._reserve_probe(first, "cell-1")
    probe._reserve_probe(second, "cell-2")

    records = [
        json.loads(line)
        for line in (tmp_path / "budget_reservations.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert sum(record["reservation_usd"] for record in records) == 1.0

    with pytest.raises(RuntimeError, match="budget ceiling"):
        probe._reserve_probe(third, "cell-3")
"""
End-to-end pipeline tests (build plan A11; study plan §6.1).

Runs every phase offline against a temporary pool with a fake embedding client
and a mock LLM.  The point of interest is the gate: ``choices`` must refuse to
start until the pool's validation gate has passed, because that phase is the
study's main API spend.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from applications.seu_sensitivity_study import pools as pools_module
from applications.seu_sensitivity_study import schemas, study_runner
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig
from applications.seu_sensitivity_study.study_runner import SEUSensitivityStudyRunner


POOL_ID = "tinypool"


def _budget_request_archive(state, cell):
    requests = []
    for index in range(state["request_count"]):
        params = {
            "model": cell.endpoint,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "prompt"}],
        }
        request = {"custom_id": f"request-{index}"}
        if cell.provider == "openai":
            request.update(method="POST", url="/v1/chat/completions", body=params)
        else:
            request["params"] = params
        requests.append(request)
    return {"request_hash": state["request_hash"], "requests": requests}


class FakeEmbeddingClient:
    """Deterministic pseudo-embeddings; no network."""

    def __init__(self, model=None):
        self.model = model

    def embed(self, texts):
        rng = np.random.default_rng(0)
        return [rng.normal(size=24).tolist() for _ in texts]


def _write_pool(tmp_path):
    items = []
    index = 0
    for label, count in (("strong", 10), ("ambiguous", 10), ("weak", 15)):
        for _ in range(count):
            index += 1
            items.append(
                {
                    "id": f"X{index:03d}",
                    "family": "main",
                    "text": f"A {label} templated vignette number {index}.",
                    "quality_label": label,
                    "attributes": {},
                    "matched_key": None,
                }
            )
    pool = {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": POOL_ID,
        "framing": "positive",
        "consequences": ["loss", "break_even", "high_return"],
        "families": {"main": {"description": "test"}},
        "items": items,
    }
    path = tmp_path / "tinypool.json"
    path.write_text(json.dumps(pool))
    return path


def _write_prompts(tmp_path):
    payload = {
        "schema_version": "1.0",
        "pool_id": POOL_ID,
        "assessment": {
            "system_prompt": "You are an analyst.",
            "user_prompt": (
                "Assess {item_text}\nOutcomes:\n{consequence_lines}\n{probability_format}"
            ),
        },
        "choice": {
            "system_prompt": "You are choosing one item.",
            "user_prompt": "{assessments_list}\n{instruction}\nANSWER: n (1-{n_max})",
            "instructions": {
                "neutral": "Choose one.",
                "seu_maximizing": "Choose the item that maximizes subjective expected value.",
                "deliberative": "Think carefully and reason step by step.",
            },
        },
    }
    path = tmp_path / "prompts_tinypool.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _responder(prompt: str, system_prompt: str | None) -> str:
    """Answer assessment prompts with a well-formed line, choices with a token."""
    if "Outcomes:" in prompt or "Possible outcomes:" in prompt:
        return "A short assessment.\nPROBABILITIES: 0.2, 0.5, 0.3"
    return "ANSWER: 1"


@pytest.fixture
def runner(tmp_path, monkeypatch, mock_client_factory):
    spec = pools_module.PoolSpec(
        pool_id=POOL_ID,
        framing="positive",
        families=("main",),
        item_file=_write_pool(tmp_path),
        prompts_file=_write_prompts(tmp_path),
    )
    monkeypatch.setitem(pools_module.POOL_SPECS, POOL_ID, spec)
    monkeypatch.setattr(
        "applications.temperature_study.llm_client.EmbeddingClient", FakeEmbeddingClient
    )
    monkeypatch.setattr(
        study_runner,
        "build_client",
        lambda cell, **kwargs: mock_client_factory(_responder),
    )

    config = SEUSensitivityStudyConfig(
        pool_ids=[POOL_ID],
        problems_per_family={POOL_ID: {"main": 8}},
        results_dir=str(tmp_path / "results"),
        target_dim=6,
    )
    return SEUSensitivityStudyRunner(config)


class TestDryRun:
    def test_reports_call_counts_without_calling(self, runner):
        summary = runner.run(dry_run=True)
        plan = summary["plan"]["pools"][POOL_ID]
        assert plan["menus"] == 8
        assert plan["cells"] == 18
        assert plan["choice_calls"] == 8 * 2 * 18
        assert not (runner.results_dir / "pools").exists()

    def test_totals_are_summed(self, runner):
        summary = runner.run(dry_run=True)
        assert summary["plan"]["totals"]["choice_calls"] == 288


def test_matched_rq5_runs_only_after_both_source_pools(monkeypatch, tmp_path):
    config = SEUSensitivityStudyConfig(
        pool_ids=["venture", "hiring"],
        results_dir=str(tmp_path / "results"),
        stan_model="h_m01_size_assessment_anchored",
    )
    runner = SEUSensitivityStudyRunner(config)
    events = []
    monkeypatch.setattr(
        runner,
        "_run_pool",
        lambda pool_id, *args, **kwargs: events.append(pool_id) or {"stan_data": {}},
    )
    monkeypatch.setattr(
        runner,
        "_phase_matched_rq5_stan_data",
        lambda: events.append("matched_rq5") or {"M_total": 2880},
    )

    summary = runner.run(phases=["stan_data"])

    assert events == ["venture", "hiring", "matched_rq5"]
    assert summary["matched_rq5"]["M_total"] == 2880


class TestPhases:
    def test_design_writes_a_valid_problem_set(self, runner):
        runner.run(phases=["design"])
        path = runner.results_dir / "pools" / POOL_ID / "problems.json"
        design = json.loads(path.read_text())
        pool = json.loads((path.parent / "pool.json").read_text())
        assert schemas.validate_problem_set(design, pool=pool) == []
        assert len(design["problems"]) == 8

    def test_embed_produces_reduced_vectors(self, runner):
        runner.run(phases=["design", "embed"])
        info = json.loads(
            (runner.results_dir / "pools" / POOL_ID / "pca_info.json").read_text()
        )
        assert info["effective_dim"] == 6
        with np.load(runner.results_dir / "pools" / POOL_ID / "embeddings_reduced.npz") as z:
            assert len(z.files) == 35

    def test_assess_writes_one_artefact_per_model(self, runner):
        runner.run(phases=["design", "assess"])
        directory = runner.results_dir / "pools" / POOL_ID / "assessments"
        assert len(list(directory.glob("*.json"))) == 6  # per model, not per cell

    def test_assessment_artefacts_are_neutral(self, runner):
        runner.run(phases=["design", "assess"])
        directory = runner.results_dir / "pools" / POOL_ID / "assessments"
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text())
            assert payload["instruction"] == schemas.ASSESSMENT_INSTRUCTION


class TestGate:
    def test_multi_pool_validation_refreshes_before_later_phases(
        self, tmp_path, monkeypatch
    ):
        config = SEUSensitivityStudyConfig(
            pool_ids=["venture", "hiring"], results_dir=str(tmp_path / "results")
        )
        runner = SEUSensitivityStudyRunner(config)
        calls = []

        def fake_run_pool(pool_id, phases, **kwargs):
            calls.append(("run", pool_id, tuple(phases)))
            return {phase: {} for phase in phases}

        def fake_validate(pool_id):
            calls.append(("refresh", pool_id))
            return {"status": "passed", "passed": True}

        monkeypatch.setattr(runner, "_run_pool", fake_run_pool)
        monkeypatch.setattr(runner, "_phase_validate", fake_validate)

        summary = runner.run(phases=["design", "validate", "choices"])

        assert calls == [
            ("run", "venture", ("design", "validate")),
            ("run", "hiring", ("design", "validate")),
            ("refresh", "venture"),
            ("refresh", "hiring"),
            ("run", "venture", ("choices",)),
            ("run", "hiring", ("choices",)),
        ]
        assert summary["pools"]["venture"]["validate"]["passed"] is True

    def test_choices_are_blocked_before_the_gate(self, runner):
        runner.run(phases=["design", "embed", "assess"])
        with pytest.raises(RuntimeError, match="validation gate"):
            runner.run(phases=["choices"])

    def test_gate_report_is_written_and_blocks_without_embeddings(self, runner):
        runner.run(phases=["design", "validate"])
        report = json.loads(
            (runner.results_dir / "pools" / POOL_ID / "gate_report.json").read_text()
        )
        assert report["passed"] is False
        assert report["status"] == "awaiting_embeddings"

    def test_gate_blocks_when_assessments_are_missing(self, runner):
        runner.run(phases=["design", "embed", "validate"])
        report = json.loads(
            (runner.results_dir / "pools" / POOL_ID / "gate_report.json").read_text()
        )
        assert report["passed"] is False
        assert report["status"] == "awaiting_assessments"
        assert "predictive_validity" in report["failed_checks"]

    def test_assess_runs_before_validate(self):
        """R3 regresses belief on embeddings, so the gate needs assessments."""
        from applications.seu_sensitivity_study.study_runner import PHASES

        assert PHASES.index("assess") < PHASES.index("validate")
        assert PHASES.index("validate") < PHASES.index("choices")

    def test_force_overrides_and_is_recorded(self, runner):
        runner.run(phases=["design", "embed", "assess", "validate"])
        summary = runner.run(phases=["choices"], force=True)
        assert summary["forced"] is True
        assert summary["pools"][POOL_ID]["choices"]

    def test_passing_gate_unblocks_choices(self, runner):
        runner.run(phases=["design", "embed", "assess"])
        gate = runner.results_dir / "pools" / POOL_ID / "gate_report.json"
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(json.dumps({"pool_id": POOL_ID, "status": "passed", "passed": True}))
        summary = runner.run(phases=["choices"])
        assert summary["forced"] is False


class TestFullPipeline:
    def _run_all(self, runner):
        runner.run(phases=["design", "embed", "assess", "validate"])
        return runner.run(phases=["choices", "stan_data"], force=True)

    def test_choice_sets_validate(self, runner):
        self._run_all(runner)
        pool_dir = runner.results_dir / "pools" / POOL_ID
        design = json.loads((pool_dir / "problems.json").read_text())
        paths = list((pool_dir / "choices").glob("*.json"))
        assert len(paths) == 18
        for path in paths:
            payload = json.loads(path.read_text())
            assert schemas.validate_choice_set(payload, problem_set=design) == []

    def test_stan_data_validates_for_both_models(self, runner):
        self._run_all(runner)
        pool_dir = runner.results_dir / "pools" / POOL_ID
        base = json.loads((pool_dir / "stan_data.json").read_text())
        sized = json.loads((pool_dir / "stan_data_size.json").read_text())
        assert schemas.validate_stan_data(base, model="h_m01") == []
        assert schemas.validate_stan_data(sized, model="h_m01_size") == []
        assert base["J"] == 18
        assert base["M_total"] == 18 * 8 * 2

    def test_assessment_anchored_config_writes_fixed_eta_payload(self, runner):
        runner = SEUSensitivityStudyRunner(SEUSensitivityStudyConfig(
            pool_ids=["venture"], problems_per_family={"venture": {"startup": 8}},
            results_dir=str(runner.results_dir), target_dim=6,
            stan_model="h_m01_size_assessment_anchored",
        ))
        with pytest.raises(ValueError, match="full fixed menus"):
            self._run_all(runner)
        stan_summary = runner._phase_stan_data("venture", include_assessment_scale_reference=False)
        pool_dir = runner.results_dir / "pools" / "venture"
        sized = json.loads((pool_dir / "stan_data_size.json").read_text())

        assert "eta" in sized
        assert "w" not in sized
        assert sized["utility_values"] == [0.0, 0.5, 1.0]
        low = json.loads((pool_dir / "stan_data_size_u035.json").read_text())
        high = json.loads((pool_dir / "stan_data_size_u065.json").read_text())
        presentation_1 = json.loads(
            (pool_dir / "stan_data_size_presentation_1.json").read_text()
        )
        presentation_2 = json.loads(
            (pool_dir / "stan_data_size_presentation_2.json").read_text()
        )
        assert low["utility_values"] == [0.0, 0.35, 1.0]
        assert high["utility_values"] == [0.0, 0.65, 1.0]
        assert presentation_1["M_total"] == presentation_2["M_total"] == 18 * 8
        assert presentation_1["I"] == presentation_2["I"]
        assert schemas.validate_stan_data(
            sized, model="h_m01_size_assessment_anchored"
        ) == []
        assert stan_summary["confirmatory_design_rank"] == 8
        assert stan_summary["confirmatory_design_required_rank"] == 8
        assert stan_summary["presentation_sensitivity_files"] == [
            "stan_data_size_presentation_1.json",
            "stan_data_size_presentation_2.json",
        ]
        assert stan_summary["analysis_contract"] == "analysis_contract.json"
        contract = json.loads((pool_dir / "analysis_contract.json").read_text())
        assert contract["primary_decisions_per_pool"] == 10
        for suffix, presentation in (("", None), ("_u035", None), ("_u065", None),
                                     ("_presentation_1", 1), ("_presentation_2", 2)):
            assembly = json.loads((pool_dir / f"stan_data_size{suffix}_assembly_report.json").read_text())
            assert assembly["design_columns"] == stan_summary["design_columns"]
            assert len(assembly["cell_ids"]) == 18
            assert assembly["presentation_id"] == presentation
            assert "assessment_scale_reference" not in assembly
        assert contract["rq5"]["status"] == "confirmatory_fit_validated"

    def test_design_matrix_rows_align_with_cells(self, runner):
        self._run_all(runner)
        base = json.loads(
            (runner.results_dir / "pools" / POOL_ID / "stan_data.json").read_text()
        )
        _, _, cell_ids = runner.config.design_matrix_for_pool(POOL_ID)
        assert base["J"] == len(cell_ids)
        assert len(base["X"]) == len(cell_ids)

    def test_diagnostics_are_written(self, runner):
        self._run_all(runner)
        payload = json.loads(
            (runner.results_dir / "pools" / POOL_ID / "diagnostics.json").read_text()
        )
        assert payload["na_table"]
        assert payload["position_flips"]
        assert "stability_subset" in payload

    def test_na_logs_are_written_per_cell(self, runner):
        self._run_all(runner)
        logs = list((runner.results_dir / "pools" / POOL_ID / "na_logs").glob("*.json"))
        assert len(logs) == 18

    def test_usage_events_survive_later_phase_summaries(self, runner):
        self._run_all(runner)
        path = runner.results_dir / "usage_events.jsonl"
        before = [json.loads(line) for line in path.read_text().splitlines()]

        runner.run(phases=["stan_data"])

        after = [json.loads(line) for line in path.read_text().splitlines()]
        assert after == before
        assert len(after) == 24  # six assessment arms plus 18 choice cells
        assert {event["phase"] for event in after} == {"assess", "choices"}
        assert all(event["collection_mode"] == "synchronous" for event in after)
        assert all("usage" in event and "artifact" in event for event in after)

    def test_batch_liability_uses_utf8_bytes_and_exact_rendered_cap(self, runner):
        cell = next(cell for cell in runner.config.cells if cell.provider == "openai")
        archive = {"requests": [{
            "custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": cell.endpoint, "max_completion_tokens": 9000,
                     "reasoning_effort": "high", "messages": [
                         {"role": "system", "content": "system"},
                         {"role": "user", "content": "\u00e9"},
                     ]},
        }]}
        estimate = runner._batch_request_liability(archive, cell)
        pricing = study_runner.pricing_for(cell.endpoint, cell.provider)
        expected = 0.5 * ((8 + 3 * 1024) * pricing["input"] + 9000 * pricing["output"]) / 1_000_000
        assert estimate["token_liability_usd"] == pytest.approx(expected)
        assert estimate["reservation_usd"] >= expected
        assert estimate["requests"][0]["content_utf8_bytes"] == 8
        assert estimate["requests"][0]["output_token_cap"] == 9000

    @pytest.mark.parametrize("change", [
        {"tools": []}, {"n": 2}, {"response_format": {"type": "json_object"}},
        {"max_tokens": None}, {"max_tokens": True}, {"max_tokens": 0},
        {"max_tokens": 1.5}, {"max_completion_tokens": 100},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"messages": [{"role": "tool", "content": "text"}]},
    ])
    def test_batch_liability_rejects_unsupported_requests(self, runner, change):
        cell = next(cell for cell in runner.config.cells if cell.provider == "openai")
        archive = {"requests": [{
            "custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": cell.endpoint, "max_tokens": 100,
                     "messages": [{"role": "user", "content": "prompt"}], **change},
        }]}
        with pytest.raises(RuntimeError, match="Unsupported|output token cap"):
            runner._batch_request_liability(archive, cell)

    def test_batch_liability_prices_all_rendered_model_arms(self, runner):
        from applications.seu_sensitivity_study.batch_client import BatchPrompt, ProviderBatchClient

        for cell in runner.config.cells_for_pool(POOL_ID):
            client = ProviderBatchClient(cell, sdk_client=object())
            requests = client.render_requests([
                BatchPrompt(custom_id="request-1", prompt="prompt", system_prompt="system", max_tokens=100, temperature=0.0)
            ])
            estimate = runner._batch_request_liability({"requests": requests}, cell)
            params = requests[0]["body" if cell.provider == "openai" else "params"]
            cap = params.get("max_completion_tokens", params.get("max_tokens"))
            pricing = study_runner.pricing_for(cell.endpoint, cell.provider)
            assert estimate["requests"][0]["input_token_bound"] == 12 + 3 * 1024
            assert estimate["requests"][0]["output_token_cap"] == cap
            assert estimate["token_liability_usd"] == pytest.approx(
                0.5 * ((12 + 3 * 1024) * pricing["input"] + cap * pricing["output"]) / 1_000_000
            )

    def test_batch_liability_rejects_missing_cap_and_unpriced_endpoint(self, runner, monkeypatch):
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        state = {"request_count": 1, "request_hash": "hash"}
        archive = _budget_request_archive(state, cell)
        params = archive["requests"][0]["body" if cell.provider == "openai" else "params"]
        del params["max_tokens"]
        with pytest.raises(RuntimeError, match="exactly one output token cap"):
            runner._batch_request_liability(archive, cell)
        params["max_tokens"] = 100
        monkeypatch.setattr(study_runner, "pricing_for", lambda *args: {"input": 0.0, "output": 0.0})
        with pytest.raises(RuntimeError, match="Unpriced"):
            runner._batch_request_liability(archive, cell)

    @pytest.mark.parametrize("change", [
        {"system": [{"type": "text", "text": "system"}]},
        {"tools": []}, {"thinking": {"type": "adaptive"}},
        {"thinking": {"type": "enabled", "budget_tokens": 200}},
        {"messages": [{"role": "user", "content": "text", "cache_control": {"type": "ephemeral"}}]},
    ])
    def test_batch_liability_rejects_unsupported_anthropic_requests(self, runner, change):
        cell = next(cell for cell in runner.config.cells if cell.provider == "anthropic")
        archive = _budget_request_archive({"request_count": 1, "request_hash": "hash"}, cell)
        archive["requests"][0]["params"].update(change)
        with pytest.raises(RuntimeError, match="Unsupported"):
            runner._batch_request_liability(archive, cell)

    def test_batch_budget_blocks_token_overrun_with_unchanged_ceiling(self, runner, monkeypatch):
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id]
        assert runner.config.batch_choice_budget_usd == 31
        state = {"submission_id": "submission-1", "request_count": 1, "request_hash": "hash"}
        archive = _budget_request_archive(state, cell)
        params = archive["requests"][0]["body" if cell.provider == "openai" else "params"]
        params["max_tokens"] = 100_000_000
        monkeypatch.setattr(runner, "_assert_production_preflight", lambda *args: archive)
        assert runner.config.batch_choice_reservation_per_request_usd < 31
        assert runner._batch_request_liability(archive, cell)["token_liability_usd"] > 31
        with pytest.raises(RuntimeError, match="budget exceeded before submission"):
            runner._reserve_batch_budget(state, cell)
        assert not (runner.results_dir / "batch_budget_reservations.jsonl").exists()

    def test_batch_budget_existing_reservation_must_cover_liability(self, runner, monkeypatch):
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id]
        state = {"submission_id": "submission-1", "request_count": 1, "request_hash": "hash"}
        archive = _budget_request_archive(state, cell)
        monkeypatch.setattr(runner, "_assert_production_preflight", lambda *args: archive)
        runner._reserve_batch_budget(state, cell)
        path = runner.results_dir / "batch_budget_reservations.jsonl"
        record = json.loads(path.read_text())
        assert record["liability"] == runner._batch_request_liability(archive, cell)
        record["reservation_usd"] *= 2
        record.pop("liability")
        path.write_text(json.dumps(record) + "\n")
        before = path.read_bytes()
        runner._reserve_batch_budget(state, cell)
        runner._write_batch_budget_report()
        assert path.read_bytes() == before
        report = json.loads((runner.results_dir / "batch_budget_report.json").read_text())
        assert report["reserved_total_usd"] == record["reservation_usd"]
        assert report["liability_assumptions"]["input_overhead_tokens_per_message"] == 1024
        with pytest.raises(RuntimeError, match="Conflicting Batch budget reservation"):
            runner._reserve_batch_budget({**state, "request_hash": "changed"}, cell)
        record["reservation_usd"] /= 4
        path.write_text(json.dumps(record) + "\n")
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match="Conflicting Batch budget reservation"):
            runner._reserve_batch_budget(state, cell)
        assert path.read_bytes() == before

    def test_batch_budget_reservation_is_idempotent_and_enforces_ceiling(
        self, runner, monkeypatch
    ):
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        second_cell = runner.config.cells_for_pool(POOL_ID)[1]
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id, second_cell.cell_id]
        runner.config.batch_choice_reservation_per_request_usd = 0.5
        runner.config.batch_choice_budget_usd = 1.0
        monkeypatch.setattr(runner, "_assert_production_preflight", _budget_request_archive)
        state = {
            "submission_id": "submission-1",
            "request_hash": "hash-1",
            "request_count": 2,
        }
        runner._reserve_batch_budget(state, cell)
        runner._reserve_batch_budget(state, cell)

        path = runner.results_dir / "batch_budget_reservations.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["reservation_usd"] == 1.0

        with pytest.raises(RuntimeError, match="already reserved for cell"):
            runner._reserve_batch_budget(
                {**state, "submission_id": "submission-2"}, cell
            )
        assert len(path.read_text().splitlines()) == 1

        with pytest.raises(RuntimeError, match="budget exceeded before submission"):
            runner._reserve_batch_budget(
                {**state, "submission_id": "submission-3"}, second_cell
            )
        assert len(path.read_text().splitlines()) == 1

    def test_truncated_batch_budget_ledger_blocks_reservation(
        self, runner, monkeypatch
    ):
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id]
        monkeypatch.setattr(runner, "_assert_production_preflight", _budget_request_archive)
        path = runner.results_dir / "batch_budget_reservations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"submission_id": "incomplete"')

        with pytest.raises(RuntimeError, match="Invalid or truncated budget ledger"):
            runner._reserve_batch_budget(
                {
                    "submission_id": "submission-2",
                    "request_hash": "hash-2",
                    "request_count": 1,
                },
                cell,
            )

        path.write_text(json.dumps({"submission_id": "partial"}) + "\n")
        with pytest.raises(RuntimeError, match="Invalid or truncated budget ledger"):
            runner._reserve_batch_budget(
                {
                    "submission_id": "submission-3",
                    "request_hash": "hash-3",
                    "request_count": 1,
                },
                cell,
            )

    def test_batch_budget_report_keeps_missing_actual_cost_unresolved(
        self, runner, monkeypatch
    ):
        cells = runner.config.cells_for_pool(POOL_ID)[:2]
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id for cell in cells]
        monkeypatch.setattr(runner, "_assert_production_preflight", _budget_request_archive)
        runner.config.batch_choice_reservation_per_request_usd = 0.1
        runner.config.batch_choice_budget_usd = 1.0
        for index, cell in enumerate(cells, start=1):
            runner._reserve_batch_budget(
                {
                    "submission_id": f"submission-{index}",
                    "request_hash": f"hash-{index}",
                    "request_count": 1,
                },
                cell,
            )

        state_path = (
            runner.checkpoint_dir / POOL_ID / "batches" / f"{cells[0].cell_id}.json"
        )
        runner._write_json(
            state_path,
            {
                "submission_id": "submission-1",
                "status": "completed",
                "batch_id": "batch-1",
                "usage": {"estimated_cost_usd": 0.04},
            },
        )
        runner._write_batch_budget_report()

        report = json.loads(
            (runner.results_dir / "batch_budget_report.json").read_text()
        )
        assert report["reserved_total_usd"] == pytest.approx(0.2)
        assert report["known_usage_estimated_total_usd"] == pytest.approx(0.04)
        assert report["usage_cost_complete"] is False
        assert report["unresolved_submission_ids"] == ["submission-2"]
        assert report["attempts"][1]["usage_estimated_cost_usd"] is None

        runner._write_json(
            runner.checkpoint_dir / POOL_ID / "batches" / f"{cells[1].cell_id}.json",
            {"submission_id": "submission-2", "status": "completed",
             "usage": {"estimated_cost_usd": 0.02, "usage_complete": False}},
        )
        runner._write_batch_budget_report()
        report = json.loads((runner.results_dir / "batch_budget_report.json").read_text())
        assert report["known_usage_estimated_total_usd"] == pytest.approx(0.06)
        assert report["usage_cost_complete"] is False
        assert report["unresolved_submission_ids"] == ["submission-2"]
        assert report["attempts"][1]["usage_cost_known"] is True

    def test_batch_budget_requires_explicit_cell_wave_authorization(self, runner):
        cells = runner.config.cells_for_pool(POOL_ID)[:2]
        state = {
            "submission_id": "submission-1",
            "request_hash": "hash-1",
            "request_count": 1,
        }
        with pytest.raises(RuntimeError, match="batch_wave_id is unset"):
            runner._reserve_batch_budget(state, cells[0])

        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cells[1].cell_id]
        with pytest.raises(RuntimeError, match="not authorized in wave"):
            runner._reserve_batch_budget(state, cells[0])
        assert not (runner.results_dir / "batch_budget_reservations.jsonl").exists()

    def test_production_preflight_stages_and_binds_exact_inputs(
        self, runner, monkeypatch
    ):
        self._run_all(runner)
        repository_root = study_runner.Path(study_runner.__file__).resolve().parents[2]
        monkeypatch.setattr(
            study_runner,
            "_clean_repository_identity",
            lambda: (repository_root, "test-commit"),
        )
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        runner.config.collection_mode = "batch"
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id]
        gate = {"pool_id": POOL_ID, "status": "passed", "passed": True}
        gate_path = runner.results_dir / "pools" / POOL_ID / "gate_report.json"
        runner._write_json(gate_path, gate)
        monkeypatch.setattr(runner, "_phase_validate", lambda pool_id: gate)

        evidence = runner.run_production_preflight()
        manifest_path = (
            runner.results_dir
            / "production_stages"
            / "test-wave"
            / "preflight_manifest.json"
        )
        staged_problem = manifest_path.parent / "pools" / POOL_ID / "problems.json"
        request_archive = json.loads(
            (manifest_path.parent / "requests" / f"{cell.cell_id}.json").read_text()
        )
        assert manifest_path.exists()
        assert staged_problem.stat().st_mode & 0o222 == 0
        assert manifest_path.parent.stat().st_mode & 0o222 == 0
        assert evidence["request_hashes"][cell.cell_id]
        assert evidence["git_commit"] == "test-commit"
        assert evidence["repository_hashes"]
        assert evidence["toolchain"]["python"]
        assert len(request_archive["requests"]) == 16
        assert len(request_archive["mapping"]) == 16
        assert request_archive["request_hash"] == evidence["request_hashes"][cell.cell_id]

        state = {
            "request_hash": evidence["request_hashes"][cell.cell_id],
            "request_count": len(request_archive["requests"]),
            "submission_id": "submission-1",
        }
        assert runner._assert_production_preflight(state, cell) == request_archive
        assert evidence["batch_budget"]["cells"][cell.cell_id] == runner._batch_request_liability(request_archive, cell)
        with pytest.raises(RuntimeError, match="request count does not match"):
            runner._assert_production_preflight({**state, "request_count": 1}, cell)
        with pytest.raises(RuntimeError, match="Production stage already exists"):
            runner.run_production_preflight()
        with pytest.raises(RuntimeError, match="Rendered Batch request hash"):
            runner._assert_production_preflight(
                {**state, "request_hash": "changed"}, cell
            )

        runner.config.max_choice_tokens += 1
        with pytest.raises(RuntimeError, match="config hash does not match"):
            runner._assert_production_preflight(state, cell)
        runner.config.max_choice_tokens -= 1

        original_toolchain = study_runner.provenance.toolchain_versions
        monkeypatch.setattr(
            study_runner.provenance,
            "toolchain_versions",
            lambda: {"python": "changed"},
        )
        with pytest.raises(RuntimeError, match="toolchain does not match"):
            runner._assert_production_preflight(state, cell)
        monkeypatch.setattr(
            study_runner.provenance, "toolchain_versions", original_toolchain
        )

        staged_problem_content = staged_problem.read_text()
        staged_problem.chmod(0o644)
        staged_problem.write_text(staged_problem_content + "\n")
        with pytest.raises(RuntimeError, match="staged artifact hash does not match"):
            runner._assert_production_preflight(state, cell)
        staged_problem.write_text(staged_problem_content)
        staged_problem.chmod(0o444)

        source_problem = runner.results_dir / "pools" / POOL_ID / "problems.json"
        source_problem.write_text(source_problem.read_text() + "\n")
        with pytest.raises(RuntimeError, match="artifact hash does not match"):
            runner._assert_production_preflight(state, cell)

    @pytest.mark.parametrize("existing_reserved", [0.0, 30.999999])
    def test_production_preflight_blocks_wave_liability_before_staging(
        self, runner, monkeypatch, existing_reserved
    ):
        self._run_all(runner)
        repository_root = study_runner.Path(study_runner.__file__).resolve().parents[2]
        monkeypatch.setattr(study_runner, "_clean_repository_identity", lambda: (repository_root, "test-commit"))
        cells = runner.config.cells_for_pool(POOL_ID)[:2]
        runner.config.collection_mode = "batch"
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id for cell in cells]
        assert runner.config.batch_choice_budget_usd == 31
        gate = {"pool_id": POOL_ID, "status": "passed", "passed": True}
        runner._write_json(runner._pool_dir(POOL_ID) / "gate_report.json", gate)
        monkeypatch.setattr(runner, "_phase_validate", lambda pool_id: gate)
        original_evidence = study_runner.ChoiceCollector.batch_request_evidence

        def request_evidence(collector, client):
            archive = original_evidence(collector, client)
            if not existing_reserved:
                for request in archive["requests"]:
                    params = request["body" if collector.cell.provider == "openai" else "params"]
                    cap_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
                    params[cap_key] = 100_000_000
            return archive

        monkeypatch.setattr(study_runner.ChoiceCollector, "batch_request_evidence", request_evidence)
        ledger_path = runner.results_dir / "batch_budget_reservations.jsonl"
        if existing_reserved:
            ledger_path.write_text(json.dumps({
                "submission_id": "prior-submission", "cell_id": "prior-cell",
                "reservation_usd": existing_reserved,
            }) + "\n")
        before = ledger_path.read_bytes() if ledger_path.exists() else None
        with pytest.raises(RuntimeError, match="budget exceeded before production staging"):
            runner.run_production_preflight()
        assert not (runner.results_dir / "production_stages").exists()
        assert (ledger_path.read_bytes() if ledger_path.exists() else None) == before

    def test_production_preflight_rejects_failed_fresh_gate(self, runner, monkeypatch):
        runner.run(phases=["design", "embed", "assess"])
        repository_root = study_runner.Path(study_runner.__file__).resolve().parents[2]
        monkeypatch.setattr(
            study_runner,
            "_clean_repository_identity",
            lambda: (repository_root, "test-commit"),
        )
        cell = runner.config.cells_for_pool(POOL_ID)[0]
        runner.config.collection_mode = "batch"
        runner.config.batch_wave_id = "test-wave"
        runner.config.batch_wave_cell_ids = [cell.cell_id]
        with pytest.raises(RuntimeError, match="gate did not pass"):
            runner.run_production_preflight()

    def test_rerun_is_idempotent(self, runner):
        self._run_all(runner)
        summary = runner.run(phases=["choices"], force=True)
        assert all(value == "cached" for value in summary["pools"][POOL_ID]["choices"].values())

    def test_batch_cached_choice_rejects_changed_effective_request(
        self, runner, monkeypatch
    ):
        self._run_all(runner)
        choices_dir = runner.results_dir / "pools" / POOL_ID / "choices"
        for path in choices_dir.glob("*.json"):
            payload = json.loads(path.read_text())
            payload["request_hash"] = "old-effective-request"
            path.write_text(json.dumps(payload))

        class FakeBatchClient:
            def __init__(self, cell, **kwargs):
                pass

            def request_hash(self, requests):
                return "changed-effective-request"

        runner.config.collection_mode = "batch"
        monkeypatch.setattr(study_runner, "ProviderBatchClient", FakeBatchClient)
        with pytest.raises(RuntimeError, match="request identity does not match"):
            runner.run(phases=["choices"], force=True)

    def test_manifest_validates(self, runner):
        runner.run(phases=["design", "embed"])
        manifest = runner.write_manifest()
        assert schemas.validate_run_manifest(manifest) == []
        assert POOL_ID in manifest["pool_ids"]
        assert manifest["pca_info"][POOL_ID]["effective_dim"] == 6


class TestPhaseSelection:
    def test_unknown_phase_raises(self, runner):
        with pytest.raises(ValueError, match="Unknown phase"):
            runner.run(phases=["embedd"])

    def test_missing_prerequisite_is_explained(self, runner):
        with pytest.raises(FileNotFoundError, match="'design' phase"):
            runner.run(phases=["embed"])

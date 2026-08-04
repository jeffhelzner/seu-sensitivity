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
    if "Outcomes:" in prompt:
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

    def test_rerun_is_idempotent(self, runner):
        self._run_all(runner)
        summary = runner.run(phases=["choices"], force=True)
        assert all(value == "cached" for value in summary["pools"][POOL_ID]["choices"].values())

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

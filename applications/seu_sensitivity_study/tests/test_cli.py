import argparse
import json

from applications.seu_sensitivity_study import cli
from applications.seu_sensitivity_study.study_runner import SEUSensitivityStudyRunner


def test_preflight_command_stages_locally(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_preflight(self):
        captured["results_dir"] = str(self.results_dir)
        return {"batch_wave_id": "test-wave", "aggregate_hash": "abc123"}

    monkeypatch.setattr(
        SEUSensitivityStudyRunner, "run_production_preflight", fake_preflight
    )
    cli.cmd_preflight(
        argparse.Namespace(config=None, output_dir=str(tmp_path / "results"))
    )

    assert captured["results_dir"] == str(tmp_path / "results")
    assert json.loads(capsys.readouterr().out)["batch_wave_id"] == "test-wave"
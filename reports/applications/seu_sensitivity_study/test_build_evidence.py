"""Focused offline checks for the portable evidence exporter."""

import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_evidence", HERE / "_build_evidence.py")
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = yaml.safe_load((HERE / "data/design_evidence.yml").read_text(encoding="utf-8"))

    def test_bundle_validates(self):
        EXPORTER.validate(self.data)
        EXPORTER.check_outputs(self.data)

    def test_default_works_with_only_portable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "reports/applications/seu_sensitivity_study"
            for relative in ("_build_evidence.py", "data/design_evidence.yml", *self.data["outputs"]):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE / relative, target)
            result = subprocess.run([sys.executable, "-B", str(destination / "_build_evidence.py")],
                                    cwd=directory, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validated (offline)", result.stdout)
            target = destination / "_evidence.qmd"
            target.write_text("changed", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(destination / "_build_evidence.py"), "--check"],
                                    cwd=directory, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Generated output changed", result.stderr)

    def test_rejects_bad_probability(self):
        data = copy.deepcopy(self.data)
        data["pools"]["venture"]["probabilities"]["gpt-4o"]["V001"] = [0.8, 0.8, 0.8]
        with self.assertRaisesRegex(ValueError, "simplex"):
            EXPORTER.validate(data)

    def test_exact_fit_set(self):
        for mutation in ("duplicate", "missing", "midpoint", "scope"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                fits = data["planned_fit_variants"]
                if mutation == "duplicate":
                    fits[-1] = copy.deepcopy(fits[0])
                elif mutation == "missing":
                    fits.pop()
                elif mutation == "midpoint":
                    fits[0]["utility_middle"] = 0.35
                else:
                    fits[0]["scope"] = "other"
                with self.assertRaisesRegex(ValueError, "planned fit set"):
                    EXPORTER.validate(data)
        data = copy.deepcopy(self.data)
        data["planned_fit_variants"].reverse()
        EXPORTER.validate(data)

    def test_rejects_nonexample_probability_mutation(self):
        data = copy.deepcopy(self.data)
        data["figure_integrity"] = EXPORTER.figure_integrity(data)
        original = EXPORTER.numerical_figure_inputs(data)
        data["pools"]["venture"]["probabilities"]["gpt-4o-mini"]["V001"] = [0, 0, 1]
        EXPORTER.validate(data)
        self.assertNotEqual(original[EXPORTER.FIGURES[0]],
                            EXPORTER.numerical_figure_inputs(data)[EXPORTER.FIGURES[0]])
        with self.assertRaisesRegex(ValueError, "Figure input"):
            EXPORTER.check_outputs(data)

    def test_figure_digest_is_canonical(self):
        self.assertEqual(EXPORTER.canonical_digest({"first": 1, "second": 2}),
                         EXPORTER.canonical_digest({"second": 2, "first": 1}))

    def test_header_metadata_and_missing_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chains = root / "chains/main"
            chains.mkdir(parents=True)
            header = ("# model = test_model\n# num_warmup = 500\n# num_samples = 500\n"
                      "# delta = 0.95\n# max_depth = 12\n# thin = 1 (Default)\n"
                      "# num_chains = 1 (Default)\n# id = {chain}\nlp__,value\n0,1\n")
            for chain in (1, 2):
                (chains / f"chain{chain}.csv").write_text(header.format(chain=chain), encoding="utf-8")
            sources = EXPORTER.Sources()
            diagnostics = {"chain_files": ["/old/private/chain1.csv", "/old/private/chain2.csv"]}
            with patch.object(EXPORTER, "ROOT", root):
                sampling = EXPORTER.sampling_metadata(sources, root, diagnostics)
                self.assertEqual(sampling["num_warmup"], 500)
                self.assertEqual(sampling["num_samples"], 500)
                self.assertEqual(sampling["n_chains"], 2)
                self.assertEqual(sampling["max_depth"], 12)
                self.assertEqual(sampling["model"], "test_model")
                EXPORTER.publication_check(sources.manifest())
                self.assertEqual({entry["scope"] for entry in sources.manifest()}, {"cmdstan_header"})
                diagnostics["chain_files"][1] = "missing.csv"
                self.assertEqual(EXPORTER.sampling_metadata(sources, root, diagnostics)["status"], "unavailable")

    def test_rejects_changed_plotting_source(self):
        with patch.object(EXPORTER.inspect, "getsource", return_value="changed plotting implementation"):
            with self.assertRaisesRegex(ValueError, "plotting source"):
                EXPORTER.check_outputs(self.data)

    def test_rejects_changed_exporter(self):
        data = copy.deepcopy(self.data)
        data["exporter_integrity"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Exporter source"):
            EXPORTER.check_outputs(data)

    def test_full_recovery_tables(self):
        rendered = EXPORTER.recovery_tables(self.data)
        for name, campaign in self.data["recovery"].items():
            section = rendered.split(f"#### {name}: sampling and full parameter recovery\n", 1)[1].split("####", 1)[0]
            self.assertIn("Mean 90% interval width", section)
            for parameter, summary in campaign["coverage"].items():
                self.assertIn(f"| `{parameter}` |", section)
                self.assertIn(f"{summary['mean_interval_width']:.4f}", section)

    def test_retained_sampling_schedules(self):
        expected = {"venture": [3, 13, 16, 28, 33], "hiring": [3, 12, 17, 21, 38],
                    "matched_rq5": [2, 4, 6, 8, 13, 14, 16, 17, 26, 38]}
        for name, replacements in expected.items():
            campaign = self.data["recovery"][name]
            self.assertNotIn("sampling", campaign)
            self.assertIn("last_launch_config", campaign)
            for row in campaign["iterations"]:
                sampling = row["sampling"]
                expected_draws = 1000 if row["iteration"] in replacements else 500
                self.assertEqual((sampling["num_warmup"], sampling["num_samples"]), (expected_draws, expected_draws))
                self.assertEqual((sampling["n_chains"], sampling["max_depth"], sampling["delta"]), (4, 12, 0.95))

    def test_scientific_source_scope(self):
        paths = {row["path"] for row in self.data["sources"]}
        for relative in ("models/h_m01_size_assessment_anchored.stan",
                         "applications/seu_sensitivity_study/data_preparation.py",
                         "applications/seu_sensitivity_study/PREREGISTRATION.md"):
            self.assertIn(relative, paths)
            self.assertTrue(any(path.endswith("/repository/" + relative) for path in paths))
        self.assertTrue(all(row["stage_equality"] == "current equals frozen staged source"
                            for row in self.data["scientific_sources"]))

    def test_rejects_absolute_source_path(self):
        data = copy.deepcopy(self.data)
        data["sources"][0]["path"] = "/tmp/private.json"
        with self.assertRaisesRegex(ValueError, "provenance"):
            EXPORTER.validate(data)

    def test_rejects_raw_response_field(self):
        data = copy.deepcopy(self.data)
        data["raw_response"] = "not for publication"
        with self.assertRaisesRegex(ValueError, "Private field"):
            EXPORTER.validate(data)

    def test_rejects_misaligned_example(self):
        data = copy.deepcopy(self.data)
        data["examples"][0]["item_order"].reverse()
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            EXPORTER.validate(data)

    def test_rejects_sampler_failure(self):
        data = copy.deepcopy(self.data)
        data["recovery"]["venture"]["iterations"][15]["min_ess_tail"] = 399
        with self.assertRaisesRegex(ValueError, "sampler gates"):
            EXPORTER.validate(data)

    def test_rejects_incorrect_sign_reversal(self):
        data = copy.deepcopy(self.data)
        contrast = next(row for row in data["contrasts"] if row["contrast_id"] == "rq1_openai_flagship_minus_small")
        contrast["coefficients"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "duplication"):
            EXPORTER.validate(data)

    def test_eta_gap_uses_actual_model_specific_menu(self):
        pool = {"probabilities": {"model": {"first": [0, 1, 0], "second": [0, 0, 1], "unused": [1, 0, 0]}},
                "menus": [{"item_ids": ["first", "second"], "menu_size": 2}]}
        self.assertEqual(EXPORTER.eta_gaps(pool, "model", 2), [0.5])

    def test_import_does_not_load_collection_modules(self):
        self.assertNotIn("applications.seu_sensitivity_study.client", sys.modules)
        self.assertNotIn("applications.seu_sensitivity_study.batch_client", sys.modules)


if __name__ == "__main__":
    unittest.main()
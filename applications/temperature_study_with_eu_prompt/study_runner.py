"""
Study Runner — pipeline orchestration for the Temperature Study with EU Prompt.

Simplified pipeline that reuses data from the base temperature study:
  Load   → Problems, assessments, and embeddings from base study
  Phase 1 → Choice collection with EU-maximization prompt
  Phase 2 → NA filtering + Stan data assembly
  Phase 3 → Model fitting (optional, requires cmdstanpy)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import StudyConfig
from .choice_collector import ChoiceCollector
from .data_preparation import (
    load_base_study_data,
    prepare_stan_data,
    save_stan_data,
    save_na_log,
)
from applications.temperature_study.assessment_collector import AssessmentCollector

logger = logging.getLogger(__name__)


class EUPromptStudyRunner:
    """
    Pipeline for the EU-prompt temperature study.

    Reuses problems, assessments, and embeddings from the base study.
    Only collects new choices using the EU-maximization prompt.
    """

    def __init__(self, config: StudyConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._choice_collector: Optional[ChoiceCollector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_collection: bool = False,
        skip_model_fitting: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the EU-prompt study pipeline.

        Args:
            skip_collection: If True, skip choice collection and load
                existing choices from disk.
            skip_model_fitting: If True, stop after data preparation.

        Returns:
            Summary dict with paths to all output files and run metadata.
        """
        run_start = datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "started_at": run_start.isoformat(),
            "phases": {},
        }

        # ── Load base study data ──
        logger.info("═══ Loading base study data ═══")
        problems, per_temp_assess, per_temp_reduced = load_base_study_data(
            self.config
        )
        summary["base_study"] = {
            "results_dir": self.config.base_study_results_dir,
            "num_problems": len(problems),
            "temperatures": self.config.temperatures,
        }

        if not skip_collection:
            # ── Phase 1: Choice Collection (EU prompt) ──
            per_temp_choices = self._run_choice_collection(
                problems, per_temp_assess
            )
            summary["phases"]["choice_collection"] = {
                "temperatures": list(per_temp_choices.keys()),
                "na_summary": ChoiceCollector.summarize_na(per_temp_choices),
            }
        else:
            per_temp_choices = self._load_choices()
            summary["phases"]["choice_collection"] = {"loaded_from_disk": True}

        # ── Phase 2: Data Preparation ──
        stan_outputs = self._run_data_preparation(
            problems, per_temp_reduced, per_temp_choices
        )
        summary["phases"]["data_preparation"] = stan_outputs

        # ── Phase 3: Model Fitting (optional) ──
        if self.config.fit_models and not skip_model_fitting:
            fit_results = self._run_model_fitting(stan_outputs)
            summary["phases"]["model_fitting"] = fit_results

        # ── Finalise ──
        run_end = datetime.now(timezone.utc)
        summary["finished_at"] = run_end.isoformat()
        summary["duration_seconds"] = (run_end - run_start).total_seconds()

        # Save run summary
        summary_path = self.results_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Run complete. Summary: %s", summary_path)

        return summary

    # ------------------------------------------------------------------
    # Phase 1: Choice Collection
    # ------------------------------------------------------------------

    def _run_choice_collection(
        self,
        problems: List[Dict[str, Any]],
        per_temp_assess: Dict[float, Dict[str, Any]],
    ) -> Dict[float, Dict[str, Any]]:
        logger.info("═══ Phase 1: EU-Prompt Choice Collection ═══")
        collector = self._get_choice_collector()

        # Build assessments_per_temp: {temp -> {claim_id -> text}}
        assessments_per_temp: Dict[float, Dict[str, str]] = {}
        for temp, assess_dict in per_temp_assess.items():
            assessments_per_temp[temp] = (
                AssessmentCollector.get_assessment_texts(assess_dict)
            )

        per_temp = collector.collect_all_temperatures(
            problems, assessments_per_temp=assessments_per_temp
        )

        # Save
        collector.save_all(per_temp)

        logger.info(
            "Choice usage: %s",
            json.dumps(collector.get_usage_summary(), indent=2),
        )
        return per_temp

    # ------------------------------------------------------------------
    # Phase 2: Data Preparation
    # ------------------------------------------------------------------

    def _run_data_preparation(
        self,
        problems: List[Dict[str, Any]],
        per_temp_reduced: Dict[float, Dict[str, np.ndarray]],
        per_temp_choices: Dict[float, Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 2: Data Preparation ═══")
        output_info: Dict[str, Any] = {"per_temperature": {}}

        for temp in self.config.temperatures:
            temp_str = f"{temp:.1f}".replace(".", "_")
            logger.info("  Preparing Stan data for T=%.1f", temp)

            stan_data, na_log = prepare_stan_data(
                self.config,
                problems,
                per_temp_reduced[temp],
                per_temp_choices[temp],
                temp,
            )

            stan_path = save_stan_data(
                stan_data,
                self.results_dir / f"stan_data_T{temp_str}.json",
            )
            na_path = save_na_log(
                na_log,
                self.results_dir / f"na_removal_log_T{temp_str}.json",
            )

            output_info["per_temperature"][str(temp)] = {
                "stan_data": str(stan_path),
                "na_log": str(na_path),
                "M": stan_data["M"],
                "R": stan_data["R"],
                "D": stan_data["D"],
                "na_removed": na_log["removed_observations"],
            }

        return output_info

    # ------------------------------------------------------------------
    # Phase 3: Model Fitting (optional)
    # ------------------------------------------------------------------

    def _run_model_fitting(
        self, phase2_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 3: Model Fitting ═══")

        try:
            from cmdstanpy import CmdStanModel
        except ImportError:
            logger.warning("cmdstanpy not installed — skipping model fitting")
            return {"skipped": True, "reason": "cmdstanpy not available"}

        from analysis.posterior_predictive_checks import PosteriorPredictiveChecker

        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / f"{self.config.stan_model}.stan"
        )
        if not model_path.exists():
            logger.error("Stan model not found: %s", model_path)
            return {"skipped": True, "reason": f"Model file not found: {model_path}"}

        logger.info("Compiling Stan model: %s", model_path)
        model = CmdStanModel(stan_file=str(model_path))

        fit_results: Dict[str, Any] = {}
        for temp_str, info in phase2_output.get("per_temperature", {}).items():
            temp = float(temp_str)
            logger.info("  Fitting model at T=%.1f", temp)

            with open(info["stan_data"]) as f:
                stan_data = json.load(f)

            fit = model.sample(
                data=stan_data,
                chains=4,
                iter_warmup=1000,
                iter_sampling=1000,
                seed=self.config.seed,
                show_progress=True,
            )

            # ── Persist fit artefacts ──
            ts = f"{temp:.1f}".replace(".", "_")
            fit_dir = self.results_dir / f"fit_T{ts}"
            fit_dir.mkdir(parents=True, exist_ok=True)

            fit.save_csvfiles(dir=str(fit_dir))
            logger.info("  Saved CSV draw files → %s", fit_dir)

            summary_df = fit.summary()
            summary_df.to_csv(fit_dir / "summary.csv")
            logger.info("  Saved parameter summary → summary.csv")

            diagnostics_text = fit.diagnose()
            (fit_dir / "diagnostics.txt").write_text(diagnostics_text)
            logger.info("  Saved diagnostics → diagnostics.txt")

            checker = PosteriorPredictiveChecker(fit, stan_data)
            ppc = checker.to_dict()
            with open(fit_dir / "ppc.json", "w") as f:
                json.dump(ppc, f, indent=2)
            ppc_summary = checker.summary()
            (fit_dir / "ppc_summary.txt").write_text(ppc_summary)
            logger.info("  Saved PPC → ppc.json, ppc_summary.txt")

            alpha_draws = fit.stan_variable("alpha")
            np.savez_compressed(
                fit_dir / "alpha_draws.npz", alpha=alpha_draws
            )

            fit_results[temp_str] = {
                "alpha_mean": float(np.mean(alpha_draws)),
                "alpha_median": float(np.median(alpha_draws)),
                "alpha_sd": float(np.std(alpha_draws)),
                "alpha_q05": float(np.quantile(alpha_draws, 0.05)),
                "alpha_q95": float(np.quantile(alpha_draws, 0.95)),
                "output_dir": str(fit_dir),
                "ppc_p_values": ppc["p_values"],
                "diagnostics": diagnostics_text,
            }

            logger.info(
                "  T=%.1f: α mean=%.3f, median=%.3f, 90%% CI=[%.3f, %.3f]",
                temp,
                fit_results[temp_str]["alpha_mean"],
                fit_results[temp_str]["alpha_median"],
                fit_results[temp_str]["alpha_q05"],
                fit_results[temp_str]["alpha_q95"],
            )
            for stat, pval in ppc["p_values"].items():
                logger.info("    PPC %s: p=%.3f", stat, pval)

        # Save cross-temperature summary
        summary_out = {
            temp_str: {
                k: v
                for k, v in info.items()
                if k != "diagnostics"
            }
            for temp_str, info in fit_results.items()
        }
        with open(self.results_dir / "fit_summary.json", "w") as f:
            json.dump(summary_out, f, indent=2)
        logger.info("Saved cross-temperature summary → fit_summary.json")

        return fit_results

    # ------------------------------------------------------------------
    # Fit-only entry point
    # ------------------------------------------------------------------

    def fit_only(self) -> Dict[str, Any]:
        """
        Run model fitting on existing Stan data files.

        Expects ``stan_data_T{temp}.json`` files to already exist in the
        results directory.
        """
        logger.info("Fitting models from existing Stan data in %s", self.results_dir)

        phase2_output: Dict[str, Any] = {"per_temperature": {}}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            stan_path = self.results_dir / f"stan_data_T{ts}.json"
            if not stan_path.exists():
                raise FileNotFoundError(
                    f"Stan data not found: {stan_path}. "
                    "Run the pipeline first."
                )
            phase2_output["per_temperature"][str(temp)] = {
                "stan_data": str(stan_path),
            }

        return self._run_model_fitting(phase2_output)

    # ------------------------------------------------------------------
    # Loading helpers (for skip_collection mode)
    # ------------------------------------------------------------------

    def _load_choices(self) -> Dict[float, Dict[str, Any]]:
        logger.info("Loading saved EU-prompt choices…")
        per_temp: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            path = self.results_dir / f"choices_T{ts}.json"
            per_temp[temp] = ChoiceCollector.load_choices(path)
        return per_temp

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def _get_choice_collector(self) -> ChoiceCollector:
        if self._choice_collector is None:
            self._choice_collector = ChoiceCollector(self.config)
        return self._choice_collector

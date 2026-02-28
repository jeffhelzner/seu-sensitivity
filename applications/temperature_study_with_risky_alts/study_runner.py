"""
Study Runner — pipeline orchestration for the Temperature Study with Risky Alternatives.

Runs the risky-choice extension pipeline:
  Phase 1 → Risky problem generation
  Phase 2 → Risky choice collection (at each temperature)
  Phase 3 → NA filtering + risky Stan data assembly + merge with uncertain data
  Phase 4 → Model fitting (optional, requires cmdstanpy)

Supports checkpoint/resume: each phase saves its outputs and can be
skipped if those outputs already exist.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import StudyConfig
from .risky_problem_generator import RiskyProblemGenerator
from .risky_choice_collector import RiskyChoiceCollector
from .data_preparation import (
    RiskyStanDataBuilder,
    build_augmented_stan_data,
    filter_valid_risky_choices,
    validate_augmented_stan_data,
    save_augmented_stan_data,
    save_risky_na_log,
)

logger = logging.getLogger(__name__)


class RiskyStudyRunner:
    """
    End-to-end pipeline for the risky alternatives extension.

    Instantiate with a :class:`StudyConfig`, then call :meth:`run`.
    """

    def __init__(self, config: StudyConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_study_dir = Path(config.temperature_study_results_dir)

        # Components — lazily created during run()
        self._generator: Optional[RiskyProblemGenerator] = None
        self._choice_collector: Optional[RiskyChoiceCollector] = None

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
        Execute the risky alternatives pipeline.

        Args:
            skip_collection: If True, skip Phases 1–2 and load existing
                data from disk (for re-running data prep only).
            skip_model_fitting: If True, stop after Phase 3.

        Returns:
            Summary dict with paths to all output files and run metadata.
        """
        run_start = datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "started_at": run_start.isoformat(),
            "phases": {},
        }

        # ── Phase 1: Risky Problem Generation ──
        problems = self._run_phase1(skip=skip_collection)
        summary["phases"]["phase1"] = {
            "num_problems": len(problems),
            "output": str(self._problems_path()),
        }

        if not skip_collection:
            # ── Phase 2: Risky Choice Collection ──
            per_temp_choices = self._run_phase2(problems)
            summary["phases"]["phase2_risky_choices"] = {
                "temperatures": list(per_temp_choices.keys()),
                "na_summary": RiskyChoiceCollector.summarize_na(per_temp_choices),
            }
        else:
            per_temp_choices = self._load_risky_choices()

        # ── Phase 3: Data Preparation (merge uncertain + risky) ──
        stan_outputs = self._run_phase3(problems, per_temp_choices)
        summary["phases"]["phase3_data_prep"] = stan_outputs

        # ── Phase 4: Model Fitting (optional) ──
        if self.config.fit_models and not skip_model_fitting:
            fit_results = self._run_phase4(stan_outputs)
            summary["phases"]["phase4_model_fitting"] = fit_results

        # ── Finalise ──
        run_end = datetime.now(timezone.utc)
        summary["finished_at"] = run_end.isoformat()
        summary["duration_seconds"] = (run_end - run_start).total_seconds()

        summary_path = self.results_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Run complete. Summary: %s", summary_path)

        return summary

    # ------------------------------------------------------------------
    # Phase 1: Risky Problem Generation
    # ------------------------------------------------------------------

    def _run_phase1(self, *, skip: bool = False) -> List[Dict[str, Any]]:
        path = self._problems_path()

        if skip and path.exists():
            logger.info("Phase 1 skipped — loading risky problems from %s", path)
            problems, _ = RiskyProblemGenerator.load_problems(path)
            return problems

        logger.info("═══ Phase 1: Risky Problem Generation ═══")
        gen = self._get_generator()
        problems = gen.generate_problems_from_config(self.config)
        gen.save_problems(problems, path)
        return problems

    # ------------------------------------------------------------------
    # Phase 2: Risky Choice Collection
    # ------------------------------------------------------------------

    def _run_phase2(
        self, problems: List[Dict[str, Any]]
    ) -> Dict[float, Dict[str, Any]]:
        logger.info("═══ Phase 2: Risky Choice Collection ═══")
        collector = self._get_choice_collector()

        per_temp = collector.collect_all_temperatures(problems)
        collector.save_all(per_temp)

        logger.info(
            "Risky choice usage: %s",
            json.dumps(collector.get_usage_summary(), indent=2),
        )
        return per_temp

    # ------------------------------------------------------------------
    # Phase 3: Data Preparation (Merge Uncertain + Risky)
    # ------------------------------------------------------------------

    def _run_phase3(
        self,
        problems: List[Dict[str, Any]],
        per_temp_risky_choices: Dict[float, Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 3: Data Preparation ═══")
        output_info: Dict[str, Any] = {"per_temperature": {}}

        # Load risky alternatives for probability data
        gen = self._get_generator()
        risky_alts = gen.alternatives

        builder = RiskyStanDataBuilder(self.config)

        for temp in self.config.temperatures:
            temp_str = f"{temp:.1f}".replace(".", "_")
            logger.info("  Preparing augmented Stan data for T=%.1f", temp)

            # ── 3a. Load existing uncertain Stan data ──
            uncertain_path = (
                self.temp_study_dir / f"stan_data_T{temp_str}.json"
            )
            if not uncertain_path.exists():
                raise FileNotFoundError(
                    f"Uncertain Stan data not found: {uncertain_path}. "
                    "Ensure the original temperature study has been run."
                )
            with open(uncertain_path) as f:
                uncertain_stan_data = json.load(f)
            logger.info(
                "  Loaded uncertain data: M=%d, R=%d, D=%d",
                uncertain_stan_data["M"],
                uncertain_stan_data["R"],
                uncertain_stan_data["D"],
            )

            # ── 3b. Filter risky NAs ──
            valid_risky, na_log = filter_valid_risky_choices(
                per_temp_risky_choices[temp]
            )
            na_path = save_risky_na_log(
                na_log,
                self.results_dir / f"risky_na_removal_log_T{temp_str}.json",
            )

            # ── 3c. Build risky Stan data block ──
            risky_stan_data = builder.build(valid_risky, risky_alts, problems)

            # Validate risky block
            risky_issues = RiskyStanDataBuilder.validate_risky_stan_data(
                risky_stan_data
            )
            if risky_issues:
                for issue in risky_issues:
                    logger.error(
                        "Risky Stan data issue (T=%.1f): %s", temp, issue
                    )
                raise ValueError(
                    f"Risky Stan data validation failed for T={temp}: "
                    f"{risky_issues}"
                )

            # ── 3d. Merge uncertain + risky ──
            augmented = build_augmented_stan_data(
                uncertain_stan_data, risky_stan_data
            )

            # Validate merged data
            aug_issues = validate_augmented_stan_data(augmented)
            if aug_issues:
                for issue in aug_issues:
                    logger.error(
                        "Augmented Stan data issue (T=%.1f): %s", temp, issue
                    )
                raise ValueError(
                    f"Augmented Stan data validation failed for T={temp}: "
                    f"{aug_issues}"
                )

            # ── 3e. Save ──
            aug_path = save_augmented_stan_data(
                augmented,
                self.results_dir / f"stan_data_augmented_T{temp_str}.json",
            )

            output_info["per_temperature"][str(temp)] = {
                "augmented_stan_data": str(aug_path),
                "na_log": str(na_path),
                "M": augmented["M"],
                "R": augmented["R"],
                "D": augmented["D"],
                "N": augmented["N"],
                "S": augmented["S"],
                "risky_na_removed": na_log["removed_observations"],
            }

        return output_info

    # ------------------------------------------------------------------
    # Phase 4: Model Fitting (optional)
    # ------------------------------------------------------------------

    _DEFAULT_MODELS = ["m_11", "m_21", "m_31"]

    def _run_phase4(
        self,
        phase3_output: Dict[str, Any],
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 4: Model Fitting ═══")

        if models is None:
            models = self._DEFAULT_MODELS

        try:
            from cmdstanpy import CmdStanModel
        except ImportError:
            logger.warning("cmdstanpy not installed — skipping model fitting")
            return {"skipped": True, "reason": "cmdstanpy not available"}

        models_dir = (
            Path(__file__).resolve().parent.parent.parent / "models"
        )

        # Fit specified models on each temperature
        fit_results: Dict[str, Any] = {}
        for model_name in models:
            model_path = models_dir / f"{model_name}.stan"
            if not model_path.exists():
                logger.warning(
                    "Stan model not found: %s — skipping", model_path
                )
                continue

            logger.info("Compiling Stan model: %s", model_path)
            model = CmdStanModel(stan_file=str(model_path))

            model_results: Dict[str, Any] = {}
            for temp_str, info in phase3_output.get(
                "per_temperature", {}
            ).items():
                temp = float(temp_str)
                logger.info(
                    "  Fitting %s at T=%.1f", model_name, temp
                )

                with open(info["augmented_stan_data"]) as f:
                    stan_data = json.load(f)

                fit = model.sample(
                    data=stan_data,
                    chains=4,
                    iter_warmup=1000,
                    iter_sampling=1000,
                    seed=self.config.seed,
                    show_progress=True,
                )

                # Save fit artifacts
                ts = f"{temp:.1f}".replace(".", "_")
                fit_dir = (
                    self.results_dir / f"fit_{model_name}_T{ts}"
                )
                fit_dir.mkdir(parents=True, exist_ok=True)

                fit.save_csvfiles(dir=str(fit_dir))
                summary_df = fit.summary()
                summary_df.to_csv(fit_dir / "summary.csv")
                diagnostics_text = fit.diagnose()
                (fit_dir / "diagnostics.txt").write_text(diagnostics_text)

                alpha_draws = fit.stan_variable("alpha")
                np.savez_compressed(
                    fit_dir / "alpha_draws.npz", alpha=alpha_draws
                )

                model_results[temp_str] = {
                    "alpha_mean": float(np.mean(alpha_draws)),
                    "alpha_median": float(np.median(alpha_draws)),
                    "alpha_sd": float(np.std(alpha_draws)),
                    "alpha_q05": float(np.quantile(alpha_draws, 0.05)),
                    "alpha_q95": float(np.quantile(alpha_draws, 0.95)),
                    "output_dir": str(fit_dir),
                    "diagnostics": diagnostics_text,
                }

                logger.info(
                    "  %s T=%.1f: α mean=%.3f, median=%.3f, "
                    "90%% CI=[%.3f, %.3f]",
                    model_name,
                    temp,
                    model_results[temp_str]["alpha_mean"],
                    model_results[temp_str]["alpha_median"],
                    model_results[temp_str]["alpha_q05"],
                    model_results[temp_str]["alpha_q95"],
                )

            fit_results[model_name] = model_results

        # Save cross-model summary
        summary_out = {
            model_name: {
                temp_str: {
                    k: v for k, v in info.items() if k != "diagnostics"
                }
                for temp_str, info in model_results.items()
            }
            for model_name, model_results in fit_results.items()
            if isinstance(model_results, dict) and not model_results.get("skipped")
        }
        with open(self.results_dir / "fit_summary.json", "w") as f:
            json.dump(summary_out, f, indent=2)
        logger.info("Saved cross-model fit summary → fit_summary.json")

        return fit_results

    # ------------------------------------------------------------------
    # Fit-only entry point
    # ------------------------------------------------------------------

    def fit_only(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run Phase 4 (model fitting) on existing augmented Stan data files.

        Args:
            models: List of model names to fit
                    (default: ["m_11", "m_21", "m_31"]).

        Returns:
            Dict with per-model per-temperature fit summaries.
        """
        logger.info(
            "Fitting models from existing augmented Stan data in %s",
            self.results_dir,
        )

        phase3_output: Dict[str, Any] = {"per_temperature": {}}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            aug_path = self.results_dir / f"stan_data_augmented_T{ts}.json"
            if not aug_path.exists():
                raise FileNotFoundError(
                    f"Augmented Stan data not found: {aug_path}. "
                    "Run the pipeline first."
                )
            phase3_output["per_temperature"][str(temp)] = {
                "augmented_stan_data": str(aug_path),
            }

        return self._run_phase4(phase3_output, models=models)

    # ------------------------------------------------------------------
    # Loading helpers (for skip_collection mode)
    # ------------------------------------------------------------------

    def _load_risky_choices(self) -> Dict[float, Dict[str, Any]]:
        logger.info("Loading saved risky choices…")
        per_temp: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            path = self.results_dir / f"risky_choices_T{ts}.json"
            per_temp[temp] = RiskyChoiceCollector.load_choices(path)
        return per_temp

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def _get_generator(self) -> RiskyProblemGenerator:
        if self._generator is None:
            self._generator = RiskyProblemGenerator.from_config(self.config)
        return self._generator

    def _get_choice_collector(self) -> RiskyChoiceCollector:
        if self._choice_collector is None:
            self._choice_collector = RiskyChoiceCollector(
                self.config, self._get_generator()
            )
        return self._choice_collector

    # ------------------------------------------------------------------
    # Path conventions
    # ------------------------------------------------------------------

    def _problems_path(self) -> Path:
        return self.results_dir / "risky_problems.json"

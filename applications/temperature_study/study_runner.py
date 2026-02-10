"""
Study Runner — main pipeline orchestration for the Temperature Study.

Runs the full pipeline end-to-end:
  Phase 1 → Problem generation
  Phase 2 → Deliberation collection  +  Choice collection
  Phase 3 → Pooled PCA  +  NA filtering  +  Stan data assembly
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
from .problem_generator import ProblemGenerator
from .deliberation_collector import DeliberationCollector
from .choice_collector import ChoiceCollector
from .data_preparation import (
    EmbeddingReducer,
    StanDataBuilder,
    filter_valid_choices,
    save_stan_data,
    save_na_log,
    save_reduced_embeddings,
)

logger = logging.getLogger(__name__)


class TemperatureStudyRunner:
    """
    End-to-end pipeline for the temperature study.

    Instantiate with a :class:`StudyConfig`, then call :meth:`run`.
    """

    def __init__(self, config: StudyConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Components — lazily created during run()
        self._generator: Optional[ProblemGenerator] = None
        self._delib_collector: Optional[DeliberationCollector] = None
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
        Execute the full study pipeline.

        Args:
            skip_collection: If True, skip Phases 1–2 and load existing
                data from disk (for re-running data prep / analysis only).
            skip_model_fitting: If True, stop after Phase 3 (data
                preparation).  Overrides ``config.fit_models``.

        Returns:
            Summary dict with paths to all output files and run metadata.
        """
        run_start = datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "started_at": run_start.isoformat(),
            "phases": {},
        }

        # ── Phase 1: Problem Generation ──
        problems = self._run_phase1(skip=skip_collection)
        summary["phases"]["phase1"] = {
            "num_problems": len(problems),
            "output": str(self._problems_path()),
        }

        if not skip_collection:
            # ── Phase 2a: Deliberation Collection ──
            per_temp_delibs, per_temp_raw_embs = self._run_phase2a(problems)
            summary["phases"]["phase2a_deliberation"] = {
                "temperatures": list(per_temp_delibs.keys()),
                "deliberations_per_temp": {
                    str(t): d["total_deliberations"]
                    for t, d in per_temp_delibs.items()
                },
            }

            # ── Phase 2b: Choice Collection ──
            per_temp_choices = self._run_phase2b(problems)
            summary["phases"]["phase2b_choices"] = {
                "temperatures": list(per_temp_choices.keys()),
                "na_summary": ChoiceCollector.summarize_na(per_temp_choices),
            }
        else:
            per_temp_delibs, per_temp_raw_embs = self._load_deliberations()
            per_temp_choices = self._load_choices()

        # ── Phase 3: Data Preparation ──
        stan_outputs = self._run_phase3(
            problems, per_temp_delibs, per_temp_raw_embs, per_temp_choices
        )
        summary["phases"]["phase3_data_prep"] = stan_outputs

        # ── Phase 4: Model Fitting (optional) ──
        if self.config.fit_models and not skip_model_fitting:
            fit_results = self._run_phase4(stan_outputs)
            summary["phases"]["phase4_model_fitting"] = fit_results

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
    # Phase 1: Problem Generation
    # ------------------------------------------------------------------

    def _run_phase1(self, *, skip: bool = False) -> List[Dict[str, Any]]:
        path = self._problems_path()

        if skip and path.exists():
            logger.info("Phase 1 skipped — loading problems from %s", path)
            problems, _ = ProblemGenerator.load_problems(path)
            return problems

        logger.info("═══ Phase 1: Problem Generation ═══")
        gen = self._get_generator()
        problems = gen.generate_problems_from_config(self.config)
        gen.save_problems(problems, path)
        return problems

    # ------------------------------------------------------------------
    # Phase 2a: Deliberation Collection
    # ------------------------------------------------------------------

    def _run_phase2a(
        self, problems: List[Dict[str, Any]]
    ) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, np.ndarray]]]:
        logger.info("═══ Phase 2a: Deliberation Collection ═══")
        collector = self._get_delib_collector()

        per_temp_delibs: Dict[float, Dict[str, Any]] = {}
        per_temp_raw: Dict[float, Dict[str, np.ndarray]] = {}

        for temp in self.config.temperatures:
            logger.info("  Collecting deliberations at T=%.1f", temp)
            delib_dict, raw_embs = collector.collect_temperature(problems, temp)
            per_temp_delibs[temp] = delib_dict
            per_temp_raw[temp] = raw_embs

            # Save immediately (checkpoint)
            collector.save_deliberations(temp, delib_dict, raw_embs)

        logger.info(
            "Deliberation usage: %s",
            json.dumps(collector.get_usage_summary(), indent=2),
        )
        return per_temp_delibs, per_temp_raw

    # ------------------------------------------------------------------
    # Phase 2b: Choice Collection
    # ------------------------------------------------------------------

    def _run_phase2b(
        self, problems: List[Dict[str, Any]]
    ) -> Dict[float, Dict[str, Any]]:
        logger.info("═══ Phase 2b: Choice Collection ═══")
        collector = self._get_choice_collector()
        per_temp = collector.collect_all_temperatures(problems)

        # Save
        collector.save_all(per_temp)

        logger.info(
            "Choice usage: %s",
            json.dumps(collector.get_usage_summary(), indent=2),
        )
        return per_temp

    # ------------------------------------------------------------------
    # Phase 3: Data Preparation
    # ------------------------------------------------------------------

    def _run_phase3(
        self,
        problems: List[Dict[str, Any]],
        per_temp_delibs: Dict[float, Dict[str, Any]],
        per_temp_raw_embs: Dict[float, Dict[str, np.ndarray]],
        per_temp_choices: Dict[float, Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 3: Data Preparation ═══")
        output_info: Dict[str, Any] = {"per_temperature": {}}

        # ── 3a. Pooled PCA ──
        logger.info("  Fitting pooled PCA (target_dim=%d)", self.config.target_dim)
        reducer = EmbeddingReducer(
            target_dim=self.config.target_dim, seed=self.config.seed
        )
        per_temp_reduced = reducer.fit_transform_pooled(per_temp_raw_embs)
        output_info["pca_summary"] = reducer.get_summary()

        # Save reduced embeddings per temperature
        for temp, reduced in per_temp_reduced.items():
            temp_str = f"{temp:.1f}".replace(".", "_")
            save_reduced_embeddings(
                reduced,
                self.results_dir / f"embeddings_reduced_T{temp_str}.npz",
            )

        # ── 3b. Per-temperature: NA filter + Stan data ──
        builder = StanDataBuilder(self.config)

        for temp in self.config.temperatures:
            temp_str = f"{temp:.1f}".replace(".", "_")
            logger.info("  Preparing Stan data for T=%.1f", temp)

            # Filter NAs
            valid_entries, na_log = filter_valid_choices(per_temp_choices[temp])
            na_path = save_na_log(
                na_log,
                self.results_dir / f"na_removal_log_T{temp_str}.json",
            )

            # Build Stan data
            reduced_embs = per_temp_reduced[temp]
            stan_data = builder.build(valid_entries, reduced_embs, problems)

            # Validate
            issues = StanDataBuilder.validate_stan_data(stan_data)
            if issues:
                for issue in issues:
                    logger.error("Stan data issue (T=%.1f): %s", temp, issue)
                raise ValueError(
                    f"Stan data validation failed for T={temp}: {issues}"
                )

            stan_path = save_stan_data(
                stan_data,
                self.results_dir / f"stan_data_T{temp_str}.json",
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
    # Phase 4: Model Fitting (optional)
    # ------------------------------------------------------------------

    def _run_phase4(
        self, phase3_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info("═══ Phase 4: Model Fitting ═══")

        try:
            from cmdstanpy import CmdStanModel
        except ImportError:
            logger.warning("cmdstanpy not installed — skipping model fitting")
            return {"skipped": True, "reason": "cmdstanpy not available"}

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
        for temp_str, info in phase3_output.get("per_temperature", {}).items():
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

            # Save fit summary
            ts = f"{temp:.1f}".replace(".", "_")
            fit_dir = self.results_dir / f"fit_T{ts}"
            fit_dir.mkdir(parents=True, exist_ok=True)

            alpha_draws = fit.stan_variable("alpha")
            fit_results[temp_str] = {
                "alpha_mean": float(np.mean(alpha_draws)),
                "alpha_median": float(np.median(alpha_draws)),
                "alpha_sd": float(np.std(alpha_draws)),
                "alpha_q05": float(np.quantile(alpha_draws, 0.05)),
                "alpha_q95": float(np.quantile(alpha_draws, 0.95)),
                "output_dir": str(fit_dir),
            }

            logger.info(
                "  T=%.1f: α mean=%.3f, median=%.3f, 90%% CI=[%.3f, %.3f]",
                temp,
                fit_results[temp_str]["alpha_mean"],
                fit_results[temp_str]["alpha_median"],
                fit_results[temp_str]["alpha_q05"],
                fit_results[temp_str]["alpha_q95"],
            )

        return fit_results

    # ------------------------------------------------------------------
    # Fit-only entry point
    # ------------------------------------------------------------------

    def fit_only(self) -> Dict[str, Any]:
        """
        Run Phase 4 (model fitting) on existing Stan data files.

        Expects ``stan_data_T{temp}.json`` files to already exist in the
        results directory (produced by a prior ``run`` or ``prepare``).

        Returns:
            Dict with per-temperature fit summaries.
        """
        logger.info("Fitting models from existing Stan data in %s", self.results_dir)

        # Build a minimal phase3 output dict from what's on disk
        phase3_output: Dict[str, Any] = {"per_temperature": {}}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            stan_path = self.results_dir / f"stan_data_T{ts}.json"
            if not stan_path.exists():
                raise FileNotFoundError(
                    f"Stan data not found: {stan_path}. "
                    "Run the pipeline with 'run --skip-fitting' or 'prepare' first."
                )
            phase3_output["per_temperature"][str(temp)] = {
                "stan_data": str(stan_path),
            }

        return self._run_phase4(phase3_output)

    # ------------------------------------------------------------------
    # Loading helpers (for skip_collection mode)
    # ------------------------------------------------------------------

    def _load_deliberations(
        self,
    ) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, np.ndarray]]]:
        logger.info("Loading saved deliberations and embeddings…")
        per_temp_delibs: Dict[float, Dict[str, Any]] = {}
        per_temp_raw: Dict[float, Dict[str, np.ndarray]] = {}

        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            delib_path = self.results_dir / f"deliberations_T{ts}.json"
            emb_path = self.results_dir / f"embeddings_raw_T{ts}.npz"

            with open(delib_path) as f:
                per_temp_delibs[temp] = json.load(f)

            data = np.load(emb_path)
            per_temp_raw[temp] = {k: data[k] for k in data.files}

        return per_temp_delibs, per_temp_raw

    def _load_choices(self) -> Dict[float, Dict[str, Any]]:
        logger.info("Loading saved choices…")
        per_temp: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            ts = f"{temp:.1f}".replace(".", "_")
            path = self.results_dir / f"choices_T{ts}.json"
            per_temp[temp] = ChoiceCollector.load_choices(path)
        return per_temp

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def _get_generator(self) -> ProblemGenerator:
        if self._generator is None:
            self._generator = ProblemGenerator.from_config(self.config)
        return self._generator

    def _get_delib_collector(self) -> DeliberationCollector:
        if self._delib_collector is None:
            self._delib_collector = DeliberationCollector(
                self.config, self._get_generator()
            )
        return self._delib_collector

    def _get_choice_collector(self) -> ChoiceCollector:
        if self._choice_collector is None:
            self._choice_collector = ChoiceCollector(
                self.config, self._get_generator()
            )
        return self._choice_collector

    # ------------------------------------------------------------------
    # Path conventions
    # ------------------------------------------------------------------

    def _problems_path(self) -> Path:
        return self.results_dir / "problems.json"

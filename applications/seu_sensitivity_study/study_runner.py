"""
Pipeline orchestration (study plan §6.1; build plan A11).

Runs the collection pipeline per pool, as a sequence of resumable phases with
one **blocking gate** between them::

    design -> embed -> validate (GATE) -> assess -> choices -> stan_data

Two ordering facts are load-bearing.

*Assessments precede the gate's expensive half.*  Assessment calls are ~4% of
the study's API spend because they are collected once per model x pool (B2), so
the §5 predictive-validity check -- which regresses parsed assessment
probabilities on item embeddings -- can be run before committing to the ~10.8k
choice calls.

*The gate genuinely blocks.*  ``choices`` refuses to run unless the pool's gate
report says it passed.  Until the Phase B item-validation module lands, the
gate reports ``not_implemented`` and ``choices`` will not start without an
explicit ``force=True``, which is recorded in the run summary.  A gate that
could be skipped by forgetting about it is not a gate.

Layout under ``results/``::

    run_manifest.json
    run_summary.json
    pools/<pool_id>/
        pool.json  problems.json  pca_info.json  gate_report.json
        embeddings_raw.npz  embeddings_reduced.npz
        assessments/<model_slug>.json
        choices/<cell_id>.json
        na_logs/<cell_id>.json
        diagnostics.json  stan_data.json  stan_data_size.json
    _cache/  _checkpoints/
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import confirmatory_analysis, diagnostics, data_preparation
from . import pools as pools_module, problem_generation
from . import provenance, prompts as prompts_module, schemas
from .assessment_collection import AssessmentCollector
from .batch_client import ProviderBatchClient
from .choice_collection import ChoiceCollector
from .client import build_client
from .config import MODELS, CellSpec, SEUSensitivityStudyConfig, get_model_spec
from .llm_extensions import pricing_for

logger = logging.getLogger(__name__)

__all__ = ["SEUSensitivityStudyRunner", "PHASES"]


_BATCH_LIABILITY_ASSUMPTIONS = {
    "method": "text_utf8_bytes_and_output_cap_v1",
    "input_tokens_per_utf8_byte": 1,
    "input_overhead_tokens_per_message": 1024,
    "input_overhead_tokens_per_request": 1024,
    "batch_price_multiplier": 0.5,
    "formula": "max(legacy_floor, 0.5 * (input_bound * input_rate + output_cap * output_rate) / 1000000)",
    "scope": "Text-only OpenAI chat completions and Anthropic messages; system counts as a message; output cap includes reasoning/thinking; rates are USD per million tokens.",
    "exclusions": "No tools, images, audio, caching directives, multiple completions, or unknown request parameters.",
    "reservation_policy": "Reservations are never released; this bound grants no additional spending authorization.",
}


#: ``assess`` runs *before* ``validate`` on purpose.  The R3 predictive-validity
#: check (§5) regresses the parsed belief probabilities on the item embeddings,
#: so the gate has nothing to evaluate until assessments exist.  "Pre-run" in
#: the plan means **pre-choice**, and the ordering respects that: assessments
#: are collected once per model x pool (~900 calls) while choice collection is
#: ~13,680, so the gate still stands in front of the spend that matters.
PHASES: tuple[str, ...] = (
    "design",
    "embed",
    "assess",
    "validate",
    "choices",
    "stan_data",
)


class SEUSensitivityStudyRunner:
    """Orchestrates the collection pipeline."""

    def __init__(self, config: SEUSensitivityStudyConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.cache_dir = Path(config.cache_dir)
        self.checkpoint_dir = self.results_dir / "_checkpoints"

    # -- Entry point --

    def run(
        self,
        *,
        phases: Optional[Sequence[str]] = None,
        pool_ids: Optional[Sequence[str]] = None,
        cell_ids: Optional[Sequence[str]] = None,
        model_names: Optional[Sequence[str]] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute *phases* for *pool_ids*.

        Parameters
        ----------
        dry_run:
            Report the planned work -- call counts per phase -- and make no API
            calls.  Intended to be run before every real run (§12, E1).
        model_names:
            Restrict the ``assess`` phase to these models.  Assessments are keyed
            on model alone (the prompt is omitted -- that is B2 in code), so this
            is the only phase the filter applies to; ``cell_ids`` filters
            ``choices``.  Used to price a small probe before committing to the
            full model set.
        force:
            Proceed past a gate that has not passed.  Recorded in the summary
            so a forced run is never indistinguishable from a clean one.
        """
        selected_phases = list(phases) if phases else list(PHASES)
        unknown = [phase for phase in selected_phases if phase not in PHASES]
        if unknown:
            raise ValueError(f"Unknown phase(s) {unknown}; available: {list(PHASES)}")

        selected_pools = list(pool_ids) if pool_ids else list(self.config.pool_ids)
        selected_models = list(model_names) if model_names else None
        if selected_models:
            known = {spec.name for spec in MODELS}
            unknown_models = [m for m in selected_models if m not in known]
            if unknown_models:
                raise ValueError(
                    f"Unknown model(s) {unknown_models}; available: {sorted(known)}"
                )
        summary: Dict[str, Any] = {
            "phases": selected_phases,
            "pools": {},
            "dry_run": dry_run,
            "forced": force,
        }
        if selected_models:
            summary["models"] = selected_models

        if dry_run:
            summary["plan"] = self._dry_run_plan(selected_pools, selected_phases)
            logger.info("Dry run: %s", json.dumps(summary["plan"], indent=2))
            return summary

        if "validate" in selected_phases and len(selected_pools) > 1:
            validate_index = selected_phases.index("validate")
            through_validate = selected_phases[: validate_index + 1]
            after_validate = selected_phases[validate_index + 1 :]

            for pool_id in selected_pools:
                summary["pools"][pool_id] = self._run_pool(
                    pool_id,
                    through_validate,
                    cell_ids=cell_ids,
                    model_names=selected_models,
                    force=force,
                )

            # The first pool cannot compare against current sibling reports
            # until every pool has completed its initial validation pass.
            for pool_id in selected_pools:
                summary["pools"][pool_id]["validate"] = self._phase_validate(pool_id)

            for pool_id in selected_pools:
                if after_validate:
                    summary["pools"][pool_id].update(
                        self._run_pool(
                            pool_id,
                            after_validate,
                            cell_ids=cell_ids,
                            model_names=selected_models,
                            force=force,
                        )
                    )
        else:
            for pool_id in selected_pools:
                summary["pools"][pool_id] = self._run_pool(
                    pool_id,
                    selected_phases,
                    cell_ids=cell_ids,
                    model_names=selected_models,
                    force=force,
                )

        if (
            "stan_data" in selected_phases
            and self.config.stan_model == "h_m01_size_assessment_anchored"
            and {"venture", "hiring"}.issubset(selected_pools)
        ):
            summary["matched_rq5"] = self._phase_matched_rq5_stan_data()

        self._write_json(self.results_dir / "run_summary.json", summary)
        return summary

    # -- Per-pool driver --

    def _run_pool(
        self,
        pool_id: str,
        phases: Sequence[str],
        *,
        cell_ids: Optional[Sequence[str]],
        model_names: Optional[Sequence[str]] = None,
        force: bool,
    ) -> Dict[str, Any]:
        logger.info("=== pool %s ===", pool_id)
        result: Dict[str, Any] = {}
        pool_dir = self._pool_dir(pool_id)
        pool_dir.mkdir(parents=True, exist_ok=True)

        if "design" in phases:
            result["design"] = self._phase_design(pool_id)
        if "embed" in phases:
            result["embed"] = self._phase_embed(pool_id)
        if "assess" in phases:
            result["assess"] = self._phase_assess(pool_id, model_names=model_names)
        if "validate" in phases:
            result["validate"] = self._phase_validate(pool_id)
        if "choices" in phases:
            self._require_gate(pool_id, force=force)
            result["choices"] = self._phase_choices(pool_id, cell_ids=cell_ids)
        if "stan_data" in phases:
            result["stan_data"] = self._phase_stan_data(pool_id)
        return result

    # -- Phases --

    def _phase_design(self, pool_id: str) -> Dict[str, Any]:
        pool = pools_module.load_pool(pool_id)
        self._write_json(self._pool_dir(pool_id) / "pool.json", pool)

        problem_set = problem_generation.generate_problem_set(
            pool,
            problems_per_family=self.config.problems_for(pool_id),
            seed=self.config.seed,
            menu_sizes=self.config.menu_sizes,
            num_presentations=self.config.num_presentations,
            presentation_mode=self.config.presentation_mode,
        )
        self._write_json(self._pool_dir(pool_id) / "problems.json", problem_set)
        return {
            "items": len(pool["items"]),
            "menus": len(problem_set["problems"]),
            "observations_per_cell": len(problem_set["problems"])
            * self.config.num_presentations,
        }

    def _phase_embed(self, pool_id: str) -> Dict[str, Any]:
        from applications.temperature_study.llm_client import EmbeddingClient

        pool = self._load_pool_artifact(pool_id)
        raw = data_preparation.embed_pool_items(
            pool, EmbeddingClient(model=self.config.embedding_model)
        )
        reduced, info = data_preparation.reduce_embeddings(
            raw, target_dim=self.config.target_dim, seed=self.config.seed
        )

        pool_dir = self._pool_dir(pool_id)
        np.savez(pool_dir / "embeddings_raw.npz", **raw)
        np.savez(pool_dir / "embeddings_reduced.npz", **reduced)
        self._write_json(pool_dir / "pca_info.json", info)
        return info

    def _phase_validate(self, pool_id: str) -> Dict[str, Any]:
        """
        The pre-choice gate (§5 R3, §6.3 R4).

        A missing prerequisite produces a gate report saying so rather than a
        traceback.  The report is the artefact the ``choices`` phase reads, so
        an unrunnable gate has to leave one behind: otherwise "the gate has not
        cleared" and "the gate crashed" would be indistinguishable to anyone
        inspecting the results directory.
        """
        from . import item_validation

        try:
            embeddings = self._load_reduced_embeddings(pool_id)
        except FileNotFoundError:
            embeddings = {}

        report = item_validation.run_gate(
            pool=self._load_pool_artifact(pool_id),
            problem_set=self._load_problem_set(pool_id),
            reduced_embeddings=embeddings,
            assessments=self._load_all_assessments(pool_id),
            config=self.config,
        )

        self._write_json(self._pool_dir(pool_id) / "gate_report.json", report)
        logger.info("Gate for pool %s: %s", pool_id, report.get("status"))
        return report

    def _phase_assess(
        self, pool_id: str, *, model_names: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        pool = self._load_pool_artifact(pool_id)
        prompt_sets = prompts_module.load_prompt_sets(pool_id)
        out_dir = self._pool_dir(pool_id) / "assessments"
        out_dir.mkdir(parents=True, exist_ok=True)

        wanted = set(model_names) if model_names else None
        if wanted is not None:
            logger.warning(
                "Assess phase RESTRICTED to models %s for pool %s -- the resulting "
                "assessment set is PARTIAL, so any gate run against it is a probe, "
                "not a pass",
                sorted(wanted),
                pool_id,
            )

        collected: Dict[str, Any] = {}
        for key, job in self.config.assessment_jobs().items():
            if job.pool_id != pool_id:
                continue
            if wanted is not None and job.model_name not in wanted:
                continue
            slug = get_model_spec(job.model_name).slug
            target = out_dir / f"{slug}.json"
            if target.exists():
                logger.info("Assessments already present for %s/%s", slug, pool_id)
                collected[slug] = "cached"
                continue

            client = build_client(
                job,
                cache_dir=self.cache_dir,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
            payload = AssessmentCollector(
                pool=pool,
                prompt_sets=prompt_sets,
                llm_client=client,
                model_name=job.model_name,
                max_tokens=self.config.max_assessment_tokens,
                temperature=job.temperature,
            ).collect(
                checkpoint_path=self.checkpoint_dir / pool_id / f"assess_{slug}.json"
            )
            usage = client.get_usage_summary()
            self._append_usage_event(
                {
                    "phase": "assess",
                    "collection_mode": "synchronous",
                    "pool_id": pool_id,
                    "model": job.model_name,
                    "artifact": str(target),
                    "records": len(payload["assessments"]),
                    "usage": usage,
                }
            )
            self._write_json(target, payload)
            collected[slug] = {
                "items": len(payload["assessments"]),
                "parsed": sum(1 for r in payload["assessments"] if r["parse_ok"]),
                "usage": usage,
            }
        return collected

    def _phase_choices(
        self, pool_id: str, *, cell_ids: Optional[Sequence[str]]
    ) -> Dict[str, Any]:
        problem_set = self._load_problem_set(pool_id)
        prompt_sets = prompts_module.load_prompt_sets(pool_id)
        out_dir = self._pool_dir(pool_id) / "choices"
        out_dir.mkdir(parents=True, exist_ok=True)

        cells = self.config.cells_for_pool(pool_id)
        if cell_ids:
            wanted = set(cell_ids)
            cells = [cell for cell in cells if cell.cell_id in wanted]

        collected: Dict[str, Any] = {}
        for cell in cells:
            target = out_dir / f"{cell.cell_id}.json"
            assessments = self._load_assessments(pool_id, cell.model_name)
            client = None
            if self.config.collection_mode == "synchronous":
                client = build_client(
                    cell,
                    cache_dir=self.cache_dir,
                    max_retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                )
            collector = ChoiceCollector(
                cell=cell,
                problem_set=problem_set,
                prompt_sets=prompt_sets,
                assessments=assessments,
                llm_client=client,
                max_tokens=self.config.max_choice_tokens,
            )
            batch_client = None
            if self.config.collection_mode == "batch":
                batch_client = ProviderBatchClient(
                    cell,
                    submission_reserver=lambda state, cell=cell: self._reserve_batch_budget(
                        state, cell
                    ),
                )

            if target.exists():
                cached = json.loads(target.read_text())
                schemas.check(
                    schemas.validate_choice_set(cached, problem_set=problem_set),
                    context=f"choice set {cell.cell_id!r}",
                )
                if batch_client is not None:
                    expected_hash = collector.batch_request_hash(batch_client)
                    if cached.get("request_hash") != expected_hash:
                        raise RuntimeError(
                            f"Cached choice artifact request identity does not match "
                            f"effective requests for {cell.cell_id}; refusing reuse"
                        )
                logger.info("Choices already present for cell %s", cell.cell_id)
                collected[cell.cell_id] = "cached"
                continue

            checkpoint_path = (
                self.checkpoint_dir / pool_id / f"choices_{cell.cell_id}.json"
            )
            if self.config.collection_mode == "batch":
                batch_state_path = (
                    self.checkpoint_dir
                    / pool_id
                    / "batches"
                    / f"{cell.cell_id}.json"
                )
                try:
                    payload = collector.collect_batch(
                        batch_client=batch_client,
                        state_path=batch_state_path,
                        checkpoint_path=checkpoint_path,
                    )
                finally:
                    self._write_batch_budget_report()
                if payload is None:
                    collected[cell.cell_id] = "batch_pending"
                    continue
                usage = batch_client.last_usage or batch_client.recover_usage(
                    batch_state_path
                )
            else:
                payload = collector.collect(checkpoint_path=checkpoint_path)
                usage = client.get_usage_summary()
            _, na_log = data_preparation.filter_resolved_choices(payload)
            self._append_usage_event(
                {
                    "phase": "choices",
                    "collection_mode": self.config.collection_mode,
                    "pool_id": pool_id,
                    "cell_id": cell.cell_id,
                    "model": cell.model_name,
                    "artifact": str(target),
                    "records": len(payload["choices"]),
                    "usage": usage,
                }
            )
            self._write_json(target, payload)
            self._write_json(
                self._pool_dir(pool_id) / "na_logs" / f"{cell.cell_id}.json", na_log
            )
            collected[cell.cell_id] = {
                "observations": len(payload["choices"]),
                "na_rate": na_log["na_rate"],
                "usage": usage,
            }
        return collected

    def _phase_stan_data(
        self, pool_id: str, *, include_assessment_scale_reference: bool = True
    ) -> Dict[str, Any]:
        pool = self._load_pool_artifact(pool_id)
        problem_set = self._load_problem_set(pool_id)
        reduced = self._load_reduced_embeddings(pool_id)
        choice_sets = self._load_all_choice_sets(pool_id)

        design_matrix, column_names, cell_ids = self.config.design_matrix_for_pool(pool_id)
        cells = self.config.cells_for_pool(pool_id)
        pool_dir = self._pool_dir(pool_id)

        anchored = self.config.stan_model == "h_m01_size_assessment_anchored"
        if anchored and not include_assessment_scale_reference:
            logger.warning("Omitting assessment_scale reference; outputs are not valid for A6 reporting")
        assessment_probabilities = None
        cell_model_names = None
        if anchored:
            model_names = sorted({cell.model_name for cell in cells})
            assessment_probabilities = {
                model_name: self._load_assessment_probabilities(pool_id, model_name)
                for model_name in model_names
            }
            cell_model_names = [cell.model_name for cell in cells]

        outputs: Dict[str, Any] = {"design_columns": column_names}
        analysis_contract_path = pool_dir / "analysis_contract.json"
        self._write_json(
            analysis_contract_path,
            confirmatory_analysis.contract_manifest(column_names, design_matrix),
        )
        outputs["analysis_contract"] = analysis_contract_path.name
        for include_size, filename in ((False, "stan_data.json"), (True, "stan_data_size.json")):
            anchored_kwargs = {}
            if anchored and include_size:
                anchored_kwargs = {
                    "include_assessment_scale_reference": include_assessment_scale_reference,
                    "assessment_probabilities": assessment_probabilities,
                    "cell_model_names": cell_model_names,
                    "utility_values": [
                        0.0,
                        self.config.primary_utility_middle,
                        1.0,
                    ],
                    "design_column_names": column_names,
                }
            stan_data, report = data_preparation.build_stan_data(
                pool=pool,
                problem_set=problem_set,
                choice_sets=choice_sets,
                reduced_embeddings=reduced,
                design_matrix=design_matrix,
                cell_ids=cell_ids,
                K=self.config.K,
                include_menu_size=include_size,
                **anchored_kwargs,
            )
            self._write_json(pool_dir / filename, stan_data)
            if anchored and include_size:
                self._write_json(pool_dir / f"{Path(filename).stem}_assembly_report.json", report)
            if not include_size:
                outputs["M_total"] = stan_data["M_total"]
                outputs["overall_na_rate"] = report["overall_na_rate"]
            elif anchored:
                outputs["confirmatory_design_rank"] = report[
                    "confirmatory_design_rank"
                ]
                outputs["confirmatory_design_required_rank"] = report[
                    "confirmatory_design_required_rank"
                ]

        if anchored:
            sensitivity_files = []
            for middle in self.config.utility_middle_values:
                if middle == self.config.primary_utility_middle:
                    continue
                stan_data, report = data_preparation.build_stan_data(
                    pool=pool,
                    problem_set=problem_set,
                    choice_sets=choice_sets,
                    reduced_embeddings=reduced,
                    design_matrix=design_matrix,
                    cell_ids=cell_ids,
                    K=self.config.K,
                    include_menu_size=True,
                    include_assessment_scale_reference=include_assessment_scale_reference,
                    assessment_probabilities=assessment_probabilities,
                    cell_model_names=cell_model_names,
                    utility_values=[0.0, middle, 1.0],
                    design_column_names=column_names,
                )
                label = f"{round(middle * 100):03d}"
                filename = f"stan_data_size_u{label}.json"
                self._write_json(pool_dir / filename, stan_data)
                self._write_json(pool_dir / f"{Path(filename).stem}_assembly_report.json", report)
                sensitivity_files.append(filename)
            outputs["utility_sensitivity_files"] = sensitivity_files

            presentation_files = []
            for presentation_id in (1, 2):
                stan_data, report = data_preparation.build_stan_data(
                    pool=pool,
                    problem_set=problem_set,
                    choice_sets=choice_sets,
                    reduced_embeddings=reduced,
                    design_matrix=design_matrix,
                    cell_ids=cell_ids,
                    K=self.config.K,
                    include_menu_size=True,
                    include_assessment_scale_reference=include_assessment_scale_reference,
                    assessment_probabilities=assessment_probabilities,
                    cell_model_names=cell_model_names,
                    utility_values=[0.0, self.config.primary_utility_middle, 1.0],
                    design_column_names=column_names,
                    presentation_id=presentation_id,
                )
                filename = f"stan_data_size_presentation_{presentation_id}.json"
                self._write_json(pool_dir / filename, stan_data)
                self._write_json(pool_dir / f"{Path(filename).stem}_assembly_report.json", report)
                presentation_files.append(filename)
                outputs[f"presentation_{presentation_id}_design_rank"] = report[
                    "confirmatory_design_rank"
                ]
            outputs["presentation_sensitivity_files"] = presentation_files

        subset, retention = diagnostics.size_balanced_stability_subset(
            choice_sets, seed=self.config.seed
        )
        self._write_json(
            pool_dir / "diagnostics.json",
            {
                "na_table": diagnostics.na_table(choice_sets),
                "position_flips": [
                    diagnostics.position_flip_summary(cs) for cs in choice_sets.values()
                ],
                "stability_subset": {"problem_ids": subset, **retention},
            },
        )
        outputs["stability_retention"] = retention["retention_after_balance"]
        return outputs

    def _phase_matched_rq5_stan_data(self) -> Dict[str, Any]:
        """Build the joint assessment-anchored matched-task RQ5 re-slice."""
        cells = self.config.cells_for_pool("venture") + self.config.cells_for_pool(
            "hiring"
        )
        design_matrix, column_names = confirmatory_analysis.matched_rq5_design(cells)
        cell_ids = [cell.cell_id for cell in cells]
        cell_model_names = [cell.model_name for cell in cells]

        probabilities: Dict[str, Dict[str, List[float]]] = {}
        for model in MODELS:
            probabilities[model.name] = {
                **self._load_assessment_probabilities("venture", model.name),
                **self._load_assessment_probabilities("hiring", model.name),
            }

        output_dir = self.results_dir / "matched_rq5"
        contract = confirmatory_analysis.matched_rq5_contract(cells)
        self._write_json(output_dir / "analysis_contract.json", contract)

        common = {
            "venture_pool": self._load_pool_artifact("venture"),
            "hiring_pool": self._load_pool_artifact("hiring"),
            "venture_problem_set": self._load_problem_set("venture"),
            "hiring_problem_set": self._load_problem_set("hiring"),
            "venture_choice_sets": self._load_all_choice_sets("venture"),
            "hiring_choice_sets": self._load_all_choice_sets("hiring"),
            "assessment_probabilities": probabilities,
            "design_matrix": design_matrix,
            "cell_ids": cell_ids,
            "cell_model_names": cell_model_names,
            "design_column_names": column_names,
            "K": self.config.K,
            "include_assessment_scale_reference": True,
        }
        files = []
        primary_report = None
        for middle in self.config.utility_middle_values:
            stan_data, report = data_preparation.build_matched_rq5_stan_data(
                **common,
                utility_values=[0.0, middle, 1.0],
            )
            if middle == self.config.primary_utility_middle:
                filename = "stan_data_size.json"
                primary_report = report
            else:
                filename = f"stan_data_size_u{round(middle * 100):03d}.json"
            self._write_json(output_dir / filename, stan_data)
            self._write_json(output_dir / f"{Path(filename).stem}_assembly_report.json", report)
            files.append(filename)

        presentation_files = []
        for presentation_id in (1, 2):
            stan_data, report = data_preparation.build_matched_rq5_stan_data(
                **common,
                utility_values=[0.0, self.config.primary_utility_middle, 1.0],
                presentation_id=presentation_id,
            )
            filename = f"stan_data_size_presentation_{presentation_id}.json"
            self._write_json(output_dir / filename, stan_data)
            self._write_json(output_dir / f"{Path(filename).stem}_assembly_report.json", report)
            presentation_files.append(filename)

        assert primary_report is not None
        self._write_json(output_dir / "assembly_report.json", primary_report)
        return {
            "analysis_contract": "analysis_contract.json",
            "stan_data_files": files,
            "presentation_sensitivity_files": presentation_files,
            "matched_item_pairs": primary_report["matched_item_pairs"],
            "paired_menus_per_task": primary_report["paired_menus_per_task"],
            "M_total": sum(primary_report["na_logs"][cell_id]["resolved"] for cell_id in cell_ids),
            "confirmatory_design_rank": primary_report["confirmatory_design_rank"],
            "confirmatory_design_required_rank": primary_report[
                "confirmatory_design_required_rank"
            ],
        }

    # -- Gate enforcement --

    def _require_gate(self, pool_id: str, *, force: bool) -> None:
        path = self._pool_dir(pool_id) / "gate_report.json"
        report = json.loads(path.read_text()) if path.exists() else None

        if report and report.get("passed"):
            return

        status = (report or {}).get("status", "missing")
        message = (
            f"Pool {pool_id!r} has not cleared the pre-choice validation gate "
            f"(status: {status}). Choice collection is the study's main API spend "
            f"(§12) and §5 blocks it on the predictive-validity check."
        )
        if not force:
            raise RuntimeError(message + " Pass force=True to override deliberately.")
        logger.warning("%s Proceeding because force=True.", message)

    # -- Dry run --

    def _dry_run_plan(
        self, pool_ids: Sequence[str], phases: Sequence[str]
    ) -> Dict[str, Any]:
        plan: Dict[str, Any] = {"pools": {}, "totals": {}}
        total_choice_calls = 0
        total_assessment_calls = 0

        for pool_id in pool_ids:
            try:
                pool = self._load_pool_artifact(pool_id)
                n_items = len(pool["items"])
            except FileNotFoundError:
                n_items = None

            menus = sum(self.config.problems_for(pool_id).values())
            n_cells = len(self.config.cells_for_pool(pool_id))
            choice_calls = menus * self.config.num_presentations * n_cells
            assessment_calls = (n_items or 0) * len({c.model_name for c in self.config.cells_for_pool(pool_id)})

            plan["pools"][pool_id] = {
                "items": n_items,
                "menus": menus,
                "cells": n_cells,
                "assessment_calls": assessment_calls,
                "choice_calls": choice_calls,
                "observations_per_cell": menus * self.config.num_presentations,
            }
            total_choice_calls += choice_calls
            total_assessment_calls += assessment_calls

        plan["totals"] = {
            "assessment_calls": total_assessment_calls,
            "choice_calls": total_choice_calls,
            "phases": list(phases),
        }
        return plan

    # -- Manifest --

    def write_manifest(self, **kwargs: Any) -> Dict[str, Any]:
        """Build and persist the provenance manifest (§6.5)."""
        prompt_sets = {
            pool_id: prompts_module.load_prompt_sets(pool_id)
            for pool_id in self.config.pool_ids
        }
        pca_info = {}
        for pool_id in self.config.pool_ids:
            path = self._pool_dir(pool_id) / "pca_info.json"
            if path.exists():
                pca_info[pool_id] = json.loads(path.read_text())

        manifest = provenance.build_run_manifest(
            self.config,
            prompt_hashes=prompts_module.prompt_hashes(prompt_sets),
            pca_info=pca_info,
            **kwargs,
        )
        manifest["batch_liability_assumptions"] = dict(_BATCH_LIABILITY_ASSUMPTIONS)
        self._write_json(self.results_dir / "run_manifest.json", manifest)
        return manifest

    def run_production_preflight(self) -> Dict[str, Any]:
        """Stage and bind exact prerequisites for one authorized Batch wave."""
        if self.config.collection_mode != "batch":
            raise RuntimeError("Production preflight requires collection_mode=batch")
        if not self.config.batch_wave_id or not self.config.batch_wave_cell_ids:
            raise RuntimeError("Production preflight requires an authorized Batch wave")

        cells_by_id = {cell.cell_id: cell for cell in self.config.cells}
        cells = [cells_by_id[cell_id] for cell_id in self.config.batch_wave_cell_ids]
        wave_pool_ids = list(dict.fromkeys(cell.pool_id for cell in cells))
        repository_root, git_commit = _clean_repository_identity()
        repository_sources = _production_source_paths(
            repository_root, self.config.stan_model
        )
        repository_hashes = {
            str(path.relative_to(repository_root)): _sha256_file(path)
            for path in repository_sources
        }
        for pool_id in self.config.pool_ids:
            self._phase_validate(pool_id)
        if len(self.config.pool_ids) > 1:
            for pool_id in self.config.pool_ids:
                self._phase_validate(pool_id)

        source_paths = []
        gate_hashes = {}
        assessment_jobs = list(self.config.assessment_jobs().values())
        for pool_id in wave_pool_ids:
            pool_dir = self._pool_dir(pool_id)
            gate_path = pool_dir / "gate_report.json"
            gate = json.loads(gate_path.read_text())
            if not gate.get("passed"):
                raise RuntimeError(
                    f"Production preflight gate did not pass for pool {pool_id}"
                )
            required = [
                pool_dir / "pool.json",
                pool_dir / "problems.json",
                pool_dir / "embeddings_raw.npz",
                pool_dir / "embeddings_reduced.npz",
                pool_dir / "pca_info.json",
                gate_path,
            ]
            assessment_paths = sorted((pool_dir / "assessments").glob("*.json"))
            expected_assessments = sum(
                job.pool_id == pool_id for job in assessment_jobs
            )
            if len(assessment_paths) != expected_assessments:
                raise RuntimeError(
                    f"Production preflight requires {expected_assessments} assessment "
                    f"artifacts for pool {pool_id}; found {len(assessment_paths)}"
                )
            required.extend(assessment_paths)
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Production preflight missing artifacts: {missing}")
            source_paths.extend(required)
            gate_hashes[pool_id] = _sha256_file(gate_path)

        source_hashes = {
            str(path.relative_to(self.results_dir)): _sha256_file(path)
            for path in source_paths
        }
        prompt_sets = {
            pool_id: prompts_module.load_prompt_sets(pool_id)
            for pool_id in wave_pool_ids
        }
        prompt_hashes = prompts_module.prompt_hashes(prompt_sets)
        request_hashes = {}
        request_evidence = {}
        for cell in cells:
            collector = ChoiceCollector(
                cell=cell,
                problem_set=self._load_problem_set(cell.pool_id),
                prompt_sets=prompt_sets[cell.pool_id],
                assessments=self._load_assessments(cell.pool_id, cell.model_name),
                llm_client=None,
                max_tokens=self.config.max_choice_tokens,
            )
            batch_client = ProviderBatchClient(cell, sdk_client=object())
            cell_evidence = collector.batch_request_evidence(batch_client)
            request_hashes[cell.cell_id] = cell_evidence["request_hash"]
            request_evidence[cell.cell_id] = cell_evidence

        liabilities = {
            cell.cell_id: self._batch_request_liability(request_evidence[cell.cell_id], cell)
            for cell in cells
        }
        reservations = self._load_batch_budget_records(
            self.results_dir / "batch_budget_reservations.jsonl"
        )
        reserved_total = math.fsum(record["reservation_usd"] for record in reservations)
        reserved_cells = {record["cell_id"]: record for record in reservations}
        additional_liability = 0.0
        for cell in cells:
            liability = liabilities[cell.cell_id]["reservation_usd"]
            existing = reserved_cells.get(cell.cell_id)
            if existing is not None:
                if (
                    existing.get("wave_id") != self.config.batch_wave_id
                    or existing.get("request_hash") != request_hashes[cell.cell_id]
                    or existing.get("request_count") != liabilities[cell.cell_id]["request_count"]
                    or existing["reservation_usd"] < liability
                ):
                    raise RuntimeError(
                        f"Conflicting Batch budget reservation for {cell.cell_id}; "
                        "replacement attempts require amended authorization"
                    )
            else:
                additional_liability += liability
        headroom = self.config.batch_choice_budget_usd - reserved_total
        if additional_liability > headroom:
            raise RuntimeError(
                "Batch choice budget exceeded before production staging: "
                f"reserved=${reserved_total:.6f}, requested=${additional_liability:.6f}, "
                f"ceiling=${self.config.batch_choice_budget_usd:.2f}"
            )
        budget_evidence = {
            "assumptions": dict(_BATCH_LIABILITY_ASSUMPTIONS),
            "budget_ceiling_usd": self.config.batch_choice_budget_usd,
            "reserved_total_usd": reserved_total,
            "reservation_headroom_usd": headroom,
            "wave_reservation_usd": math.fsum(item["reservation_usd"] for item in liabilities.values()),
            "additional_reservation_usd": additional_liability,
            "cells": liabilities,
        }

        generated_artifacts = {
            "preflight_config.json": self.config.to_dict(),
            **{
                f"requests/{cell_id}.json": payload
                for cell_id, payload in request_evidence.items()
            },
        }
        generated_hashes = {
            relative_path: _sha256_json(payload)
            for relative_path, payload in generated_artifacts.items()
        }

        evidence = {
            "schema_version": 2,
            "wave_id": self.config.batch_wave_id,
            "authorized_cell_ids": list(self.config.batch_wave_cell_ids),
            "config_hash": _sha256_json(self.config.to_dict()),
            "git_commit": git_commit,
            "repository_hashes": repository_hashes,
            "toolchain": provenance.toolchain_versions(),
            "source_hashes": source_hashes,
            "generated_hashes": generated_hashes,
            "prompt_hashes": prompt_hashes,
            "gate_hashes": gate_hashes,
            "request_hashes": request_hashes,
            "batch_budget": budget_evidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence["aggregate_hash"] = _sha256_json(evidence)

        stage_dir = self.results_dir / "production_stages" / self.config.batch_wave_id
        if stage_dir.exists():
            raise RuntimeError(
                f"Production stage already exists for wave {self.config.batch_wave_id}; "
                "use a new wave ID"
            )
        temporary = stage_dir.with_name(stage_dir.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        for source in source_paths:
            destination = temporary / source.relative_to(self.results_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for source in repository_sources:
            destination = temporary / "repository" / source.relative_to(repository_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for relative_path, payload in generated_artifacts.items():
            self._write_json(temporary / relative_path, payload)
        self._write_json(temporary / "preflight_manifest.json", evidence)
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        temporary.replace(stage_dir)
        for path in sorted(stage_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        stage_dir.chmod(0o555)
        return evidence

    # -- Artefact IO --

    def _pool_dir(self, pool_id: str) -> Path:
        return self.results_dir / "pools" / pool_id

    def _load_pool_artifact(self, pool_id: str) -> Dict[str, Any]:
        path = self._pool_dir(pool_id) / "pool.json"
        if not path.exists():
            raise FileNotFoundError(f"Run the 'design' phase for pool {pool_id!r} first")
        return json.loads(path.read_text())

    def _load_problem_set(self, pool_id: str) -> Dict[str, Any]:
        path = self._pool_dir(pool_id) / "problems.json"
        if not path.exists():
            raise FileNotFoundError(f"Run the 'design' phase for pool {pool_id!r} first")
        return json.loads(path.read_text())

    def _load_reduced_embeddings(self, pool_id: str) -> Dict[str, np.ndarray]:
        path = self._pool_dir(pool_id) / "embeddings_reduced.npz"
        if not path.exists():
            raise FileNotFoundError(f"Run the 'embed' phase for pool {pool_id!r} first")
        with np.load(path) as payload:
            return {key: payload[key] for key in payload.files}

    def _load_assessments(self, pool_id: str, model_name: str) -> Dict[str, str]:
        slug = get_model_spec(model_name).slug
        path = self._pool_dir(pool_id) / "assessments" / f"{slug}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No assessments for {model_name}/{pool_id}; run the 'assess' phase first"
            )
        payload = json.loads(path.read_text())
        return {record["item_id"]: record["text"] for record in payload["assessments"]}

    def _load_assessment_probabilities(
        self, pool_id: str, model_name: str
    ) -> Dict[str, List[float]]:
        slug = get_model_spec(model_name).slug
        path = self._pool_dir(pool_id) / "assessments" / f"{slug}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No assessments for {model_name}/{pool_id}; run the 'assess' phase first"
            )
        payload = json.loads(path.read_text())
        probabilities: Dict[str, List[float]] = {}
        for record in payload["assessments"]:
            if not record.get("parse_ok") or record.get("probabilities") is None:
                raise ValueError(
                    f"Assessment {model_name}/{pool_id}/{record['item_id']} has no "
                    "parsed probabilities; anchored Stan data cannot be built"
                )
            probabilities[record["item_id"]] = record["probabilities"]
        return probabilities

    def _load_all_assessments(self, pool_id: str) -> Dict[str, Dict[str, Any]]:
        directory = self._pool_dir(pool_id) / "assessments"
        if not directory.exists():
            return {}
        return {
            path.stem: json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
        }

    def _load_all_choice_sets(self, pool_id: str) -> Dict[str, Dict[str, Any]]:
        directory = self._pool_dir(pool_id) / "choices"
        if not directory.exists():
            raise FileNotFoundError(f"Run the 'choices' phase for pool {pool_id!r} first")
        return {
            path.stem: json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
        }

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
        tmp.replace(path)

    def _append_usage_event(self, event: Mapping[str, Any]) -> None:
        path = self.results_dir / "usage_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            key: event.get(key)
            for key in ("phase", "pool_id", "cell_id", "model", "artifact")
        }
        event_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if path.exists():
            with open(path) as handle:
                if any(
                    json.loads(line).get("event_id") == event_id
                    for line in handle
                    if line.strip()
                ):
                    return
        record = {
            "event_id": event_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "collection_mode": "synchronous",
            **event,
        }
        line = json.dumps(record, default=_json_default, sort_keys=True) + "\n"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _batch_request_liability(
        self, request_archive: Mapping[str, Any], cell: Any
    ) -> Dict[str, Any]:
        requests = request_archive.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RuntimeError("Batch liability requires a nonempty rendered request archive")
        if cell.provider not in {"openai", "anthropic"}:
            raise RuntimeError("Unsupported Batch liability provider")
        pricing = pricing_for(cell.endpoint, cell.provider)
        if any(
            isinstance(pricing.get(kind), bool)
            or not isinstance(pricing.get(kind), (int, float))
            or not math.isfinite(pricing[kind])
            or pricing[kind] <= 0
            for kind in ("input", "output")
        ):
            raise RuntimeError(f"Unpriced Batch liability endpoint: {cell.endpoint}")
        per_request = []
        custom_ids = set()
        for request in requests:
            if not isinstance(request, dict):
                raise RuntimeError("Unsupported rendered Batch request structure")
            custom_id = request.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id or custom_id in custom_ids:
                raise RuntimeError("Invalid or duplicate rendered Batch custom_id")
            custom_ids.add(custom_id)
            if cell.provider == "openai":
                if (
                    set(request) != {"custom_id", "method", "url", "body"}
                    or request["method"] != "POST"
                    or request["url"] != "/v1/chat/completions"
                ):
                    raise RuntimeError("Unsupported OpenAI Batch request structure")
                params = request["body"]
                allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "reasoning_effort"}
                cap_keys = {"max_tokens", "max_completion_tokens"}
                roles = {"system", "developer", "user", "assistant"}
            else:
                if set(request) != {"custom_id", "params"}:
                    raise RuntimeError("Unsupported Anthropic Batch request structure")
                params = request["params"]
                allowed = {"model", "messages", "max_tokens", "temperature", "system", "thinking"}
                cap_keys = {"max_tokens"}
                roles = {"user", "assistant"}
            if not isinstance(params, dict) or set(params) - allowed:
                raise RuntimeError("Unsupported Batch liability request parameters")
            if params.get("model") != cell.endpoint:
                raise RuntimeError("Batch liability endpoint does not match cell")
            caps = cap_keys.intersection(params)
            if len(caps) != 1:
                raise RuntimeError("Batch liability requires exactly one output token cap")
            output_cap = params[next(iter(caps))]
            if isinstance(output_cap, bool) or not isinstance(output_cap, int) or output_cap <= 0:
                raise RuntimeError("Batch liability requires a positive integer output token cap")
            if "temperature" in params and (
                isinstance(params["temperature"], bool)
                or not isinstance(params["temperature"], (int, float))
                or not math.isfinite(params["temperature"])
                or not 0 <= params["temperature"] <= (2 if cell.provider == "openai" else 1)
            ):
                raise RuntimeError("Unsupported Batch liability temperature")
            if "reasoning_effort" in params and (
                params["reasoning_effort"] not in ("low", "medium", "high")
                or "max_completion_tokens" not in params
            ):
                raise RuntimeError("Unsupported Batch liability reasoning parameters")
            if "thinking" in params:
                thinking = params["thinking"]
                if (
                    not isinstance(thinking, dict)
                    or set(thinking) != {"type", "budget_tokens"}
                    or thinking["type"] != "enabled"
                    or isinstance(thinking["budget_tokens"], bool)
                    or not isinstance(thinking["budget_tokens"], int)
                    or not 0 < thinking["budget_tokens"] < output_cap
                ):
                    raise RuntimeError("Unsupported Batch liability thinking parameters")
            messages = params.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RuntimeError("Unsupported Batch liability messages")
            contents = []
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or not isinstance(message["role"], str)
                    or message["role"] not in roles
                    or not isinstance(message["content"], str)
                ):
                    raise RuntimeError("Unsupported Batch liability text-only message structure")
                contents.append(message["content"])
            if "system" in params:
                if not isinstance(params["system"], str):
                    raise RuntimeError("Unsupported Batch liability text-only system content")
                contents.append(params["system"])
            content_bytes = sum(len(content.encode("utf-8")) for content in contents)
            input_bound = content_bytes + 1024 * len(contents) + 1024
            liability_usd = 0.5 * (
                input_bound * pricing["input"] + output_cap * pricing["output"]
            ) / 1_000_000
            per_request.append({
                "custom_id": custom_id,
                "content_utf8_bytes": content_bytes,
                "message_count": len(contents),
                "input_token_bound": input_bound,
                "output_token_cap": output_cap,
                "token_liability_usd": liability_usd,
                "reservation_usd": max(liability_usd, self.config.batch_choice_reservation_per_request_usd),
            })
        return {
            "assumptions": dict(_BATCH_LIABILITY_ASSUMPTIONS),
            "provider": cell.provider,
            "model": cell.endpoint,
            "pricing_per_million_tokens_usd": dict(pricing),
            "legacy_floor_per_request_usd": self.config.batch_choice_reservation_per_request_usd,
            "request_count": len(requests),
            "token_liability_usd": math.fsum(item["token_liability_usd"] for item in per_request),
            "reservation_usd": math.fsum(item["reservation_usd"] for item in per_request),
            "requests": per_request,
        }

    def _reserve_batch_budget(
        self, state: Mapping[str, Any], cell: Any
    ) -> None:
        if not self.config.batch_wave_id:
            raise RuntimeError(
                "Batch submission is not authorized: batch_wave_id is unset"
            )
        if cell.cell_id not in self.config.batch_wave_cell_ids:
            raise RuntimeError(
                f"Batch cell {cell.cell_id} is not authorized in wave "
                f"{self.config.batch_wave_id}"
            )
        request_archive = self._assert_production_preflight(state, cell)
        liability = self._batch_request_liability(request_archive, cell)
        path = self.results_dir / "batch_budget_reservations.jsonl"
        lock_path = path.with_suffix(path.suffix + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Batch budget ledger is already being modified: {path}"
                ) from error
            records = self._load_batch_budget_records(path)
            cell_ids = {record["cell_id"] for record in records}

            submission_id = str(state["submission_id"])
            reservation_usd = liability["reservation_usd"]
            reserved_usd = math.fsum(record["reservation_usd"] for record in records)
            if reserved_usd > self.config.batch_choice_budget_usd:
                raise RuntimeError("Batch choice budget exceeded before submission: existing reservations exceed ceiling")
            existing = [
                record
                for record in records
                if record.get("submission_id") == submission_id
            ]
            if existing:
                if (
                    existing[0]["reservation_usd"] < reservation_usd
                    or any(existing[0].get(key) != value for key, value in {
                        "cell_id": cell.cell_id,
                        "wave_id": self.config.batch_wave_id,
                        "request_hash": state["request_hash"],
                        "request_count": liability["request_count"],
                        "provider": cell.provider,
                        "model": cell.endpoint,
                    }.items())
                ):
                    raise RuntimeError(
                        f"Conflicting Batch budget reservation for {submission_id}"
                    )
                return

            if cell.cell_id in cell_ids:
                raise RuntimeError(
                    f"Batch budget already reserved for cell {cell.cell_id}; "
                    "replacement attempts require amended authorization"
                )

            if reserved_usd + reservation_usd > self.config.batch_choice_budget_usd:
                raise RuntimeError(
                    "Batch choice budget exceeded before submission: "
                    f"reserved=${reserved_usd:.6f}, requested=${reservation_usd:.6f}, "
                    f"ceiling=${self.config.batch_choice_budget_usd:.2f}"
                )
            record = {
                "submission_id": submission_id,
                "wave_id": self.config.batch_wave_id,
                "request_hash": state["request_hash"],
                "request_count": int(state["request_count"]),
                "reservation_usd": reservation_usd,
                "liability": liability,
                "budget_ceiling_usd": self.config.batch_choice_budget_usd,
                "reservation_rate_usd": (
                    self.config.batch_choice_reservation_per_request_usd
                ),
                "cell_id": cell.cell_id,
                "provider": cell.provider,
                "model": cell.endpoint,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            line = json.dumps(record, sort_keys=True) + "\n"
            ledger = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(ledger, line.encode("utf-8"))
                os.fsync(ledger)
            finally:
                os.close(ledger)
        finally:
            os.close(descriptor)

    def _assert_production_preflight(
        self, state: Mapping[str, Any], cell: Any
    ) -> Dict[str, Any]:
        wave_id = self.config.batch_wave_id
        if not wave_id:
            raise RuntimeError("Batch submission is not authorized: batch_wave_id is unset")
        manifest_path = (
            self.results_dir / "production_stages" / wave_id / "preflight_manifest.json"
        )
        if not manifest_path.exists():
            raise RuntimeError(
                f"Production preflight has not been staged for wave {wave_id}"
            )
        manifest = json.loads(manifest_path.read_text())
        signed_evidence = {
            key: value
            for key, value in manifest.items()
            if key != "aggregate_hash"
        }
        if manifest.get("aggregate_hash") != _sha256_json(signed_evidence):
            raise RuntimeError("Production preflight manifest hash does not match")
        if manifest.get("wave_id") != wave_id:
            raise RuntimeError("Production preflight wave identity does not match config")
        if manifest.get("config_hash") != _sha256_json(self.config.to_dict()):
            raise RuntimeError("Production preflight config hash does not match")
        repository_root, git_commit = _clean_repository_identity()
        if manifest.get("git_commit") != git_commit:
            raise RuntimeError("Production preflight Git commit does not match")
        if manifest.get("toolchain") != provenance.toolchain_versions():
            raise RuntimeError("Production preflight toolchain does not match")
        if cell.cell_id not in manifest.get("authorized_cell_ids", []):
            raise RuntimeError(
                f"Batch cell {cell.cell_id} is not authorized by production preflight"
            )
        if manifest.get("request_hashes", {}).get(cell.cell_id) != state.get(
            "request_hash"
        ):
            raise RuntimeError(
                f"Rendered Batch request hash does not match production preflight for {cell.cell_id}"
            )
        current_prompts = prompts_module.prompt_hashes(
            {cell.pool_id: prompts_module.load_prompt_sets(cell.pool_id)}
        )
        expected_prompts = {
            key: value
            for key, value in manifest.get("prompt_hashes", {}).items()
            if key.startswith(f"{cell.pool_id}/")
        }
        if current_prompts != expected_prompts:
            raise RuntimeError("Production preflight prompt hashes do not match")
        for relative_path, expected_hash in manifest.get(
            "repository_hashes", {}
        ).items():
            source_path = repository_root / relative_path
            staged_path = manifest_path.parent / "repository" / relative_path
            if not source_path.exists() or _sha256_file(source_path) != expected_hash:
                raise RuntimeError(
                    f"Production preflight repository hash does not match: {relative_path}"
                )
            if not staged_path.exists() or _sha256_file(staged_path) != expected_hash:
                raise RuntimeError(
                    f"Production staged repository hash does not match: {relative_path}"
                )
        generated_paths = ["preflight_config.json", f"requests/{cell.cell_id}.json"]
        request_archive = None
        for relative_path in generated_paths:
            generated_path = manifest_path.parent / relative_path
            expected_hash = manifest.get("generated_hashes", {}).get(relative_path)
            if not generated_path.exists():
                raise RuntimeError(
                    f"Production staged generated artifact is missing: {relative_path}"
                )
            try:
                actual_hash = _sha256_json(json.loads(generated_path.read_text()))
            except (json.JSONDecodeError, OSError) as error:
                raise RuntimeError(
                    f"Production staged generated artifact is invalid: {relative_path}"
                ) from error
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Production staged generated artifact hash does not match: {relative_path}"
                )
            if relative_path.startswith("requests/"):
                request_archive = json.loads(generated_path.read_text())
        if request_archive is None or len(request_archive.get("requests", [])) != int(
            state.get("request_count", -1)
        ):
            raise RuntimeError(
                "Batch request count does not match production staged request archive"
            )
        if request_archive.get("request_hash") != state.get("request_hash"):
            raise RuntimeError("Rendered Batch request archive hash does not match production preflight")
        for relative_path, expected_hash in manifest.get("source_hashes", {}).items():
            if not relative_path.startswith(f"pools/{cell.pool_id}/"):
                continue
            path = self.results_dir / relative_path
            if not path.exists() or _sha256_file(path) != expected_hash:
                raise RuntimeError(
                    f"Production preflight artifact hash does not match: {relative_path}"
                )
            staged_path = manifest_path.parent / relative_path
            if not staged_path.exists() or _sha256_file(staged_path) != expected_hash:
                raise RuntimeError(
                    f"Production staged artifact hash does not match: {relative_path}"
                )
        return request_archive

    @staticmethod
    def _load_batch_budget_records(path: Path) -> List[Dict[str, Any]]:
        records = []
        submission_ids = set()
        cell_ids = set()
        if not path.exists():
            return records
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid or truncated budget ledger at line {line_number}: {path}"
                    ) from error
                submission_id = record.get("submission_id")
                reservation = record.get("reservation_usd")
                cell_id = record.get("cell_id")
                if (
                    not isinstance(submission_id, str)
                    or not submission_id
                    or not isinstance(cell_id, str)
                    or not cell_id
                    or isinstance(reservation, bool)
                    or not isinstance(reservation, (int, float))
                    or not math.isfinite(reservation)
                    or reservation <= 0
                    or submission_id in submission_ids
                    or cell_id in cell_ids
                ):
                    raise RuntimeError(
                        f"Invalid or truncated budget ledger at line {line_number}: {path}"
                    )
                submission_ids.add(submission_id)
                cell_ids.add(cell_id)
                records.append(record)
        return records

    def _write_batch_budget_report(self) -> None:
        ledger_path = self.results_dir / "batch_budget_reservations.jsonl"
        reservations = self._load_batch_budget_records(ledger_path)
        states = {}
        for state_path in sorted(self.checkpoint_dir.glob("*/batches/*.json")):
            state = json.loads(state_path.read_text())
            submission_id = state.get("submission_id")
            if submission_id:
                states[submission_id] = (state_path, state)

        attempts = []
        usage_estimated_costs = []
        for reservation in reservations:
            state_entry = states.get(reservation["submission_id"])
            state_path, state = state_entry if state_entry else (None, {})
            usage = state.get("usage") or {}
            usage_estimated_cost = usage.get("estimated_cost_usd")
            if isinstance(usage_estimated_cost, (int, float)) and math.isfinite(
                usage_estimated_cost
            ):
                usage_estimated_costs.append(float(usage_estimated_cost))
            else:
                usage_estimated_cost = None
            attempts.append(
                {
                    **reservation,
                    "batch_state": str(state_path) if state_path else None,
                    "status": state.get("status", "state_missing"),
                    "batch_id": state.get("batch_id"),
                    "usage_estimated_cost_usd": usage_estimated_cost,
                    "usage_cost_known": usage_estimated_cost is not None,
                    "usage_complete": usage_estimated_cost is not None
                    and usage.get("usage_complete", state.get("status") == "completed")
                    and not any(state.get(key) for key in (
                        "failed_custom_ids", "duplicate_custom_ids", "provider_errors"
                    )),
                }
            )

        reserved_total = sum(float(item["reservation_usd"]) for item in reservations)
        report = {
            "liability_assumptions": dict(_BATCH_LIABILITY_ASSUMPTIONS),
            "budget_ceiling_usd": self.config.batch_choice_budget_usd,
            "reserved_total_usd": reserved_total,
            "reservation_headroom_usd": self.config.batch_choice_budget_usd
            - reserved_total,
            "known_usage_estimated_total_usd": sum(usage_estimated_costs),
            "usage_cost_complete": all(attempt["usage_complete"] for attempt in attempts),
            "unresolved_submission_ids": [
                attempt["submission_id"]
                for attempt in attempts
                if not attempt["usage_complete"]
            ],
            "attempts": attempts,
        }
        self._write_json(self.results_dir / "batch_budget_report.json", report)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, default=_json_default, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clean_repository_identity() -> tuple[Path, str]:
    repository_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Production preflight requires a clean tracked worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository_root, commit


def _production_source_paths(repository_root: Path, stan_model: str) -> List[Path]:
    package_dir = repository_root / "applications" / "seu_sensitivity_study"
    paths = sorted(package_dir.glob("*.py"))
    relative_paths = [
        "applications/seu_sensitivity_study/PREREGISTRATION.md",
        "applications/seu_sensitivity_study/configs/preregistered.yaml",
        "environment.yml",
        "requirements.txt",
        f"models/{stan_model}.stan",
    ]
    paths.extend(repository_root / relative_path for relative_path in relative_paths)
    paths = sorted(set(path.resolve() for path in paths))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Production preflight missing source files: {missing}")
    return paths

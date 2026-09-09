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

import hashlib
import json
import logging
import os
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

logger = logging.getLogger(__name__)

__all__ = ["SEUSensitivityStudyRunner", "PHASES"]


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
                batch_client = ProviderBatchClient(cell)

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
                payload = collector.collect_batch(
                    batch_client=batch_client,
                    state_path=batch_state_path,
                    checkpoint_path=checkpoint_path,
                )
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

    def _phase_stan_data(self, pool_id: str) -> Dict[str, Any]:
        pool = self._load_pool_artifact(pool_id)
        problem_set = self._load_problem_set(pool_id)
        reduced = self._load_reduced_embeddings(pool_id)
        choice_sets = self._load_all_choice_sets(pool_id)

        design_matrix, column_names, cell_ids = self.config.design_matrix_for_pool(pool_id)
        cells = self.config.cells_for_pool(pool_id)
        pool_dir = self._pool_dir(pool_id)

        anchored = self.config.stan_model == "h_m01_size_assessment_anchored"
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
                stan_data, _ = data_preparation.build_stan_data(
                    pool=pool,
                    problem_set=problem_set,
                    choice_sets=choice_sets,
                    reduced_embeddings=reduced,
                    design_matrix=design_matrix,
                    cell_ids=cell_ids,
                    K=self.config.K,
                    include_menu_size=True,
                    assessment_probabilities=assessment_probabilities,
                    cell_model_names=cell_model_names,
                    utility_values=[0.0, middle, 1.0],
                    design_column_names=column_names,
                )
                label = f"{round(middle * 100):03d}"
                filename = f"stan_data_size_u{label}.json"
                self._write_json(pool_dir / filename, stan_data)
                sensitivity_files.append(filename)
            outputs["utility_sensitivity_files"] = sensitivity_files

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
            files.append(filename)

        assert primary_report is not None
        self._write_json(output_dir / "assembly_report.json", primary_report)
        return {
            "analysis_contract": "analysis_contract.json",
            "stan_data_files": files,
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
        self._write_json(self.results_dir / "run_manifest.json", manifest)
        return manifest

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

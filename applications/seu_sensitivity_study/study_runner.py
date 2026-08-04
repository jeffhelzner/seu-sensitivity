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

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import diagnostics, data_preparation, pools as pools_module, problem_generation
from . import provenance, prompts as prompts_module, schemas
from .assessment_collection import AssessmentCollector
from .choice_collection import ChoiceCollector
from .client import build_client
from .config import CellSpec, SEUSensitivityStudyConfig, get_model_spec

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
        force:
            Proceed past a gate that has not passed.  Recorded in the summary
            so a forced run is never indistinguishable from a clean one.
        """
        selected_phases = list(phases) if phases else list(PHASES)
        unknown = [phase for phase in selected_phases if phase not in PHASES]
        if unknown:
            raise ValueError(f"Unknown phase(s) {unknown}; available: {list(PHASES)}")

        selected_pools = list(pool_ids) if pool_ids else list(self.config.pool_ids)
        summary: Dict[str, Any] = {
            "phases": selected_phases,
            "pools": {},
            "dry_run": dry_run,
            "forced": force,
        }

        if dry_run:
            summary["plan"] = self._dry_run_plan(selected_pools, selected_phases)
            logger.info("Dry run: %s", json.dumps(summary["plan"], indent=2))
            return summary

        for pool_id in selected_pools:
            summary["pools"][pool_id] = self._run_pool(
                pool_id, selected_phases, cell_ids=cell_ids, force=force
            )

        self._write_json(self.results_dir / "run_summary.json", summary)
        return summary

    # -- Per-pool driver --

    def _run_pool(
        self,
        pool_id: str,
        phases: Sequence[str],
        *,
        cell_ids: Optional[Sequence[str]],
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
            result["assess"] = self._phase_assess(pool_id)
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

    def _phase_assess(self, pool_id: str) -> Dict[str, Any]:
        pool = self._load_pool_artifact(pool_id)
        prompt_sets = prompts_module.load_prompt_sets(pool_id)
        out_dir = self._pool_dir(pool_id) / "assessments"
        out_dir.mkdir(parents=True, exist_ok=True)

        collected: Dict[str, Any] = {}
        for key, job in self.config.assessment_jobs().items():
            if job.pool_id != pool_id:
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
            self._write_json(target, payload)
            collected[slug] = {
                "items": len(payload["assessments"]),
                "parsed": sum(1 for r in payload["assessments"] if r["parse_ok"]),
                "usage": client.get_usage_summary(),
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
            if target.exists():
                logger.info("Choices already present for cell %s", cell.cell_id)
                collected[cell.cell_id] = "cached"
                continue

            assessments = self._load_assessments(pool_id, cell.model_name)
            client = build_client(
                cell,
                cache_dir=self.cache_dir,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
            payload = ChoiceCollector(
                cell=cell,
                problem_set=problem_set,
                prompt_sets=prompt_sets,
                assessments=assessments,
                llm_client=client,
                max_tokens=self.config.max_choice_tokens,
            ).collect(
                checkpoint_path=self.checkpoint_dir / pool_id / f"choices_{cell.cell_id}.json"
            )
            self._write_json(target, payload)

            _, na_log = data_preparation.filter_resolved_choices(payload)
            self._write_json(
                self._pool_dir(pool_id) / "na_logs" / f"{cell.cell_id}.json", na_log
            )
            collected[cell.cell_id] = {
                "observations": len(payload["choices"]),
                "na_rate": na_log["na_rate"],
                "usage": client.get_usage_summary(),
            }
        return collected

    def _phase_stan_data(self, pool_id: str) -> Dict[str, Any]:
        pool = self._load_pool_artifact(pool_id)
        problem_set = self._load_problem_set(pool_id)
        reduced = self._load_reduced_embeddings(pool_id)
        choice_sets = self._load_all_choice_sets(pool_id)

        design_matrix, column_names, cell_ids = self.config.design_matrix_for_pool(pool_id)
        pool_dir = self._pool_dir(pool_id)

        outputs: Dict[str, Any] = {"design_columns": column_names}
        for include_size, filename in ((False, "stan_data.json"), (True, "stan_data_size.json")):
            stan_data, report = data_preparation.build_stan_data(
                pool=pool,
                problem_set=problem_set,
                choice_sets=choice_sets,
                reduced_embeddings=reduced,
                design_matrix=design_matrix,
                cell_ids=cell_ids,
                K=self.config.K,
                include_menu_size=include_size,
            )
            self._write_json(pool_dir / filename, stan_data)
            if not include_size:
                outputs["M_total"] = stan_data["M_total"]
                outputs["overall_na_rate"] = report["overall_na_rate"]

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

"""
Study runner for the alignment study.

Orchestrates the 4-phase pipeline across all 18 cells:
  Phase 1: Generate problems (shared across cells)
  Phase 2a: Collect assessments per cell
  Phase 2b: Collect choices per cell
  Phase 3: Build stacked Stan data
  Phase 4: Fit hierarchical model (optional)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from applications.temperature_study.problem_generator import ProblemGenerator
from applications.temperature_study.llm_client import EmbeddingClient
from applications.temperature_study.data_preparation import (
    EmbeddingReducer,
    filter_valid_choices,
    save_stan_data,
)

from .config import AlignmentStudyConfig, CellSpec
from .data_preparation import HierarchicalStanDataBuilder
from .llm_extensions import create_alignment_llm_client
from .choice_collection import AlignmentChoiceCollector

logger = logging.getLogger(__name__)


class AlignmentStudyRunner:
    """Orchestrates the full alignment study pipeline."""

    def __init__(self, config: AlignmentStudyConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        skip_collection: bool = False,
        cells_to_run: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Parameters
        ----------
        skip_collection : bool
            If True, load existing assessment/choice data instead of
            collecting new data via API calls.
        cells_to_run : list of str, optional
            If provided, only run these cell IDs.
        """
        # Filter cells if requested
        if cells_to_run:
            active_cells = [c for c in self.config.cells if c.cell_id in cells_to_run]
        else:
            active_cells = self.config.cells

        output = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # Phase 1: Generate or load shared problems
        problems, claims = self._run_phase1()
        output["num_problems"] = len(problems)
        output["num_claims"] = len(claims)

        if skip_collection:
            logger.info("Skipping data collection (skip_collection=True)")
            # Load from saved files
            per_cell_assessments = {}
            per_cell_raw_embeddings = {}
            per_cell_choices = {}
            for cell in active_cells:
                cell_dir = self.results_dir / "cells" / cell.cell_id
                if (cell_dir / "assessments.json").exists():
                    with open(cell_dir / "assessments.json") as f:
                        per_cell_assessments[cell.cell_id] = json.load(f)
                if (cell_dir / "choices.json").exists():
                    with open(cell_dir / "choices.json") as f:
                        per_cell_choices[cell.cell_id] = json.load(f)
        else:
            # Phase 2: Collect data per cell
            per_cell_assessments = {}
            per_cell_raw_embeddings = {}
            per_cell_choices = {}

            for cell in active_cells:
                logger.info("Processing cell: %s", cell.cell_id)
                cell_dir = self.results_dir / "cells" / cell.cell_id
                cell_dir.mkdir(parents=True, exist_ok=True)

                # Phase 2a: Assessments + embeddings
                assessments, raw_embeddings = self._run_phase2a_cell(cell, claims)
                per_cell_assessments[cell.cell_id] = assessments
                per_cell_raw_embeddings[cell.cell_id] = raw_embeddings

                # Save
                with open(cell_dir / "assessments.json", "w") as f:
                    json.dump(assessments, f, indent=2)

                # Phase 2b: Choices
                choices = self._run_phase2b_cell(cell, problems, assessments)
                per_cell_choices[cell.cell_id] = choices

                with open(cell_dir / "choices.json", "w") as f:
                    json.dump(choices, f, indent=2)

        # Phase 3: Build stacked Stan data
        phase3_output = self._run_phase3(
            problems, per_cell_assessments, per_cell_raw_embeddings,
            per_cell_choices, active_cells,
        )
        output["phase3"] = phase3_output

        return output

    def _run_phase1(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate or load shared problems."""
        problems_file = self.results_dir / "shared" / "problems.json"
        claims_file = Path(self.config.claims_file)

        # Load claims
        with open(claims_file) as f:
            claims_data = json.load(f)
        claims = claims_data.get("claims", claims_data)

        if problems_file.exists():
            logger.info("Loading existing problems from %s", problems_file)
            with open(problems_file) as f:
                problems = json.load(f)
        else:
            logger.info("Generating new problems")
            generator = ProblemGenerator(
                claims_pool=claims,
                num_problems=self.config.num_problems,
                min_alternatives=self.config.min_alternatives,
                max_alternatives=self.config.max_alternatives,
                num_presentations=self.config.num_presentations,
                seed=self.config.seed,
            )
            problems = generator.generate()

            problems_file.parent.mkdir(parents=True, exist_ok=True)
            with open(problems_file, "w") as f:
                json.dump(problems, f, indent=2)

        return problems, claims

    def _run_phase2a_cell(
        self,
        cell: CellSpec,
        claims: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Collect assessments for a single cell.

        Returns (assessments_dict, raw_embeddings).
        """
        llm_client = create_alignment_llm_client(
            cell,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
        )

        collector = AlignmentChoiceCollector(
            config=self.config,
            cell=cell,
            llm_client=llm_client,
        )

        assessments = collector.collect_assessments(claims)

        # Embed assessments
        embedding_client = EmbeddingClient(model=self.config.embedding_model)
        assessment_texts = list(assessments.values())
        assessment_keys = list(assessments.keys())

        raw_vectors = embedding_client.embed(assessment_texts)
        raw_embeddings = {
            key: np.array(vec) for key, vec in zip(assessment_keys, raw_vectors)
        }

        return assessments, raw_embeddings

    def _run_phase2b_cell(
        self,
        cell: CellSpec,
        problems: List[Dict[str, Any]],
        assessments: Dict[str, str],
    ) -> Dict[str, Any]:
        """Collect choices for a single cell."""
        llm_client = create_alignment_llm_client(
            cell,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
        )

        collector = AlignmentChoiceCollector(
            config=self.config,
            cell=cell,
            llm_client=llm_client,
        )

        return collector.collect_choices(problems, assessments)

    def _run_phase3(
        self,
        problems: List[Dict[str, Any]],
        per_cell_assessments: Dict[str, Dict[str, Any]],
        per_cell_raw_embeddings: Dict[str, Dict[str, np.ndarray]],
        per_cell_choices: Dict[str, Dict[str, Any]],
        active_cells: List[CellSpec],
    ) -> Dict[str, Any]:
        """
        Build stacked Stan data.

        Steps:
        1. Pool all raw embeddings across all cells.
        2. Fit PCA on pooled set.
        3. Project each cell's embeddings.
        4. Filter valid choices per cell.
        5. Build stacked data via HierarchicalStanDataBuilder.
        6. Save everything.
        """
        output_info = {}

        # 1. Pool embeddings
        pooled_embeddings = {}
        for cell_id, raw_embs in per_cell_raw_embeddings.items():
            for claim_id, vec in raw_embs.items():
                key = f"{claim_id}_{cell_id}"
                pooled_embeddings[key] = vec

        if not pooled_embeddings:
            logger.warning("No raw embeddings available; skipping Phase 3 PCA")
            return output_info

        # 2. Fit PCA
        reducer = EmbeddingReducer(
            target_dim=self.config.target_dim,
            seed=self.config.seed,
        )
        reducer.fit(pooled_embeddings)

        # 3. Project per cell — build shared reduced embeddings keyed by claim_id
        # Use first cell's embeddings as representative (all cells embed same claims)
        first_cell_embs = next(iter(per_cell_raw_embeddings.values()))
        reduced_embeddings = {}
        for claim_id, raw_vec in first_cell_embs.items():
            reduced = reducer.pca.transform(raw_vec.reshape(1, -1))[0]
            reduced_embeddings[claim_id] = reduced

        # 4. Filter valid choices per cell
        per_cell_valid = {}
        cell_ids = []
        for cell in active_cells:
            cell_id = cell.cell_id
            choices = per_cell_choices.get(cell_id, {})

            # Build valid choice list
            valid = []
            for problem in problems:
                pid = problem["problem_id"]
                if pid in choices and choices[pid].get("choice") is not None:
                    valid.append({
                        "problem_id": pid,
                        "alternatives": problem["alternatives"],
                        "choice": choices[pid]["choice"],
                    })

            per_cell_valid[cell_id] = valid
            cell_ids.append(cell_id)
            output_info[f"valid_choices_{cell_id}"] = len(valid)

        # 5. Build stacked data
        X, col_names = self.config.get_design_matrix()
        # Filter X to only active cells
        all_cell_ids = [c.cell_id for c in self.config.cells]
        active_indices = [all_cell_ids.index(cid) for cid in cell_ids]
        X_active = X[active_indices]

        builder = HierarchicalStanDataBuilder(self.config)
        stan_data = builder.build(
            per_cell_valid_choices=per_cell_valid,
            reduced_embeddings=reduced_embeddings,
            problems=problems,
            design_matrix=X_active,
            cell_ids=cell_ids,
        )

        # 6. Save
        stan_data_file = self.results_dir / "stan_data" / "h_m01_data.json"
        stan_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stan_data_file, "w") as f:
            json.dump(stan_data, f, indent=2)

        output_info["stan_data_file"] = str(stan_data_file)
        output_info["M_total"] = stan_data["M_total"]
        output_info["J"] = stan_data["J"]
        output_info["design_columns"] = col_names

        return output_info

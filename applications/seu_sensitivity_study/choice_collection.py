"""
Choice collection (study plan §6.1 step 3, §3.2, §6.4).

This is where the prompt manipulation is applied -- and *only* here.  The cell's
prompt condition selects one choice instruction; the assessments inserted into
the prompt were collected once per model x pool under the neutral instruction
and are shared across the three prompt cells (B2, §3.2).

Two correctness properties the previous implementation lacked:

* **Presentations are actually iterated.**  One observation is produced per
  (menu, presentation), and the assessments are inserted *in presentation
  order*, so position 1 of the prompt is position 1 of the design.  The
  recorded ``chosen_item_id`` is resolved through that same order, which is
  what lets :func:`schemas.validate_choice_set` catch a counterbalancing bug
  instead of silently corrupting ``y``.
* **Resolution path is recorded per observation** (§6.4), so parser-induced NA
  can be separated from genuine refusal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import schemas
from .config import CellSpec
from .parsing import parse_choice_response
from .prompts import PromptSet

logger = logging.getLogger(__name__)

__all__ = ["ChoiceCollector"]


class ChoiceCollector:
    """Collects one cell's choices over its pool's fixed menu design."""

    def __init__(
        self,
        *,
        cell: CellSpec,
        problem_set: Mapping[str, Any],
        prompt_sets: Mapping[str, PromptSet],
        assessments: Mapping[str, str],
        llm_client: Any,
        max_tokens: int = 64,
    ):
        if problem_set["pool_id"] != cell.pool_id:
            raise ValueError(
                f"problem set is for pool {problem_set['pool_id']!r} but cell "
                f"{cell.cell_id!r} belongs to {cell.pool_id!r}"
            )
        self.cell = cell
        self.problem_set = problem_set
        self.prompt_sets = prompt_sets
        self.assessments = assessments
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    # -- Collection --

    def collect(
        self,
        *,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 25,
        on_progress: Optional[Callable[[int, int], None]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect every (menu, presentation) observation for this cell.

        Resumes from *checkpoint_path* when present.  Because the choice phase
        is ~10,800 calls (§12), an interrupted run that could not resume would
        be the single most expensive failure mode in the study.
        """
        self._check_assessments_complete()

        done: Dict[tuple, Dict[str, Any]] = {}
        if checkpoint_path is not None:
            done = _load_checkpoint(checkpoint_path)
            if done:
                logger.info(
                    "Resuming cell %s: %d observation(s) already collected",
                    self.cell.cell_id,
                    len(done),
                )

        jobs = [
            (problem, presentation)
            for problem in self.problem_set["problems"]
            for presentation in problem["presentations"]
        ]

        records: List[Dict[str, Any]] = []
        for index, (problem, presentation) in enumerate(jobs, start=1):
            key = (problem["id"], presentation["presentation_id"])
            if key in done:
                records.append(done[key])
            else:
                records.append(self._collect_one(problem, presentation))
                if checkpoint_path is not None and index % checkpoint_every == 0:
                    _write_checkpoint(checkpoint_path, records)
            if on_progress is not None:
                on_progress(index, len(jobs))

        payload = {
            "schema_version": schemas.SCHEMA_VERSION,
            "cell_id": self.cell.cell_id,
            "pool_id": self.cell.pool_id,
            "model_name": self.cell.model_name,
            "prompt_condition": self.cell.prompt_condition,
            "answer_format_version": schemas.ANSWER_TOKEN_VERSION,
            "choices": records,
        }

        if validate:
            schemas.check(
                schemas.validate_choice_set(payload, problem_set=dict(self.problem_set)),
                context=f"choice set {self.cell.cell_id!r}",
            )

        self._log_summary(records)
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, records)
        return payload

    def _collect_one(
        self, problem: Mapping[str, Any], presentation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        order: List[str] = list(presentation["order"])
        prompt_set = self._prompt_set_for(problem["family"])

        # Assessments are inserted in presentation order; the parsed position
        # is then resolved back through the same list.
        assessments_in_order = [self.assessments[item_id] for item_id in order]
        prompt = prompt_set.render_choice(
            self.cell.prompt_condition, assessments_in_order
        )

        response = self.llm_client.generate(
            prompt,
            system_prompt=prompt_set.choice_system.strip(),
            temperature=self.cell.temperature,
            max_tokens=self.max_tokens,
        )

        position, resolution_path = parse_choice_response(response, len(order))
        if position is None:
            logger.warning(
                "Cell %s, %s/p%s: unresolved response %r",
                self.cell.cell_id,
                problem["id"],
                presentation["presentation_id"],
                (response or "")[:120],
            )

        return {
            "problem_id": problem["id"],
            "presentation_id": presentation["presentation_id"],
            "menu_size": problem["menu_size"],
            "difficulty_stratum": problem["difficulty_stratum"],
            "family": problem["family"],
            "chosen_position": position,
            "chosen_item_id": order[position - 1] if position is not None else None,
            "resolution_path": resolution_path,
            "raw_response": response,
        }

    # -- Helpers --

    def _prompt_set_for(self, family: str) -> PromptSet:
        try:
            return self.prompt_sets[family]
        except KeyError:
            raise KeyError(
                f"No prompts resolved for family {family!r} in pool "
                f"{self.cell.pool_id!r}; have {sorted(self.prompt_sets)}"
            ) from None

    def _check_assessments_complete(self) -> None:
        needed = {
            item_id
            for problem in self.problem_set["problems"]
            for item_id in problem["item_ids"]
        }
        missing = sorted(needed - set(self.assessments))
        if missing:
            raise ValueError(
                f"Cell {self.cell.cell_id!r}: no assessment for {len(missing)} item(s) "
                f"used by the design, e.g. {missing[:5]}. Run the assessment phase "
                f"for {self.cell.model_name}/{self.cell.pool_id} first."
            )

    def _log_summary(self, records: List[Dict[str, Any]]) -> None:
        total = len(records)
        by_path: Dict[str, int] = {}
        for record in records:
            path = record["resolution_path"]
            by_path[path] = by_path.get(path, 0) + 1
        na = by_path.get("unresolved", 0)
        logger.info(
            "Cell %s: %d observation(s); NA %d (%.1f%%); resolution paths %s",
            self.cell.cell_id,
            total,
            na,
            100.0 * na / total if total else 0.0,
            by_path,
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> Dict[tuple, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable choice checkpoint %s: %s", path, error)
        return {}
    return {
        (record["problem_id"], record["presentation_id"]): record
        for record in payload.get("choices", [])
    }


def _write_checkpoint(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump({"choices": records}, handle, indent=2)
    tmp.replace(path)

"""
Assessment collection (study plan §6.1 step 2, §3.2 B2, §5).

Assessments are collected **once per model x pool under the neutral
instruction** and shared across that model's three prompt cells.  Two things
follow, and both are enforced here rather than left to the caller:

* the collector takes no prompt condition -- it always renders the pool's
  single neutral assessment prompt, so a treatment prompt cannot reach the
  belief-side channel (B2);
* the artefact it writes is keyed by model x pool, not by cell, which is the
  ~3x cost saving that makes the §5 predictive-validity gate affordable
  *before* the expensive choice phase.

Each response is parsed for the structured ``PROBABILITIES:`` line.  Those
numbers are the predictive-validity check's outcome (§5) and are never fed to
Stan; an unparseable line is recorded, not retried into submission.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import schemas
from .parsing import parse_probabilities
from .prompts import PromptSet

logger = logging.getLogger(__name__)

__all__ = ["AssessmentCollector"]


class AssessmentCollector:
    """Collects one model's assessments over one pool's items."""

    def __init__(
        self,
        *,
        pool: Mapping[str, Any],
        prompt_sets: Mapping[str, PromptSet],
        llm_client: Any,
        model_name: str,
        max_tokens: int = 400,
        temperature: Optional[float] = 0.0,
        keep_probability_line: bool = True,
    ):
        """
        Parameters
        ----------
        keep_probability_line:
            Whether the structured ``PROBABILITIES:`` line stays in the text
            that later gets inserted into choice prompts.  Default ``True``,
            matching the plan's reading that the assessment *as written* is the
            choice context (§5, §6.1 step 2).

            Worth knowing when interpreting RQ2: leaving the line in hands
            every arm an explicit probability vector, which partially pre-empts
            what the ``seu_maximizing`` instruction supplies.  It is uniform
            across cells, so contrasts between arms remain interpretable and
            RQ4/RQ5 (which read on contrasts) are unaffected -- but the
            *magnitude* of the RQ2 prompt effect is conditioned on this choice.
        """
        self.pool = pool
        self.prompt_sets = prompt_sets
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.keep_probability_line = keep_probability_line

        self.consequences: List[str] = list(pool["consequences"])
        self.k = len(self.consequences)

    # -- Collection --

    def collect(
        self,
        *,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect assessments for every item in the pool.

        Resumes from *checkpoint_path* when present, so an interrupted run does
        not re-spend on items already assessed.
        """
        items = list(self.pool["items"])
        done: Dict[str, Dict[str, Any]] = {}
        if checkpoint_path is not None:
            done = _load_checkpoint(checkpoint_path)
            if done:
                logger.info(
                    "Resuming assessments for %s/%s: %d of %d already collected",
                    self.model_name,
                    self.pool["pool_id"],
                    len(done),
                    len(items),
                )

        records: List[Dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            item_id = item["id"]
            if item_id in done:
                records.append(done[item_id])
            else:
                records.append(self._assess_item(item))
                if checkpoint_path is not None and index % checkpoint_every == 0:
                    _write_checkpoint(checkpoint_path, records)
            if on_progress is not None:
                on_progress(index, len(items))

        payload = {
            "schema_version": schemas.SCHEMA_VERSION,
            "pool_id": self.pool["pool_id"],
            "model_name": self.model_name,
            "instruction": schemas.ASSESSMENT_INSTRUCTION,
            "keep_probability_line": self.keep_probability_line,
            "assessments": records,
        }

        if validate:
            schemas.check(
                schemas.validate_assessment_set(payload, pool=dict(self.pool)),
                context=(
                    f"assessment set {self.model_name}/{self.pool['pool_id']}"
                ),
            )

        parsed = sum(1 for record in records if record["parse_ok"])
        logger.info(
            "Collected %d assessments for %s/%s (%d/%d probability lines parsed)",
            len(records),
            self.model_name,
            self.pool["pool_id"],
            parsed,
            len(records),
        )
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, records)
        return payload

    def _assess_item(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        prompt_set = self._prompt_set_for(item["family"])
        prompt = prompt_set.render_assessment(item["text"], self.consequences)

        response = self.llm_client.generate(
            prompt,
            system_prompt=prompt_set.assessment_system.strip(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        probabilities = parse_probabilities(response, self.k)
        if probabilities is None:
            logger.warning(
                "Unparseable probability line for %s/%s item %s",
                self.model_name,
                self.pool["pool_id"],
                item["id"],
            )

        return {
            "item_id": item["id"],
            "family": item["family"],
            "text": _assessment_text(response, self.keep_probability_line),
            "probabilities": probabilities,
            "parse_ok": probabilities is not None,
            "raw_response": response,
        }

    def _prompt_set_for(self, family: str) -> PromptSet:
        try:
            return self.prompt_sets[family]
        except KeyError:
            raise KeyError(
                f"No prompts resolved for family {family!r} in pool "
                f"{self.pool['pool_id']!r}; have {sorted(self.prompt_sets)}"
            ) from None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assessment_text(response: str, keep_probability_line: bool) -> str:
    if keep_probability_line:
        return response.strip()
    kept = [
        line
        for line in response.splitlines()
        if not line.strip().upper().startswith("PROBABILITIES")
    ]
    return "\n".join(kept).strip()


def _load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable assessment checkpoint %s: %s", path, error)
        return {}
    return {record["item_id"]: record for record in payload.get("assessments", [])}


def _write_checkpoint(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump({"assessments": records}, handle, indent=2)
    tmp.replace(path)

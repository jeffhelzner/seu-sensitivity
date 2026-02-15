"""
Choice Collector for the Temperature Study.

Sends choice prompts for each problem × presentation × temperature,
using assessment texts (not raw claim descriptions) so that the
choice agent operates on the same information encoded into w[r].
Parses responses with NA-safe logic and records position metadata.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config import StudyConfig
from .llm_client import LLMClient, create_llm_client, parse_choice
from .problem_generator import ProblemGenerator

logger = logging.getLogger(__name__)


class ChoiceCollector:
    """
    Collect choices per temperature with NA-safe parsing.

    For each problem × presentation × temperature, the choice prompt is
    sent with assessment texts in the presentation's shuffled order.
    Responses are parsed strictly; failures are recorded as NA
    (``valid=false``).
    """

    def __init__(
        self,
        config: StudyConfig,
        generator: ProblemGenerator,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.generator = generator
        self._llm: Optional[LLMClient] = llm_client

        # Load prompt templates
        with open(config.prompts_file) as f:
            self._prompts = yaml.safe_load(f)

        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm_client(
                provider=self.config.provider,
                model=self.config.llm_model,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_temperature(
        self,
        problems: List[Dict[str, Any]],
        temperature: float,
        *,
        assessments: Optional[Dict[str, str]] = None,
        checkpoint_every: int = 50,
    ) -> Dict[str, Any]:
        """
        Collect choices for all problems × presentations at one temperature.

        Args:
            problems: List of problem dicts with ``presentations``.
            temperature: The LLM temperature for this condition.
            assessments: Mapping of claim_id → assessment text.  When
                provided, the choice prompt presents these assessment
                texts instead of raw claim descriptions.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict conforming to the ``choices_T{temp}.json`` schema from
            DESIGN.md §5.3.
        """
        system_prompt = self._prompts["choice"]["system"]
        user_template = self._prompts["choice"]["user"]

        all_choices: List[Dict[str, Any]] = []
        valid_count = 0
        na_count = 0

        for problem in problems:
            claim_ids = problem["claim_ids"]
            num_alts = problem["num_alternatives"]
            num_range_str = ProblemGenerator.num_range_str(num_alts)

            for pres in problem["presentations"]:
                ordered_ids = pres["order"]

                if assessments is not None:
                    list_str = ProblemGenerator.format_choice_assessments_list(
                        ordered_ids, assessments
                    )
                else:
                    # Fallback: use raw claim descriptions formatted as
                    # assessments (for tests or backward compatibility)
                    list_str = self.generator.format_choice_claims_list(
                        ordered_ids
                    )

                user_prompt = user_template.format(
                    assessments_list=list_str,
                    num_range=num_range_str,
                )

                raw_response = self.llm.generate(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=64,
                )

                parsed = parse_choice(raw_response, num_alts)

                if parsed is not None:
                    position_chosen = parsed  # 1-indexed
                    claim_chosen = ordered_ids[parsed - 1]
                    valid = True
                    valid_count += 1
                else:
                    position_chosen = None
                    claim_chosen = None
                    valid = False
                    na_count += 1
                    logger.warning(
                        "NA choice at T=%.1f, problem=%s, pres=%d: %r",
                        temperature,
                        problem["id"],
                        pres["presentation_id"],
                        raw_response,
                    )

                all_choices.append(
                    {
                        "problem_id": problem["id"],
                        "presentation_id": pres["presentation_id"],
                        "claim_order": ordered_ids,
                        "position_chosen": position_chosen,
                        "claim_chosen": claim_chosen,
                        "valid": valid,
                        "raw_response": raw_response,
                    }
                )

                # Checkpoint
                total_so_far = valid_count + na_count
                if total_so_far % checkpoint_every == 0:
                    logger.info(
                        "T=%.1f: %d/%d choices collected (%d valid, %d NA)",
                        temperature,
                        total_so_far,
                        self._expected_choices(problems),
                        valid_count,
                        na_count,
                    )
                    self._save_checkpoint(
                        temperature, all_choices, valid_count, na_count
                    )

        total = valid_count + na_count
        na_rate = na_count / total if total > 0 else 0.0

        logger.info(
            "T=%.1f complete: %d total, %d valid, %d NA (rate=%.3f)",
            temperature,
            total,
            valid_count,
            na_count,
            na_rate,
        )

        return {
            "temperature": temperature,
            "total_choices": total,
            "valid_choices": valid_count,
            "na_choices": na_count,
            "na_rate": round(na_rate, 4),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "choices": all_choices,
        }

    def collect_all_temperatures(
        self,
        problems: List[Dict[str, Any]],
        *,
        assessments_per_temp: Optional[Dict[float, Dict[str, str]]] = None,
        checkpoint_every: int = 50,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Collect choices across all configured temperatures.

        Args:
            problems: List of problem dicts with ``presentations``.
            assessments_per_temp: Optional mapping of temperature →
                {claim_id → assessment text}.  When provided, assessment
                texts are used in the choice prompt.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict mapping temperature → choices dict.
        """
        results: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            logger.info("Starting choice collection at T=%.1f", temp)
            assess = (
                assessments_per_temp[temp]
                if assessments_per_temp is not None
                else None
            )
            results[temp] = self.collect_temperature(
                problems, temp,
                assessments=assess,
                checkpoint_every=checkpoint_every,
            )
            logger.info(
                "T=%.1f cost so far: $%.4f",
                temp,
                self.llm.get_estimated_cost(),
            )
        return results

    # ------------------------------------------------------------------
    # Saving / Loading
    # ------------------------------------------------------------------

    def save_choices(
        self, temperature: float, choices_dict: Dict[str, Any]
    ) -> Path:
        """Save choices JSON for one temperature."""
        temp_str = f"{temperature:.1f}".replace(".", "_")
        json_path = self.results_dir / f"choices_T{temp_str}.json"
        with open(json_path, "w") as f:
            json.dump(choices_dict, f, indent=2)
        logger.info("Saved choices to %s", json_path)
        return json_path

    def save_all(
        self, per_temp: Dict[float, Dict[str, Any]]
    ) -> List[Path]:
        """Save choices for every temperature."""
        paths: List[Path] = []
        for temp, choices_dict in per_temp.items():
            paths.append(self.save_choices(temp, choices_dict))
        return paths

    @staticmethod
    def load_choices(filepath: str | Path) -> Dict[str, Any]:
        """Load a ``choices_T{temp}.json`` file."""
        with open(filepath) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # NA analysis
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_na(
        per_temp: Dict[float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize NA rates across temperatures.

        Useful for DESIGN.md §6 analysis — systematically different NA
        rates across temperatures are themselves informative.
        """
        summary: Dict[str, Any] = {"per_temperature": {}, "overall": {}}
        total_valid = 0
        total_na = 0

        for temp in sorted(per_temp):
            d = per_temp[temp]
            summary["per_temperature"][f"T={temp}"] = {
                "valid": d["valid_choices"],
                "na": d["na_choices"],
                "total": d["total_choices"],
                "na_rate": d["na_rate"],
            }
            total_valid += d["valid_choices"]
            total_na += d["na_choices"]

        total = total_valid + total_na
        summary["overall"] = {
            "valid": total_valid,
            "na": total_na,
            "total": total,
            "na_rate": round(total_na / total, 4) if total > 0 else 0.0,
        }
        return summary

    @staticmethod
    def get_na_entries(
        choices_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return all NA choice entries from a single temperature's data."""
        return [c for c in choices_dict["choices"] if not c["valid"]]

    # ------------------------------------------------------------------
    # Position bias helpers
    # ------------------------------------------------------------------

    @staticmethod
    def position_choice_rates(
        choices_dict: Dict[str, Any],
    ) -> Dict[int, float]:
        """
        Compute the rate at which each position was chosen (valid only).

        Returns:
            Dict mapping 1-indexed position → proportion (e.g. {1: 0.45, 2: 0.30, 3: 0.25}).
        """
        valid_entries = [c for c in choices_dict["choices"] if c["valid"]]
        if not valid_entries:
            return {}

        # Determine max number of alternatives
        max_pos = max(len(c["claim_order"]) for c in valid_entries)
        counts = {pos: 0 for pos in range(1, max_pos + 1)}

        for c in valid_entries:
            counts[c["position_chosen"]] = counts.get(
                c["position_chosen"], 0
            ) + 1

        total = len(valid_entries)
        return {pos: round(count / total, 4) for pos, count in counts.items()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_choices(problems: List[Dict[str, Any]]) -> int:
        """Total expected choices = sum of presentations across all problems."""
        return sum(len(p.get("presentations", [])) for p in problems)

    def _save_checkpoint(
        self,
        temperature: float,
        choices: List[Dict[str, Any]],
        valid: int,
        na: int,
    ) -> None:
        temp_str = f"{temperature:.1f}".replace(".", "_")
        ckpt_path = self.results_dir / f"choices_T{temp_str}_partial.json"
        with open(ckpt_path, "w") as f:
            json.dump(
                {
                    "temperature": temperature,
                    "partial": True,
                    "total_choices": valid + na,
                    "valid_choices": valid,
                    "na_choices": na,
                    "choices": choices,
                },
                f,
                indent=2,
            )
        logger.debug("Checkpoint saved: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return LLM usage summary."""
        if self._llm is not None:
            return self._llm.get_usage_summary()
        return {}

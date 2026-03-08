"""
Choice Collector for the Temperature Study with EU Prompt.

Sends choice prompts using the EU-maximization prompt variant.
Reuses assessment texts from the base temperature study — the only
difference is the choice prompt template, which explicitly instructs
the LLM to maximize expected utility with respect to the three
defined outcomes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config import StudyConfig

# Reuse LLM infrastructure from the base temperature study
from applications.temperature_study.llm_client import (
    LLMClient,
    create_llm_client,
    parse_choice,
)
from applications.temperature_study.problem_generator import ProblemGenerator

logger = logging.getLogger(__name__)


class ChoiceCollector:
    """
    Collect choices per temperature using the EU-maximization prompt.

    For each problem × presentation × temperature, the choice prompt is
    sent with assessment texts in the presentation's shuffled order.
    The prompt explicitly instructs EU maximization.
    Responses are parsed strictly; failures are recorded as NA.
    """

    def __init__(
        self,
        config: StudyConfig,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self._llm: Optional[LLMClient] = llm_client

        # Load EU-prompt templates
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
        assessments: Dict[str, str],
        *,
        checkpoint_every: int = 50,
    ) -> Dict[str, Any]:
        """
        Collect choices for all problems × presentations at one temperature.

        Args:
            problems: List of problem dicts with ``presentations``.
            temperature: The LLM temperature for this condition.
            assessments: Mapping of claim_id → assessment text from
                the base study.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict conforming to the ``choices_T{temp}.json`` schema.
        """
        system_prompt = self._prompts["choice"]["system"]
        user_template = self._prompts["choice"]["user"]

        all_choices: List[Dict[str, Any]] = []
        valid_count = 0
        na_count = 0

        for problem in problems:
            num_alts = problem["num_alternatives"]
            num_range_str = ProblemGenerator.num_range_str(num_alts)

            for pres in problem["presentations"]:
                ordered_ids = pres["order"]

                list_str = ProblemGenerator.format_choice_assessments_list(
                    ordered_ids, assessments
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
        assessments_per_temp: Dict[float, Dict[str, str]],
        *,
        checkpoint_every: int = 50,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Collect choices across all configured temperatures.

        Args:
            problems: List of problem dicts with ``presentations``.
            assessments_per_temp: Mapping of temperature →
                {claim_id → assessment text} from the base study.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict mapping temperature → choices dict.
        """
        results: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            logger.info("Starting EU-prompt choice collection at T=%.1f", temp)
            results[temp] = self.collect_temperature(
                problems, temp,
                assessments=assessments_per_temp[temp],
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
        """Summarize NA rates across temperatures."""
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

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return LLM usage summary."""
        if self._llm is not None:
            return self._llm.get_usage_summary()
        return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_choices(problems: List[Dict[str, Any]]) -> int:
        return sum(len(p["presentations"]) for p in problems)

    def _save_checkpoint(
        self,
        temperature: float,
        choices: List[Dict[str, Any]],
        valid: int,
        na: int,
    ) -> None:
        temp_str = f"{temperature:.1f}".replace(".", "_")
        ckpt_path = self.results_dir / f"checkpoint_choices_T{temp_str}.json"
        with open(ckpt_path, "w") as f:
            json.dump(
                {
                    "temperature": temperature,
                    "valid_count": valid,
                    "na_count": na,
                    "choices_so_far": choices,
                },
                f,
                indent=2,
            )
        logger.debug("Checkpoint saved: %s", ckpt_path)

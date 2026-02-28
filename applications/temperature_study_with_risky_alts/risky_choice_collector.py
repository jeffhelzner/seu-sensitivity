"""
Risky Choice Collector for the Temperature Study with Risky Alternatives.

Sends choice prompts for each risky problem × presentation × temperature.
Each prompt presents risky alternatives with explicitly stated objective
probability distributions. Parses responses with NA-safe logic.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import StudyConfig
from .risky_problem_generator import RiskyProblemGenerator

# Import shared LLM infrastructure from the temperature_study module
from applications.temperature_study.llm_client import (
    LLMClient,
    create_llm_client,
    parse_choice,
)

logger = logging.getLogger(__name__)


class RiskyChoiceCollector:
    """
    Collect risky choices per temperature with NA-safe parsing.

    For each risky problem × presentation × temperature, the choice prompt
    presents risky alternatives with their objective probability descriptions.
    Responses are parsed strictly; failures are recorded as NA.
    """

    def __init__(
        self,
        config: StudyConfig,
        generator: RiskyProblemGenerator,
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
        checkpoint_every: int = 50,
    ) -> Dict[str, Any]:
        """
        Collect risky choices for all problems × presentations at one temperature.

        Args:
            problems: List of risky problem dicts with ``presentations``.
            temperature: The LLM temperature for this condition.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict conforming to the risky choices schema.
        """
        system_prompt = self._prompts["risky_choice"]["system"]
        user_template = self._prompts["risky_choice"]["user"]

        all_choices: List[Dict[str, Any]] = []
        valid_count = 0
        na_count = 0

        for problem in problems:
            alt_ids = problem["alternative_ids"]
            num_alts = problem["num_alternatives"]
            num_range_str = RiskyProblemGenerator.num_range_str(num_alts)

            for pres in problem["presentations"]:
                ordered_ids = pres["order"]

                # Format the alternatives list with descriptions
                list_str = self.generator.format_alternatives_list(ordered_ids)

                user_prompt = user_template.format(
                    alternatives_list=list_str,
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
                    alt_chosen = ordered_ids[parsed - 1]
                    valid = True
                    valid_count += 1
                else:
                    position_chosen = None
                    alt_chosen = None
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
                        "alternative_order": ordered_ids,
                        "position_chosen": position_chosen,
                        "alternative_chosen": alt_chosen,
                        "valid": valid,
                        "raw_response": raw_response,
                    }
                )

                # Checkpoint
                total_so_far = valid_count + na_count
                if total_so_far % checkpoint_every == 0:
                    logger.info(
                        "T=%.1f: %d/%d risky choices collected "
                        "(%d valid, %d NA)",
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
            "T=%.1f risky complete: %d total, %d valid, %d NA (rate=%.3f)",
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
        checkpoint_every: int = 50,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Collect risky choices across all configured temperatures.

        Args:
            problems: List of risky problem dicts with ``presentations``.
            checkpoint_every: Save intermediate results every N choices.

        Returns:
            Dict mapping temperature → choices dict.
        """
        results: Dict[float, Dict[str, Any]] = {}
        for temp in self.config.temperatures:
            logger.info("Starting risky choice collection at T=%.1f", temp)
            results[temp] = self.collect_temperature(
                problems, temp,
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
        """Save risky choices JSON for one temperature."""
        temp_str = f"{temperature:.1f}".replace(".", "_")
        json_path = self.results_dir / f"risky_choices_T{temp_str}.json"
        with open(json_path, "w") as f:
            json.dump(choices_dict, f, indent=2)
        logger.info("Saved risky choices to %s", json_path)
        return json_path

    def save_all(
        self, per_temp: Dict[float, Dict[str, Any]]
    ) -> List[Path]:
        """Save risky choices for every temperature."""
        paths: List[Path] = []
        for temp, choices_dict in per_temp.items():
            paths.append(self.save_choices(temp, choices_dict))
        return paths

    @staticmethod
    def load_choices(filepath: str | Path) -> Dict[str, Any]:
        """Load a ``risky_choices_T{temp}.json`` file."""
        with open(filepath) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # NA analysis
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_na(
        per_temp: Dict[float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize NA rates across temperatures for risky choices."""
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
        ckpt_path = self.results_dir / f"risky_choices_T{temp_str}_partial.json"
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

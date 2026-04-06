"""
Prompt-aware choice collection for the alignment study.

Wraps ChoiceCollector with prompt-condition awareness, selecting
the appropriate prompt templates based on the cell's prompt_condition.
"""

from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from applications.temperature_study.llm_client import LLMClient, parse_choice
from applications.temperature_study.problem_generator import ProblemGenerator

from .config import AlignmentStudyConfig, CellSpec

logger = logging.getLogger(__name__)


class AlignmentChoiceCollector:
    """
    Wraps ChoiceCollector with prompt-condition awareness.

    Loads the multi-condition prompts.yaml, extracts templates for
    the specified condition, and collects choices using the cell-specific
    LLM client.
    """

    def __init__(
        self,
        config: AlignmentStudyConfig,
        cell: CellSpec,
        llm_client: LLMClient,
    ):
        self.config = config
        self.cell = cell
        self.llm_client = llm_client

        # Load prompt templates
        with open(config.prompts_file) as f:
            all_prompts = yaml.safe_load(f)

        condition = cell.prompt_condition
        if condition not in all_prompts:
            raise ValueError(
                f"Prompt condition '{condition}' not found in {config.prompts_file}. "
                f"Available: {list(all_prompts.keys())}"
            )

        self.prompts = all_prompts[condition]
        self.system_prompt = self.prompts["system_prompt"].strip()
        self.assessment_prompt_template = self.prompts["assessment_prompt"].strip()
        self.choice_prompt_template = self.prompts["choice_prompt"].strip()

    def collect_assessments(
        self,
        claims: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Collect assessments for all claims using the cell's LLM and prompt condition.

        Parameters
        ----------
        claims : list of dict
            Each dict has at least 'claim_id' and 'description' keys.

        Returns
        -------
        dict : {claim_id -> assessment_text}
        """
        assessments = {}
        for claim in claims:
            claim_id = claim["claim_id"]
            description = claim["description"]

            prompt = self.assessment_prompt_template.format(
                claim_description=description
            )

            response = self.llm_client.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=self.cell.temperature,
            )
            assessments[claim_id] = response
            logger.debug("Assessment for %s: %s", claim_id, response[:100])

        return assessments

    def collect_choices(
        self,
        problems: List[Dict[str, Any]],
        assessments: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Collect choices for all problems using cell-specific prompts and LLM.

        Parameters
        ----------
        problems : list of dict
            Each has 'problem_id', 'alternatives' (list of claim_ids).
        assessments : dict
            {claim_id -> assessment_text}

        Returns
        -------
        dict : {problem_id -> {'choice': int or None, 'response': str}}
        """
        choices = {}
        for problem in problems:
            problem_id = problem["problem_id"]
            alternatives = problem["alternatives"]

            # Build assessments list text
            lines = []
            for idx, claim_id in enumerate(alternatives, 1):
                assessment = assessments.get(claim_id, "[Assessment unavailable]")
                lines.append(f"{idx}. {assessment}")
            assessments_list = "\n\n".join(lines)

            valid_range = f"1-{len(alternatives)}"

            prompt = self.choice_prompt_template.format(
                assessments_list=assessments_list,
                valid_range=valid_range,
            )

            response = self.llm_client.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=self.cell.temperature,
                max_tokens=64,
            )

            parsed = parse_choice(response, len(alternatives))
            choices[problem_id] = {
                "choice": parsed,
                "response": response,
                "num_alternatives": len(alternatives),
            }

            if parsed is None:
                logger.warning(
                    "Cell %s, problem %s: could not parse choice from: %r",
                    self.cell.cell_id, problem_id, response,
                )

        return choices

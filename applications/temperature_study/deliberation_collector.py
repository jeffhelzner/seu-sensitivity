"""
Deliberation Collector for the Temperature Study.

Sends deliberation prompts for each problem × claim × temperature,
records text responses, embeds them, and saves checkpointed results.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .config import StudyConfig
from .llm_client import LLMClient, EmbeddingClient, create_llm_client
from .problem_generator import ProblemGenerator

logger = logging.getLogger(__name__)


class DeliberationCollector:
    """
    Collect deliberation texts and embeddings per temperature.

    For each problem, every constituent claim receives its own deliberation
    prompt (letter-labelled, with the full claim set visible) at the
    specified temperature.  Responses are embedded and cached.
    """

    def __init__(
        self,
        config: StudyConfig,
        generator: ProblemGenerator,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.config = config
        self.generator = generator

        # Lazy initialization — only create when actually needed
        self._llm: Optional[LLMClient] = llm_client
        self._embedder: Optional[EmbeddingClient] = embedding_client

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

    @property
    def embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(
                model=self.config.embedding_model,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
        return self._embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_temperature(
        self,
        problems: List[Dict[str, Any]],
        temperature: float,
        *,
        checkpoint_every: int = 25,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Collect deliberations for all problems at a single temperature.

        Args:
            problems: List of problem dicts with ``claim_ids``.
            temperature: The LLM temperature for this condition.
            checkpoint_every: Save intermediate results every N problems.

        Returns:
            ``(deliberations_dict, raw_embeddings)`` where:

            - *deliberations_dict* follows the ``deliberations_T{temp}.json``
              schema from DESIGN.md §5.2.
            - *raw_embeddings* maps claim IDs to 1-D numpy arrays (raw
              embedding vectors before PCA).
        """
        system_prompt = self._prompts["deliberation"]["system"]
        user_template = self._prompts["deliberation"]["user"]

        all_entries: List[Dict[str, Any]] = []
        raw_embeddings: Dict[str, np.ndarray] = {}

        # Track which claims we've already collected at this temperature
        # (each unique claim needs only one deliberation per problem context)
        for prob_idx, problem in enumerate(problems):
            claim_ids = problem["claim_ids"]
            claims_list_str = self.generator.format_deliberation_claims_list(
                claim_ids
            )

            for claim_idx, cid in enumerate(claim_ids):
                target_letter = ProblemGenerator.claim_index_to_letter(claim_idx)

                user_prompt = user_template.format(
                    claims_list=claims_list_str,
                    target_letter=target_letter,
                )

                response_text = self.llm.generate(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=256,
                )

                # Embed the deliberation
                embedding_vec = self.embedder.embed_single(response_text)
                emb_key = f"{problem['id']}_{cid}"
                raw_embeddings[emb_key] = np.array(embedding_vec)

                all_entries.append(
                    {
                        "problem_id": problem["id"],
                        "claim_id": cid,
                        "target_letter": target_letter,
                        "temperature": temperature,
                        "response": response_text,
                        "embedding_key": emb_key,
                    }
                )

            # Checkpoint
            if (prob_idx + 1) % checkpoint_every == 0:
                logger.info(
                    "T=%.1f: checkpointing after %d/%d problems",
                    temperature,
                    prob_idx + 1,
                    len(problems),
                )
                self._save_checkpoint(
                    temperature, all_entries, raw_embeddings, partial=True
                )

        delib_dict = self._build_output_dict(temperature, all_entries, problems)
        return delib_dict, raw_embeddings

    def collect_all_temperatures(
        self,
        problems: List[Dict[str, Any]],
        *,
        checkpoint_every: int = 25,
    ) -> Tuple[Dict[float, Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Collect deliberations across all configured temperatures.

        Returns:
            ``(per_temp_deliberations, pooled_raw_embeddings)`` where:

            - *per_temp_deliberations* maps temperature → deliberation dict.
            - *pooled_raw_embeddings* maps ``"{problem_id}_{claim_id}_T{temp}"``
              to raw embedding arrays (all temperatures pooled for PCA).
        """
        per_temp: Dict[float, Dict[str, Any]] = {}
        pooled_raw: Dict[str, np.ndarray] = {}

        for temp in self.config.temperatures:
            logger.info(
                "Starting deliberation collection at T=%.1f", temp
            )
            delib_dict, raw_embs = self.collect_temperature(
                problems, temp, checkpoint_every=checkpoint_every
            )
            per_temp[temp] = delib_dict

            # Pool embeddings with temperature-tagged keys for later PCA
            for key, vec in raw_embs.items():
                pooled_key = f"{key}_T{temp}"
                pooled_raw[pooled_key] = vec

            logger.info(
                "T=%.1f complete: %d deliberations, cost so far: $%.4f",
                temp,
                delib_dict["total_deliberations"],
                self.llm.get_estimated_cost(),
            )

        return per_temp, pooled_raw

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_deliberations(
        self,
        temperature: float,
        delib_dict: Dict[str, Any],
        raw_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[Path, Path]:
        """
        Save deliberation JSON and raw embeddings NPZ for one temperature.

        Returns:
            ``(json_path, npz_path)``
        """
        temp_str = f"{temperature:.1f}".replace(".", "_")
        json_path = self.results_dir / f"deliberations_T{temp_str}.json"
        npz_path = self.results_dir / f"embeddings_raw_T{temp_str}.npz"

        with open(json_path, "w") as f:
            json.dump(delib_dict, f, indent=2)
        logger.info("Saved deliberations to %s", json_path)

        if self.config.save_raw_embeddings:
            np.savez_compressed(npz_path, **raw_embeddings)
            logger.info("Saved raw embeddings to %s", npz_path)

        return json_path, npz_path

    def save_all(
        self,
        per_temp: Dict[float, Dict[str, Any]],
        per_temp_embeddings: Dict[float, Dict[str, np.ndarray]],
    ) -> None:
        """Save deliberation results for every temperature."""
        for temp, delib_dict in per_temp.items():
            self.save_deliberations(temp, delib_dict, per_temp_embeddings[temp])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_output_dict(
        temperature: float,
        entries: List[Dict[str, Any]],
        problems: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "temperature": temperature,
            "total_deliberations": len(entries),
            "num_problems": len(problems),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "deliberations": entries,
        }

    def _save_checkpoint(
        self,
        temperature: float,
        entries: List[Dict[str, Any]],
        raw_embeddings: Dict[str, np.ndarray],
        partial: bool = True,
    ) -> None:
        temp_str = f"{temperature:.1f}".replace(".", "_")
        suffix = "_partial" if partial else ""
        ckpt_path = self.results_dir / f"deliberations_T{temp_str}{suffix}.json"
        with open(ckpt_path, "w") as f:
            json.dump(
                {
                    "temperature": temperature,
                    "total_deliberations": len(entries),
                    "partial": partial,
                    "deliberations": entries,
                },
                f,
                indent=2,
            )
        logger.debug("Checkpoint saved: %s (%d entries)", ckpt_path, len(entries))

    # ------------------------------------------------------------------
    # Usage summary
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return combined LLM + embedding usage and cost."""
        summary: Dict[str, Any] = {"llm": {}, "embedding": {}}
        if self._llm is not None:
            summary["llm"] = self._llm.get_usage_summary()
        if self._embedder is not None:
            summary["embedding"] = self._embedder.get_usage_summary()
        total = summary["llm"].get("estimated_cost_usd", 0) + summary[
            "embedding"
        ].get("estimated_cost_usd", 0)
        summary["total_estimated_cost_usd"] = total
        return summary

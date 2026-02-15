"""
Assessment Collector for the Temperature Study.

For each claim in the pool, sends an assessment prompt at the specified
temperature, records the text response, and embeds it.  One assessment
per (claim, temperature) — not per (problem, claim) — so that the
resulting embedding maps directly to w[r] in the Stan model.
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

logger = logging.getLogger(__name__)


class AssessmentCollector:
    """
    Collect per-claim assessment texts and embeddings at each temperature.

    Unlike the former ``DeliberationCollector``, this iterates over
    *claims* (not problem × claim), producing exactly one assessment and
    one embedding per claim per temperature.  This is consistent with
    the m_0 model where ``w[r]`` is a fixed property of alternative *r*.
    """

    def __init__(
        self,
        config: StudyConfig,
        claims: List[Dict[str, Any]],
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.config = config
        self.claims = claims
        self.claim_lookup: Dict[str, Dict[str, Any]] = {
            c["id"]: c for c in claims
        }

        # Lazy initialisation — only create when actually needed
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
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Collect assessments for all claims at a single temperature.

        Args:
            temperature: The LLM temperature for this condition.

        Returns:
            ``(assessments_dict, raw_embeddings)`` where:

            - *assessments_dict* contains the assessment text for each
              claim, keyed by claim ID.
            - *raw_embeddings* maps claim IDs to 1-D numpy arrays (raw
              embedding vectors before PCA).
        """
        system_prompt = self._prompts["assessment"]["system"]
        user_template = self._prompts["assessment"]["user"]

        all_entries: List[Dict[str, Any]] = []
        raw_embeddings: Dict[str, np.ndarray] = {}

        for claim in self.claims:
            cid = claim["id"]

            user_prompt = user_template.format(
                claim_description=claim["description"],
            )

            response_text = self.llm.generate(
                user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=256,
            )

            # Embed the assessment
            embedding_vec = self.embedder.embed_single(response_text)
            raw_embeddings[cid] = np.array(embedding_vec)

            all_entries.append(
                {
                    "claim_id": cid,
                    "temperature": temperature,
                    "response": response_text,
                    "embedding_key": cid,
                }
            )

        assessments_dict = self._build_output_dict(temperature, all_entries)
        return assessments_dict, raw_embeddings

    def collect_all_temperatures(
        self,
    ) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, np.ndarray]]]:
        """
        Collect assessments across all configured temperatures.

        Returns:
            ``(per_temp_assessments, per_temp_raw_embeddings)`` where:

            - *per_temp_assessments* maps temperature → assessments dict.
            - *per_temp_raw_embeddings* maps temperature → {claim_id → raw
              embedding array}.
        """
        per_temp: Dict[float, Dict[str, Any]] = {}
        per_temp_raw: Dict[float, Dict[str, np.ndarray]] = {}

        for temp in self.config.temperatures:
            logger.info(
                "Starting assessment collection at T=%.1f", temp
            )
            assess_dict, raw_embs = self.collect_temperature(temp)
            per_temp[temp] = assess_dict
            per_temp_raw[temp] = raw_embs

            logger.info(
                "T=%.1f complete: %d assessments, cost so far: $%.4f",
                temp,
                assess_dict["total_assessments"],
                self.llm.get_estimated_cost(),
            )

        return per_temp, per_temp_raw

    # ------------------------------------------------------------------
    # Helpers: extract assessment texts for the choice phase
    # ------------------------------------------------------------------

    @staticmethod
    def get_assessment_texts(
        assessments_dict: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Extract a mapping of claim_id → assessment text from the output dict.

        This is what the choice collector needs to build its prompts.
        """
        return {
            entry["claim_id"]: entry["response"]
            for entry in assessments_dict["assessments"]
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_assessments(
        self,
        temperature: float,
        assessments_dict: Dict[str, Any],
        raw_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[Path, Path]:
        """
        Save assessment JSON and raw embeddings NPZ for one temperature.

        Returns:
            ``(json_path, npz_path)``
        """
        temp_str = f"{temperature:.1f}".replace(".", "_")
        json_path = self.results_dir / f"assessments_T{temp_str}.json"
        npz_path = self.results_dir / f"embeddings_raw_T{temp_str}.npz"

        with open(json_path, "w") as f:
            json.dump(assessments_dict, f, indent=2)
        logger.info("Saved assessments to %s", json_path)

        if self.config.save_raw_embeddings:
            np.savez_compressed(npz_path, **raw_embeddings)
            logger.info("Saved raw embeddings to %s", npz_path)

        return json_path, npz_path

    def save_all(
        self,
        per_temp: Dict[float, Dict[str, Any]],
        per_temp_embeddings: Dict[float, Dict[str, np.ndarray]],
    ) -> None:
        """Save assessment results for every temperature."""
        for temp, assess_dict in per_temp.items():
            self.save_assessments(
                temp, assess_dict, per_temp_embeddings[temp]
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_assessments(filepath: str | Path) -> Dict[str, Any]:
        """Load an ``assessments_T{temp}.json`` file."""
        with open(filepath) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_output_dict(
        temperature: float,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "temperature": temperature,
            "total_assessments": len(entries),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "assessments": entries,
        }

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

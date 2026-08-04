"""
Response parsing for the SEU sensitivity study (study plan §6.4, §5).

Two parsers live here, both of which record *how* a response was resolved.

``parse_choice_response``
    Reads the hardened ``ANSWER: n`` token first and falls back to the
    inherited free-form heuristics only when no token is present (§6.4, §13
    item 25).  The reason for the hardening is that the free-form parser's
    error profile is **menu-size dependent**: it accepts a lone in-range digit
    anywhere in the response, so at menu size 4 a stray "8" is out of range and
    harmlessly ignored, while at size 8 the same token is in range and either
    mis-parses or forces an NA.  Hiring and procurement vignettes are exactly
    where stray in-range integers live.  Left unhardened, the reported NA rate
    -- an RQ5 quantity -- would be partly an artefact of the RQ6 factor.

``parse_probabilities``
    Extracts the structured ``PROBABILITIES:`` line the assessment prompt asks
    for.  These values are the §5 predictive-validity check's *outcome* and are
    never fed to Stan.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from applications.temperature_study.llm_client import parse_choice as _legacy_parse_choice

from .schemas import RESOLUTION_PATHS

logger = logging.getLogger(__name__)

__all__ = [
    "parse_choice_response",
    "parse_probabilities",
    "answer_token_values",
]


_ANSWER_RE = re.compile(r"ANSWER\s*[:\-=]?\s*(\d+)", re.IGNORECASE)
_PROBABILITIES_RE = re.compile(r"PROBABILITIES\s*[:\-=]?\s*(.+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+\s*%?")

#: Maximum deviation from 1.0 tolerated before the probability line is treated
#: as unparseable.  Wider than the schema's post-hoc tolerance because values
#: are renormalized here; see :func:`parse_probabilities`.
_SUM_TOLERANCE = 0.10


def answer_token_values(response: str) -> List[int]:
    """Every integer the response offers via an ``ANSWER:`` token."""
    if not response:
        return []
    return [int(value) for value in _ANSWER_RE.findall(response)]


def parse_choice_response(
    response: Optional[str], n_max: int
) -> Tuple[Optional[int], str]:
    """
    Resolve a choice response to a 1-indexed position within the presentation.

    Returns ``(position, resolution_path)`` where ``resolution_path`` is one of
    :data:`schemas.RESOLUTION_PATHS`.  Recording the path is what lets §6.4
    separate parser-induced NA from genuine refusal.

    An ``ANSWER:`` token that is ambiguous (several disagreeing values) or out
    of range resolves to ``unresolved`` rather than falling through to the
    weaker heuristics.  Falling through would let a *less* reliable parser
    override an explicit answer the model got wrong, which manufactures data
    where the honest reading is "the model did not answer usably".
    """
    if not response or not response.strip():
        return None, "unresolved"

    values = answer_token_values(response)
    if values:
        distinct = set(values)
        if len(distinct) == 1 and 1 <= values[0] <= n_max:
            return values[0], "answer_token"
        logger.debug(
            "Ambiguous or out-of-range ANSWER token(s) %s for n_max=%d", values, n_max
        )
        return None, "unresolved"

    fallback = _legacy_parse_choice(response, n_max)
    if fallback is not None:
        logger.debug("Resolved via legacy fallback parser (n_max=%d)", n_max)
        return fallback, "fallback_parse"

    return None, "unresolved"


def parse_probabilities(
    response: Optional[str], k: int, *, tolerance: float = _SUM_TOLERANCE
) -> Optional[List[float]]:
    """
    Extract the K outcome probabilities from an assessment response.

    Returns ``None`` when the line is absent, malformed, has the wrong length,
    or sums too far from one.  Otherwise the values are **renormalized** to sum
    exactly to one.

    Renormalizing a near-miss is safe here in a way it would not be elsewhere:
    these numbers are context and check-outcome only (§5, §6.1 step 2), never a
    Stan input, and the check regresses the *relative* likelihood judgment on
    the item embedding.  Rejecting a 0.2/0.5/0.4 reply outright would discard a
    perfectly usable judgment over an arithmetic slip.
    """
    if not response or k < 1:
        return None

    matches = _PROBABILITIES_RE.findall(response)
    if not matches:
        return None

    # Use the last labelled line: models sometimes illustrate the format before
    # committing to their actual answer.
    tokens = _NUMBER_RE.findall(matches[-1])
    if len(tokens) != k:
        logger.debug("Expected %d probabilities, found %d in %r", k, len(tokens), matches[-1])
        return None

    values: List[float] = []
    percent_seen = False
    for token in tokens:
        cleaned = token.strip()
        if cleaned.endswith("%"):
            percent_seen = True
            cleaned = cleaned[:-1].strip()
        try:
            values.append(float(cleaned))
        except ValueError:
            return None

    if any(value < 0 for value in values):
        return None

    total = sum(values)
    # Percentages: either explicitly marked, or unmarked but summing to ~100.
    if percent_seen or (
        total > 1.0 + tolerance and abs(total - 100.0) <= 100.0 * tolerance
    ):
        values = [value / 100.0 for value in values]
        total = sum(values)

    # The epsilon matters: 0.2 + 0.5 + 0.4 is 1.1000000000000001 in binary
    # floating point, so a bare `> tolerance` comparison rejects a reply that
    # sits exactly on the boundary.
    if total <= 0 or abs(total - 1.0) > tolerance + 1e-9:
        logger.debug("Probability line sums to %.3f, outside tolerance", total)
        return None

    return [value / total for value in values]


assert set(RESOLUTION_PATHS) == {"answer_token", "fallback_parse", "unresolved"}

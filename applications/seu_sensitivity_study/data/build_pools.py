"""
Deterministic construction of the ``venture`` and ``hiring`` item pools (§5, §6.3, §7.3).

Nothing here is free text.  Every surface string in a rendered vignette comes
from a closed level -> phrase table, so the pools are a *function* of an
authored merit-attribute grid rather than a corpus someone typed.  Three study
requirements force that discipline:

**§5 -- quality-relevant variation must dominate the embedding.**  Item-text
embeddings are the sole signal channel into ``h_m01``.  If items varied in
prose style as much as in merit, the principal embedding axis would track style
and the R4 PC1-as-quality validation would fail.  Templated rendering with a
fixed line structure leaves merit as the only thing that moves.

**§3.3 -- the matched pair must vary the task label and nothing else.**  The
same 24 merit vectors are rendered twice: once under the hiring label (the
``matched`` family of the hiring pool) and once under the procurement label
(the ``procurement`` family of the venture pool), linked by ``matched_key``.
Because both renderings are driven from one vector list, the pair cannot drift
out of alignment through an edit to one side.

**§7.3 -- hiring sterilization, enforced by construction.**  Candidates are
described by job-relevant evidence only.  There is no place in the renderer to
put a name, a pronoun, an age, a nationality, a disability, or a school: those
strings do not exist in the phrase tables, and :func:`audit_sterilization`
re-checks both the tables and the rendered output so that a later edit to a
phrase cannot quietly reintroduce a cue.

Design decisions worth knowing before reading the code
------------------------------------------------------

*The quality label is monotone in merit total.*  ``strong`` is total >= 12,
``weak`` is total <= 5, ``ambiguous`` is the band between.  The tempting
alternative -- "high total but internally conflicting counts as ambiguous" --
would place some ambiguous items above some strong items on any merit-tracking
axis, which sets the R4 check (PC1 separates the labels) up to fail *by
construction*.  Internal conflict is instead carried by ``merit_dispersion``,
which is authored deliberately per band and handed to the item-validation gate
as an eta-gap tuning handle, never as a labelling input.

*Each label draws from its own level-composition band.*  Monotone totals are
not sufficient on their own: a vector totalling 11 with four top levels and one
bottom level renders as a strong item but is labelled ambiguous.  So strong
items carry no bottom-tier phrase, weak items carry at most one mid-tier
phrase, and ambiguous items carry both a high and a low one.  The labels are
then legible from the text, which is the precondition for R4 testing
measurement rather than testing the authoring.

*Context attributes are orthogonal to merit by construction.*  Experience band
is an age proxy, which §7.3 coarsens to 2-4 / 5-8 / 9+.  Coarsening alone is
not enough: if band correlated with merit it would become a quality cue through
the correlation, which is the thing coarsening was meant to prevent.  Bands and
functions are therefore dealt out by merit *rank*, so they interleave across
the quality spread, and :func:`audit_context_orthogonality` fails the build if
the realised rank correlation exceeds a small tolerance.

*Startup and candidate families share one merit lattice.*  The two primary
families are rendered from the same 36 merit vectors under different domain
vocabularies.  That is deliberate: §6.3 wants pool difficulty equalized so that
"pool" is as close to an interchangeable relabeling as three domains allow, and
authoring the same lattice twice is the cheapest way to get there.  The matched
vectors are selected first and excluded from the primary selection, so no two
families in a pool can render the same vector.

Run ``python -m applications.seu_sensitivity_study.data.build_pools`` to
regenerate ``venture.json`` and ``hiring.json`` in place.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .. import schemas

logger = logging.getLogger(__name__)

__all__ = [
    "GRID_VERSION",
    "MERIT_DIMENSIONS",
    "STARTUP_DIMENSIONS",
    "MeritVector",
    "quality_label",
    "merit_total",
    "merit_dispersion",
    "select_vectors",
    "reservation_position",
    "build_venture_pool",
    "build_hiring_pool",
    "audit_sterilization",
    "audit_context_orthogonality",
    "SterilizationError",
    "main",
]


#: Bumped when the grid, the phrase tables, or the selection rule change in a
#: way that alters rendered text.  Recorded on every item.
GRID_VERSION = "merit-grid-v1"

_DATA_DIR = Path(__file__).resolve().parent

#: Ordinal merit levels.  Four levels per dimension keeps the lattice small
#: enough to enumerate exhaustively (4**5 = 1024) and therefore to select from
#: deterministically rather than by sampling.
LEVELS: Tuple[int, ...] = (0, 1, 2, 3)

#: Label-neutral evidence *sources*, shared by the hiring and procurement
#: renderings.  They are sources rather than domain nouns precisely so that one
#: merit vector can be read as a candidate profile and as a vendor profile
#: without either reading being a translation of the other (§3.3).
MERIT_DIMENSIONS: Tuple[str, ...] = (
    "direct_evidence",
    "track_record",
    "structured_review",
    "reference_checks",
    "scope_coverage",
)

#: The startup family is not part of the matched pair, so it gets its own
#: domain-appropriate dimensions.  The *arity* and the level scale match, so
#: one selection routine and one label rule serve both grids.
STARTUP_DIMENSIONS: Tuple[str, ...] = (
    "team_track_record",
    "traction_evidence",
    "market_validation",
    "product_maturity",
    "unit_economics",
)

MeritVector = Tuple[int, ...]

# -- Label bands (authored; monotone in total, see module docstring) ---------

STRONG_MIN_TOTAL = 12
WEAK_MAX_TOTAL = 5


def _in_strong_band(vector: Sequence[int]) -> bool:
    return min(vector) >= 2


def _in_ambiguous_band(vector: Sequence[int]) -> bool:
    return max(vector) >= 2 and min(vector) <= 1


def _in_weak_band(vector: Sequence[int]) -> bool:
    return max(vector) <= 2 and sum(1 for level in vector if level >= 2) <= 1


#: Which lattice points each label is allowed to draw from, *on top of* the
#: total-based label rule.
#:
#: These are composition constraints rather than dispersion ranges because the
#: label has to be legible from the rendered phrase mix, not merely from the
#: arithmetic.  Strong items carry no bottom-tier phrase (every level >= 2);
#: weak items carry at most one mid-tier phrase; ambiguous items carry both a
#: high and a low phrase, which is what makes them ambiguous to read rather
#: than just middling to score.  Without this, a dispersion-3 vector totalling
#: 11 would render as four top-tier lines and one bottom-tier line -- textually
#: a strong item, labelled ambiguous -- and the R4 check that PC1 separates the
#: labels would be set up to fail on authoring rather than on measurement.
BAND_PREDICATES = {
    "strong": _in_strong_band,
    "ambiguous": _in_ambiguous_band,
    "weak": _in_weak_band,
}

#: Items per family and label.  These sit well above the hard floors enforced
#: by ``schemas.validate_item_pool`` (>= 8 per family) and
#: ``problem_generation`` (>= 2 strong / >= 1 ambiguous / >= 7 weak for a
#: size-8 menu).  The weak counts in particular are generous on purpose: menu
#: items are drawn without replacement, so a family holding exactly 7 weak
#: items would put the identical filler set in *every* size-8 menu, giving the
#: size-8 eta-gap distribution zero between-menu variance -- exactly the
#: quantity §6.3 asks the gate to equalize across sizes.
PRIMARY_COUNTS: Dict[str, int] = {"strong": 10, "ambiguous": 10, "weak": 16}
MATCHED_COUNTS: Dict[str, int] = {"strong": 6, "ambiguous": 6, "weak": 12}


# ---------------------------------------------------------------------------
# The merit lattice
# ---------------------------------------------------------------------------


def merit_total(vector: Sequence[int]) -> int:
    return int(sum(vector))


def merit_dispersion(vector: Sequence[int]) -> int:
    return int(max(vector) - min(vector))


def quality_label(vector: Sequence[int]) -> str:
    """Authored quality label; a monotone function of :func:`merit_total`."""
    total = merit_total(vector)
    if total >= STRONG_MIN_TOTAL:
        return "strong"
    if total <= WEAK_MAX_TOTAL:
        return "weak"
    return "ambiguous"


def _candidate_vectors(label: str) -> List[MeritVector]:
    """Every lattice point carrying *label* and inside its composition band."""
    in_band = BAND_PREDICATES[label]
    return [
        vector
        for vector in itertools.product(LEVELS, repeat=len(MERIT_DIMENSIONS))
        if quality_label(vector) == label and in_band(vector)
    ]


def _spread(items: Sequence[MeritVector], count: int) -> List[MeritVector]:
    """
    Take *count* evenly spaced entries from *items*.

    Deterministic rather than sampled: the selection is reproducible from the
    source alone, with no seed to record and no possibility of a reroll
    changing the authored pool.
    """
    if count > len(items):
        raise ValueError(
            f"need {count} vectors but only {len(items)} lattice points satisfy the "
            f"band; widen the composition band or lower the count"
        )
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    picked: List[MeritVector] = []
    seen: Set[MeritVector] = set()
    for index in range(count):
        vector = items[round(index * step)]
        while vector in seen:  # pragma: no cover - only reachable on a tie
            vector = items[(items.index(vector) + 1) % len(items)]
        picked.append(vector)
        seen.add(vector)
    return picked


def select_vectors(
    counts: Mapping[str, int], *, exclude: Optional[Set[MeritVector]] = None
) -> Dict[str, List[MeritVector]]:
    """
    Choose merit vectors per label, skipping anything in *exclude*.

    ``exclude`` is how family disjointness is guaranteed: the matched vectors
    are selected first, then handed in here as the exclusion set for the
    primary families, so no pool can contain two items built from the same
    merit vector.
    """
    blocked = set(exclude or ())
    selected: Dict[str, List[MeritVector]] = {}
    for label in schemas.QUALITY_LABELS:
        count = counts.get(label, 0)
        if not count:
            continue
        available = [v for v in _candidate_vectors(label) if v not in blocked]
        available.sort(key=lambda v: (merit_total(v), merit_dispersion(v), v))
        selected[label] = _spread(available, count)
    return selected


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Renderer:
    """
    A closed vocabulary plus a fixed line structure.

    ``phrases`` maps ``dimension -> level -> sentence``.  Rendering never
    concatenates anything outside these tables, the header, and the item id,
    which is what makes :func:`audit_sterilization` a check of a finite set
    rather than of unbounded prose.
    """

    template_id: str
    id_prefix: str
    dimensions: Tuple[str, ...]
    line_labels: Dict[str, str]
    phrases: Dict[str, Dict[int, str]]
    reservation_template: str
    reservation_subjects: Dict[str, str]
    header_template: str
    #: ``key -> ordered field values``.  Order is load-bearing for the matched
    #: pair: the hiring and procurement renderers list parallel values in the
    #: same positions, so one merit vector lands in the same context slot under
    #: both labels.
    context_values: Dict[str, Tuple[str, ...]]

    def render(
        self, item_id: str, vector: Sequence[int], context: Mapping[str, str]
    ) -> str:
        lines = [self.header_template.format(item_id=item_id, **context), ""]
        for dimension, level in zip(self.dimensions, vector):
            lines.append(f"{self.line_labels[dimension]}: {self.phrases[dimension][level]}")

        reservation = reservation_position(vector)
        if reservation is not None:
            subject = self.reservation_subjects[self.dimensions[reservation]]
            lines.append(self.reservation_template.format(subject=subject))
        return "\n".join(lines)


def reservation_position(vector: Sequence[int]) -> Optional[int]:
    """
    Which evidence source (if any) the file records a reservation against.

    Returned as a *position* rather than a name because the same merit vector
    is rendered under two dimension vocabularies (shared evidence sources for
    the matched families, venture-specific ones for startups); a position is
    the only thing both readings agree on.

    Only fired for internally conflicting profiles -- a uniformly weak file has
    no single weakest part worth naming, and saying so anyway would add a
    quality cue the merit vector does not carry.
    """
    if merit_dispersion(vector) < 2:
        return None
    lowest = min(vector)
    if lowest > 1:
        return None
    return list(vector).index(lowest)


_HIRING_RENDERER = Renderer(
    template_id="matched.hiring.v1",
    id_prefix="H",
    dimensions=MERIT_DIMENSIONS,
    line_labels={
        "direct_evidence": "Work sample",
        "track_record": "Prior delivery record",
        "structured_review": "Structured interview panel",
        "reference_checks": "Reference checks",
        "scope_coverage": "Scope coverage",
    },
    phrases={
        "direct_evidence": {
            0: "incomplete; did not meet the stated requirements.",
            1: "met the basic requirements but showed several correctness and clarity problems.",
            2: "met all requirements and was clearly organised, with minor gaps.",
            3: "met all requirements and handled edge cases unusually thoroughly.",
        },
        "track_record": {
            0: "no documented outcomes from comparable prior work.",
            1: "documented outcomes on one comparable prior project, with mixed results.",
            2: "documented successful outcomes on several comparable prior projects.",
            3: "documented successful outcomes on a sustained series of comparable prior projects.",
        },
        "structured_review": {
            0: "scored the file in the bottom band of the rubric.",
            1: "scored the file in the lower-middle band of the rubric.",
            2: "scored the file in the upper-middle band of the rubric.",
            3: "scored the file in the top band of the rubric.",
        },
        "reference_checks": {
            0: "could not be completed.",
            1: "returned brief, non-specific responses.",
            2: "corroborated the claimed contributions.",
            3: "corroborated the claimed contributions in specific detail and were uniformly positive.",
        },
        "scope_coverage": {
            0: "covers few of the capabilities the role scope requires.",
            1: "covers some of the capabilities the role scope requires.",
            2: "covers most of the capabilities the role scope requires.",
            3: "covers all of the capabilities the role scope requires.",
        },
    },
    reservation_template="Noted reservation: reviewers flagged the {subject} as the weakest part of the file.",
    reservation_subjects={
        "direct_evidence": "work sample",
        "track_record": "delivery record",
        "structured_review": "panel score",
        "reference_checks": "reference evidence",
        "scope_coverage": "scope gap",
    },
    header_template=(
        "Candidate {item_id} -- {function}; {experience_band} years of experience "
        "in this function."
    ),
    context_values={
        "function": (
            "data platform engineering",
            "records management",
            "network operations",
            "quality assurance",
            "logistics coordination",
        ),
        "experience_band": ("2-4", "5-8", "9+"),
    },
)


_PROCUREMENT_RENDERER = Renderer(
    template_id="matched.procurement.v1",
    id_prefix="P",
    dimensions=MERIT_DIMENSIONS,
    line_labels={
        "direct_evidence": "Technical proposal",
        "track_record": "Prior contract delivery record",
        "structured_review": "Technical review board",
        "reference_checks": "Past-client references",
        "scope_coverage": "Scope coverage",
    },
    phrases={
        "direct_evidence": {
            0: "incomplete; did not meet the stated requirements.",
            1: "met the basic requirements but showed several correctness and clarity problems.",
            2: "met all requirements and was clearly organised, with minor gaps.",
            3: "met all requirements and handled edge cases unusually thoroughly.",
        },
        "track_record": {
            0: "no documented outcomes from comparable prior contracts.",
            1: "documented outcomes on one comparable prior contract, with mixed results.",
            2: "documented successful outcomes on several comparable prior contracts.",
            3: "documented successful outcomes on a sustained series of comparable prior contracts.",
        },
        "structured_review": {
            0: "scored the bid in the bottom band of the rubric.",
            1: "scored the bid in the lower-middle band of the rubric.",
            2: "scored the bid in the upper-middle band of the rubric.",
            3: "scored the bid in the top band of the rubric.",
        },
        "reference_checks": {
            0: "could not be completed.",
            1: "returned brief, non-specific responses.",
            2: "corroborated the claimed delivery history.",
            3: "corroborated the claimed delivery history in specific detail and were uniformly positive.",
        },
        "scope_coverage": {
            0: "covers few of the capabilities the statement of work requires.",
            1: "covers some of the capabilities the statement of work requires.",
            2: "covers most of the capabilities the statement of work requires.",
            3: "covers all of the capabilities the statement of work requires.",
        },
    },
    reservation_template="Noted reservation: evaluators flagged the {subject} as the weakest part of the bid.",
    reservation_subjects={
        "direct_evidence": "technical proposal",
        "track_record": "delivery record",
        "structured_review": "review board score",
        "reference_checks": "reference evidence",
        "scope_coverage": "scope gap",
    },
    header_template=(
        "Vendor {item_id} -- {function}; {experience_band} years operating in this "
        "service category."
    ),
    context_values={
        "function": (
            "data platform services",
            "records management services",
            "network operations services",
            "quality assurance services",
            "logistics coordination services",
        ),
        "experience_band": ("2-4", "5-8", "9+"),
    },
)


_STARTUP_RENDERER = Renderer(
    template_id="venture.startup.v1",
    id_prefix="V",
    dimensions=STARTUP_DIMENSIONS,
    line_labels={
        "team_track_record": "Team record",
        "traction_evidence": "Traction",
        "market_validation": "Demand evidence",
        "product_maturity": "Product maturity",
        "unit_economics": "Unit economics",
    },
    phrases={
        "team_track_record": {
            0: "founding team has no documented record of shipping a comparable product.",
            1: "founding team has shipped one comparable product, with mixed results.",
            2: "founding team has shipped several comparable products successfully.",
            3: "founding team has a sustained record of shipping comparable products successfully.",
        },
        "traction_evidence": {
            0: "no usage or revenue reported to date.",
            1: "early usage reported, well below the plan filed at the last round.",
            2: "close to the plan filed at the last round.",
            3: "well ahead of the plan filed at the last round.",
        },
        "market_validation": {
            0: "asserted but not evidenced by any signed customer.",
            1: "a small number of unpaid pilots.",
            2: "several paid pilots.",
            3: "multi-year paid commitments across several customers.",
        },
        "product_maturity": {
            0: "product exists only as a prototype with no production deployment.",
            1: "product is in limited production with recurring reliability problems.",
            2: "product is in stable production with a documented roadmap.",
            3: "product is in stable production at scale with a documented roadmap and a defensible technical position.",
        },
        "unit_economics": {
            0: "negative, with no credible path to contribution margin.",
            1: "negative but improving on the reported cohort trend.",
            2: "positive at the contribution-margin level.",
            3: "positive at the contribution-margin level and widening across cohorts.",
        },
    },
    reservation_template=(
        "Noted reservation: the investment committee flagged the {subject} as the "
        "weakest part of the file."
    ),
    reservation_subjects={
        "team_track_record": "team record",
        "traction_evidence": "traction evidence",
        "market_validation": "demand evidence",
        "product_maturity": "product maturity",
        "unit_economics": "unit economics",
    },
    header_template="Venture {item_id} -- {sector}; {stage_band} stage.",
    context_values={
        "sector": (
            "industrial software",
            "climate hardware",
            "logistics software",
            "materials processing",
            "health data infrastructure",
        ),
        "stage_band": ("pre-seed", "seed", "Series A"),
    },
)


# ---------------------------------------------------------------------------
# Context assignment
# ---------------------------------------------------------------------------


def assign_contexts(
    vectors: Sequence[MeritVector], renderer: Renderer
) -> List[Tuple[int, Dict[str, str]]]:
    """
    Deal context values out by merit rank so they interleave across quality.

    The cycle lengths are 5 (function/sector) and 3 (experience/stage band).
    Being coprime, every combination occurs, and neither field is recoverable
    from the other -- so a model cannot read one context field as a proxy for
    the other, nor either as a proxy for merit.

    Returns ``(rank, context)`` per input vector.  The rank is stored on the
    item so the matched-pair audit can check that a merit vector landed in the
    *same* context slot under both task labels.
    """
    order = sorted(range(len(vectors)), key=lambda i: (merit_total(vectors[i]), vectors[i]))
    contexts: List[Tuple[int, Dict[str, str]]] = [(0, {}) for _ in vectors]
    for rank, index in enumerate(order):
        contexts[index] = (
            rank,
            {
                key: values[rank % len(values)]
                for key, values in renderer.context_values.items()
            },
        )
    return contexts


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------


def _build_items(
    *,
    family: str,
    renderer: Renderer,
    selection: Mapping[str, List[MeritVector]],
    id_prefix: str,
    id_start: int,
    matched_keys: Optional[Mapping[MeritVector, str]] = None,
) -> List[Dict[str, Any]]:
    ordered: List[MeritVector] = []
    for label in schemas.QUALITY_LABELS:
        ordered.extend(selection.get(label, []))
    ordered.sort(key=lambda v: (-merit_total(v), v))

    contexts = assign_contexts(ordered, renderer)
    items: List[Dict[str, Any]] = []

    for offset, (vector, (rank, context)) in enumerate(zip(ordered, contexts)):
        item_id = f"{id_prefix}{id_start + offset:03d}"
        reservation = reservation_position(vector)
        reservation_name = (
            renderer.dimensions[reservation] if reservation is not None else None
        )

        items.append(
            {
                "id": item_id,
                "family": family,
                "text": renderer.render(item_id, vector, context),
                "quality_label": quality_label(vector),
                "matched_key": (
                    matched_keys.get(vector) if matched_keys is not None else None
                ),
                "attributes": {
                    "merit_vector": dict(zip(renderer.dimensions, (int(v) for v in vector))),
                    "merit_total": merit_total(vector),
                    "merit_dispersion": merit_dispersion(vector),
                    "context": dict(context),
                    "context_rank": rank,
                    "reservation": reservation_name,
                    "grid_version": GRID_VERSION,
                    "template_id": renderer.template_id,
                },
            }
        )
    return items


def _matched_key_map(selection: Mapping[str, List[MeritVector]]) -> Dict[MeritVector, str]:
    """Stable ``MK01..MKnn`` keys, ordered by descending merit then lattice order."""
    ordered: List[MeritVector] = []
    for label in schemas.QUALITY_LABELS:
        ordered.extend(selection.get(label, []))
    ordered.sort(key=lambda v: (-merit_total(v), v))
    return {vector: f"MK{index:02d}" for index, vector in enumerate(ordered, start=1)}


# ---------------------------------------------------------------------------
# Pool builders
# ---------------------------------------------------------------------------

VENTURE_CONSEQUENCES = [
    "Loss: the committed capital is not recovered",
    "Break-even: the commitment is recovered with no material gain",
    "High return: the commitment returns a substantial gain",
]

HIRING_CONSEQUENCES = [
    "Underperforms: falls short of what the role requires",
    "Meets expectations: performs at the level the role requires",
    "Exceeds expectations: performs well above the level the role requires",
]


def _selections() -> Tuple[
    Dict[str, List[MeritVector]], Dict[str, List[MeritVector]], Dict[MeritVector, str]
]:
    """Matched vectors first, then primary vectors disjoint from them."""
    matched = select_vectors(MATCHED_COUNTS)
    used = {vector for vectors in matched.values() for vector in vectors}
    primary = select_vectors(PRIMARY_COUNTS, exclude=used)
    return matched, primary, _matched_key_map(matched)


def build_venture_pool() -> Dict[str, Any]:
    matched, primary, keys = _selections()
    items = _build_items(
        family="startup",
        renderer=_STARTUP_RENDERER,
        selection=primary,
        id_prefix="V",
        id_start=1,
    ) + _build_items(
        family="procurement",
        renderer=_PROCUREMENT_RENDERER,
        selection=matched,
        id_prefix="P",
        id_start=1,
        matched_keys=keys,
    )

    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "venture",
        "framing": "positive",
        "consequences": list(VENTURE_CONSEQUENCES),
        "families": {
            "startup": {
                "description": (
                    "Early-stage venture vignettes; the RQ5 comparator backbone. "
                    "Supplies the ordinary venture menus."
                ),
                "grid": list(STARTUP_DIMENSIONS),
                "primary": True,
            },
            "procurement": {
                "description": (
                    "Vendor vignettes rendered from the same merit vectors as the "
                    "hiring pool's matched family (§3.3). Own menu stratum; never "
                    "mixed into the comparator menus."
                ),
                "grid": list(MERIT_DIMENSIONS),
                "primary": False,
                "matched_with": "hiring/matched",
            },
        },
        "items": items,
        "provenance": _provenance("venture"),
    }


def build_hiring_pool() -> Dict[str, Any]:
    matched, primary, keys = _selections()
    items = _build_items(
        family="candidates",
        renderer=_HIRING_RENDERER,
        selection=primary,
        id_prefix="H",
        id_start=1,
    ) + _build_items(
        family="matched",
        renderer=_HIRING_RENDERER,
        selection=matched,
        id_prefix="H",
        id_start=101,
        matched_keys=keys,
    )

    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": "hiring",
        "framing": "positive",
        "consequences": list(HIRING_CONSEQUENCES),
        "families": {
            "candidates": {
                "description": (
                    "Sterilized candidate profiles (§7.3); the RQ5 override probe. "
                    "Supplies the ordinary hiring menus."
                ),
                "grid": list(MERIT_DIMENSIONS),
                "primary": True,
            },
            "matched": {
                "description": (
                    "The designated matched subset (§3.3): the same merit vectors as "
                    "the venture pool's procurement family, under the hiring label. "
                    "Own menu stratum."
                ),
                "grid": list(MERIT_DIMENSIONS),
                "primary": False,
                "matched_with": "venture/procurement",
            },
        },
        "items": items,
        "provenance": _provenance("hiring"),
    }


def _provenance(pool_id: str) -> Dict[str, Any]:
    return {
        "generator": "applications.seu_sensitivity_study.data.build_pools",
        "grid_version": GRID_VERSION,
        "deterministic": True,
        "note": (
            "Rendered from a closed level->phrase vocabulary; no free text. "
            "Regenerate with `python -m "
            "applications.seu_sensitivity_study.data.build_pools`."
        ),
        "label_rule": (
            f"quality_label is monotone in merit_total: strong >= {STRONG_MIN_TOTAL}, "
            f"weak <= {WEAK_MAX_TOTAL}, ambiguous between."
        ),
        "sterilized": pool_id == "hiring",
    }


# ---------------------------------------------------------------------------
# Audits (§7.3) -- these are the enforcement, not the documentation
# ---------------------------------------------------------------------------


class SterilizationError(ValueError):
    """Raised when a rendered pool carries a cue the protocol forbids."""


_FORBIDDEN_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("gendered term", r"\b(he|him|his|she|her|hers|mr|mrs|ms|miss|sir|madam|male|female|man|woman|men|women)\b"),
    ("age cue", r"\b(age|aged|born|birth|year[- ]old|young|older|elderly|senior citizen|generation)\b"),
    ("nationality or ethnicity cue", r"\b(american|british|chinese|indian|nigerian|mexican|european|asian|african|hispanic|latino|latina|caucasian|white|black|immigrant|visa|citizenship|native speaker|accent)\b"),
    ("religion cue", r"\b(christian|muslim|jewish|hindu|buddhist|catholic|church|mosque|synagogue|temple)\b"),
    ("disability cue", r"\b(disabled|disability|impair\w*|wheelchair|accommodation|neurodiver\w*|autis\w*|adhd|chronic illness)\b"),
    ("school or institution cue", r"\b(university|college|school|alma mater|ivy|mba|phd|degree|graduated|gpa|honou?rs student)\b"),
    ("marital or family cue", r"\b(married|single parent|spouse|husband|wife|children|pregnan\w*|maternity|paternity)\b"),
)

#: Digits are allowed only in an item id and in an experience band.  Anything
#: else numeric in a hiring or procurement vignette is a potential tenure or
#: age cue, so the audit strips the two permitted forms and then requires the
#: remainder to be digit-free.
_ALLOWED_NUMERIC = (r"\b[A-Z]\d{3}\b", r"\b(?:2-4|5-8|9\+) years\b")

#: Capitalisation inside a phrase is how a proper noun would enter.  Only
#: sentence-initial capitals and these tokens are permitted.
_ALLOWED_CAPITALISED = {"Series", "A"}


def _vocabulary_capitalised_tokens() -> Set[str]:
    """
    Every capitalised token the renderers can legitimately emit.

    Derived from the closed vocabulary rather than hand-listed, so it cannot
    drift out of date when a phrase or a line label changes.  Anything
    capitalised in a rendered item that is *not* in this set came from
    somewhere the templates do not control -- which is what a smuggled name,
    nationality, or institution looks like.  A deny-list of names could never
    be complete; this allow-list is complete by construction.
    """
    allowed = set(_ALLOWED_CAPITALISED)
    for renderer in (_HIRING_RENDERER, _PROCUREMENT_RENDERER, _STARTUP_RENDERER):
        sources: List[str] = [renderer.header_template, renderer.reservation_template]
        sources.extend(renderer.line_labels.values())
        sources.extend(renderer.reservation_subjects.values())
        for levels in renderer.phrases.values():
            sources.extend(levels.values())
        for values in renderer.context_values.values():
            sources.extend(values)
        for source in sources:
            allowed.update(re.findall(r"\b[A-Z][a-z]+\b", source))
    return allowed


def audit_sterilization(pool: Mapping[str, Any]) -> None:
    """
    Fail the build if any item carries a forbidden cue (§7.3).

    Applied to *rendered* text rather than to the phrase tables alone, so it
    also catches a cue introduced by a header, a context value, or an id
    scheme -- the places a purely table-level check would miss.
    """
    problems: List[str] = []
    allowed_tokens = _vocabulary_capitalised_tokens()

    for item in pool["items"]:
        text = item["text"]
        lowered = text.lower()
        for description, pattern in _FORBIDDEN_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                problems.append(f"{item['id']}: {description} {match.group(0)!r}")

        for token in re.findall(r"\b[A-Z][a-z]+\b", text):
            if token not in allowed_tokens:
                problems.append(
                    f"{item['id']}: proper noun {token!r} is not in the renderers' "
                    f"closed vocabulary; names, nationalities and institutions all "
                    f"arrive this way"
                )

        stripped = text
        for allowed in _ALLOWED_NUMERIC:
            stripped = re.sub(allowed, "", stripped)
        leftover = re.search(r"\d", stripped)
        if leftover:
            problems.append(
                f"{item['id']}: unexpected numeral near "
                f"{stripped[max(0, leftover.start() - 20):leftover.start() + 20]!r}; "
                f"precise figures can act as tenure or age cues"
            )

    if problems:
        raise SterilizationError(
            f"pool {pool['pool_id']!r} failed the §7.3 sterilization audit "
            f"({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems[:20])
        )


def audit_phrase_tables() -> None:
    """Reject a proper noun smuggled into a level phrase."""
    problems: List[str] = []
    for renderer in (_HIRING_RENDERER, _PROCUREMENT_RENDERER, _STARTUP_RENDERER):
        for dimension, levels in renderer.phrases.items():
            for level, phrase in levels.items():
                for token in re.findall(r"\b[A-Z][a-z]+\b", phrase):
                    if token not in _ALLOWED_CAPITALISED:
                        problems.append(
                            f"{renderer.template_id}.{dimension}[{level}]: "
                            f"capitalised token {token!r}"
                        )
    if problems:
        raise SterilizationError(
            "phrase tables carry capitalised tokens that may be proper nouns:\n  - "
            + "\n  - ".join(problems)
        )


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho without a scipy dependency (average ranks, ties handled)."""

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b) ** 0.5


#: Coarsening experience is only half of §7.3.  If band correlated with merit
#: it would act as a quality cue *through* the correlation, so the build also
#: requires it to be near-orthogonal.
MAX_CONTEXT_MERIT_RHO = 0.15


def audit_context_orthogonality(
    pool: Mapping[str, Any], *, tolerance: float = MAX_CONTEXT_MERIT_RHO
) -> Dict[str, Dict[str, float]]:
    """
    Check that no context attribute predicts merit (§7.3).

    Returns the realised rank correlations so the build can record them; raises
    if any exceeds *tolerance*.
    """
    realised: Dict[str, Dict[str, float]] = {}
    problems: List[str] = []

    for family in pool["families"]:
        items = [i for i in pool["items"] if i["family"] == family]
        if len(items) < 3:
            continue
        totals = [float(i["attributes"]["merit_total"]) for i in items]
        keys = sorted({k for i in items for k in i["attributes"]["context"]})
        realised[family] = {}
        for key in keys:
            values = sorted({i["attributes"]["context"][key] for i in items})
            coded = [float(values.index(i["attributes"]["context"][key])) for i in items]
            rho = _spearman(coded, totals)
            realised[family][key] = round(rho, 4)
            if abs(rho) > tolerance:
                problems.append(
                    f"{family}.{key}: |rho| = {abs(rho):.3f} > {tolerance} against "
                    f"merit_total"
                )

    if problems:
        raise SterilizationError(
            "context attributes are not orthogonal to merit; a correlated band or "
            "function acts as a quality cue (§7.3):\n  - " + "\n  - ".join(problems)
        )
    return realised


def audit_pool(pool: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every build-time audit and return what they measured."""
    audit_phrase_tables()
    schemas.check(schemas.validate_item_pool(dict(pool)), context=f"pool {pool['pool_id']!r}")

    texts = [item["text"] for item in pool["items"]]
    if len(set(texts)) != len(texts):
        raise SterilizationError(
            f"pool {pool['pool_id']!r} contains duplicate item text; families must be "
            f"built from disjoint merit vectors"
        )

    # Applied to every pool, not just hiring.  The procurement twin of a
    # matched hiring item has to be equally free of person cues, or the paired
    # contrast would vary sterilization alongside the task label.
    audit_sterilization(pool)
    orthogonality = audit_context_orthogonality(pool)
    return {
        "n_items": len(pool["items"]),
        "by_family": _label_counts(pool),
        "context_merit_rho": orthogonality,
    }


def _label_counts(pool: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for item in pool["items"]:
        counts.setdefault(item["family"], {label: 0 for label in schemas.QUALITY_LABELS})
        counts[item["family"]][item["quality_label"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all() -> Dict[str, Dict[str, Any]]:
    """Build and audit both pools without writing anything."""
    pools = {"venture": build_venture_pool(), "hiring": build_hiring_pool()}
    for pool in pools.values():
        audit_pool(pool)
    _audit_matched_alignment(pools["hiring"], pools["venture"])
    return pools


def _audit_matched_alignment(
    hiring: Mapping[str, Any], venture: Mapping[str, Any]
) -> None:
    """The matched pair must agree on every merit vector and quality label."""
    def by_key(pool: Mapping[str, Any], family: str) -> Dict[str, Dict[str, Any]]:
        return {
            item["matched_key"]: item
            for item in pool["items"]
            if item["family"] == family and item["matched_key"]
        }

    left = by_key(hiring, "matched")
    right = by_key(venture, "procurement")
    if set(left) != set(right):
        raise SterilizationError(
            f"matched keys differ between hiring/matched and venture/procurement: "
            f"{sorted(set(left) ^ set(right))}"
        )
    for key in sorted(left):
        a, b = left[key]["attributes"], right[key]["attributes"]
        if a["merit_vector"] != b["merit_vector"]:
            raise SterilizationError(f"{key}: merit vectors differ across labels")
        if left[key]["quality_label"] != right[key]["quality_label"]:
            raise SterilizationError(f"{key}: quality labels differ across labels")
        if a["context_rank"] != b["context_rank"]:
            raise SterilizationError(
                f"{key}: context slot differs across labels ({a['context_rank']} vs "
                f"{b['context_rank']}); the pair would vary context alongside the "
                f"task label"
            )
        if a["context"]["experience_band"] != b["context"]["experience_band"]:
            raise SterilizationError(f"{key}: experience band differs across labels")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--out-dir", default=str(_DATA_DIR), help="where the pool JSON files are written"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build and audit but do not write; exits non-zero if the files on disk "
        "differ from what the grid produces",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    pools = build_all()
    status = 0

    for pool_id, pool in pools.items():
        payload = json.dumps(pool, indent=2, ensure_ascii=True) + "\n"
        target = out_dir / f"{pool_id}.json"
        if args.check:
            current = target.read_text() if target.exists() else ""
            if current != payload:
                logger.error("%s is stale; re-run build_pools", target)
                status = 1
            else:
                logger.info("%s is up to date", target)
            continue
        target.write_text(payload)
        summary = audit_pool(pool)
        logger.info(
            "wrote %s: %d items, %s", target, summary["n_items"], summary["by_family"]
        )
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

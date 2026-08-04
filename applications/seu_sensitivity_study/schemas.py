"""
Frozen data schemas for the SEU sensitivity study (study plan v0.5).

This module is the single source of truth for every on-disk artefact the
collection pipeline produces.  Each phase writes JSON conforming to one of the
schemas below, and each downstream phase validates its input before using it,
so a schema drift surfaces at the boundary that caused it rather than as a
mis-shaped Stan payload eight steps later.

Artefact chain (see study plan §6.1)::

    item pool  ->  problem set  ->  assessment set  ->  choice set  ->  stan data
                                                     \\-> run manifest

Design notes
------------
* The wire format is **plain JSON-compatible dicts**, matching the existing
  ``temperature_study`` pipeline.  Validators are free functions returning a
  list of human-readable errors; :func:`check` turns them into an exception.
* Frozen study constants (menu sizes, presentation count, resolution paths)
  live here rather than in ``config.py`` because the validators enforce them
  and the pre-registration commit (§13) pins them.
* Nothing in this module imports numpy, cmdstanpy, or any provider SDK, so it
  is cheap to import from tests and from the item-authoring scripts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "ANSWER_TOKEN_VERSION",
    "MENU_SIZES",
    "NUM_PRESENTATIONS",
    "QUALITY_LABELS",
    "RESOLUTION_PATHS",
    "PROMPT_CONDITIONS",
    "ASSESSMENT_INSTRUCTION",
    "SchemaError",
    "check",
    "validate_item_pool",
    "validate_problem_set",
    "validate_assessment_set",
    "validate_choice_set",
    "validate_stan_data",
    "validate_run_manifest",
]


# ---------------------------------------------------------------------------
# Frozen study constants (study plan §3.4, §6.2, §6.4; pinned by §13)
# ---------------------------------------------------------------------------

#: Bumped only when an artefact's shape changes incompatibly.
SCHEMA_VERSION = "1.0"

#: Identifies the choice-prompt answer format (§6.4, §13 item 25).  Pinned in
#: the run manifest so a re-analysis can tell which parser produced a choice.
ANSWER_TOKEN_VERSION = "answer-token-v1"

#: Pre-registered menu-size range (§3.4).  Frozen only after the §8.5(e) power
#: sweep and the §12 token forecast; the fallback is ``(2, 3, 4, 6)``.
MENU_SIZES: tuple[int, ...] = (2, 4, 6, 8)

#: Frozen at 2 (§6.2): the floor at which a position flip is definable, hence
#: the minimum compatible with the R6 robustness read.
NUM_PRESENTATIONS = 2

#: Authored quality labels, also used as the menu difficulty strata (§6.3).
QUALITY_LABELS: tuple[str, ...] = ("strong", "ambiguous", "weak")

#: How a choice observation was resolved (§6.4).  Recorded per observation so
#: parser-induced NA is separable from genuine refusal.
RESOLUTION_PATHS: tuple[str, ...] = (
    "answer_token",
    "fallback_parse",
    "unresolved",
)

#: Prompt arms (§3.2).  Applied at the choice step only.
PROMPT_CONDITIONS: tuple[str, ...] = (
    "neutral",
    "seu_maximizing",
    "deliberative",
)

#: Assessments are collected once per model x pool under this instruction only
#: (§3.2 B2).  The validator refuses any other value.
ASSESSMENT_INSTRUCTION = "neutral"

#: Tolerance when checking that parsed outcome probabilities sum to one.
_PROB_SUM_TOL = 0.05


class SchemaError(ValueError):
    """Raised when an artefact violates its frozen schema."""


def check(errors: Sequence[str], *, context: str = "artefact") -> None:
    """Raise :class:`SchemaError` if *errors* is non-empty."""
    if errors:
        joined = "\n  - ".join(errors)
        raise SchemaError(f"Invalid {context} ({len(errors)} problem(s)):\n  - {joined}")


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, errors: List[str], message: str) -> bool:
    if not condition:
        errors.append(message)
    return condition


def _require_keys(
    obj: Any, keys: Iterable[str], errors: List[str], context: str
) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{context}: expected an object, got {type(obj).__name__}")
        return False
    missing = [k for k in keys if k not in obj]
    if missing:
        errors.append(f"{context}: missing required key(s) {missing}")
        return False
    return True


def _check_schema_version(obj: Dict[str, Any], errors: List[str], context: str) -> None:
    version = obj.get("schema_version")
    if version is None:
        errors.append(f"{context}: missing 'schema_version'")
    elif version != SCHEMA_VERSION:
        errors.append(
            f"{context}: schema_version {version!r} != expected {SCHEMA_VERSION!r}"
        )


def _check_enum(
    value: Any, allowed: Sequence[str], errors: List[str], context: str
) -> None:
    if value not in allowed:
        errors.append(f"{context}: {value!r} not in {list(allowed)}")


# ---------------------------------------------------------------------------
# 1. Item pool  (study plan §6.3)
# ---------------------------------------------------------------------------


def validate_item_pool(pool: Any) -> List[str]:
    """
    Validate a canonical item-pool artefact.

    Expected shape::

        {
          "schema_version": "1.0",
          "pool_id": "venture",
          "framing": "positive",                 # or "negative"
          "consequences": ["loss", "break_even", "high_return"],
          "families": {"startup": {...}, "procurement": {...}},
          "items": [
            {"id": "V001", "family": "startup", "text": "...",
             "quality_label": "strong", "attributes": {...},
             "matched_key": null}
          ]
        }

    Enforced invariants
    -------------------
    * ``consequences`` is ordered worst -> best and has K >= 2 entries.
    * item ids are unique; every ``family`` is declared in ``families``.
    * every ``quality_label`` is one of :data:`QUALITY_LABELS`.
    * each family carries at least ``max(MENU_SIZES)`` items -- families are
      drawn on as separate menu strata (§3.3, §3.4), so the >= 8 constraint
      binds per family, not merely per pool.
    * ``matched_key`` values pair at most one item per family, so a matched
      merit vector never resolves ambiguously.
    """
    errors: List[str] = []
    if not _require_keys(
        pool,
        ("schema_version", "pool_id", "framing", "consequences", "families", "items"),
        errors,
        "item pool",
    ):
        return errors

    _check_schema_version(pool, errors, "item pool")
    _check_enum(pool["framing"], ("positive", "negative"), errors, "item pool.framing")

    consequences = pool["consequences"]
    if not isinstance(consequences, list) or len(consequences) < 2:
        errors.append("item pool.consequences: expected a list of at least 2 labels")
    elif len(set(consequences)) != len(consequences):
        errors.append("item pool.consequences: labels must be unique")

    families = pool["families"]
    if not isinstance(families, dict) or not families:
        errors.append("item pool.families: expected a non-empty object")
        families = {}

    items = pool["items"]
    if not isinstance(items, list) or not items:
        errors.append("item pool.items: expected a non-empty list")
        return errors

    seen_ids: set[str] = set()
    family_counts: Dict[str, int] = {name: 0 for name in families}
    matched_index: Dict[tuple[str, str], str] = {}

    for position, item in enumerate(items):
        ctx = f"item pool.items[{position}]"
        if not _require_keys(
            item, ("id", "family", "text", "quality_label"), errors, ctx
        ):
            continue

        item_id = item["id"]
        if item_id in seen_ids:
            errors.append(f"{ctx}: duplicate item id {item_id!r}")
        seen_ids.add(item_id)

        if not isinstance(item["text"], str) or not item["text"].strip():
            errors.append(f"{ctx}: 'text' must be a non-empty string")

        _check_enum(item["quality_label"], QUALITY_LABELS, errors, f"{ctx}.quality_label")

        family = item["family"]
        if family not in families:
            errors.append(f"{ctx}: family {family!r} not declared in pool.families")
        else:
            family_counts[family] += 1

        matched_key = item.get("matched_key")
        if matched_key is not None:
            if not isinstance(matched_key, str):
                errors.append(f"{ctx}: 'matched_key' must be a string or null")
            else:
                slot = (family, matched_key)
                if slot in matched_index:
                    errors.append(
                        f"{ctx}: matched_key {matched_key!r} already used by "
                        f"{matched_index[slot]!r} in family {family!r}"
                    )
                else:
                    matched_index[slot] = item_id

    largest_menu = max(MENU_SIZES)
    for family, count in family_counts.items():
        if count and count < largest_menu:
            errors.append(
                f"item pool.families[{family!r}]: {count} item(s) but menus of size "
                f"{largest_menu} are drawn from this family alone (§3.4)"
            )

    return errors


# ---------------------------------------------------------------------------
# 2. Problem set  (study plan §6.1 step 1, §6.2)
# ---------------------------------------------------------------------------


def validate_problem_set(
    problem_set: Any, *, pool: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Validate a per-pool problem (menu) design.

    Expected shape::

        {
          "schema_version": "1.0",
          "pool_id": "venture",
          "design_seed": 42,
          "menu_sizes": [2, 4, 6, 8],
          "num_presentations": 2,
          "problems": [
            {"id": "VEN0001", "family": "startup", "item_ids": [...],
             "menu_size": 4, "difficulty_stratum": "ambiguous",
             "presentations": [{"presentation_id": 1, "order": [...]},
                               {"presentation_id": 2, "order": [...]}]}
          ]
        }

    Enforced invariants
    -------------------
    * ``menu_size == len(item_ids)`` and is drawn from the declared range.
    * menus contain no repeated item.
    * every menu carries exactly ``num_presentations`` presentations, each a
      permutation of ``item_ids``, and the orderings are pairwise distinct --
      an identical repeat would make the position-flip statistic vacuous
      (§3.1).
    * menu sizes are **balanced** across the declared range (§3.4); imbalance
      is reported because the RQ6 slope is identified from that spread.
    * if *pool* is supplied, every referenced item exists and menus do not mix
      families (matched items form their own stratum, §3.3).
    """
    errors: List[str] = []
    if not _require_keys(
        problem_set,
        ("schema_version", "pool_id", "design_seed", "menu_sizes", "num_presentations", "problems"),
        errors,
        "problem set",
    ):
        return errors

    _check_schema_version(problem_set, errors, "problem set")

    declared_sizes = problem_set["menu_sizes"]
    if not isinstance(declared_sizes, list) or not declared_sizes:
        errors.append("problem set.menu_sizes: expected a non-empty list")
        declared_sizes = list(MENU_SIZES)

    n_presentations = problem_set["num_presentations"]
    if n_presentations != NUM_PRESENTATIONS:
        errors.append(
            f"problem set.num_presentations: {n_presentations} != frozen "
            f"{NUM_PRESENTATIONS} (§6.2)"
        )

    problems = problem_set["problems"]
    if not isinstance(problems, list) or not problems:
        errors.append("problem set.problems: expected a non-empty list")
        return errors

    known_items: Optional[Dict[str, str]] = None
    if pool is not None:
        known_items = {item["id"]: item.get("family") for item in pool.get("items", [])}

    seen_problem_ids: set[str] = set()
    size_counts: Dict[int, int] = {int(size): 0 for size in declared_sizes}

    for position, problem in enumerate(problems):
        ctx = f"problem set.problems[{position}]"
        if not _require_keys(
            problem,
            ("id", "family", "item_ids", "menu_size", "difficulty_stratum", "presentations"),
            errors,
            ctx,
        ):
            continue

        problem_id = problem["id"]
        if problem_id in seen_problem_ids:
            errors.append(f"{ctx}: duplicate problem id {problem_id!r}")
        seen_problem_ids.add(problem_id)

        item_ids = problem["item_ids"]
        if not isinstance(item_ids, list):
            errors.append(f"{ctx}: 'item_ids' must be a list")
            continue
        if len(set(item_ids)) != len(item_ids):
            errors.append(f"{ctx}: menu repeats an item")

        menu_size = problem["menu_size"]
        if menu_size != len(item_ids):
            errors.append(
                f"{ctx}: menu_size {menu_size} != len(item_ids) {len(item_ids)}"
            )
        if menu_size in size_counts:
            size_counts[menu_size] += 1
        else:
            errors.append(
                f"{ctx}: menu_size {menu_size} not in declared range {declared_sizes}"
            )

        _check_enum(
            problem["difficulty_stratum"],
            QUALITY_LABELS,
            errors,
            f"{ctx}.difficulty_stratum",
        )

        if known_items is not None:
            unknown = [i for i in item_ids if i not in known_items]
            if unknown:
                errors.append(f"{ctx}: item id(s) not in pool: {unknown}")
            families = {known_items[i] for i in item_ids if i in known_items}
            if len(families) > 1:
                errors.append(
                    f"{ctx}: menu mixes families {sorted(families)}; matched items "
                    f"must form their own stratum (§3.3)"
                )
            elif families and problem["family"] not in families:
                errors.append(
                    f"{ctx}: declared family {problem['family']!r} does not match "
                    f"item families {sorted(families)}"
                )

        errors.extend(_validate_presentations(problem, item_ids, ctx))

    _report_size_balance(size_counts, len(problems), errors)
    return errors


def _validate_presentations(
    problem: Dict[str, Any], item_ids: List[str], ctx: str
) -> List[str]:
    errors: List[str] = []
    presentations = problem["presentations"]
    if not isinstance(presentations, list):
        errors.append(f"{ctx}.presentations: expected a list")
        return errors

    if len(presentations) != NUM_PRESENTATIONS:
        errors.append(
            f"{ctx}.presentations: {len(presentations)} present, expected "
            f"{NUM_PRESENTATIONS} (§6.2)"
        )

    expected_items = sorted(item_ids)
    seen_orders: List[tuple[str, ...]] = []
    seen_pids: set[int] = set()

    for index, presentation in enumerate(presentations):
        pctx = f"{ctx}.presentations[{index}]"
        if not _require_keys(presentation, ("presentation_id", "order"), errors, pctx):
            continue

        pid = presentation["presentation_id"]
        if pid in seen_pids:
            errors.append(f"{pctx}: duplicate presentation_id {pid}")
        seen_pids.add(pid)

        order = presentation["order"]
        if not isinstance(order, list):
            errors.append(f"{pctx}.order: expected a list")
            continue
        if sorted(order) != expected_items:
            errors.append(f"{pctx}.order: not a permutation of the menu's item_ids")
            continue
        seen_orders.append(tuple(order))

    if len(set(seen_orders)) != len(seen_orders):
        errors.append(
            f"{ctx}.presentations: orderings are not pairwise distinct; an identical "
            f"repeat makes the position-flip statistic vacuous (§3.1)"
        )
    return errors


def _report_size_balance(
    size_counts: Dict[int, int], total: int, errors: List[str]
) -> None:
    """
    Menu sizes must be balanced: the RQ6 slope lives on that spread (§3.4).

    A balanced design puts ``floor(total/n)`` or ``ceil(total/n)`` menus at
    every size -- exactly the slack needed when the menu count is not a multiple
    of the number of sizes, and no more.
    """
    if not size_counts or total == 0:
        return
    empty = [size for size, count in size_counts.items() if count == 0]
    if empty:
        errors.append(
            f"problem set: no menus at size(s) {sorted(empty)}; the RQ6 slope is "
            f"identified from the size spread (§3.4)"
        )
        return

    n_sizes = len(size_counts)
    low, remainder = divmod(total, n_sizes)
    high = low + (1 if remainder else 0)
    for size, count in sorted(size_counts.items()):
        if not low <= count <= high:
            expected = f"{low}" if low == high else f"{low} or {high}"
            errors.append(
                f"problem set: menu sizes are not balanced -- size {size} has {count} "
                f"menu(s), expected {expected} (§3.4)"
            )


# ---------------------------------------------------------------------------
# 3. Assessment set  (study plan §6.1 step 2, §5)
# ---------------------------------------------------------------------------


def validate_assessment_set(
    assessment_set: Any, *, pool: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Validate assessments for one model x pool.

    Expected shape::

        {
          "schema_version": "1.0",
          "pool_id": "venture",
          "model_name": "gpt-4o",
          "instruction": "neutral",
          "assessments": [
            {"item_id": "V001", "text": "...", "probabilities": [0.2, 0.5, 0.3],
             "parse_ok": true, "raw_response": "..."}
          ]
        }

    Enforced invariants
    -------------------
    * ``instruction`` is exactly ``"neutral"`` -- assessments are collected once
      per model x pool under the neutral prompt and shared across the three
      prompt cells (§3.2 B2).  Any other value means the prompt manipulation
      leaked into the belief-side channel.
    * ``probabilities`` is either ``null`` (unparseable) or a length-K vector of
      values in [0, 1] summing to 1 within tolerance.
    * ``parse_ok`` agrees with whether probabilities are present.
    * if *pool* is supplied, every pool item is assessed exactly once.
    """
    errors: List[str] = []
    if not _require_keys(
        assessment_set,
        ("schema_version", "pool_id", "model_name", "instruction", "assessments"),
        errors,
        "assessment set",
    ):
        return errors

    _check_schema_version(assessment_set, errors, "assessment set")

    if assessment_set["instruction"] != ASSESSMENT_INSTRUCTION:
        errors.append(
            f"assessment set.instruction: {assessment_set['instruction']!r} != "
            f"{ASSESSMENT_INSTRUCTION!r}; assessments must be collected under the "
            f"neutral instruction only (§3.2 B2)"
        )

    expected_k: Optional[int] = None
    if pool is not None:
        expected_k = len(pool.get("consequences", []))

    assessments = assessment_set["assessments"]
    if not isinstance(assessments, list) or not assessments:
        errors.append("assessment set.assessments: expected a non-empty list")
        return errors

    seen_items: set[str] = set()
    for position, record in enumerate(assessments):
        ctx = f"assessment set.assessments[{position}]"
        if not _require_keys(
            record, ("item_id", "text", "probabilities", "parse_ok"), errors, ctx
        ):
            continue

        item_id = record["item_id"]
        if item_id in seen_items:
            errors.append(f"{ctx}: duplicate assessment for item {item_id!r}")
        seen_items.add(item_id)

        probabilities = record["probabilities"]
        parse_ok = record["parse_ok"]

        if probabilities is None:
            if parse_ok:
                errors.append(f"{ctx}: parse_ok is true but probabilities are null")
            continue

        if not parse_ok:
            errors.append(f"{ctx}: parse_ok is false but probabilities are present")

        if not isinstance(probabilities, list):
            errors.append(f"{ctx}.probabilities: expected a list or null")
            continue
        if expected_k is not None and len(probabilities) != expected_k:
            errors.append(
                f"{ctx}.probabilities: length {len(probabilities)} != K={expected_k}"
            )
        if any(not isinstance(p, (int, float)) or p < 0 or p > 1 for p in probabilities):
            errors.append(f"{ctx}.probabilities: values must lie in [0, 1]")
            continue
        total = math.fsum(probabilities)
        if abs(total - 1.0) > _PROB_SUM_TOL:
            errors.append(f"{ctx}.probabilities: sum {total:.3f} is not 1 +/- {_PROB_SUM_TOL}")

    if pool is not None:
        pool_ids = {item["id"] for item in pool.get("items", [])}
        missing = sorted(pool_ids - seen_items)
        if missing:
            errors.append(f"assessment set: no assessment for item(s) {missing[:10]}")
        extra = sorted(seen_items - pool_ids)
        if extra:
            errors.append(f"assessment set: assessment for unknown item(s) {extra[:10]}")

    return errors


# ---------------------------------------------------------------------------
# 4. Choice set  (study plan §6.1 step 3, §6.4)
# ---------------------------------------------------------------------------


def validate_choice_set(
    choice_set: Any, *, problem_set: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Validate choices for one cell.

    Expected shape::

        {
          "schema_version": "1.0",
          "cell_id": "gpt_4o_neutral_venture",
          "pool_id": "venture",
          "model_name": "gpt-4o",
          "prompt_condition": "neutral",
          "answer_format_version": "answer-token-v1",
          "choices": [
            {"problem_id": "VEN0001", "presentation_id": 1, "menu_size": 4,
             "difficulty_stratum": "ambiguous", "chosen_position": 3,
             "chosen_item_id": "V007", "resolution_path": "answer_token",
             "raw_response": "ANSWER: 3"}
          ]
        }

    Enforced invariants
    -------------------
    * ``chosen_position`` is 1-indexed **within the presentation order** and
      lies in ``[1, menu_size]``, or is ``null`` for an NA.
    * ``chosen_item_id`` agrees with the presentation order at that position --
      this is the check that catches a counterbalancing bug, which would
      otherwise silently corrupt ``y``.
    * an NA has ``resolution_path == "unresolved"`` and vice versa, so the NA
      accounting in §6.4 cannot drift from the resolution-path accounting.
    * every (problem, presentation) pair in the design appears exactly once.
    """
    errors: List[str] = []
    if not _require_keys(
        choice_set,
        (
            "schema_version",
            "cell_id",
            "pool_id",
            "model_name",
            "prompt_condition",
            "answer_format_version",
            "choices",
        ),
        errors,
        "choice set",
    ):
        return errors

    _check_schema_version(choice_set, errors, "choice set")
    _check_enum(
        choice_set["prompt_condition"],
        PROMPT_CONDITIONS,
        errors,
        "choice set.prompt_condition",
    )
    if choice_set["answer_format_version"] != ANSWER_TOKEN_VERSION:
        errors.append(
            f"choice set.answer_format_version: "
            f"{choice_set['answer_format_version']!r} != {ANSWER_TOKEN_VERSION!r}"
        )

    orders: Dict[tuple[str, int], List[str]] = {}
    expected_pairs: set[tuple[str, int]] = set()
    if problem_set is not None:
        for problem in problem_set.get("problems", []):
            for presentation in problem.get("presentations", []):
                key = (problem["id"], presentation["presentation_id"])
                orders[key] = presentation["order"]
                expected_pairs.add(key)

    choices = choice_set["choices"]
    if not isinstance(choices, list) or not choices:
        errors.append("choice set.choices: expected a non-empty list")
        return errors

    seen_pairs: set[tuple[str, int]] = set()
    for position, record in enumerate(choices):
        ctx = f"choice set.choices[{position}]"
        if not _require_keys(
            record,
            (
                "problem_id",
                "presentation_id",
                "menu_size",
                "difficulty_stratum",
                "chosen_position",
                "chosen_item_id",
                "resolution_path",
                "raw_response",
            ),
            errors,
            ctx,
        ):
            continue

        key = (record["problem_id"], record["presentation_id"])
        if key in seen_pairs:
            errors.append(f"{ctx}: duplicate observation for {key}")
        seen_pairs.add(key)

        _check_enum(
            record["resolution_path"], RESOLUTION_PATHS, errors, f"{ctx}.resolution_path"
        )
        _check_enum(
            record["difficulty_stratum"],
            QUALITY_LABELS,
            errors,
            f"{ctx}.difficulty_stratum",
        )

        chosen_position = record["chosen_position"]
        is_na = chosen_position is None
        unresolved = record["resolution_path"] == "unresolved"
        if is_na != unresolved:
            errors.append(
                f"{ctx}: chosen_position={chosen_position!r} is inconsistent with "
                f"resolution_path={record['resolution_path']!r}"
            )
        if is_na:
            if record["chosen_item_id"] is not None:
                errors.append(f"{ctx}: NA observation must have null chosen_item_id")
            continue

        menu_size = record["menu_size"]
        if not isinstance(chosen_position, int) or not 1 <= chosen_position <= menu_size:
            errors.append(
                f"{ctx}: chosen_position {chosen_position!r} outside [1, {menu_size}]"
            )
            continue

        order = orders.get(key)
        if order is None:
            if problem_set is not None:
                errors.append(f"{ctx}: {key} is not in the problem design")
            continue
        if len(order) != menu_size:
            errors.append(
                f"{ctx}: menu_size {menu_size} != design order length {len(order)}"
            )
            continue
        expected_item = order[chosen_position - 1]
        if record["chosen_item_id"] != expected_item:
            errors.append(
                f"{ctx}: chosen_item_id {record['chosen_item_id']!r} != "
                f"{expected_item!r} at position {chosen_position} of the presentation "
                f"order (counterbalancing bug?)"
            )

    if problem_set is not None:
        missing = sorted(expected_pairs - seen_pairs)
        if missing:
            errors.append(
                f"choice set: {len(missing)} design observation(s) absent, e.g. "
                f"{missing[:5]}"
            )

    return errors


# ---------------------------------------------------------------------------
# 5. Stan data  (study plan §6.1 step 5, §4)
# ---------------------------------------------------------------------------

_STAN_REQUIRED = (
    "J",
    "K",
    "D",
    "R",
    "P",
    "w",
    "M_total",
    "cell",
    "I",
    "y",
    "X",
    "M_per_cell",
)


def validate_stan_data(stan_data: Any, *, model: str = "h_m01") -> List[str]:
    """
    Validate a stacked Stan payload against the ``h_m01`` data block.

    ``model="h_m01_size"`` additionally requires the per-observation centered
    menu-size covariate ``s`` (§4).

    Enforced invariants
    -------------------
    * declared dimensions agree with the actual array shapes;
    * ``cell`` entries are 1-indexed into ``[1, J]`` and ``M_per_cell`` sums to
      ``M_total`` and matches the realised per-cell counts;
    * each row of ``I`` selects at least two alternatives, and ``y`` indexes
      within that row's active set -- ``y`` is 1-indexed *within the active
      set*, which is the single easiest thing to get wrong when stacking.
    """
    errors: List[str] = []
    required = list(_STAN_REQUIRED)
    if model == "h_m01_size":
        required.append("s")
    if not _require_keys(stan_data, required, errors, f"stan data ({model})"):
        return errors

    J = stan_data["J"]
    K = stan_data["K"]
    D = stan_data["D"]
    R = stan_data["R"]
    P = stan_data["P"]
    M_total = stan_data["M_total"]

    for name, value, minimum in (("J", J, 1), ("K", K, 2), ("D", D, 1), ("R", R, 2), ("P", P, 1)):
        if not isinstance(value, int) or value < minimum:
            errors.append(f"stan data.{name}: expected int >= {minimum}, got {value!r}")

    w = stan_data["w"]
    if not isinstance(w, list) or len(w) != R:
        errors.append(f"stan data.w: expected {R} rows, got {len(w) if isinstance(w, list) else w!r}")
    elif any(not isinstance(row, list) or len(row) != D for row in w):
        errors.append(f"stan data.w: every row must have length D={D}")

    X = stan_data["X"]
    if not isinstance(X, list) or len(X) != J:
        errors.append(f"stan data.X: expected {J} rows, got {len(X) if isinstance(X, list) else X!r}")
    elif any(not isinstance(row, list) or len(row) != P for row in X):
        errors.append(f"stan data.X: every row must have length P={P}")

    cell = stan_data["cell"]
    I = stan_data["I"]
    y = stan_data["y"]
    for name, value in (("cell", cell), ("I", I), ("y", y)):
        if not isinstance(value, list) or len(value) != M_total:
            errors.append(
                f"stan data.{name}: expected {M_total} entries, got "
                f"{len(value) if isinstance(value, list) else value!r}"
            )
            return errors

    if model == "h_m01_size":
        s = stan_data["s"]
        if not isinstance(s, list) or len(s) != M_total:
            errors.append(f"stan data.s: expected {M_total} entries")

    realised_counts = [0] * (J if isinstance(J, int) and J > 0 else 0)
    for m in range(M_total):
        ctx = f"stan data[obs {m}]"
        j = cell[m]
        if not isinstance(j, int) or not 1 <= j <= J:
            errors.append(f"{ctx}: cell {j!r} outside [1, {J}]")
        else:
            realised_counts[j - 1] += 1

        row = I[m]
        if not isinstance(row, list) or len(row) != R:
            errors.append(f"{ctx}: I row must have length R={R}")
            continue
        if any(v not in (0, 1) for v in row):
            errors.append(f"{ctx}: I row must be binary")
            continue
        active = sum(row)
        if active < 2:
            errors.append(f"{ctx}: menu has {active} alternative(s), need >= 2")
            continue
        if not isinstance(y[m], int) or not 1 <= y[m] <= active:
            errors.append(
                f"{ctx}: y={y[m]!r} outside [1, {active}] (y is 1-indexed within the "
                f"active set)"
            )

    M_per_cell = stan_data["M_per_cell"]
    if not isinstance(M_per_cell, list) or len(M_per_cell) != J:
        errors.append(f"stan data.M_per_cell: expected {J} entries")
    else:
        if sum(M_per_cell) != M_total:
            errors.append(
                f"stan data.M_per_cell: sums to {sum(M_per_cell)}, expected M_total="
                f"{M_total}"
            )
        if realised_counts and list(M_per_cell) != realised_counts:
            errors.append(
                f"stan data.M_per_cell {list(M_per_cell)} disagrees with the realised "
                f"per-cell counts {realised_counts}"
            )

    return errors


# ---------------------------------------------------------------------------
# 6. Run manifest  (study plan §6.5)
# ---------------------------------------------------------------------------

_MANIFEST_REQUIRED = (
    "schema_version",
    "run_id",
    "started_at",
    "answer_format_version",
    "menu_sizes",
    "num_presentations",
    "design_seed",
    "embedding_model",
    "pca_target_dim",
    "prompt_hashes",
    "models",
)


def validate_run_manifest(manifest: Any) -> List[str]:
    """
    Validate the provenance manifest (§6.5).

    Every model entry must pin the exact endpoint id, the access date, and the
    **full** request parameters -- for the reasoning tier those settings are
    part of the treatment (§3.1), so an unpinned ``reasoning_effort`` or
    ``budget_tokens`` makes the run unreproducible.
    """
    errors: List[str] = []
    if not _require_keys(manifest, _MANIFEST_REQUIRED, errors, "run manifest"):
        return errors

    _check_schema_version(manifest, errors, "run manifest")

    if manifest["answer_format_version"] != ANSWER_TOKEN_VERSION:
        errors.append(
            f"run manifest.answer_format_version: "
            f"{manifest['answer_format_version']!r} != {ANSWER_TOKEN_VERSION!r}"
        )
    if manifest["num_presentations"] != NUM_PRESENTATIONS:
        errors.append(
            f"run manifest.num_presentations: {manifest['num_presentations']} != "
            f"frozen {NUM_PRESENTATIONS} (§6.2)"
        )

    models = manifest["models"]
    if not isinstance(models, list) or not models:
        errors.append("run manifest.models: expected a non-empty list")
        return errors

    for position, entry in enumerate(models):
        ctx = f"run manifest.models[{position}]"
        if not _require_keys(
            entry,
            ("model_name", "endpoint_id", "provider", "accessed_at", "request_params"),
            errors,
            ctx,
        ):
            continue
        if not isinstance(entry["request_params"], dict) or not entry["request_params"]:
            errors.append(
                f"{ctx}.request_params: must pin the full request parameters (§3.1)"
            )
        if entry.get("substituted_for") is not None and not entry.get("substitution_reason"):
            errors.append(
                f"{ctx}: a deprecation substitution must record 'substitution_reason' "
                f"(§3.1)"
            )

    return errors

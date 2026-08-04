"""
Menu (problem) design generation for the SEU sensitivity study (§6.1 step 1, §6.3).

One fixed design is generated per pool and shared across all 18 cells of that
pool, so cells differ only by model and prompt -- never by which menus they saw.

Two design decisions here are load-bearing for RQ6 and R6 and are therefore
implemented explicitly rather than left to random sampling.

**Menu composition is size-aware (§6.3).**  Menus are not drawn uniformly from
the pool.  Each menu is built from a fixed number of *contenders* plus filler
items drawn from the lowest quality tier.  Because the contender count does not
vary with menu size, the best-vs-next-best quality gap is held roughly constant
as menus grow.  Drawing uniformly instead would let order statistics shrink the
top-two gap at size 8 relative to size 2 -- a size-correlated geometry shift
produced by the draw count alone, which is exactly the artefact the RQ6 slope
must not be confounded with.

**Presentations are reversals, not random permutations (§3.1).**  With
``num_presentations`` frozen at 2, a reversal is the best available probe of
position bias: every menu size in the pre-registered range is even, so under
reversal *every* item strictly changes position (item at position *i* moves to
*n+1-i*).  Two randomly drawn permutations would leave some items in place and
would waste part of the only positional contrast the design affords.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import schemas
from .schemas import MENU_SIZES, NUM_PRESENTATIONS, QUALITY_LABELS

logger = logging.getLogger(__name__)

__all__ = [
    "StratumRecipe",
    "DEFAULT_RECIPES",
    "PresentationMode",
    "allocate_menu_plan",
    "generate_problem_set",
]


PresentationMode = str  # "reverse" | "random"


@dataclass(frozen=True)
class StratumRecipe:
    """
    How to compose a menu of a given difficulty stratum, at any menu size.

    Attributes
    ----------
    stratum:
        The difficulty label recorded on the menu (one of
        :data:`schemas.QUALITY_LABELS`).
    contenders:
        Quality labels of the top items, which set the best-vs-next-best gap.
        Its length is **independent of menu size** -- that is what keeps the gap
        stable as menus grow.
    filler_label:
        Quality label used for every remaining slot.
    """

    stratum: str
    contenders: Tuple[str, ...]
    filler_label: str

    def compose(self, menu_size: int) -> List[str]:
        """Return the list of quality labels a menu of *menu_size* needs."""
        if menu_size < len(self.contenders):
            raise ValueError(
                f"stratum {self.stratum!r} needs at least {len(self.contenders)} "
                f"items but menu size is {menu_size}"
            )
        filler = [self.filler_label] * (menu_size - len(self.contenders))
        return list(self.contenders) + filler


#: Default composition recipes.  The three strata differ in the top-two gap:
#: wide (one clear winner over filler), near-tied (two equally strong
#: contenders), and medium (a middling winner over filler).  Ambiguous menus
#: are where alpha is identified; all-easy menus are uninformative (§6.3).
DEFAULT_RECIPES: Tuple[StratumRecipe, ...] = (
    StratumRecipe(stratum="strong", contenders=("strong",), filler_label="weak"),
    StratumRecipe(
        stratum="ambiguous", contenders=("strong", "strong"), filler_label="weak"
    ),
    StratumRecipe(stratum="weak", contenders=("ambiguous",), filler_label="weak"),
)


# ---------------------------------------------------------------------------
# Balanced allocation
# ---------------------------------------------------------------------------


def allocate_menu_plan(
    num_problems: int,
    menu_sizes: Sequence[int],
    strata: Sequence[str],
) -> List[Tuple[int, str]]:
    """
    Spread *num_problems* menus as evenly as possible over size x stratum.

    Balance across menu sizes is a pre-registered design property (§3.4): the
    RQ6 slope is identified from the size spread, so an unbalanced design
    silently costs power.  The split is therefore **nested** -- menus are
    divided evenly across sizes first, then evenly across strata within each
    size.  Distributing round-robin over the flat size x stratum grid instead
    would let the remainder cluster in the first sizes, skewing the size margin
    by as much as one menu per stratum while each individual cell still looked
    balanced.
    """
    if num_problems <= 0:
        return []
    if not menu_sizes or not strata:
        raise ValueError("menu_sizes and strata must both be non-empty")

    plan: List[Tuple[int, str]] = []
    for size, size_count in zip(menu_sizes, _split_evenly(num_problems, len(menu_sizes))):
        for stratum, count in zip(strata, _split_evenly(size_count, len(strata))):
            plan.extend([(size, stratum)] * count)
    return plan


def _split_evenly(total: int, parts: int) -> List[int]:
    """Split *total* into *parts* counts differing by at most one."""
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_problem_set(
    pool: Mapping[str, Any],
    *,
    problems_per_family: Mapping[str, int] | int,
    seed: int,
    menu_sizes: Sequence[int] = MENU_SIZES,
    num_presentations: int = NUM_PRESENTATIONS,
    recipes: Sequence[StratumRecipe] = DEFAULT_RECIPES,
    presentation_mode: PresentationMode = "reverse",
    families: Optional[Iterable[str]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Build the fixed menu design for one pool.

    Parameters
    ----------
    pool:
        A canonical pool dict (see :mod:`pools`).
    problems_per_family:
        Menus to generate for each family.  Families are separate menu strata
        (§3.3): the matched procurement family never mixes into the ordinary
        venture menus, so the RQ5 comparator stays a pure primary-family
        quantity.  An ``int`` applies the same count to every selected family.
    seed:
        Design seed, recorded in the artefact and in the run manifest (§6.5).
    presentation_mode:
        ``"reverse"`` (default, see module docstring) or ``"random"``.

    Returns
    -------
    dict
        A problem-set artefact conforming to
        :func:`schemas.validate_problem_set`.
    """
    pool_id = pool["pool_id"]
    selected = list(families) if families is not None else list(pool["families"])

    if isinstance(problems_per_family, int):
        counts = {family: problems_per_family for family in selected}
    else:
        counts = {family: int(problems_per_family.get(family, 0)) for family in selected}

    by_family = _items_by_family_and_label(pool, selected)
    strata = [recipe.stratum for recipe in recipes]
    _validate_strata(strata)
    recipe_by_stratum = {recipe.stratum: recipe for recipe in recipes}

    rng = random.Random(seed)
    problems: List[Dict[str, Any]] = []
    counter = 0
    prefix = pool_id[:3].upper()

    for family in selected:
        plan = allocate_menu_plan(counts.get(family, 0), menu_sizes, strata)
        if not plan:
            continue
        _check_family_capacity(pool_id, family, by_family[family], menu_sizes, recipes)

        for menu_size, stratum in plan:
            counter += 1
            item_ids = _draw_menu(
                by_family[family], recipe_by_stratum[stratum], menu_size, rng
            )
            rng.shuffle(item_ids)
            problems.append(
                {
                    "id": f"{prefix}{counter:04d}",
                    "family": family,
                    "item_ids": item_ids,
                    "menu_size": menu_size,
                    "difficulty_stratum": stratum,
                    "presentations": _build_presentations(
                        item_ids, num_presentations, presentation_mode, rng
                    ),
                }
            )

    rng.shuffle(problems)

    problem_set = {
        "schema_version": schemas.SCHEMA_VERSION,
        "pool_id": pool_id,
        "design_seed": seed,
        "menu_sizes": list(menu_sizes),
        "num_presentations": num_presentations,
        "presentation_mode": presentation_mode,
        "problems_per_family": counts,
        "problems": problems,
    }

    if validate:
        schemas.check(
            schemas.validate_problem_set(problem_set, pool=dict(pool)),
            context=f"problem set for pool {pool_id!r}",
        )

    logger.info(
        "Generated %d menus for pool %r (sizes %s, %d presentation(s), mode %r)",
        len(problems),
        pool_id,
        list(menu_sizes),
        num_presentations,
        presentation_mode,
    )
    return problem_set


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_strata(strata: Sequence[str]) -> None:
    unknown = [s for s in strata if s not in QUALITY_LABELS]
    if unknown:
        raise ValueError(
            f"recipe stratum/strata {unknown} not in {list(QUALITY_LABELS)}"
        )
    if len(set(strata)) != len(strata):
        raise ValueError(f"duplicate stratum in recipes: {list(strata)}")


def _items_by_family_and_label(
    pool: Mapping[str, Any], families: Sequence[str]
) -> Dict[str, Dict[str, List[str]]]:
    """``family -> quality_label -> [item_id, ...]``."""
    known = set(pool["families"])
    unknown = [family for family in families if family not in known]
    if unknown:
        raise ValueError(
            f"pool {pool['pool_id']!r} has no family/families {unknown}; "
            f"available: {sorted(known)}"
        )

    grouped: Dict[str, Dict[str, List[str]]] = {
        family: {label: [] for label in QUALITY_LABELS} for family in families
    }
    for item in pool["items"]:
        family = item["family"]
        if family in grouped:
            grouped[family][item["quality_label"]].append(item["id"])
    return grouped


def _check_family_capacity(
    pool_id: str,
    family: str,
    by_label: Mapping[str, List[str]],
    menu_sizes: Sequence[int],
    recipes: Sequence[StratumRecipe],
) -> None:
    """Fail early and loudly if a recipe cannot be satisfied at the largest size."""
    largest = max(menu_sizes)
    for recipe in recipes:
        needed: Dict[str, int] = {}
        for label in recipe.compose(largest):
            needed[label] = needed.get(label, 0) + 1
        for label, count in needed.items():
            available = len(by_label.get(label, []))
            if available < count:
                raise ValueError(
                    f"pool {pool_id!r} family {family!r}: stratum {recipe.stratum!r} "
                    f"needs {count} {label!r} item(s) for a size-{largest} menu but "
                    f"the family has {available}. Author more {label!r} items, or "
                    f"reduce the menu-size range (§3.4 fallback)."
                )


def _draw_menu(
    by_label: Mapping[str, List[str]],
    recipe: StratumRecipe,
    menu_size: int,
    rng: random.Random,
) -> List[str]:
    """Draw one menu's items without replacement, following *recipe*."""
    chosen: List[str] = []
    used: set[str] = set()
    for label in recipe.compose(menu_size):
        candidates = [item_id for item_id in by_label[label] if item_id not in used]
        if not candidates:
            raise ValueError(
                f"exhausted {label!r} items while composing a size-{menu_size} "
                f"{recipe.stratum!r} menu"
            )
        pick = rng.choice(candidates)
        chosen.append(pick)
        used.add(pick)
    return chosen


def _build_presentations(
    item_ids: Sequence[str],
    num_presentations: int,
    mode: PresentationMode,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Build position-counterbalanced orderings.

    ``"reverse"`` yields the base order and its reversal (see module docstring).
    For more than two presentations the extra orderings are distinct random
    permutations, so the function still honours a widened design.
    """
    base = list(item_ids)
    orders: List[List[str]] = [base]

    if mode == "reverse":
        if num_presentations >= 2:
            orders.append(list(reversed(base)))
    elif mode != "random":
        raise ValueError(f"unknown presentation_mode {mode!r}")

    attempts = 0
    max_attempts = 200
    while len(orders) < num_presentations:
        candidate = base[:]
        rng.shuffle(candidate)
        if candidate not in orders:
            orders.append(candidate)
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"could not build {num_presentations} distinct orderings for a menu "
                f"of size {len(base)} (only {_factorial(len(base))} exist)"
            )

    return [
        {"presentation_id": index + 1, "order": order}
        for index, order in enumerate(orders[:num_presentations])
    ]


def _factorial(n: int) -> int:
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result

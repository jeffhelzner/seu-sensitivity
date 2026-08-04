"""
Tests for the authored pools and the deterministic builder.

The point of these is that the §7.3 sterilization protocol and the §3.3 matched
pair are enforced by executable checks rather than by authoring discipline: a
later edit to a phrase table or a merit count has to fail a test, not merely
contradict a comment.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from applications.seu_sensitivity_study import pools, problem_generation, schemas
from applications.seu_sensitivity_study.data import build_pools

DATA_DIR = Path(build_pools.__file__).resolve().parent


class TestMeritGrid:
    def test_label_is_monotone_in_total(self):
        """Non-monotone labels would set R4 up to fail on authoring, not measurement."""
        for label in schemas.QUALITY_LABELS:
            totals = [
                build_pools.merit_total(v) for v in build_pools._candidate_vectors(label)
            ]
            assert totals, f"no lattice points for {label}"

        strong = [build_pools.merit_total(v) for v in build_pools._candidate_vectors("strong")]
        ambiguous = [
            build_pools.merit_total(v) for v in build_pools._candidate_vectors("ambiguous")
        ]
        weak = [build_pools.merit_total(v) for v in build_pools._candidate_vectors("weak")]
        assert min(strong) > max(ambiguous)
        assert min(ambiguous) > max(weak)

    def test_bands_are_textually_distinct(self):
        """Strong items carry no bottom phrase; weak items carry at most one mid phrase."""
        assert all(min(v) >= 2 for v in build_pools._candidate_vectors("strong"))
        assert all(
            max(v) >= 2 and min(v) <= 1
            for v in build_pools._candidate_vectors("ambiguous")
        )
        assert all(
            max(v) <= 2 and sum(1 for level in v if level >= 2) <= 1
            for v in build_pools._candidate_vectors("weak")
        )

    def test_selection_is_deterministic(self):
        assert build_pools.select_vectors(build_pools.MATCHED_COUNTS) == (
            build_pools.select_vectors(build_pools.MATCHED_COUNTS)
        )

    def test_exclusion_makes_families_disjoint(self):
        matched = build_pools.select_vectors(build_pools.MATCHED_COUNTS)
        used = {v for vectors in matched.values() for v in vectors}
        primary = build_pools.select_vectors(build_pools.PRIMARY_COUNTS, exclude=used)
        assert not used & {v for vectors in primary.values() for v in vectors}

    def test_selection_raises_when_the_band_is_too_small(self):
        with pytest.raises(ValueError, match="composition band"):
            build_pools.select_vectors({"strong": 10_000})

    def test_reservation_only_fires_on_conflicting_profiles(self):
        assert build_pools.reservation_position((3, 3, 3, 3, 3)) is None  # uniform strong
        assert build_pools.reservation_position((1, 1, 1, 1, 1)) is None  # uniform weak
        assert build_pools.reservation_position((3, 3, 3, 0, 2)) == 3  # names the gap


class TestBuiltPools:
    @pytest.fixture(scope="class")
    def built(self):
        return build_pools.build_all()

    def test_files_on_disk_match_the_generator(self):
        """The committed JSON must be reproducible from the grid, not hand-edited."""
        assert build_pools.main(["--check"]) == 0

    @pytest.mark.parametrize("pool_id", ["venture", "hiring"])
    def test_schema_valid(self, built, pool_id):
        assert schemas.validate_item_pool(built[pool_id]) == []

    @pytest.mark.parametrize("pool_id", ["venture", "hiring"])
    def test_capacity_supports_the_default_recipes(self, built, pool_id):
        """>= 2 strong / >= 1 ambiguous / >= 7 weak per family for a size-8 menu."""
        for family, items in pools.items_by_family(built[pool_id]).items():
            counts = Counter(item["quality_label"] for item in items)
            assert len(items) >= max(schemas.MENU_SIZES)
            assert counts["strong"] >= 2
            assert counts["ambiguous"] >= 1
            assert counts["weak"] >= 7

    @pytest.mark.parametrize("pool_id", ["venture", "hiring"])
    def test_weak_filler_has_draw_variance_at_the_largest_size(self, built, pool_id):
        """Exactly 7 weak items would make size-8 filler a constant, not a draw."""
        for items in pools.items_by_family(built[pool_id]).values():
            weak = sum(1 for item in items if item["quality_label"] == "weak")
            assert weak > max(schemas.MENU_SIZES) - 1

    @pytest.mark.parametrize("pool_id", ["venture", "hiring"])
    def test_menus_generate(self, built, pool_id):
        problem_set = problem_generation.generate_problem_set(
            built[pool_id], problems_per_family=24, seed=7
        )
        assert schemas.validate_problem_set(problem_set, pool=built[pool_id]) == []


class TestMatchedPair:
    @pytest.fixture(scope="class")
    def built(self):
        return build_pools.build_all()

    def test_same_merit_vectors_under_both_labels(self, built):
        pairs = pools.matched_pairs(built["hiring"], built["venture"])
        assert len(pairs) == sum(build_pools.MATCHED_COUNTS.values())
        for hiring_item, venture_item in pairs:
            assert (
                hiring_item["attributes"]["merit_vector"]
                == venture_item["attributes"]["merit_vector"]
            )
            assert hiring_item["quality_label"] == venture_item["quality_label"]
            assert (
                hiring_item["attributes"]["context"]["experience_band"]
                == venture_item["attributes"]["context"]["experience_band"]
            )

    def test_only_the_task_label_differs(self, built):
        """The pair must vary the label, not the vocabulary of the evidence."""
        hiring_item, venture_item = pools.matched_pairs(built["hiring"], built["venture"])[0]
        assert hiring_item["text"].startswith("Candidate")
        assert venture_item["text"].startswith("Vendor")
        assert hiring_item["text"] != venture_item["text"]

    def test_matched_families_are_separate_strata(self, built):
        assert built["hiring"]["families"]["matched"]["matched_with"] == "venture/procurement"
        assert built["venture"]["families"]["procurement"]["matched_with"] == "hiring/matched"

    def test_alignment_audit_catches_a_drifted_twin(self, built):
        drifted = copy.deepcopy(built["venture"])
        for item in drifted["items"]:
            if item["matched_key"] == "MK01":
                item["attributes"]["merit_vector"]["track_record"] = 0
                break
        with pytest.raises(build_pools.SterilizationError, match="merit vectors differ"):
            build_pools._audit_matched_alignment(built["hiring"], drifted)


class TestSterilization:
    @pytest.fixture(scope="class")
    def built(self):
        return build_pools.build_all()

    @pytest.mark.parametrize("pool_id", ["venture", "hiring"])
    def test_rendered_pools_pass(self, built, pool_id):
        build_pools.audit_sterilization(built[pool_id])

    @pytest.mark.parametrize(
        "injected",
        [
            "The candidate, Sarah, interviewed well.",
            "He performed strongly in the panel.",
            "Graduated from a top university.",
            "The candidate is 42 years old.",
            "A British national with 18 years of experience.",
            "Requested a workplace accommodation.",
        ],
    )
    def test_audit_catches_an_injected_cue(self, built, injected):
        polluted = copy.deepcopy(built["hiring"])
        polluted["items"][0]["text"] += "\n" + injected
        with pytest.raises(build_pools.SterilizationError):
            build_pools.audit_sterilization(polluted)

    def test_experience_bands_are_coarse(self, built):
        bands = {
            item["attributes"]["context"]["experience_band"]
            for item in built["hiring"]["items"]
        }
        assert bands == {"2-4", "5-8", "9+"}

    def test_context_is_orthogonal_to_merit(self, built):
        """A band correlated with merit would be a quality cue despite coarsening."""
        realised = build_pools.audit_context_orthogonality(built["hiring"])
        for family, correlations in realised.items():
            for key, rho in correlations.items():
                assert abs(rho) <= build_pools.MAX_CONTEXT_MERIT_RHO, (family, key, rho)

    def test_orthogonality_audit_catches_a_correlated_band(self, built):
        rigged = copy.deepcopy(built["hiring"])
        for item in rigged["items"]:
            total = item["attributes"]["merit_total"]
            item["attributes"]["context"]["experience_band"] = (
                "9+" if total >= 12 else "5-8" if total >= 6 else "2-4"
            )
        with pytest.raises(build_pools.SterilizationError, match="not orthogonal"):
            build_pools.audit_context_orthogonality(rigged)

    def test_phrase_tables_carry_no_proper_nouns(self):
        build_pools.audit_phrase_tables()


class TestInsuranceLabels:
    @pytest.fixture(scope="class")
    def sidecar(self):
        return json.loads((DATA_DIR / "insurance_quality_labels.json").read_text())

    def test_covers_every_legacy_claim(self, sidecar):
        pool = pools.load_pool("insurance")
        assert set(sidecar["labels"]) == {item["id"] for item in pool["items"]}

    def test_labels_are_valid(self, sidecar):
        assert set(sidecar["labels"].values()) <= set(schemas.QUALITY_LABELS)

    def test_every_label_is_justified(self, sidecar):
        """These are a reading of frozen text, so each one has to be defensible."""
        assert set(sidecar["justifications"]) == set(sidecar["labels"])

    def test_supports_the_default_recipes(self):
        counts = Counter(
            item["quality_label"] for item in pools.load_pool("insurance")["items"]
        )
        assert counts["strong"] >= 2
        assert counts["ambiguous"] >= 1
        assert counts["weak"] >= 7

    def test_legacy_text_is_untouched(self):
        legacy = json.loads(
            (
                Path(pools.__file__).resolve().parent.parent
                / "temperature_study"
                / "data"
                / "claims.json"
            ).read_text()
        )
        adapted = {item["id"]: item["text"] for item in pools.load_pool("insurance")["items"]}
        for claim in legacy["claims"]:
            assert adapted[claim["id"]] == claim["description"]

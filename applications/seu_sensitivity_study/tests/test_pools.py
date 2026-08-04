"""
Tests for the item-pool registry and loaders (study plan §3.3, §6.3).
"""

from __future__ import annotations

import json

import pytest

from applications.seu_sensitivity_study import pools, schemas


class TestRegistry:
    def test_three_pools_registered(self):
        assert pools.available_pools() == ["insurance", "venture", "hiring"]

    def test_venture_hosts_the_matched_family(self):
        spec = pools.get_pool_spec("venture")
        assert spec.families == ("startup", "procurement")
        assert spec.primary_family == "startup"

    def test_framing_valence_recorded(self):
        assert pools.get_pool_spec("insurance").framing == "negative"
        assert pools.get_pool_spec("hiring").framing == "positive"

    def test_unknown_pool_raises(self):
        with pytest.raises(KeyError, match="Unknown pool"):
            pools.get_pool_spec("ellsberg")


class TestInsuranceAdapter:
    """The legacy claims file is adapted, never rewritten (§6.3)."""

    def test_missing_sidecar_explains_why_labels_are_needed(self, monkeypatch, tmp_path):
        spec = pools.get_pool_spec("insurance")
        monkeypatch.setitem(
            pools.POOL_SPECS,
            "insurance",
            pools.PoolSpec(
                pool_id=spec.pool_id,
                framing=spec.framing,
                families=spec.families,
                item_file=spec.item_file,
                prompts_file=spec.prompts_file,
                legacy_adapter=spec.legacy_adapter,
                label_sidecar=tmp_path / "absent.json",
            ),
        )
        with pytest.raises(FileNotFoundError, match="PC1 validation"):
            pools.load_pool("insurance")

    def test_adapted_pool_validates_and_preserves_text(self, monkeypatch, tmp_path):
        spec = pools.get_pool_spec("insurance")
        with open(spec.item_file) as handle:
            legacy = json.load(handle)

        # A placeholder labelling that satisfies the per-family capacity rule.
        cycle = ["strong", "ambiguous", "weak", "weak"]
        labels = {
            claim["id"]: cycle[index % len(cycle)]
            for index, claim in enumerate(legacy["claims"])
        }
        sidecar = tmp_path / "labels.json"
        sidecar.write_text(json.dumps({"labels": labels}))

        monkeypatch.setitem(
            pools.POOL_SPECS,
            "insurance",
            pools.PoolSpec(
                pool_id=spec.pool_id,
                framing=spec.framing,
                families=spec.families,
                item_file=spec.item_file,
                prompts_file=spec.prompts_file,
                legacy_adapter=spec.legacy_adapter,
                label_sidecar=sidecar,
            ),
        )

        pool = pools.load_pool("insurance")
        assert schemas.validate_item_pool(pool) == []
        assert pool["consequences"] == legacy["consequences"]
        assert len(pool["items"]) == len(legacy["claims"])

        by_id = {item["id"]: item for item in pool["items"]}
        for claim in legacy["claims"]:
            assert by_id[claim["id"]]["text"] == claim["description"]

    def test_incomplete_sidecar_names_the_gap(self, monkeypatch, tmp_path):
        spec = pools.get_pool_spec("insurance")
        sidecar = tmp_path / "labels.json"
        sidecar.write_text(json.dumps({"labels": {"C001": "strong"}}))
        monkeypatch.setitem(
            pools.POOL_SPECS,
            "insurance",
            pools.PoolSpec(
                pool_id=spec.pool_id,
                framing=spec.framing,
                families=spec.families,
                item_file=spec.item_file,
                prompts_file=spec.prompts_file,
                legacy_adapter=spec.legacy_adapter,
                label_sidecar=sidecar,
            ),
        )
        with pytest.raises(ValueError, match="missing quality label"):
            pools.load_pool("insurance")


class TestViews:
    def test_items_by_family(self, two_family_pool):
        grouped = pools.items_by_family(two_family_pool)
        assert set(grouped) == {"startup", "procurement"}
        assert all(i["family"] == "startup" for i in grouped["startup"])

    def test_matched_index_only_covers_keyed_items(self, two_family_pool):
        index = pools.matched_index(two_family_pool)
        assert all(item["family"] == "procurement" for item in index.values())

    def test_matched_pairs_link_the_two_labels(self, hiring_like_pool, two_family_pool):
        pairs = pools.matched_pairs(hiring_like_pool, two_family_pool)
        assert pairs
        for hiring_item, procurement_item in pairs:
            assert hiring_item["matched_key"] == procurement_item["matched_key"]
            assert hiring_item["family"] == "candidates"
            assert procurement_item["family"] == "procurement"

    def test_unpaired_keys_are_excluded(self, hiring_like_pool, two_family_pool, caplog):
        hiring_like_pool["items"][0]["matched_key"] = "merit-orphan"
        with caplog.at_level("WARNING"):
            pairs = pools.matched_pairs(hiring_like_pool, two_family_pool)
        keys = {h["matched_key"] for h, _ in pairs}
        assert "merit-orphan" not in keys
        assert "only one label" in caplog.text

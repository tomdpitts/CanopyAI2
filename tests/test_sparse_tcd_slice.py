"""Offline guard on the sparse-canopy TCD slice: it must never contain a tile the model saw.

Runs without Modal, without a GPU and without reading the 2.8 GB of imagery -- it checks the
built manifests against the frozen cohort records. Skips cleanly if the slice is not built.
"""
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "boxinst_commonality_tcd_04")
SLICE = os.path.join(REPO, "data/tcd_sparse")
SM = os.path.join(SLICE, "slice_manifest.json")
HFM = os.path.join(PKG, "modal_sparse_tcd_multiseed/manifest_sparse.json")

pytestmark = pytest.mark.skipif(not os.path.exists(SM),
                                reason="sparse slice not built (build_slice.py)")


@pytest.fixture(scope="module")
def sm():
    return json.load(open(SM))


@pytest.fixture(scope="module")
def seen():
    """The 900, from BOTH frozen records -- which must agree with each other."""
    man = json.load(open(os.path.join(PKG, "modal_tcd_multiseed/phase4/manifest.json")))
    gt = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    assert set(man["feat_traintile"]) == set(gt) == set(man["feat_traintile"]) | set(gt)
    assert len(gt) == 900
    return {t: man["feat_traintile"][t]["image_id"] for t in gt}


def test_disjoint_by_tile_id(sm, seen):
    assert not set(sm["tiles"]) & set(seen)


def test_disjoint_by_image_id(sm, seen):
    assert not {v["image_id"] for v in sm["tiles"].values()} & set(seen.values())


def test_disjoint_by_pixels(sm, seen):
    """The gate ids alone cannot see: the same ortho crop under a different image_id."""
    ref = json.load(open(os.path.join(PKG, "modal_tcd_multiseed/phase4/manifest.json")))
    seen_sha = {r["rgb_sha1"] for r in ref["feat_traintile"].values() if "rgb_sha1" in r}
    assert seen_sha, "no reference hashes to compare against"
    assert not {v["rgb_sha1"] for v in sm["tiles"].values()} & seen_sha


def test_every_tile_is_in_the_declared_biome_set(sm):
    chosen = set(sm["criteria"]["biomes"])
    assert {v["biome"] for v in sm["tiles"].values()} <= chosen


def test_manifests_and_gt_describe_the_same_tiles(sm):
    hf = json.load(open(HFM))
    gt = json.load(open(os.path.join(SLICE, "sparse_gt.json")))
    assert set(sm["tiles"]) == set(hf["feat_test"]) == set(gt)
    assert sm["counts"]["n"] == len(sm["tiles"])
    assert not hf["feat_traintile"], "the slice is evaluation-only; no train half"


def test_scene_leakage_is_recorded_not_hidden(sm):
    """Tile-disjoint is not scene-disjoint. The flag must exist so the honest cut is possible."""
    assert sm["criteria"]["scene_disjoint"] is False
    clean = sum(v["scene_clean"] for v in sm["tiles"].values())
    assert clean == sm["counts"]["scene_clean"] and 0 < clean < len(sm["tiles"])

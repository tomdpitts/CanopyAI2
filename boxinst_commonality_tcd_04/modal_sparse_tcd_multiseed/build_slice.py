"""Build the sparse-canopy OAM-TCD test slice: open-canopy biomes, none of the seen 900.

Purpose: measure the settled phase-4 detector + beta=0.5 masker on OPEN-CANOPY imagery
WITHOUT retraining. The 900 tiles it was fitted on are ~80% closed-canopy forest, so the
439-tile headline says nothing about the dryland/savanna regime the LACE paper claims.

Selection = biome in SPARSE (WWF/Olson codes 7,8,9,10,12,13) AND not one of the 900.
Sources are the whole local corpus: the unseen part of data/tcd/train, plus the matching
tiles of the official 439 (data/tcd/test). Those 16 were never TRAINED on but WERE used to
tune mask_thr and the alpha/kappa knobs, so every tile records `source` and results can be
quoted with or without them.

NON-DESTRUCTIVE: reads data/tcd/{train,test}, writes only under data/tcd_sparse/. Tiles are
COPIED (not symlinked) so the slice survives independently of the mirror.

Exclusion is by tile id (your call), enforced by three independent gates that must all pass:
  1. tile id      vs both seen-900 records (manifest.json AND train_tiles_gt.json)
  2. image_id     vs the seen-900 image_ids
  3. rgb sha1     vs the seen-900 pixels -- the only gate that can catch the same ortho
                  crop republished under a different image_id
The slice is therefore TILE-disjoint from training. It is NOT scene-disjoint: 130 of the 221
train-pool tiles sit in a 300 m spatial cluster containing a training tile (adjacent crops of
the same OpenAerialMap ortho). Every tile carries `scene_clean` so the honest cut is one
filter away -- report it.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.build_slice --dry-run
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.build_slice
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from boxinst_commonality_tcd_04.prepare_test import anns_of, seg_rings
from boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.tile_index import (
    BIOME, DIRS, HERE, PHASE4_MANIFEST, PKG, REPO, SPARSE_DEFAULT, candidates, load,
    seen_tiles)

Image.MAX_IMAGE_PIXELS = None

DST = os.path.join(REPO, "data/tcd_sparse")
DST_TEST = os.path.join(DST, "test")
SLICE_MANIFEST = os.path.join(DST, "slice_manifest.json")
GT_OUT = os.path.join(DST, "sparse_gt.json")
HF_MANIFEST = os.path.join(HERE, "manifest_sparse.json")
SHA_CACHE = os.path.join(HERE, "rgb_sha1.json")


def tif_path(rec):
    return os.path.join(DIRS[rec["split"]], rec.get("src_tid", rec["tid"]) + ".tif")


def meta_path(rec):
    return os.path.join(DIRS[rec["split"]],
                        rec.get("src_tid", rec["tid"]) + "_meta.json")


def readable(path):
    """A tif that is present and decodable. Three tiles in the local mirror (image_id 0,1,2)
    are ZERO BYTES from a truncated download -- the HF rows are fine, only the local copies
    are empty, so this is a mirror defect, not a dataset one."""
    try:
        if os.path.getsize(path) == 0:
            return False
        Image.open(path).verify()
        return True
    except Exception:
        return False


def rgb_stats(path):
    """(sha1 of the RGB array, valid_frac). Matches the hash phase4/manifest.json stores,
    so the 20 sha1s recorded there are a live correctness check on this function.

    valid_frac = share of non-black pixels: OAM orthos are rotated, clipped mosaics and many
    tiles carry a large no-data wedge, which no annotation field reveals."""
    a = np.asarray(Image.open(path).convert("RGB"))
    return (hashlib.sha1(a.tobytes()).hexdigest(),
            float((a[::8, ::8].max(2) > 0).mean()), list(a.shape))


def hashes(recs, label, cache):
    """sha1 + valid_frac per tile, memoised on disk (~12 MB read each, so worth caching)."""
    todo = [r for r in recs if r["tid"] not in cache]
    t0 = time.time()
    for k, r in enumerate(todo):
        if not readable(tif_path(r)):
            cache[r["tid"]] = {"rgb_sha1": None, "valid_frac": None, "rgb_shape": None}
            continue
        sha, valid, shape = rgb_stats(tif_path(r))
        cache[r["tid"]] = {"rgb_sha1": sha, "valid_frac": round(valid, 4),
                           "rgb_shape": shape}
        if (k + 1) % 100 == 0 or k + 1 == len(todo):
            el = time.time() - t0
            print(f"  [{label}] {k + 1}/{len(todo)}  {el / (k + 1):.2f}s/tile  "
                  f"ETA {(len(todo) - k - 1) * el / (k + 1) / 60:.1f} min", flush=True)
    return cache


def build_gt(recs):
    """{tid: {trees, canopy, W, H}} -- the exact shape of the frozen test_gt.json, built with
    prepare_test's own polygon/RLE handling. prepare_test.main() is NOT called: it globs
    data/tcd/test and would overwrite the frozen 439 GT."""
    gt, n_trees, n_canopy, n_rle, n_empty = {}, 0, 0, 0, 0
    for r in recs:
        meta = json.load(open(meta_path(r)))
        H, W = meta["height"], meta["width"]
        trees, canopy = [], []
        for a in anns_of(meta, 2):
            n_rle += isinstance(a["segmentation"], dict)
            rs = [x for x in seg_rings(a["segmentation"], H, W) if len(x) >= 6]
            if rs:
                trees.append(max(rs, key=len))
        for a in anns_of(meta, 1):
            n_rle += isinstance(a["segmentation"], dict)
            canopy += [x for x in seg_rings(a["segmentation"], H, W) if len(x) >= 6]
        gt[r["tid"]] = {"trees": trees, "canopy": canopy, "W": W, "H": H}
        n_trees += len(trees); n_canopy += len(canopy); n_empty += not trees
    print(f"[gt] {len(gt)} tiles ({n_empty} with no ITC), {n_trees} tree rings, "
          f"{n_canopy} canopy rings, {n_rle} RLE anns decoded")
    return gt, n_trees


def gates(recs, seen, cache, biomes, seen_hashable):
    """Independent disjointness proofs. Raises on any failure."""
    seen_tids, seen_iids = set(seen), set(seen.values())

    bad = sorted(r["tid"] for r in recs if r["tid"] in seen_tids)
    assert not bad, f"GATE 1 FAILED — tile id overlap with the seen 900: {bad[:10]}"
    print(f"[gate 1] tile id      OK — 0 / {len(recs)} in the seen 900")

    bad = sorted(r["image_id"] for r in recs if r["image_id"] in seen_iids)
    assert not bad, f"GATE 2 FAILED — image_id overlap with the seen 900: {bad[:10]}"
    print(f"[gate 2] image_id     OK — 0 / {len(recs)} in the seen 900")

    ref = json.load(open(PHASE4_MANIFEST))
    checked = 0
    for tiles in ref.values():
        for tid, rec in tiles.items():
            if "rgb_sha1" in rec and tid in cache:
                assert cache[tid]["rgb_sha1"] == rec["rgb_sha1"], \
                    f"hashing disagrees with phase4/manifest.json on {tid}"
                checked += 1
    assert checked >= 20, f"only {checked} recorded sha1s available to self-check"
    nohash = sorted(r["tid"] for r in recs if cache[r["tid"]]["rgb_sha1"] is None)
    assert not nohash, f"GATE 3 FAILED — candidate pixels unreadable, cannot clear: {nohash}"
    hashable = {r["tid"] for r in seen_hashable}
    assert len(hashable) >= len(seen_tids) - 5, \
        f"too much of the 900 unreadable ({len(hashable)}/{len(seen_tids)}) to trust gate 3"
    seen_sha = {cache[t]["rgb_sha1"] for t in hashable}
    dup = sorted(r["tid"] for r in recs if cache[r["tid"]]["rgb_sha1"] in seen_sha)
    assert not dup, f"GATE 3 FAILED — pixel-identical to a training tile: {dup[:10]}"
    print(f"[gate 3] rgb sha1     OK — 0 / {len(recs)} pixel-identical to "
          f"{len(hashable)} / {len(seen_tids)} of the 900 "
          f"(hasher self-checked against {checked} recorded sha1s)")

    bad = sorted({r["biome"] for r in recs} - set(biomes))
    assert not bad, f"GATE 4 FAILED — biome outside the chosen set: {bad}"
    print(f"[gate 4] biome        OK — all in {sorted(biomes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomes", default=",".join(map(str, SPARSE_DEFAULT)))
    ap.add_argument("--dry-run", action="store_true",
                    help="select + run every gate, write nothing")
    ap.add_argument("--inject-seen", default=None, metavar="TID",
                    help="smuggle a known training tile into the selection; the named gate "
                         "must then FAIL. Proves the gates are live, not vacuous.")
    ap.add_argument("--inject-gate", type=int, default=1, choices=(1, 2, 3),
                    help="1: same tile id. 2: id disguised, image_id kept. 3: id AND "
                         "image_id disguised, pixels kept — the republished-ortho case.")
    ap.add_argument("--no-official-test", action="store_true",
                    help="drop the tiles that come from the official 439")
    a = ap.parse_args()
    biomes = tuple(int(x) for x in a.biomes.split(","))

    idx, seen = load(), seen_tiles()
    recs = candidates(idx, biomes, include_official_test=not a.no_official_test)

    # Preflight: a candidate we cannot read cannot be copied or evaluated. Drop loudly.
    unreadable = [r for r in recs if not readable(tif_path(r))]
    if unreadable:
        print("[preflight] DROPPED — unreadable tif in the local mirror:")
        for r in unreadable:
            print(f"    {r['tid']} (image_id {r['image_id']}, biome {r['biome']}, "
                  f"{r['n_crowns']} crowns)")
        recs = [r for r in recs if readable(tif_path(r))]
    if a.inject_seen:
        hit = [r for r in idx["tiles"] if r["tid"] == a.inject_seen]
        assert hit and hit[0]["seen"], f"{a.inject_seen} is not one of the seen 900"
        r = dict(hit[0], source="INJECTED", src_tid=a.inject_seen,
                 biome=biomes[0], scene_clean=True)
        if a.inject_gate >= 2:                       # disguise the tile id
            r["tid"] = "INJECTED_" + a.inject_seen
        if a.inject_gate >= 3:                       # and the image_id; pixels unchanged
            r["image_id"] = -1
        recs = recs + [r]
        print(f"[!] injected {a.inject_seen} as {r['tid']} (image_id {r['image_id']}) "
              f"— gate {a.inject_gate} is expected to FAIL")

    by_b = {b: sum(r["biome"] == b for r in recs) for b in sorted(biomes)}
    by_s = {s: sum(r["source"] == s for r in recs) for s in {r["source"] for r in recs}}
    print(f"[select] {len(recs)} tiles  biomes={by_b}  source={by_s}")

    cache = json.load(open(SHA_CACHE)) if os.path.exists(SHA_CACHE) else {}
    n0 = len(cache)
    seen_all = [r for r in idx["tiles"] if r["seen"]]
    seen_recs = [r for r in seen_all if readable(tif_path(r))]
    if len(seen_recs) != len(seen_all):
        miss = sorted(r["tid"] for r in seen_all if r not in seen_recs)
        print(f"[preflight] {len(miss)} of the 900 have no readable local tif and so are "
              f"outside gate 3's pixel comparison: {miss}. They ARE still covered by gates "
              f"1 and 2 (the model saw them via HF, not via this mirror).")
    print(f"[hash] {len(seen_recs) + len(recs) - n0} tiles to read "
          f"({n0} cached) — sha1 + no-data fraction")
    hashes(seen_recs, "seen-900", cache)
    hashes(recs, "candidates", cache)
    if len(cache) != n0 and not a.dry_run:
        json.dump(cache, open(SHA_CACHE, "w"))

    gates(recs, seen, cache, biomes, seen_recs)

    v = np.array([cache[r["tid"]]["valid_frac"] for r in recs], dtype=float)
    print(f"[nodata] valid_frac: median {np.median(v) * 100:.0f}%  "
          f"<90%: {(v < .9).sum()}  <70%: {(v < .7).sum()}  <50%: {(v < .5).sum()} "
          f"— recorded, NOT filtered")

    if a.dry_run:
        print("[dry-run] all gates passed; nothing written")
        return

    os.makedirs(DST_TEST, exist_ok=True)
    for k, r in enumerate(recs):
        shutil.copy2(tif_path(r), os.path.join(DST_TEST, r["tid"] + ".tif"))
        shutil.copy2(meta_path(r), os.path.join(DST_TEST, r["tid"] + "_meta.json"))
        if (k + 1) % 50 == 0 or k + 1 == len(recs):
            print(f"  [copy] {k + 1}/{len(recs)}", flush=True)

    gt, n_trees = build_gt(recs)
    json.dump(gt, open(GT_OUT, "w"))

    # phase-4-shaped manifest: the isolated Modal app joins to HF restor/tcd by image_id,
    # never by filename. Empty train half -- this slice is evaluation-only by design.
    json.dump({"feat_traintile": {},
               "feat_test": {r["tid"]: {"image_id": r["image_id"], "width": r["width"],
                                        "height": r["height"], "n_cat2": r["n_crowns"],
                                        "rgb_sha1": cache[r["tid"]]["rgb_sha1"],
                                        "rgb_shape": cache[r["tid"]]["rgb_shape"]}
                              for r in recs}},
              open(HF_MANIFEST, "w"))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      text=True).strip()
    except Exception:
        sha = None
    out = {
        "criteria": {
            "biomes": sorted(biomes),
            "biome_names": {str(b): BIOME.get(b, f"code {b}") for b in sorted(biomes)},
            "exclude": "the 900 seen tiles, by tile_id + image_id + rgb_sha1",
            "sources": "unseen part of data/tcd/train + matching tiles of the official 439",
            "scene_disjoint": False,
            "dropped_unreadable": [r["tid"] for r in unreadable],
            "scene_link_m": idx["meta"]["scene_link_m"],
            "coverage": "canopy_frac/crown_frac are RLE-safe UNION areas; the COCO `area` "
                        "field is 0 on RLE anns and must not be summed",
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "src_commit": sha,
        },
        "counts": {
            "n": len(recs), "by_biome": by_b, "by_source": by_s,
            "scene_clean": sum(r["scene_clean"] for r in recs),
            "gt_tree_rings": n_trees,
            "n_crowns": sum(r["n_crowns"] for r in recs),
        },
        "tiles": {r["tid"]: {k: r[k] for k in (
            "image_id", "split", "source", "biome", "biome_name", "country", "bounds",
            "crs", "width", "height", "n_crowns", "canopy_frac", "crown_frac",
            "scene_id", "scene_clean")} | cache[r["tid"]] for r in recs},
    }
    json.dump(out, open(SLICE_MANIFEST, "w"), indent=1)
    print(f"[done] {len(recs)} tiles -> {os.path.relpath(DST, REPO)}  "
          f"({out['counts']['scene_clean']} scene-clean, {n_trees} GT crowns)")


if __name__ == "__main__":
    main()

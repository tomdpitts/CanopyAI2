"""Tier B split manifest: the exact tiles LACE used, by ID, with file and pixel hashes.

Every Tier B run (B1 BoxInst, B2 BoxTeacher, B3 MAL) trains, selects and is tested on the
tiles named here and nothing else. The containers re-hash every file they read against this
manifest before the first GPU second, so "same images as LACE" is checked, not assumed.

Sources of truth (read, never written):
  train/val  train_tiles_gt.json `partition`   -- the file LACE's detector and masker read
             (modal_tcd_multiseed/phase4/phase4_modal.py TRAIN_GT); 792 train + 108 val = 900
  val GT     modal_tcd_multiseed/phase4/val_gt.json  -- polygon GT for the 108 val tiles
  test       test_gt.json                       -- the frozen 439
  sparse     data/tcd_sparse/sparse_gt.json     -- the 236-tile sparse slice (16 overlap test)

Cross-checks (all must pass or nothing is written):
  * 792 / 108 / 439 / 236 exact counts; train and val disjoint; neither meets test or sparse
  * val_gt.json keys == the 108 val ids
  * LACE's deployed feature cache `tcd04-phase4-vol:feat_4p_train` (a saved listing) holds
    exactly the 900 train+val ids -- independent evidence that LACE trained on these tiles
  * every image is a 2048x2048 RGB TIFF; file sha256 and decoded-RGB sha256 recorded
  * each tile's HF `restor/tcd` image_id (from its meta.json) equals the phase-4 manifest's
    (modal_tcd_multiseed/phase4/manifest.json), the join LACE's features were extracted by

LACE's pixels came from the HF `restor/tcd` copy on the `canopyai-deepforest-data` volume,
joined by image_id (phase4_modal.py). Tier B stages its images from that SAME copy
(tierb_modal.py::stage) and checks them against the local hashes recorded here. A local file
that is empty or unreadable is recorded as `local_missing` rather than hashed; its HF pixels
are then the only copy and the stage records their hash.

Usage (repo root):
    .venv/bin/modal volume ls tcd04-phase4-vol feat_4p_train > \
        boxinst_commonality_tcd_04/box_supervised_baselines/tierb/lace_feat_4p_train_listing.txt
    PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.box_supervised_baselines.tierb.splits
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))            # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "splits_manifest.json")
LISTING = os.path.join(HERE, "lace_feat_4p_train_listing.txt")

TRAIN_GT = os.path.join(PKG, "train_tiles_gt.json")
PH4_MANIFEST = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
TEST_GT = os.path.join(PKG, "test_gt.json")
SPARSE_GT = os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")

SRC = {"train": os.path.join(REPO, "data", "tcd", "train"),
       "val": os.path.join(REPO, "data", "tcd", "train"),
       "test": os.path.join(REPO, "data", "tcd", "test"),
       "sparse": os.path.join(REPO, "data", "tcd_sparse", "test")}
COUNTS = {"train": 792, "val": 108, "test": 439, "sparse": 236}


def _hash_one(args):
    split, tid, path = args
    meta = json.load(open(path[:-4] + "_meta.json"))
    rec = {"image_id": int(meta["image_id"]), "hf_split": "test" if "_val_" in tid else "train"}
    b = open(path, "rb").read()
    if len(b) == 0:
        return split, tid, dict(rec, local_missing="0-byte file")
    im = Image.open(io.BytesIO(b))
    mode, size = im.mode, im.size
    rgb = np.asarray(im.convert("RGB"))
    return split, tid, {**rec, "file_sha256": hashlib.sha256(b).hexdigest(),
                        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "bytes": len(b), "mode": mode, "size": list(size)}


def ids():
    tg = json.load(open(TRAIN_GT))
    out = {"train": sorted(t for t, v in tg.items() if v["partition"] == "train"),
           "val": sorted(t for t, v in tg.items() if v["partition"] == "val"),
           "test": sorted(json.load(open(TEST_GT))),
           "sparse": sorted(json.load(open(SPARSE_GT)))}
    for k, n in COUNTS.items():
        assert len(out[k]) == n, f"{k}: {len(out[k])} != {n}"
    tr, va, te, sp = (set(out[k]) for k in ("train", "val", "test", "sparse"))
    assert not tr & va, "train/val overlap"
    assert not (tr | va) & te, "train/val meet test"
    assert not (tr | va) & sp, "train/val meet sparse"
    assert set(json.load(open(VAL_GT))) == va, "val_gt.json keys != val ids"
    return out


def lace_listing_check(out):
    """The saved `modal volume ls` of LACE's deployed train-feature cache must name exactly the
    900 train+val tiles. Returns the check record."""
    assert os.path.exists(LISTING), f"save the listing first (see module doc): {LISTING}"
    names = [ln.strip() for ln in open(LISTING) if ln.strip().endswith(".npy")]
    lace = {os.path.basename(n)[:-4] for n in names}
    want = set(out["train"]) | set(out["val"])
    assert lace == want, (f"LACE feature cache != 900 ids: {len(lace - want)} extra, "
                          f"{len(want - lace)} missing")
    return {"source": "tcd04-phase4-vol:feat_4p_train (modal volume ls, saved listing)",
            "n": len(lace), "equal_to_train_plus_val": True}


def main():
    t0 = time.time()
    out = ids()
    lace = lace_listing_check(out)
    jobs = [(s, t, os.path.join(SRC[s], t + ".tif")) for s in COUNTS for t in out[s]]
    missing = [p for _, _, p in jobs if not os.path.exists(p)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    hashes = {s: {} for s in COUNTS}
    with Pool(8) as pool:
        for k, (s, t, h) in enumerate(pool.imap_unordered(_hash_one, jobs, chunksize=4)):
            assert "local_missing" in h or (h["mode"] == "RGB" and h["size"] == [2048, 2048]), \
                (s, t, h)
            hashes[s][t] = h
            if (k + 1) % 200 == 0 or k + 1 == len(jobs):
                print(f"  hashed {k+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    # the 16 sparse tiles that are also test tiles must be byte-identical in both copies
    shared = sorted(set(out["sparse"]) & set(out["test"]))
    for t in shared:
        assert hashes["sparse"][t]["file_sha256"] == hashes["test"][t]["file_sha256"], t
    # image_id join must agree with the manifest LACE's features were extracted by
    ph4 = json.load(open(PH4_MANIFEST))
    for s_, key in (("train", "feat_traintile"), ("val", "feat_traintile"),
                    ("test", "feat_test")):
        for t in out[s_]:
            assert ph4[key][t]["image_id"] == hashes[s_][t]["image_id"], (s_, t)
    iids = [(h["hf_split"], h["image_id"]) for s_ in ("train", "val", "test")
            for h in hashes[s_].values()]
    assert len(iids) == len(set(iids)), "duplicate HF image_id across train/val/test"
    local_missing = sorted(f"{s_}/{t}" for s_ in COUNTS for t, h in hashes[s_].items()
                           if "local_missing" in h)
    print(f"local_missing: {local_missing}", flush=True)
    man = {"created": time.strftime("%Y-%m-%d %H:%M"),
           "counts": {k: len(v) for k, v in out.items()},
           "sources": {"train_val": os.path.relpath(TRAIN_GT, REPO),
                       "val_gt": os.path.relpath(VAL_GT, REPO),
                       "test": os.path.relpath(TEST_GT, REPO),
                       "sparse": os.path.relpath(SPARSE_GT, REPO),
                       "images": {k: os.path.relpath(v, REPO) for k, v in SRC.items()}},
           "checks": {"disjoint": "train/val disjoint; neither meets test or sparse",
                      "val_gt_keys_equal_val_ids": True,
                      "lace_feature_cache": lace,
                      "sparse_test_overlap": {"n": len(shared),
                                              "byte_identical": True, "ids": shared},
                      "image_id_matches_phase4_manifest": True,
                      "local_missing": local_missing,
                      "all_2048_rgb": "every local file that exists"},
           "ids": out, "hashes": hashes}
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    json.dump(man, open(OUT, "w"), indent=1)
    print(f"-> {OUT}  {man['counts']}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

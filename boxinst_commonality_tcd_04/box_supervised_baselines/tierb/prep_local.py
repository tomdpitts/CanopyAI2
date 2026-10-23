"""Build Tier B's small derived inputs LOCALLY, deterministically, from files already in the repo.
They ship inside the Modal images (content-hashed layers), with a manifest of every source and
output sha256 so a later reader can prove what the runs consumed. Refuses to overwrite.

  tierb_data/canopy/<tid>.png          LACE's canopy raster (train_detector_tiles.canopy_px, verbatim
                                       as tierb_common.canopy_px) for the 792 training tiles, 0/1 L-mode
  tierb_data/mal_train_boxes.json      LACE's 792-tile training boxes as COCO xywh (MAL's input)
  tierb_data/val_gt_crowd_2048.json    the 108 val tiles' crowd-arm GT for score_lite (selection)
  tierb_data/boxsets/<name>.json       boxes B3 masks:
      val_gt / test_gt / sparse_gt     gtbox_lib.gt_boxes_2048 of every GT crown (tab:gtbox boxes)
      lace_s{0,1,2}                    LACE detector boxes + s (native_raster/guided_ablate_inputs; = A3)
      selvabox                         seam-fixed SelvaBox boxes + s (seamfix preds; = A2/A3)
      lace_sparse_s{0,1,2}             LACE detector boxes + s on the sparse 236 (detscore preds)
  tierb_data/DATA_MANIFEST.json

    PYTHONPATH=.:boxinst_commonality_tcd_04/box_supervised_baselines \
        .venv/bin/python -m tierb.prep_local
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from PIL import Image

from tierb import gtbox_pure as GP
from tierb import score_lite as SL
from tierb import tierb_common as TC

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "tierb_data")
NR = os.path.join(PKG, "native_raster")

SRC = {
    "train_gt": os.path.join(PKG, "train_tiles_gt.json"),
    "val_gt": os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json"),
    "test_gt": os.path.join(PKG, "test_gt.json"),
    "sparse_gt": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json"),
    **{f"lace_s{s}": os.path.join(NR, "guided_ablate_inputs", f"detscore_s{s}.json") for s in (0, 1, 2)},
    "selvabox": os.path.join(NR, "seamfix", "preds_selvabox_seamfix_on_r2048.json"),
    **{f"lace_sparse_s{s}": os.path.join(NR, f"preds_lace_sparse_s{s}_r2048_cgf_r12_tau035_detscore.json")
       for s in (0, 1, 2)},
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def xywh(b):
    x0, y0, x1, y1 = (float(v) for v in b)
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    os.makedirs(os.path.join(OUT, "canopy"))
    os.makedirs(os.path.join(OUT, "boxsets"))
    man = json.load(open(os.path.join(HERE, "splits_manifest.json")))
    rec = {"created": time.strftime("%Y-%m-%d %H:%M"), "sources": {k: {"path": os.path.relpath(v, REPO),
                                                                      "sha256": sha(v)} for k, v in SRC.items()},
           "outputs": {}}
    tg = json.load(open(SRC["train_gt"]))

    # canopy rasters (training tiles only: the ignore regions of LACE's detector loss)
    can = {}
    for t in man["ids"]["train"]:
        m = TC.canopy_px(tg[t]["canopy"])
        p = os.path.join(OUT, "canopy", f"{t}.png")
        Image.fromarray(m.astype(np.uint8)).save(p, optimize=True)
        assert np.array_equal(np.asarray(Image.open(p)), m.astype(np.uint8))
        can[t] = {"sha256": sha(p), "canopy_frac": round(float(m.mean()), 6)}
    rec["outputs"]["canopy"] = {"n": len(can), "mean_canopy_frac": round(float(np.mean([v["canopy_frac"] for v in can.values()])), 6),
                                "files": can}

    # MAL training boxes
    imgs, anns, aid = [], [], 1
    for i, t in enumerate(man["ids"]["train"]):
        imgs.append({"id": i + 1, "file_name": f"{t}.png", "height": 2048, "width": 2048})
        for b in tg[t]["boxes"]:
            w = xywh(b)
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "bbox": w,
                         "iscrowd": 0, "area": w[2] * w[3]})
            aid += 1
    p = os.path.join(OUT, "mal_train_boxes.json")
    json.dump({"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "tree"}]}, open(p, "w"))
    rec["outputs"]["mal_train_boxes"] = {"images": len(imgs), "annotations": len(anns), "sha256": sha(p)}

    # val crowd GT for score_lite (checkpoint selection, B1/B2)
    vg = json.load(open(SRC["val_gt"]))
    vg = {t: vg[t] for t in man["ids"]["val"]}
    gd, _ = SL.build_gt_crowd(vg, 2048)
    p = os.path.join(OUT, "val_gt_crowd_2048.json")
    json.dump(SL.gt_to_json(gd), open(p, "w"))
    rec["outputs"]["val_gt_crowd_2048"] = {"images": len(gd["images"]), "annotations": len(gd["annotations"]),
                                          "sha256": sha(p)}

    # box sets for B3
    def write_set(name, tiles, meta):
        p = os.path.join(OUT, "boxsets", f"{name}.json")
        json.dump({"meta": meta, "tiles": tiles}, open(p, "w"))
        rec["outputs"][f"boxset_{name}"] = {"tiles": len(tiles), "boxes": sum(len(v["xyxy"]) for v in tiles.values()),
                                           "sha256": sha(p), **meta}

    for name, split in (("val_gt", "val"), ("test_gt", "test"), ("sparse_gt", "sparse")):
        g = json.load(open(SRC[name]))
        tiles = {}
        for t in man["ids"][split]:
            b = GP.gt_boxes_2048(g[t]["trees"]).astype(np.float64).tolist()
            tiles[t] = {"xyxy": b, "xywh": [xywh(x) for x in b], "scores": None}
        write_set(name, tiles, {"kind": "gt", "split": split, "rule": "gtbox_lib.gt_boxes_2048 (crown order)"})

    for name in ("lace_s0", "lace_s1", "lace_s2", "selvabox", "lace_sparse_s0", "lace_sparse_s1", "lace_sparse_s2"):
        d = json.load(open(SRC[name]))
        d = d.get("preds", d)
        split = "sparse" if "sparse" in name else "test"
        assert set(d) == set(man["ids"][split]), (name, len(d))
        tiles = {t: {"xyxy": [[float(v) for v in b] for b in d[t]["boxes_2048"]],
                     "xywh": [xywh(b) for b in d[t]["boxes_2048"]],
                     "scores": [float(s) for s in d[t]["scores"]]} for t in man["ids"][split]}
        write_set(name, tiles, {"kind": "detector", "split": split, "ranking": "detector score s"})

    json.dump(rec, open(os.path.join(OUT, "DATA_MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in rec["outputs"].items()},
                     indent=1))


if __name__ == "__main__":
    main()

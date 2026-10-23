"""Fix 1, parameter-free: every LACE test detection whose deployed guided mask is EMPTY gets the ellipse
inscribed in its box; boxes and scores (s') unchanged. Rescored with the frozen protocol (2048, maxDets 600,
floor 0, canopy crowd), 3 seeds. Writes smallcrown/preds_lace_s{s}_r2048_cgf_emptyfallback.json and
smallcrown/scored_*.json (refuses to overwrite)."""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import score_coco as SC
from boxinst_commonality_tcd_04.native_raster.tierA.a3_ellipse import ellipse_rle
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
paths = []
for s in (0, 1, 2):
    fp = os.path.join(HERE, f"preds_lace_s{s}_r2048_cgf_emptyfallback.json")
    paths.append(fp)
    if os.path.exists(fp):
        continue
    D = json.load(open(os.path.join(NR, f"preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))
    n = 0
    for t, r in D["preds"].items():
        for i, m in enumerate(r["masks_rle"]):
            if mu.area({"size": m["size"], "counts": m["counts"].encode()}) == 0:
                r["masks_rle"][i] = ellipse_rle(r["boxes_2048"][i]); n += 1
    D["meta"] = dict(D.get("meta", {}), empty_mask_fallback="ellipse-in-box", n_replaced=n)
    json.dump(D, open(fp, "w")); print(f"s{s}: replaced {n}", flush=True)
os.chdir("/Users/tompitts/dphil/CanopyAI2")
for p, r in zip(paths, SC.score_many(paths, "boxinst_commonality_tcd_04/test_gt.json", res=2048, max_dets=600, score_floor=0.0)):
    o = p.replace("preds_", "scored_"); assert not os.path.exists(o), o
    json.dump(r, open(o, "w")); c = r["canopy_neutral_crowd"]
    print(os.path.basename(p), "AP50", c["mask"]["AP50"], "AP50:95", c["mask"]["AP50_95"], flush=True)

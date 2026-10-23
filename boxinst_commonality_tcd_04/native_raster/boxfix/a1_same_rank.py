"""LACE boxes on the 439 test tiles: SAM 3 zero-shot masks (A1, tierA/a1_tiles_s{s}) ranked by LACE's product
(published scores verbatim), 3 seeds -- the same-ranking comparison against LACE's own masks (0.647).
Frozen protocol (2048, maxDets 600, floor 0, canopy crowd). Writes boxfix/preds_a1_sam3zs_lace_product_s{s}_r2048.json
and scored_*.json (refuses to overwrite)."""
import json, os
import numpy as np
from boxinst_commonality_tcd_04 import score_coco as SC
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR)
paths = []
for s in (0, 1, 2):
    fp = os.path.join(HERE, f"preds_a1_sam3zs_lace_product_s{s}_r2048.json"); paths.append(fp)
    if os.path.exists(fp):
        continue
    pub = json.load(open(os.path.join(NR, f"preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))["preds"]
    preds = {}
    for t, P in pub.items():
        a = json.load(open(os.path.join(NR, "tierA", f"a1_tiles_s{s}", t + ".json")))
        assert a["boxes_2048"] == P["boxes_2048"] and len(a["masks_rle"]) == len(P["scores"]), t
        preds[t] = {"boxes_2048": P["boxes_2048"], "scores": P["scores"], "masks_rle": a["masks_rle"]}
    json.dump({"meta": {"mask_res": 2048, "model": f"LACE s{s} boxes + SAM 3 zs per-subtile masks (A1), LACE product ranking"},
               "preds": preds}, open(fp, "w"))
os.chdir("/Users/tompitts/dphil/CanopyAI2")
aps = []
for p, r in zip(paths, SC.score_many(paths, os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=0.0)):
    o = p.replace("preds_", "scored_"); assert not os.path.exists(o), o
    json.dump(r, open(o, "w")); m = r["canopy_neutral_crowd"]["mask"]; aps.append(m["AP50"])
    print(os.path.basename(p), m, flush=True)
print(f"SAM masks @ LACE product: AP50 {np.mean(aps):.4f} +- {np.std(aps, ddof=1):.4f}", flush=True)

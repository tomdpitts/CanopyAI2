"""Assemble the per-subtile SAM 3 run (selvabox_persub_local.py) and score it, all 439 tiles, 2048,
maxDets 600 and 1600. Two rankings: detector score (as the current Table 1 row) and the released
sqrt(det x SAM). Boxes/scores asserted identical to the seam-fixed file. Floor 0 for both: every
det score is already >= 0.05 from the selection, so this equals floor 0.05 for the det ranking."""
import json, os
import numpy as np
from boxinst_commonality_tcd_04 import score_coco as SC
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
ref = json.load(open(os.path.join(NR, "seamfix", "preds_selvabox_seamfix_on_r2048.json")))["preds"]
gt = json.load(open(os.path.join(os.path.dirname(NR), "test_gt.json")))
P, G = {}, {}
for t in sorted(gt):
    r = json.load(open(os.path.join(HERE, "tiles_mps", t + ".json")))
    assert r["boxes_2048"] == ref[t]["boxes_2048"] and r["scores"] == ref[t]["scores"], t
    P[t] = {k: r[k] for k in ("boxes_2048", "scores", "sam_scores", "masks_rle")}
    det = np.asarray(r["scores"], float); sam = np.clip(np.asarray(r["sam_scores"], float), 0, None)
    G[t] = dict(P[t], scores=np.round(np.sqrt(det * sam), 6).tolist())
assert min(min(v["scores"]) for v in P.values() if v["scores"]) >= 0.05
meta = {"model": "SelvaBox -> SAM 3 FT, seam-fixed selection, SAM prompted PER SOURCE SUBTILE (CanopyRS "
                 "placement), local MPS, raw masks (no CanopyRS mask post-processing)", "mask_res": 2048}
paths = []
for name, preds, rk in (("preds_selvabox_persub_r2048.json", P, "det"), ("preds_selvabox_persub_r2048_rankgmean.json", G, "gmean")):
    fp = os.path.join(HERE, name); assert not os.path.exists(fp), fp
    json.dump({"meta": dict(meta, rank_key=rk), "preds": preds}, open(fp, "w")); paths.append(fp)
os.chdir("/Users/tompitts/dphil/CanopyAI2")
for md in (600, 1600):
    for p, r in zip(paths, SC.score_many(paths, "boxinst_commonality_tcd_04/test_gt.json", res=2048, max_dets=md, score_floor=0.0)):
        o = p.replace("preds_", "scored_").replace(".json", f"_md{md}.json"); assert not os.path.exists(o), o
        json.dump(r, open(o, "w")); c = r["canopy_neutral_crowd"]
        print(f"[md{md}] {os.path.basename(p)}: mask AP50 {c['mask']['AP50']}  AP50:95 {c['mask']['AP50_95']}  box AP50 {c['box']['AP50']}", flush=True)

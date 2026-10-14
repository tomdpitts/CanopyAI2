"""Audit 2026-10-14: re-rank the EXISTING SelvaBox->SAM3 2048 preds by the released CanopyRS ranking keys.
Masks, boxes and detection set unchanged (det floor 0.05 + NMS 0.7 + keep 1600 already applied upstream).
  sam   : score = SAM 3 iou_score      (CanopyRS segmenter COCO 'score' = segmenter_score; tile-level benchmark)
  gmean : score = sqrt(det * sam)      (CanopyRS aggregator-2 weighted_geometric_mean 0.5/0.5; no ioa-NMS, no 0.6 thr)
Scored with score_floor=0 (floor already applied on the detector score), maxDets 600 and 1600, res 2048."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/Users/tompitts/dphil/CanopyAI2")
from boxinst_commonality_tcd_04 import score_coco as SC
SRC = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04/native_raster/preds_selvabox_bench_r2048.json"
OUT = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(SRC))
paths = []
for key in ("sam", "gmean"):
    fp = os.path.join(OUT, f"preds_selvabox_bench_r2048_rank{key}.json")
    assert not os.path.exists(fp), fp
    P = {}
    for t, r in d["preds"].items():
        det = np.asarray(r["scores"], np.float64); sam = np.asarray(r["sam_scores"], np.float64)
        assert len(det) == len(sam)
        s = sam if key == "sam" else np.sqrt(det * np.clip(sam, 0, None))
        P[t] = {"boxes_2048": r["boxes_2048"], "scores": np.round(s, 6).tolist(), "masks_rle": r["masks_rle"]}
    json.dump({"meta": dict(d["meta"], rank_key=key, audit="2026-10-14 selvabox rerank"), "preds": P}, open(fp, "w"))
    paths.append(fp); print("wrote", fp, flush=True)
del d
os.chdir("/Users/tompitts/dphil/CanopyAI2")
for md in (600, 1600):
    res = SC.score_many(paths, "boxinst_commonality_tcd_04/test_gt.json", res=2048, max_dets=md, score_floor=0.0)
    for p, r in zip(paths, res):
        o = p.replace(".json", f"_scored_md{md}.json"); assert not os.path.exists(o)
        json.dump(r, open(o, "w"))
        cn = r["canopy_neutral_crowd"]
        print(f"[md{md}] {os.path.basename(p)}: mask AP50 {cn['mask']['AP50']} AP50:95 {cn['mask']['AP50_95']} box AP50 {cn['box']['AP50']}", flush=True)

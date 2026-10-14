"""EVIDENCE ONLY (not in the paper): the SelvaBox auditor's re-rank (audit_2026-10-14/selvabox_evidence/
rerank_score.py) applied to the SEAM-FIXED file. score = sqrt(det * SAM iou), CanopyRS aggregator-2 key
(weighted_geometric_mean 0.5/0.5); ioa-disambiguate NMS and the 0.6 threshold NOT applied. 26,986 of the
307,500 SAM scores are from local MPS SAM 3 (seamfix README), so this is provisional."""
import json, os
import numpy as np
from boxinst_commonality_tcd_04 import score_coco as SC
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "preds_selvabox_seamfix_on_r2048.json")
fp = os.path.join(HERE, "preds_selvabox_seamfix_on_r2048_rankgmean.json")
assert not os.path.exists(fp), fp
d = json.load(open(SRC)); P = {}
for t, r in d["preds"].items():
    det = np.asarray(r["scores"], np.float64); sam = np.asarray(r["sam_scores"], np.float64)
    assert len(det) == len(sam) and not np.isnan(sam).any()
    P[t] = {"boxes_2048": r["boxes_2048"], "scores": np.round(np.sqrt(det * np.clip(sam, 0, None)), 6).tolist(),
            "masks_rle": r["masks_rle"]}
json.dump({"meta": dict(d["meta"], rank_key="gmean"), "preds": P}, open(fp, "w")); del d
for md in (600, 1600):
    o = fp.replace(".json", f"_scored_md{md}.json"); assert not os.path.exists(o)
    r = SC.score_many([fp], "boxinst_commonality_tcd_04/test_gt.json", res=2048, max_dets=md, score_floor=0.0)[0]
    json.dump(r, open(o, "w")); c = r["canopy_neutral_crowd"]
    print(f"[md{md}] gmean: mask AP50 {c['mask']['AP50']} AP50:95 {c['mask']['AP50_95']} box AP50 {c['box']['AP50']}", flush=True)

"""Edge vs interior split of the SAM 3 - LACE mask-IoU gap (2026-11-04; pairing as mask_diag.py) -> results/mask_diag_edge_split_test.json"""
import json, os, numpy as np, multiprocessing as mp
from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fig_sam_vs_lace as F
from boxinst_commonality_tcd_04.dino_det.mask_diag import SAMD
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
if __name__ == "__main__":
    G = json.load(open(R.TEST_GT)); St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")))["preds"]; tids = sorted(St)
    gd, t2i = SL.build_gt_crowd(G, 2048); by = {i: [] for i in t2i.values()}
    for a in gd["annotations"]:
        if not a["iscrowd"]: by[a["image_id"]].append(a["segmentation"])
    jobs = [(t, [SL._as_bytes(r) for r in St[t]["masks_rle"]], [SL._as_bytes(r) for r in json.load(open(os.path.join(SAMD, t + ".json")))["masks_rle"]], St[t]["boxes_2048"], by[t2i[t]]) for t in tids]
    with mp.get_context("spawn").Pool(14) as p: rows = sum(p.map(F._tile, jobs, chunksize=4), [])
    edge = np.array([min(b[0], b[1], 2048 - b[2], 2048 - b[3]) <= 3 for b in (St[r[0]]["boxes_2048"][r[1]] for r in rows)])
    il = np.array([r[3] for r in rows]); is_ = np.array([r[4] for r in rows]); d = is_ - il
    out = {}
    for k, m in (("edge", edge), ("interior", ~edge), ("all", np.ones_like(edge))):
        out[k] = {"n": int(m.sum()), "frac": round(float(m.mean()), 3), "lace_iou": round(float(il[m].mean()), 3), "sam_iou": round(float(is_[m].mean()), 3),
                  "gap": round(float(d[m].mean()), 4), "share_of_total_gap": round(float(d[m].sum() / d.sum()), 3)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(R.OUT, "mask_diag_edge_split_test.json"), "w"), indent=1)

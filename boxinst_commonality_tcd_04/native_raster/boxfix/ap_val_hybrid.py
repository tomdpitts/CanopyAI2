"""Val-only fallback check (pre-agreed): boxfix_v1 masks only where its radius rule binds (box short side < 30 px,
since r = min(12, floor(0.4*short)) < 12 iff short < 30), deployed masks elsewhere. Built from the ap_val per-tile
renders (no re-render). Product and s rankings, all/A/B halves, frozen protocol. -> boxfix/ap_val_hybrid_s0_r2048.json"""
import json, os, tempfile
import numpy as np
from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
HERE = os.path.dirname(os.path.abspath(__file__)); TD = "/Users/tompitts/dphil/feat_cache/boxfix_ap_val_tiles"
OUT = os.path.join(HERE, "ap_val_hybrid_s0_r2048.json"); assert not os.path.exists(OUT)
gt = json.load(open(VAL_GT)); tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
P = {r: {} for r in ("product", "s")}; n_sw = n = 0
for t in tids:
    rec = json.load(open(os.path.join(TD, t + ".json")))
    b = np.asarray(rec["boxes_2048"]).reshape(-1, 4); short = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    ms = [rec["masks_rle"]["boxfix_v1"][i] if short[i] < 30 else rec["masks_rle"]["deployed"][i] for i in range(len(b))]
    n_sw += int((short < 30).sum()); n += len(b)
    for r in P:
        P[r][t] = {"boxes_2048": rec["boxes_2048"], "scores": rec[f"scores_{r}"], "canopy_ignore": [False] * len(b), "masks_rle": ms}
fp = os.path.join(HERE, "preds_val_s0_r2048_hybrid30_product.json"); assert not os.path.exists(fp)
json.dump({"meta": {"mask_res": 2048, "arm": "hybrid30", "rank": "product"}, "preds": P["product"]}, open(fp, "w"))
res = {"n_det": n, "n_switched": n_sw, "AP": {}}
for h, ts in {"all": tids, "A": tids[0::2], "B": tids[1::2]}.items():
    sub = {t: gt[t] for t in ts}
    gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False); gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
    for r in P:
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump({"meta": {"mask_res": 2048}, "preds": {t: P[r][t] for t in ts}}, open(tmp, "w"))
        c = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, cu, ts), 2048, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]; os.remove(tmp)
        res["AP"].setdefault(r, {})[h] = {"mask": c["mask"], "box": c["box"]}
        print(f"  {h} hybrid30 {r:8s} mask AP50 {c['mask']['AP50']:.4f} AP75 {c['mask']['AP75']:.4f} AP50:95 {c['mask']['AP50_95']:.4f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1); print(f"switched {n_sw}/{n}")

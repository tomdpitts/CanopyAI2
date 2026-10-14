"""Environment gate for the locally (MPS) prompted SelvaBox masks. On every 10th test tile (44),
re-prompt EVERY stored box with local SAM 3 and score the subset twice: stored A100 masks vs local
masks, same boxes, same scores, same GT subset. The AP difference bounds what the environment can
move in a row where only 26,986 of 307,500 masks are local.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.seamfix.selvabox_env_gate
"""
import json
import os
import tempfile
import time

import numpy as np

from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.native_raster import selvabox_seamfix_local as L

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "selvabox_env_gate.json")
CK = os.path.join(HERE, "selvabox_env_gate_tiles")      # per-tile checkpoint (local masks @2048)


def main(device="mps"):
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    os.makedirs(CK, exist_ok=True)
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = sorted(gt)[::10]
    sam, proc = L.load_sam(device)
    stored, local = {}, {}
    t0 = time.time()
    for n, t in enumerate(tids):
        r = json.load(open(os.path.join(L.SRC_TILES, t + ".json")))
        stored[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"], "masks_rle": r["masks_rle_2048"]}
        fp = os.path.join(CK, t + ".json")
        if not os.path.exists(fp):
            B = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
            m = []
            if len(B):
                o = L.sam_masks(sam, proc, L.load_rgb(t), B,
                                [list(range(i, min(i + 64, len(B)))) for i in range(0, len(B), 64)], device)
                m = [o[j]["m2048"] for j in range(len(B))]
            json.dump(m, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        local[t] = dict(stored[t], masks_rle=json.load(open(fp)))
        print(f"  {n+1}/{len(tids)} {t} {len(r['scores'])} boxes  {time.time()-t0:.0f}s", flush=True)
    sub = {t: gt[t] for t in tids}
    gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
    res = {"tiles": tids, "device": device, "n_boxes": sum(len(v["scores"]) for v in stored.values())}
    for name, P in (("stored_A100", stored), ("local_" + device, local)):
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump({"meta": {"mask_res": 2048}, "preds": P}, open(tmp, "w"))
        c = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, cu, tids), 2048, S.MAX_DETS, 0.05)
        os.remove(tmp)
        c = c["canopy_neutral_crowd"]
        res[name] = {"AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"], "box_AP50": c["box"]["AP50"]}
        print(name, res[name], flush=True)
    res["delta_AP50"] = round(res["local_" + device]["AP50"] - res["stored_A100"]["AP50"], 4)
    res["delta_AP50_95"] = round(res["local_" + device]["AP50_95"] - res["stored_A100"]["AP50_95"], 4)
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT, "delta AP50", res["delta_AP50"], "AP50:95", res["delta_AP50_95"])


if __name__ == "__main__":
    main()

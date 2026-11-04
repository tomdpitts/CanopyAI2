"""Environment gate for the SelvaBox row under sqrt(det * SAM) ranking (2026-11-03). selvabox_env_gate.py
gated the local (MPS) masks under det ranking only; ranking by gmean also consumes the SAM score, which
moves up to 0.287 per box on MPS (selvabox_verify_reuse_local_mps.json). On the same 44 tiles, re-prompt
EVERY stored box with local SAM 3 keeping masks AND SAM scores, then score: stored A100 vs local, det and
gmean ranking. The gmean AP delta bounds what the environment can move in the mixed row
(26,986 of 307,500 masks + scores local).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.seamfix.selvabox_env_gate_gmean
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
OUT = os.path.join(HERE, "selvabox_env_gate_gmean.json")
CK = os.path.join(HERE, "selvabox_env_gate_gmean_tiles")   # per-tile checkpoint: local masks @2048 + SAM scores
PRIOR = os.path.join(HERE, "selvabox_env_gate_tiles")       # masks from the 2026-10-14 gate (reproducibility check)


def gmean(det, sam):
    return np.round(np.sqrt(np.asarray(det, np.float64) * np.clip(np.asarray(sam, np.float64), 0, None)), 6).tolist()


def main(device="mps"):
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    os.makedirs(CK, exist_ok=True)
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = sorted(gt)[::10]
    sam, proc = L.load_sam(device)
    arms = {k: {} for k in ("A100_det", "local_det", "A100_gmean", "local_gmean")}
    sd, same_prior, n_prior = [], 0, 0
    t0 = time.time()
    for n, t in enumerate(tids):
        r = json.load(open(os.path.join(L.SRC_TILES, t + ".json")))
        nb = len(r["scores"])
        assert len(r["sam_scores"]) == nb and all(s is not None for s in r["sam_scores"]), t
        fp = os.path.join(CK, t + ".json")
        if not os.path.exists(fp):
            B = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
            rec = {"masks_rle": [], "sam_scores": []}
            if nb:
                o = L.sam_masks(sam, proc, L.load_rgb(t), B,
                                [list(range(i, min(i + 64, nb))) for i in range(0, nb, 64)], device)
                rec = {"masks_rle": [o[j]["m2048"] for j in range(nb)], "sam_scores": [o[j]["sam"] for j in range(nb)]}
            json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        loc = json.load(open(fp))
        assert len(loc["sam_scores"]) == nb
        sd += [abs(a - b) for a, b in zip(loc["sam_scores"], r["sam_scores"])]
        pp = os.path.join(PRIOR, t + ".json")
        if os.path.exists(pp):
            pm = json.load(open(pp)); n_prior += len(pm)
            same_prior += sum(a["counts"] == b["counts"] for a, b in zip(pm, loc["masks_rle"]))
        base = {"boxes_2048": r["boxes_2048"]}
        arms["A100_det"][t] = dict(base, scores=r["scores"], masks_rle=r["masks_rle_2048"])
        arms["local_det"][t] = dict(base, scores=r["scores"], masks_rle=loc["masks_rle"])
        arms["A100_gmean"][t] = dict(base, scores=gmean(r["scores"], r["sam_scores"]), masks_rle=r["masks_rle_2048"])
        arms["local_gmean"][t] = dict(base, scores=gmean(r["scores"], loc["sam_scores"]), masks_rle=loc["masks_rle"])
        print(f"  {n+1}/{len(tids)} {t} {nb} boxes  {time.time()-t0:.0f}s", flush=True)
    sub = {t: gt[t] for t in tids}
    gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
    sd = np.asarray(sd)
    res = {"tiles": tids, "device": device, "n_boxes": int(len(sd)), "max_dets": S.MAX_DETS,
           "score_floor": {"det": 0.05, "gmean": 0.0},
           "sam_score_absdiff": {"mean": float(sd.mean()), "median": float(np.median(sd)),
                                 "p99": float(np.quantile(sd, 0.99)), "max": float(sd.max())},
           "local_masks_identical_to_2026-10-14_gate": [int(same_prior), int(n_prior)]}
    for name, P in arms.items():
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump({"meta": {"mask_res": 2048}, "preds": P}, open(tmp, "w"))
        floor = 0.05 if name.endswith("det") else 0.0
        c = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, cu, tids), 2048, S.MAX_DETS, floor)
        os.remove(tmp)
        c = c["canopy_neutral_crowd"]
        res[name] = {"AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"], "box_AP50": c["box"]["AP50"]}
        print(name, res[name], flush=True)
    for k in ("det", "gmean"):
        res[f"delta_{k}"] = {m: round(res[f"local_{k}"][m] - res[f"A100_{k}"][m], 4) for m in ("AP50", "AP50_95", "box_AP50")}
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT, "delta_det", res["delta_det"], "delta_gmean", res["delta_gmean"], flush=True)


if __name__ == "__main__":
    main()

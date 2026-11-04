"""Mask IoU, LACE (D1) vs SAM 3 zs on the SAME D1 boxes (dino_det/SAM3_D1.md). Per split:
  agree_all  : mean IoU(LACE mask, SAM mask) over every D1 detection (s >= 0.05).
  agree_tp   : same, over detections whose box matches a GT tree (box IoU >= 0.5).
  vs GT      : for each GT tree, the D1 detection with the highest box IoU (>= 0.5); mean mask IoU of LACE's and of SAM's
               mask for that one detection with the GT mask (same detection for both maskers -> paired, ranking-free).
Output: results/sam3_d1/mask_iou_{split}.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_d1_iou
"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S


def xywh(b):
    b = np.asarray(b, float).reshape(-1, 4); return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def main():
    for split in ("test", "sparse"):
        L = json.load(open(os.path.join(S.RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json")))["preds"]
        M = json.load(open(os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json")))["preds"]
        g = json.load(open(S.SPLIT[split]["gt"])); gd, t2i = SL.build_gt_crowd(g, 2048)
        gts = {}
        for a in gd["annotations"]:
            if not a["iscrowd"]:
                gts.setdefault(a["image_id"], []).append(a)
        agree_all, agree_tp, gl, gs = [], [], [], []
        for t in sorted(g):
            assert L[t]["boxes_2048"] == M[t]["boxes_2048"], t
            n = len(M[t]["masks_rle"])
            if n == 0:
                continue
            lm = [dict(r, counts=r["counts"].encode()) for r in L[t]["masks_rle"]]
            sm = [dict(r, counts=r["counts"].encode()) for r in M[t]["masks_rle"]]
            a = np.array([mu.iou([lm[i]], [sm[i]], [0])[0, 0] for i in range(n)]); agree_all += a.tolist()
            G = gts.get(t2i[t], [])
            if not G:
                continue
            bi = mu.iou(xywh(M[t]["boxes_2048"]).tolist(), [x["bbox"] for x in G], [0] * len(G))  # (n_det, n_gt)
            agree_tp += a[bi.max(1) >= 0.5].tolist()
            for j, x in enumerate(G):
                i = int(bi[:, j].argmax())
                if bi[i, j] >= 0.5:
                    gl.append(mu.iou([lm[i]], [x["segmentation"]], [0])[0, 0]); gs.append(mu.iou([sm[i]], [x["segmentation"]], [0])[0, 0])
        gl, gs = np.array(gl), np.array(gs)
        rec = {"split": split, "agree_all": {"n": len(agree_all), "mean": round(float(np.mean(agree_all)), 4), "median": round(float(np.median(agree_all)), 4)},
               "agree_tp": {"n": len(agree_tp), "mean": round(float(np.mean(agree_tp)), 4), "median": round(float(np.median(agree_tp)), 4)},
               "vs_gt": {"n_gt_matched": len(gl), "lace_mean": round(float(gl.mean()), 4), "sam_mean": round(float(gs.mean()), 4),
                         "lace_minus_sam": round(float((gl - gs).mean()), 4), "frac_lace_better": round(float((gl > gs).mean()), 4)}}
        json.dump(rec, open(os.path.join(S.OUT, f"mask_iou_{split}.json"), "w"), indent=1); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

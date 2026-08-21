"""Box-source eval + prototype geometry for the collapse ablation.

Runs INSIDE the Modal container. Two things live here:

1. `prototype_stats` -- effective rank / pairwise cosine of the fitted foreground
   prototypes. Identical formulas to
   `boxinst_commonality_tcd_04/ablation/scripts/t05_convergence.py::prototype_stats`;
   duplicated rather than imported because `ablation/` is not in the Modal image. The
   Stage-1 sanity gate checks this reproduces the deployed model's 1.66 / 0.9935.

2. `eval_boxsource` -- score a masker with boxes taken either from the GT polygons
   (perfect boxes) or from the detector (imprecise boxes). This is the axis the whole
   experiment turns on, and no existing entry point offers it: `phase4_lib_tcd.py:77`
   `eval_4p_selfmask` always runs the detector. The GT-box path is ported from
   `mps_tcd_multiseed_4phase/masker_lab/sweep.py:76-98`.

METRIC CHOICE
    `mean_crown_iou` is the primary number for the GT-vs-predicted comparison, not AP.
    Under GT boxes recall is perfect by construction, so AP mostly measures the ordering of
    scores rather than mask quality, and is not commensurable with the predicted-box AP.
    Mean per-crown mask IoU is defined identically in both regimes:
      GT boxes        -> boxes are 1:1 with GT crowns, IoU per crown directly
      predicted boxes -> greedy score-ordered BOX-IoU>=0.5 match, then mask IoU on the match
    Both AP50 and mAP50-95 are still reported, for continuity with everything else.
"""
import json
import os

import numpy as np


def gt_boxes(polys):
    """Tight box per GT polygon, 2048 tile coords. From masker_lab/sweep.py:68."""
    if not polys:
        return np.zeros((0, 4), np.float32)
    out = []
    for t in polys:
        a = np.asarray(t, float).reshape(-1, 2)
        out.append([*a.min(0), *a.max(0)])
    return np.array(out, np.float32).reshape(-1, 4)


def box_iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, 0][:, None], b[:, 0][None]); iy0 = np.maximum(a[:, 1][:, None], b[:, 1][None])
    ix1 = np.minimum(a[:, 2][:, None], b[:, 2][None]); iy1 = np.minimum(a[:, 3][:, None], b[:, 3][None])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def prototype_stats(C):
    """K, effective rank, pairwise cosine. Mirrors ablation/scripts/t05_convergence.py."""
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sv = np.linalg.svd(Cn, compute_uv=False)
    ps = sv / max(sv.sum(), 1e-12)
    G = Cn @ Cn.T
    off = G[~np.eye(len(C), dtype=bool)] if len(C) > 1 else np.array([1.0])
    cbar = Cn.mean(0)
    cbar = cbar / (np.linalg.norm(cbar) + 1e-12)
    return {"K": int(len(C)),
            "effective_rank": round(float(np.exp(-(ps * np.log(ps + 1e-12)).sum())), 4),
            "pairwise_cos_mean": round(float(off.mean()), 4),
            "pairwise_cos_max": round(float(off.max()), 4),
            "frac_pairs_over_0.9": round(float((off > 0.9).mean()), 4),
            "C_bar": [round(float(x), 6) for x in cbar]}


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def eval_boxsource(em_path, feat_dir, gt_path, box_source, out_json,
                   ckpt_path=None, mask_thr=0.25, device="cuda",
                   prior_weight=None, kappa_scale=None, limit=None, exclude=()):
    """Score one masker under one box source.

    box_source='gt'   -> boxes are the GT polygons' tight boxes (perfect boxes)
    box_source='pred' -> boxes from the detector at ckpt_path (imprecise boxes)

    `exclude` drops tiles by id -- used to report a masker-fit-clean val subset, since
    the deployed 120-tile masker fit overlaps the 108 val tiles by 11.
    """
    import torch
    from dapt.decode import decode
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.detector import STRIDE8

    assert box_source in ("gt", "pred")
    masker = E.TCDMasker(em_path)
    gt = json.load(open(gt_path))
    tiles = [t for t in sorted(gt)
             if gt[t]["trees"] and os.path.exists(os.path.join(feat_dir, t + ".npy"))
             and t not in set(exclude)]
    if limit:
        tiles = tiles[:limit]

    model = None
    if box_source == "pred":
        # Detector4Phase subclasses Detector8 and lives in phase4_lib_tcd (line 33), NOT in
        # boxinst_commonality_tcd_04.detector -- that module only defines Detector8.
        from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import (
            Detector4Phase)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                               ).to(device).eval()
        model.load_state_dict(ck["state"])

    print(f"[eval_boxsource] {box_source} boxes, {len(tiles)} tiles, "
          f"em={os.path.basename(em_path)}", flush=True)

    ious = []
    P_masks, P_scores, G_masks, Ign = [], [], [], []
    n_match = 0
    with torch.no_grad(), np.errstate(all="ignore"):
        for k, tid in enumerate(tiles):
            feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
            g = feat.shape[-1]
            polys = gt[tid]["trees"]
            gb = gt_boxes(polys)
            gm = np.array(E.raster(polys))
            can = np.array(E.raster(gt[tid]["canopy"]))
            can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)

            if box_source == "gt":
                bx = gb
                sc = np.ones(len(bx), np.float32)
            else:
                det = model(torch.from_numpy(feat)[None].to(device))
                bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
                bx, sc = bx.numpy(), sc.numpy()

            zn = masker.project(feat)
            pm = E.pred_instance_masks(masker, zn, g, bx, mask_thr=mask_thr,
                                       prior_weight=prior_weight, kappa_scale=kappa_scale)

            if box_source == "gt":
                for i in range(len(pm)):            # boxes are 1:1 with GT crowns
                    u = (pm[i] | gm[i]).sum()
                    ious.append(float((pm[i] & gm[i]).sum() / u) if u else 0.0)
                    n_match += 1
            else:                                    # greedy box-IoU match, then mask IoU
                bmat = box_iou_mat(bx, gb)
                if len(sc) and bmat.shape[1]:
                    order = np.argsort(-sc); taken = np.zeros(bmat.shape[1], bool)
                    for i in order:
                        j = int(np.argmax(bmat[i]))
                        if bmat[i, j] >= 0.5 and not taken[j]:
                            taken[j] = True
                            u = (pm[i] | gm[j]).sum()
                            ious.append(float((pm[i] & gm[j]).sum() / u) if u else 0.0)
                            n_match += 1

            ign = (np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5 for m in pm])
                   if len(pm) else np.zeros(0, bool))
            P_masks.append(pm); P_scores.append(sc); G_masks.append(gm); Ign.append(ign)
            del pm, gm, can, feat
            if (k + 1) % 25 == 0 or k + 1 == len(tiles):
                print(f"    {k + 1}/{len(tiles)}", flush=True)

    per_iou = [E.mask_ap(P_masks, P_scores, G_masks, Ign, t) for t in E.IOU_50_95]
    res = {"em": os.path.basename(em_path), "box_source": box_source,
           "n_tiles": len(tiles), "n_matched_crowns": n_match,
           "n_gt": int(sum(len(g_) for g_ in G_masks)),
           "mean_crown_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
           "median_crown_iou": round(float(np.median(ious)) if ious else 0.0, 4),
           "mask_mAP50": round(float(per_iou[0]), 4),
           "mask_mAP50_95": round(float(np.nanmean(per_iou)), 4),
           "per_iou": [round(float(x), 4) for x in per_iou],
           "mask_thr": mask_thr, "prior_weight": prior_weight, "kappa_scale": kappa_scale,
           "n_excluded": len(set(exclude))}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    print(f"  -> crownIoU {res['mean_crown_iou']}  AP50 {res['mask_mAP50']}  "
          f"AP50-95 {res['mask_mAP50_95']}", flush=True)
    return res

"""SAM 3 box-prompted instance seg on OAM-TCD 439 — ablation vs our commonality masker.

Apples-to-apples: the SAME seed-0 detector boxes (read from the saved β=0.5-fixed
predictions) prompt SAM 3 instead of our EM masker. Everything downstream is IDENTICAL to
`phase4_lib_tcd.eval_4p_selfmask`: same GT rasterisation, same canopy-ignore, same
COCO-101pt mask-AP core, same operating threshold, AP RANKED BY OUR DETECTOR SCORES (so the
only thing that changes between the two runs is the mask shape, not the boxes or the
ranking). The scoring core below is copied VERBATIM from
`boxinst_commonality_tcd_04/evaluate.py` (+ `_instance_pr` from `phase4_lib_tcd.py`) so the
numbers are directly comparable byte-for-byte — see `_parity_check` for the guard.

Crowns can spill past the (tight) box, so SAM masks are scored UNCROPPED (headline). A
box-clipped variant is also reported as a sensitivity check (isolates whether SAM's
out-of-box spill helps or hurts), never as the headline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# ---- scoring core: VERBATIM from boxinst_commonality_tcd_04/evaluate.py ----
RES = 512                                # scoring raster (2048 / 4)
SCALE = 2048 / RES
IOU_50_95 = np.arange(0.5, 0.96, 0.05)


def raster(polys, res=RES, scale=SCALE):
    out = []
    for poly in polys:
        if not poly or len(poly) < 6:
            out.append(np.zeros((res, res), bool)); continue
        img = Image.new("L", (res, res), 0)
        pts = (np.asarray(poly, np.float32).reshape(-1, 2) / scale)
        ImageDraw.Draw(img).polygon([tuple(v) for v in pts], fill=1)
        out.append(np.asarray(img, bool))
    return out


def mask_iou(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = pred.reshape(len(pred), -1).astype(np.float32)
    g = gt.reshape(len(gt), -1).astype(np.float32)
    inter = p @ g.T
    return inter / (p.sum(1)[:, None] + g.sum(1)[None] - inter + 1e-9)


def _greedy_ap(Ious, Scores, Ignore, n_gt, iou_thr):
    """Shared COCO 101-pt AP core (verbatim). Identical matching for our masker and SAM,
    so the mask-AP gap is purely the mask shape."""
    scores_all, tp_all = [], []
    for iou, ps, ign in zip(Ious, Scores, Ignore):
        if len(ps) == 0:
            continue
        order = np.argsort(-ps)
        iou, ps, ign = iou[order], ps[order], ign[order]
        matched = np.zeros(iou.shape[1], bool)
        tp = np.zeros(len(ps), bool); keep = np.ones(len(ps), bool)
        for i in range(len(ps)):
            j = int(np.argmax(iou[i])) if iou.shape[1] else -1
            if j >= 0 and iou[i, j] >= iou_thr and not matched[j]:
                matched[j] = True; tp[i] = True
            elif ign[i]:
                keep[i] = False
        scores_all.append(ps[keep]); tp_all.append(tp[keep])
    if n_gt == 0:
        return float("nan")
    if not scores_all:
        return 0.0
    s = np.concatenate(scores_all); tp = np.concatenate(tp_all)
    o = np.argsort(-s); tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n_gt, tpc / (tpc + fpc + 1e-9)
    return float(sum((prec[rec >= r].max() if np.any(rec >= r) else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


def _instance_pr(Ious, Scores, Ignore, n_gt, iou_thr, op_thr):
    """Instance P/R/F1 at op_thr and at best-F1 (verbatim from phase4_lib_tcd)."""
    scores_all, tp_all = [], []
    for iou, ps, ign in zip(Ious, Scores, Ignore):
        if len(ps) == 0:
            continue
        order = np.argsort(-ps)
        iou, ps, ign = iou[order], ps[order], ign[order]
        matched = np.zeros(iou.shape[1], bool)
        tp = np.zeros(len(ps), bool); keep = np.ones(len(ps), bool)
        for i in range(len(ps)):
            j = int(np.argmax(iou[i])) if iou.shape[1] else -1
            if j >= 0 and iou[i, j] >= iou_thr and not matched[j]:
                matched[j] = True; tp[i] = True
            elif ign[i]:
                keep[i] = False
        scores_all.append(ps[keep]); tp_all.append(tp[keep])
    z = {"P": 0.0, "R": 0.0, "F1": 0.0}
    if not scores_all or n_gt == 0:
        return {"op": z, "best": {**z, "thr": 0.0}, "maxR": 0.0}
    s = np.concatenate(scores_all); tp = np.concatenate(tp_all)
    o = np.argsort(-s); s, tp = s[o], tp[o]
    tpc = np.cumsum(tp); fpc = np.cumsum(~tp)
    prec = tpc / (tpc + fpc + 1e-9); rec = tpc / n_gt
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    bi = int(np.argmax(f1))
    best = {"P": round(float(prec[bi]), 4), "R": round(float(rec[bi]), 4),
            "F1": round(float(f1[bi]), 4), "thr": round(float(s[bi]), 3)}
    m = s >= op_thr
    if m.any():
        oi = int(np.where(m)[0][-1])
        op = {"P": round(float(prec[oi]), 4), "R": round(float(rec[oi]), 4),
              "F1": round(float(f1[oi]), 4)}
    else:
        op = z
    return {"op": op, "best": best, "maxR": round(float(rec.max()), 4)}
# ---- end verbatim core ----


def _parity_check():
    """Assert the copied core matches the live evaluate.py when it is importable
    (skipped inside the SAM Modal image, which does not ship the boxinst package)."""
    try:
        from boxinst_commonality_tcd_04 import evaluate as E
    except Exception:
        return "skipped (evaluate.py not importable here)"
    rng = np.random.default_rng(0)
    pm = rng.random((5, RES, RES)) > 0.5
    gm = rng.random((4, RES, RES)) > 0.5
    assert np.allclose(mask_iou(pm, gm), E.mask_iou(pm, gm)), "mask_iou drift"
    assert E.RES == RES and E.SCALE == SCALE, "RES/SCALE drift"
    return "OK (mask_iou + RES/SCALE match evaluate.py)"


def sam_masks_for_tile(model, processor, pil_img, boxes_2048, chunk=64):
    """Box-prompt SAM 3 with each detector box -> (N, RES, RES) bool masks (UNCROPPED,
    downsampled from the 2048 tile by 4x4 area-majority) + SAM's per-mask score.

    boxes_2048: (N,4) xyxy in 2048px tile coords (the saved seed-0 detections)."""
    import torch
    n = len(boxes_2048)
    if n == 0:
        return np.zeros((0, RES, RES), bool), np.zeros(0, np.float32)
    W, H = pil_img.size
    assert (H, W) == (2048, 2048), f"expected 2048 tile, got {(H, W)}"
    state = processor.set_image(pil_img)
    masks_512, sam_sc = [], []
    with torch.inference_mode():
        for i in range(0, n, chunk):
            bx = np.asarray(boxes_2048[i:i + chunk], np.float32)
            m, sc, _ = model.predict_inst(state, point_coords=None, point_labels=None,
                                          box=bx, multimask_output=False)
            m = np.asarray(m).reshape(len(bx), H, W)          # (b,2048,2048) bool
            # 4x4 area-majority downsample 2048 -> 512 (>=50% of the block is mask)
            m = m.reshape(len(bx), RES, 4, RES, 4).mean((2, 4)) >= 0.5
            masks_512.append(m)
            sam_sc.append(np.asarray(sc, np.float32).reshape(len(bx), -1)[:, 0])
    return np.concatenate(masks_512, 0), np.concatenate(sam_sc, 0)


def _box_rect_512(box_2048):
    x0, y0, x1, y1 = np.asarray(box_2048) / SCALE
    r = np.zeros((RES, RES), bool)
    r[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
    return r


def evaluate_sam(preds, gt, get_rgb, op_thr, chunk=64, log_every=25):
    """Run SAM 3 over all tiles in `preds` and score UNCROPPED + box-CLIPPED, ranked by
    BOTH our detector scores (primary, fair) and SAM's own scores (secondary).

    preds: {tile: {"boxes_2048", "scores", ...}} (saved seed-0 β=0.5-fixed detections).
    gt:    test_gt.json {tile: {"trees", "canopy", ...}}.
    get_rgb(tile) -> PIL 2048 RGB.
    Mirrors eval_4p_selfmask's P_masks/P_scores/G_masks/Ign_mask construction exactly."""
    tiles = [t for t in preds if t in gt]
    Pm_un, Pm_cl, P_det, P_sam, G_masks = [], [], [], [], []
    Ign_un, Ign_cl = [], []
    sem = {"un": [0, 0, 0], "cl": [0, 0, 0]}     # tp, fp, fn
    for k, tid in enumerate(tiles):
        boxes = np.asarray(preds[tid]["boxes_2048"], np.float32).reshape(-1, 4)
        det_sc = np.asarray(preds[tid]["scores"], np.float32)
        img = get_rgb(tid)
        masks, sam_sc = sam_masks_for_tile(model_g, processor_g, img, boxes, chunk)
        clipped = np.array([m & _box_rect_512(b) for m, b in zip(masks, boxes)]) \
            if len(masks) else masks.copy()
        gm = np.array(raster(gt[tid]["trees"]))
        can = np.array(raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
        Pm_un.append(masks); Pm_cl.append(clipped)
        P_det.append(det_sc); P_sam.append(sam_sc); G_masks.append(gm)
        for tag, pm, ign_list in (("un", masks, Ign_un), ("cl", clipped, Ign_cl)):
            ign = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                            for m in pm]) if len(pm) else np.zeros(0, bool)
            ign_list.append(ign)
            op = pm[det_sc >= op_thr] if len(pm) else pm
            pf = (op.any(0) if len(op) else np.zeros((RES, RES), bool)) & ~can
            gf = (gm.any(0) if len(gm) else np.zeros((RES, RES), bool)) & ~can
            sem[tag][0] += int((pf & gf).sum()); sem[tag][1] += int((pf & ~gf).sum())
            sem[tag][2] += int((~pf & gf).sum())
        if (k + 1) % log_every == 0 or k + 1 == len(tiles):
            print(f"  [sam3] {k+1}/{len(tiles)} tiles", flush=True)

    n_gt = int(sum(len(g) for g in G_masks))

    def _mask_table(P_masks, Ign, scores):
        mI = [mask_iou(pm, gm) for pm, gm in zip(P_masks, G_masks)]
        tab = {"mask_mAP50": round(_greedy_ap(mI, scores, Ign, n_gt, 0.5), 4),
               "mask_mAP50_95": round(float(np.nanmean(
                   [_greedy_ap(mI, scores, Ign, n_gt, t) for t in IOU_50_95])), 4),
               "iou0.50": _instance_pr(mI, scores, Ign, n_gt, 0.5, op_thr)}
        return tab

    def _sem(tag):
        tp, fp, fn = sem[tag]
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        return {"F1": round(2 * p * r / (p + r + 1e-9), 4),
                "P": round(p, 4), "R": round(r, 4)}

    res = {
        "model": "sam3_image (box-prompt)", "n_tiles": len(tiles), "n_gt_trees": n_gt,
        "op_thr": op_thr, "ap_ranking": "detector_scores (primary)",
        "uncropped": {"det_score_ranked": _mask_table(Pm_un, Ign_un, P_det),
                      "sam_score_ranked": _mask_table(Pm_un, Ign_un, P_sam),
                      "semantic": _sem("un")},
        "box_clipped": {"det_score_ranked": _mask_table(Pm_cl, Ign_cl, P_det),
                        "sam_score_ranked": _mask_table(Pm_cl, Ign_cl, P_sam),
                        "semantic": _sem("cl")},
    }
    return res


# module-level model handles (set by the Modal function before evaluate_sam)
model_g = None
processor_g = None

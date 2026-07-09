"""Evaluate predicted instance MASKS against held-out TCD polygons.

This is the only place gt_polys.json is read, and it is used for EVALUATION ONLY —
training never sees a polygon. Polygons are rasterized to 512-px instance masks;
predictions come from one forward pass (centre-peak NMS -> per-instance dynamic
mask -> threshold). Metrics:

  mask mAP50 / mAP50-95 : COCO-style AP with MASK IoU (predicted instance mask vs
                          GT polygon mask), greedy score-ordered matching.
  box  mAP50            : same but box IoU (reference; the training signal).
  area F1 / P / R       : pixel-level foreground agreement (union of pred masks vs
                          union of GT masks), aggregated over the partition —
                          "how well does total tree area match".

Usage:
    .venv/bin/python -m boxinst_tcd.eval_masks tcd_A tcd_B tcd_C
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import torch.nn.functional as F

from dapt.backbone import pick_device
from dapt.eval import average_precision, iou_matrix           # box AP reuse
from boxinst.infer_viz import detect_instances
from boxinst.losses import gather_instances
from boxinst.model import BoxInstHead, dynamic_masks, signature_cosine
from boxinst_tcd.cache import COMM, FEAT, key
from boxinst_tcd.prepare import OUT, RES

IOU_50_95 = np.arange(0.5, 0.96, 0.05)


def raster_polys(polys, res=RES):
    """list of flat [x,y,...] -> list of bool (res,res) instance masks."""
    out = []
    for poly in polys:
        if not poly or len(poly) < 6:
            out.append(np.zeros((res, res), bool))
            continue
        img = Image.new("L", (res, res), 0)
        ImageDraw.Draw(img).polygon([tuple(v) for v in
                                     np.array(poly).reshape(-1, 2)], fill=1)
        out.append(np.asarray(img, bool))
    return out


def mask_iou_matrix(pred, gt):
    """pred:(N,H,W) bool, gt:(M,H,W) bool -> (N,M) IoU."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = pred.reshape(len(pred), -1).astype(np.float32)
    g = gt.reshape(len(gt), -1).astype(np.float32)
    inter = p @ g.T
    ap, agi = p.sum(1)[:, None], g.sum(1)[None, :]
    return inter / (ap + agi - inter + 1e-9)


def mask_ap(pred_masks, pred_scores, gt_masks, iou_thr):
    """COCO 101-pt AP with mask IoU, over a list of tiles."""
    scores_all, tp_all, n_gt = [], [], 0
    for pm, ps, gm in zip(pred_masks, pred_scores, gt_masks):
        n_gt += len(gm)
        if len(pm) == 0:
            continue
        order = np.argsort(-ps)
        pm, ps = pm[order], ps[order]
        iou = mask_iou_matrix(pm, gm)
        matched = np.zeros(len(gm), bool)
        tp = np.zeros(len(pm), bool)
        for i in range(len(pm)):
            if iou.shape[1] == 0:
                break
            j = np.argmax(iou[i])
            if iou[i, j] >= iou_thr and not matched[j]:
                matched[j] = True; tp[i] = True
        scores_all.append(ps); tp_all.append(tp)
    if n_gt == 0:
        return float("nan")
    if not scores_all:
        return 0.0
    s = np.concatenate(scores_all); tp = np.concatenate(tp_all)
    o = np.argsort(-s); tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n_gt, tpc / (tpc + fpc)
    return float(sum((prec[rec >= r].max() if np.any(rec >= r) else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


@torch.no_grad()
def gt_prompted(model, feat, lda, gt_boxes, mask_thr, z=None):
    """Prompt the mask head at each GT box's centre cell (as in training) and
    return per-instance mask prob (N,RES,RES). Decouples mask quality from the
    (weak) detector: this is 'given the right box, how good is the mask'."""
    det, ctrl, fmask = model(feat, lda)
    inst = gather_instances([gt_boxes], feat.device)
    if inst is None:
        return np.zeros((0, RES, RES), bool)
    img_idx, bx, cells, centers = inst
    params = ctrl[0, :, cells[:, 0], cells[:, 1]].T
    sig = None
    if getattr(model, "sig_ch", 0) and z is not None:
        sig = signature_cosine(z.expand(1, -1, -1, -1), img_idx, bx)
    logits = dynamic_masks(fmask.expand(len(bx), -1, -1, -1), centers, params, sig=sig)
    probs = torch.sigmoid(F.interpolate(logits[:, None], size=(RES, RES),
                          mode="bilinear", align_corners=False))[:, 0]
    return (probs.cpu().numpy() >= mask_thr)


@torch.no_grad()
def run(ckpt, device):
    ck = torch.load(os.path.join(OUT, "artifacts", ckpt + ".pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cc = 1 if cfg.get("commonality_channel") else 0
    model = BoxInstHead(cfg["in_dim"], commonality_ch=cc,
                        det_tower=cfg.get("det_tower", 2),
                        use_sig=cfg.get("use_sig", False)).to(device).eval()
    model.load_state_dict(ck["state"])
    feat_dir = cfg.get("feat_dir", FEAT)
    comm_dir = cfg.get("comm_dir", COMM)
    split = json.load(open(os.path.join(OUT, "split.json")))
    polys = json.load(open(os.path.join(OUT, "gt_polys.json")))     # EVAL ONLY
    gtbx = json.load(open(os.path.join(OUT, "boxes.json")))
    st, mt = cfg["score_thr"], cfg["mask_thr"]
    test = [p for p, t in split["tiles"].items() if t["partition"] == "test"]

    P_masks, P_scores, P_boxes = [], [], []
    G_masks, G_boxes = [], []
    px_tp = px_fp = px_fn = 0
    gp_ious = []                       # GT-prompted per-instance mask IoU
    gp_tp = gp_fp = gp_fn = 0          # GT-prompted pixel area
    for p in test:
        feat = torch.from_numpy(np.load(os.path.join(feat_dir, key(p) + ".npy"))
                                ).float()[None].to(device)
        lda = None
        z = None
        if cc:
            lda = torch.from_numpy(np.load(os.path.join(comm_dir, key(p) + ".npz"))
                                   ["lda"]).float()[None].to(device)
        if getattr(model, "sig_ch", 0):
            z = torch.from_numpy(np.load(os.path.join(comm_dir, key(p) + ".npz"))
                                 ["z"]).float()[None].to(device)
        boxes, scores, probs = detect_instances(model, feat, st, cfg["nms_iou"],
                                                lda=lda, z=z)
        pm = (probs.numpy() >= mt)
        P_masks.append(pm); P_scores.append(scores.numpy()); P_boxes.append(boxes.numpy())
        gm = np.array(raster_polys(polys[p]))
        gbx = np.array(gtbx[p], np.float32).reshape(-1, 4)
        G_masks.append(gm); G_boxes.append(gbx)
        # pixel area F1 at operating threshold (score already applied in detect)
        pf = pm.any(0) if len(pm) else np.zeros((RES, RES), bool)
        gf = gm.any(0) if len(gm) else np.zeros((RES, RES), bool)
        px_tp += int((pf & gf).sum()); px_fp += int((pf & ~gf).sum())
        px_fn += int((~pf & gf).sum())
        # GT-box-prompted masks: mask quality decoupled from the (weak) detector
        gpm = gt_prompted(model, feat, lda, gbx, mt, z=z)
        if len(gpm) and len(gm):
            iou = mask_iou_matrix(gpm, gm)
            gp_ious.extend(float(iou[i, i]) for i in range(min(len(gpm), len(gm))))
        gpf = gpm.any(0) if len(gpm) else np.zeros((RES, RES), bool)
        gp_tp += int((gpf & gf).sum()); gp_fp += int((gpf & ~gf).sum())
        gp_fn += int((~gpf & gf).sum())

    mask50 = mask_ap(P_masks, P_scores, G_masks, 0.5)
    mask5095 = float(np.nanmean([mask_ap(P_masks, P_scores, G_masks, t)
                                 for t in IOU_50_95]))
    box_preds = list(zip(P_boxes, P_scores))
    box50 = average_precision(box_preds, G_boxes, 0.5)
    prec = px_tp / (px_tp + px_fp + 1e-9)
    rec = px_tp / (px_tp + px_fn + 1e-9)
    areaf1 = 2 * prec * rec / (prec + rec + 1e-9)
    gp_prec = gp_tp / (gp_tp + gp_fp + 1e-9)
    gp_rec = gp_tp / (gp_tp + gp_fn + 1e-9)
    gp_areaf1 = 2 * gp_prec * gp_rec / (gp_prec + gp_rec + 1e-9)
    ntree = sum(len(g) for g in G_masks)
    return {"mask_mAP50": round(mask50, 4), "mask_mAP50_95": round(mask5095, 4),
            "box_mAP50": round(box50, 4), "area_F1": round(areaf1, 4),
            "area_P": round(prec, 4), "area_R": round(rec, 4),
            "gtprompt_mask_mIoU": round(float(np.mean(gp_ious)) if gp_ious else 0.0, 4),
            "gtprompt_mask_IoU50": round(float(np.mean(np.array(gp_ious) >= 0.5))
                                         if gp_ious else 0.0, 4),
            "gtprompt_area_F1": round(gp_areaf1, 4), "n_gt_trees": ntree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    rows = {}
    print("FULL PIPELINE (detect + mask)          |  GT-BOX-PROMPTED (mask quality)")
    print(f"{'cfg':7s} {'maskmAP50':>9} {'maskAP5095':>10} {'boxmAP50':>8} "
          f"{'areaF1':>7} | {'mIoU':>6} {'IoU>.5':>6} {'areaF1':>7}")
    for c in args.ckpts:
        r = run(c, device); rows[c] = r
        print(f"{c:7s} {r['mask_mAP50']:9.3f} {r['mask_mAP50_95']:10.3f} "
              f"{r['box_mAP50']:8.3f} {r['area_F1']:7.3f} | "
              f"{r['gtprompt_mask_mIoU']:6.3f} {r['gtprompt_mask_IoU50']:6.3f} "
              f"{r['gtprompt_area_F1']:7.3f}")
    json.dump(rows, open(os.path.join(OUT, "mask_results.json"), "w"), indent=2)
    print(f"n_gt_trees(test)={rows[args.ckpts[0]]['n_gt_trees']} "
          f"-> boxinst_tcd/mask_results.json")


if __name__ == "__main__":
    main()

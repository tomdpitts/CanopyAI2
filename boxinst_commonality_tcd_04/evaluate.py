"""Evaluate the full pipeline on the OAM-TCD 439 TEST benchmark.

Pipeline per 2048 test tile: stitched 128x128 web features -> 8px detector ->
decoded ITC boxes (2048 coords) -> training-free EM mask per box -> instance masks.
Scored vs held-out polygons (test_gt.json), at a downsampled resolution for memory.

Metrics:
  mask mAP50 / mAP50-95 : COCO-style instance AP with MASK IoU (the headline; the
                          number to beat is 43.2 mAP50).
  box  mAP50            : reference (box IoU).
  semantic F1 / P / R   : pixel foreground agreement, union(pred) vs union(GT tree).
Canopy (category 1) is IGNORE everywhere: a predicted crown lying in canopy and
matching no labelled tree is dropped (not a false positive); canopy pixels are
excluded from the semantic-F1 denominator. No polygon is read outside this file.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.evaluate --det det_d8 [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from dapt.backbone import pick_device
from dapt.decode import decode
from boxinst_commonality_tcd_04.cache_test import cache_dir
from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8
from boxinst_commonality_tcd_04.em import MODEL_PATH, TCDMasker
from boxinst_commonality_tcd_04.prepare_test import OUT

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


def pred_instance_masks(masker, zn, g, boxes, res=RES, scale=SCALE, mask_thr=0.5):
    """EM posterior per box -> (N,res,res) bool instance masks at scoring res."""
    out = []
    for b in boxes:
        idx, r = masker.box_mask(zn, g, b)
        grid = np.zeros(g * g, np.float32); grid[idx] = r
        prob = np.array(Image.fromarray(grid.reshape(g, g)).resize(
            (res, res), Image.BILINEAR))
        m = prob >= mask_thr
        # confine to the box (posterior can bleed one pad-cell past the border)
        x0, y0, x1, y1 = (np.asarray(b) / scale)
        box_m = np.zeros((res, res), bool)
        box_m[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
        out.append(m & box_m)
    return np.array(out) if out else np.zeros((0, res, res), bool)


def _greedy_ap(Ious, Scores, Ignore, n_gt, iou_thr):
    """Shared COCO 101-pt AP core. Per tile: iou (Npred,Ngt), scores (Npred,),
    ignore (Npred,) bool = 'unmatched pred sits in canopy -> drop, not FP'.
    Identical matching for box and mask AP so their gap is purely the IoU type."""
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
                keep[i] = False                       # unmatched, in canopy -> ignore
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


def mask_ap(P_masks, P_scores, G_masks, Ignore, iou_thr):
    Ious = [mask_iou(pm, gm) for pm, gm in zip(P_masks, G_masks)]
    return _greedy_ap(Ious, P_scores, Ignore, sum(len(g) for g in G_masks), iou_thr)


def box_ap(P_boxes, P_scores, G_boxes, Ignore, iou_thr):
    from dapt.eval import iou_matrix
    Ious = [iou_matrix(pb, gb) for pb, gb in zip(P_boxes, G_boxes)]
    return _greedy_ap(Ious, P_scores, Ignore, sum(len(g) for g in G_boxes), iou_thr)


def add_downscale(model, tid, args, bx, sc, device):
    """Big-tree arm: run the detector on the 0.5x feature grid (cache_test_down),
    scale its boxes x2 to 2048 coords, and cross-scale NMS-merge with the native
    detections. Adds big-tree candidates (rare -> low FP risk); duplicate mid
    crowns are deduped by NMS. Masks stay native, so only detection changes."""
    from torchvision.ops import nms
    from boxinst_commonality_tcd_04.cache_test_down import cache_dir as dcache, DOWN
    fp = os.path.join(dcache(args.arm), tid + ".npy")
    if not os.path.exists(fp):
        return bx, sc
    dfeat = np.load(fp).astype(np.float32)
    ddet = model(torch.from_numpy(dfeat)[None].to(device))
    dbx, dsc = decode(ddet.cpu(), score_thr=args.eval_score_thr, stride=STRIDE8,
                      topk=args.topk)
    dbx = dbx.numpy() * (2048.0 / DOWN)          # 0.5x grid -> 2048 coords
    dsc = dsc.numpy()
    ab = np.concatenate([bx, dbx]) if len(dbx) else bx
    asc = np.concatenate([sc, dsc]) if len(dsc) else sc
    if len(ab) == 0:
        return bx, sc
    keep = nms(torch.from_numpy(ab).float(), torch.from_numpy(asc).float(),
               args.ms_nms_iou).numpy()
    return ab[keep], asc[keep]


@torch.no_grad()
def run(args):
    device = pick_device(args.device)
    ck = torch.load(os.path.join(OUT, "artifacts", args.det + ".pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if args.op_thr is None:
        args.op_thr = cfg["score_thr"]          # val-picked operating threshold
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    masker = TCDMasker(args.model)
    gt = json.load(open(os.path.join(OUT, "test_gt.json")))
    cdir = cache_dir(args.arm)
    tiles = [t for t in sorted(gt)
             if os.path.exists(os.path.join(cdir, t + ".npy"))]
    if args.limit:
        tiles = tiles[:args.limit]

    P_masks, P_scores, P_boxes, G_masks, G_boxes = [], [], [], [], []
    Ign_mask, Ign_box = [], []              # per-pred 'in canopy' (COCO ignore)
    sem_tp = sem_fp = sem_fn = 0
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(cdir, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        det = model(torch.from_numpy(feat)[None].to(device))
        # AP integrates the full PR curve by score rank, so decode with a LOW
        # threshold to keep the recall tail — using the operating score_thr here
        # truncates low-confidence true positives and under-counts mAP.
        bx, sc = decode(det.cpu(), score_thr=args.eval_score_thr, stride=STRIDE8,
                        topk=args.topk)
        bx, sc = bx.numpy(), sc.numpy()
        if args.multiscale:
            bx, sc = add_downscale(model, tid, args, bx, sc, device)
        zn = masker.project(feat)
        pm = pred_instance_masks(masker, zn, g, bx)
        gm = np.array(raster(gt[tid]["trees"]))
        can = np.array(raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
        pbox = bx / SCALE
        P_masks.append(pm); P_scores.append(sc); P_boxes.append(pbox)
        G_masks.append(gm)
        G_boxes.append(np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                                  *(np.asarray(t).reshape(-1, 2).max(0))]
                                 for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                       / SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        # canopy-ignore flags (same rule, one per pred): >50% of the pred's
        # mask / box area sits in canopy. Identical treatment for both APs.
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        Ign_mask.append(ign_m); Ign_box.append(ign_b)
        # semantic F1 (canopy excluded from both masks) at the OPERATING
        # threshold (val-picked, cfg score_thr) — NOT the low AP-decode
        # threshold, whose low-confidence tail is for the PR curve only.
        # mAP is unaffected (it uses the full decode above).
        op = pm[sc >= args.op_thr] if len(pm) else pm
        pf = (op.any(0) if len(op) else np.zeros((RES, RES), bool)) & ~can
        gf = (gm.any(0) if len(gm) else np.zeros((RES, RES), bool)) & ~can
        sem_tp += int((pf & gf).sum()); sem_fp += int((pf & ~gf).sum())
        sem_fn += int((~pf & gf).sum())
        if (k + 1) % 20 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles", flush=True)

    m50 = mask_ap(P_masks, P_scores, G_masks, Ign_mask, 0.5)
    m5095 = float(np.nanmean([mask_ap(P_masks, P_scores, G_masks, Ign_mask, t)
                              for t in IOU_50_95]))
    # box AP with the SAME canopy-ignore + matching -> box50 - m50 isolates the
    # box->mask conversion cost (the detector ceiling vs the EM mask cost)
    box50 = box_ap(P_boxes, P_scores, G_boxes, Ign_box, 0.5)
    box5095 = float(np.nanmean([box_ap(P_boxes, P_scores, G_boxes, Ign_box, t)
                                for t in IOU_50_95]))
    prec = sem_tp / (sem_tp + sem_fp + 1e-9); rec = sem_tp / (sem_tp + sem_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    res = {"n_tiles": len(tiles), "n_gt_trees": int(sum(len(g) for g in G_masks)),
           "mask_mAP50": round(m50, 4), "mask_mAP50_95": round(m5095, 4),
           "box_mAP50": round(box50, 4), "box_mAP50_95": round(box5095, 4),
           "box_minus_mask_mAP50": round(box50 - m50, 4),
           "semantic_F1": round(f1, 4),
           "semantic_P": round(prec, 4), "semantic_R": round(rec, 4),
           "det": args.det, "em": os.path.basename(args.model)}
    print(json.dumps(res, indent=2))
    tag = args.det.replace("det_", "")
    json.dump(res, open(os.path.join(OUT, "artifacts", f"eval_{tag}.json"), "w"),
              indent=2)
    print(f"\nMASK mAP50 = {m50:.3f}   BOX mAP50 = {box50:.3f}  (same canopy-ignore; "
          f"gap = box->mask cost = {box50 - m50:.3f})", flush=True)
    print(f"target 0.432   semantic F1 = {f1:.3f}   [{len(tiles)} tiles]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="detector tag, e.g. det_d8")
    ap.add_argument("--model", default=MODEL_PATH, help="EM npz")
    ap.add_argument("--arm", default="web")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument("--eval_score_thr", type=float, default=0.05,
                    help="low threshold for AP recall tail (NOT the operating thr)")
    ap.add_argument("--op_thr", type=float, default=None,
                    help="operating threshold for semantic F1 (default: the "
                         "checkpoint's val-picked score_thr)")
    ap.add_argument("--multiscale", action="store_true",
                    help="add the 0.5x downscale detection arm for big trees "
                         "(needs cache_test_down; inference-only, masks unchanged)")
    ap.add_argument("--ms_nms_iou", type=float, default=0.5,
                    help="cross-scale NMS IoU when merging native + downscale")
    ap.add_argument("--device", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

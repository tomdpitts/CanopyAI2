"""4-phase real-8px detector (L24) train + single-scale box+mask eval for TCD.

The ONLY differences vs the native/interp path:
  - detector features are (1024,256,256) REAL-8px L24 (phase4_features_tcd.interleave),
    not (1024,128,128) interpolated internally;
  - Detector4Phase = Detector8 minus the internal bilinear upsample (features already
    arrive at 8px / 256-grid).
Everything else — TargetConfig(grid=256, stride=8), encode/decode, det_loss, Adam
lr1e-3 wd1e-4, cosine, bs3, eval_every5, early-stop(min12,p2,delta5e-3) — is reused
VERBATIM from train_detector_tiles via monkeypatch, so this is a clean real-vs-interp
A/B at layer 24. The box->mask EM masker stays the FIXED 4096-dim vault model, fed the
FULL 4096-dim native test features (only the detector sees the 1024-dim real-8px grid).

Eval is single-scale (no downscale arm) -> compare to native single-scale 0.499 mask /
0.555 box, and to the interp-L24 probe (0.502 / 0.540).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.decode import decode
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8


class Detector4Phase(Detector8):
    """Detector8 on a pre-assembled REAL 8px grid: identical modules & params, NO
    internal interpolate. Input (B,1024,256,256) -> heads predict (5,256,256) @ stride 8."""

    def forward(self, feat):
        x = self.tower(self.stem(feat))
        x = self.up(x)                              # no interpolate (already 8px)
        return torch.cat([self.hm(x), self.reg(x)], dim=1)


def train_4p(feat_dir, out_dir, gt_path=None, tag="phase4_L24_s0", seed=0, epochs=40,
             device="cuda"):
    """Reuse train_detector_tiles.train VERBATIM but with Detector4Phase + the 4-phase
    L24 cache. in_dim auto-detects 1024 from the features; canvas 2048 -> target grid 256."""
    import boxinst_commonality_tcd_04.train_detector_tiles as T
    ckpt = os.path.join(out_dir, f"det_{tag}.pt")
    if os.path.exists(ckpt):
        print(f"[{tag}] TRAIN: ckpt exists, skip -> {ckpt}", flush=True)
        return torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    T.Detector8 = Detector4Phase                    # <-- the only model swap
    T.cache_dir = lambda arm: feat_dir              # TileData reads this cache
    T.ART = out_dir
    args = argparse.Namespace(
        tag=tag, epochs=epochs, seed=seed, lr=1e-3, wd=1e-4, bs=3, width=256, tower=3,
        eval_every=5, arm="web", device=device, canvas=2048,
        early_stop=True, min_epochs=12, es_patience=2, es_min_delta=0.005)
    print(f"[{tag}] TRAIN Detector4Phase real-8px L24 (det_t8 recipe+ES) -> {ckpt}",
          flush=True)
    T.train(args)
    return torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]


@torch.no_grad()
def eval_4p_selfmask(ckpt_path, feat4p_test_dir, test_gt_path, em_path, out_json,
                     mask_thr=0.25, device="cuda"):
    """Single-scale box+mask eval where the EM masker reads the SAME 4-phase L24
    features as the detector (self-mask) — masker refit on 4-phase cells, so both
    detection and mask conversion are in-distribution. No native-4096 needed.

    mask_thr = the P(fg) cut in pred_instance_masks. Default 0.5; lowering to ~0.25
    grows masks to recover SMALL crowns that under-cover under imprecise PREDICTED
    boxes (validated on the real det_t8 pipeline: +0.043 mAP50 / +0.035 mAP50-95,
    peak at 0.25). Emits the two tables via _full_metrics."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    op_thr = cfg["score_thr"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                           ).to(device).eval()
    model.load_state_dict(ck["state"])
    masker = E.TCDMasker(em_path)
    gt = json.load(open(test_gt_path))
    tiles = [t for t in sorted(gt)
             if os.path.exists(os.path.join(feat4p_test_dir, t + ".npy"))]
    print(f"[eval_selfmask] {len(tiles)} tiles (op_thr={op_thr}, masker s={masker.s})",
          flush=True)

    P_masks, P_scores, P_boxes, G_masks, G_boxes = [], [], [], [], []
    Ign_mask, Ign_box = [], []
    sem_tp = sem_fp = sem_fn = 0
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(feat4p_test_dir, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]                                    # 256 (masker AND detector)
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
        zn = masker.project(feat)                            # self-mask: same 4-phase feats
        pm = E.pred_instance_masks(masker, zn, g, bx, mask_thr=mask_thr)
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        pbox = bx / E.SCALE
        P_masks.append(pm); P_scores.append(sc); P_boxes.append(pbox)
        G_masks.append(gm)
        G_boxes.append(np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                                  *(np.asarray(t).reshape(-1, 2).max(0))]
                                 for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                       / E.SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        Ign_mask.append(ign_m); Ign_box.append(ign_b)
        op = pm[sc >= op_thr] if len(pm) else pm
        pf = (op.any(0) if len(op) else np.zeros((E.RES, E.RES), bool)) & ~can
        gf = (gm.any(0) if len(gm) else np.zeros((E.RES, E.RES), bool)) & ~can
        sem_tp += int((pf & gf).sum()); sem_fp += int((pf & ~gf).sum())
        sem_fn += int((~pf & gf).sum())
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles", flush=True)

    det, seg = _full_metrics(P_boxes, P_scores, G_boxes, Ign_box, P_masks, G_masks,
                             Ign_mask, op_thr, box_ious=(0.4, 0.5), mask_ious=(0.5,))
    prec = sem_tp / (sem_tp + sem_fp + 1e-9); rec = sem_tp / (sem_tp + sem_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    res = {"variant": "4phase_L24_selfmask", "scale": "single", "op_thr": op_thr,
           "mask_thr": mask_thr, "em": os.path.basename(em_path),
           "n_tiles": len(tiles), "n_gt_trees": int(sum(len(x) for x in G_masks)),
           "mask_mAP50": seg["mask_mAP50"], "mask_mAP50_95": seg["mask_mAP50_95"],
           "box_mAP50": det["box_mAP50"], "box_mAP50_95": det["box_mAP50_95"],
           "box_minus_mask_mAP50": round(det["box_mAP50"] - seg["mask_mAP50"], 4),
           "detection": det, "instance_seg": seg,
           "semantic_F1": round(f1, 4), "semantic_P": round(prec, 4),
           "semantic_R": round(rec, 4), "best_epoch": cfg.get("best_epoch")}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"[eval_selfmask] mask_thr={mask_thr} INSTANCE-SEG mask mAP50={seg['mask_mAP50']} "
          f"mAP50-95={seg['mask_mAP50_95']} (β=0@0.5 was 0.5794/0.1948)", flush=True)
    return res


@torch.no_grad()
def eval_4p(ckpt_path, feat4p_test_dir, native_test_dir, test_gt_path, em_path,
            out_json, device="cuda"):
    """Single-scale box+mask eval. Detector reads the 4-phase L24 (1024,256,256) grid;
    the FIXED EM masker reads the full 4096-dim native (4096,128,128) grid."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    op_thr = cfg["score_thr"]
    assert cfg["in_dim"] == 1024, f"expected 1024-dim detector, got {cfg['in_dim']}"
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                           ).to(device).eval()
    model.load_state_dict(ck["state"])
    masker = E.TCDMasker(em_path)
    gt = json.load(open(test_gt_path))
    tiles = [t for t in sorted(gt)
             if os.path.exists(os.path.join(feat4p_test_dir, t + ".npy"))
             and os.path.exists(os.path.join(native_test_dir, t + ".npy"))]
    print(f"[eval_4p] {len(tiles)} tiles (op_thr={op_thr})", flush=True)

    P_masks, P_scores, P_boxes, G_masks, G_boxes = [], [], [], [], []
    Ign_mask, Ign_box = [], []
    sem_tp = sem_fp = sem_fn = 0
    for k, tid in enumerate(tiles):
        det_feat = np.load(os.path.join(feat4p_test_dir, tid + ".npy")).astype(np.float32)
        full = np.load(os.path.join(native_test_dir, tid + ".npy")).astype(np.float32)
        g = full.shape[-1]                                    # 128 (masker grid)
        det = model(torch.from_numpy(det_feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()                       # single-scale, no downscale
        zn = masker.project(full)                             # full-4096 for EM masks
        pm = E.pred_instance_masks(masker, zn, g, bx)
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        pbox = bx / E.SCALE
        P_masks.append(pm); P_scores.append(sc); P_boxes.append(pbox)
        G_masks.append(gm)
        G_boxes.append(np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                                  *(np.asarray(t).reshape(-1, 2).max(0))]
                                 for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                       / E.SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        Ign_mask.append(ign_m); Ign_box.append(ign_b)
        op = pm[sc >= op_thr] if len(pm) else pm
        pf = (op.any(0) if len(op) else np.zeros((E.RES, E.RES), bool)) & ~can
        gf = (gm.any(0) if len(gm) else np.zeros((E.RES, E.RES), bool)) & ~can
        sem_tp += int((pf & gf).sum()); sem_fp += int((pf & ~gf).sum())
        sem_fn += int((~pf & gf).sum())
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles", flush=True)

    m50 = E.mask_ap(P_masks, P_scores, G_masks, Ign_mask, 0.5)
    m5095 = float(np.nanmean([E.mask_ap(P_masks, P_scores, G_masks, Ign_mask, t)
                              for t in E.IOU_50_95]))
    box50 = E.box_ap(P_boxes, P_scores, G_boxes, Ign_box, 0.5)
    box5095 = float(np.nanmean([E.box_ap(P_boxes, P_scores, G_boxes, Ign_box, t)
                                for t in E.IOU_50_95]))
    prec = sem_tp / (sem_tp + sem_fp + 1e-9); rec = sem_tp / (sem_tp + sem_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    res = {"variant": "4phase_L24_real8px", "scale": "single", "op_thr": op_thr,
           "n_tiles": len(tiles), "n_gt_trees": int(sum(len(x) for x in G_masks)),
           "mask_mAP50": round(m50, 4), "mask_mAP50_95": round(m5095, 4),
           "box_mAP50": round(box50, 4), "box_mAP50_95": round(box5095, 4),
           "box_minus_mask_mAP50": round(box50 - m50, 4),
           "semantic_F1": round(f1, 4), "semantic_P": round(prec, 4),
           "semantic_R": round(rec, 4), "best_epoch": cfg.get("best_epoch"),
           "em": os.path.basename(em_path)}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"[eval_4p] mask mAP50={m50:.4f} box mAP50={box50:.4f} "
          f"(interp-L24 ref 0.502/0.540; native-4096 0.499/0.555)", flush=True)
    return res


def _instance_pr(Ious, Scores, Ignore, n_gt, iou_thr, op_thr):
    """Instance P/R/F1 from greedy matching at iou_thr (mirrors evaluate._greedy_ap's
    matching + canopy-ignore exactly), reported BOTH at the val-picked op_thr and at the
    best-F1 operating point (the DeepForest/NEON convention). Also returns maxR."""
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


def _full_metrics(P_boxes, P_scores, G_boxes, Ign_box, P_masks, G_masks, Ign_mask,
                  op_thr, box_ious=(0.4, 0.5), mask_ious=(0.5,)):
    """Detection (box) + instance-seg (mask) tables from precomputed preds. Box metrics
    are detector-only (masker-independent). AP reuses evaluate._greedy_ap on the same
    IoU matrices, so it matches eval_4p exactly."""
    from dapt.eval import iou_matrix
    bI = [iou_matrix(pb, gb) for pb, gb in zip(P_boxes, G_boxes)]
    mI = [E.mask_iou(pm, gm) for pm, gm in zip(P_masks, G_masks)]
    n_box = int(sum(len(g) for g in G_boxes)); n_mask = int(sum(len(g) for g in G_masks))
    det = {"box_mAP50": round(E._greedy_ap(bI, P_scores, Ign_box, n_box, 0.5), 4),
           "box_mAP40": round(E._greedy_ap(bI, P_scores, Ign_box, n_box, 0.4), 4),
           "box_mAP50_95": round(float(np.nanmean(
               [E._greedy_ap(bI, P_scores, Ign_box, n_box, t) for t in E.IOU_50_95])), 4)}
    for t in box_ious:
        det[f"iou{t:.2f}"] = _instance_pr(bI, P_scores, Ign_box, n_box, t, op_thr)
    seg = {"mask_mAP50": round(E._greedy_ap(mI, P_scores, Ign_mask, n_mask, 0.5), 4),
           "mask_mAP50_95": round(float(np.nanmean(
               [E._greedy_ap(mI, P_scores, Ign_mask, n_mask, t) for t in E.IOU_50_95])), 4)}
    for t in mask_ious:
        seg[f"iou{t:.2f}"] = _instance_pr(mI, P_scores, Ign_mask, n_mask, t, op_thr)
    return det, seg


class BlendMasker:
    """Resolution-gated ensemble of two self-mask TCDMaskers (both fit on 4-phase L24 8px
    cells): pfg = (1-γ)·pfg_fill + γ·pfg_carve, with γ = sigmoid((crown_size_px − s0)/τ).
    Small crowns → fill masker (β=0, no collapse); large crowns → carve masker (β=0.5).
    Blending the two *probability maps* per crown beats either alone (keeps β=0's interior
    fill + adds β=0.5's corner carve) and, unlike appearance-scaling, actually realises the
    sigmoid-γ gain. Drop-in for evaluate.pred_instance_masks (project + box_mask). Both
    maskers are seed-independent, so this composes with any per-seed detector for a band."""

    def __init__(self, em_fill, em_carve, s0=65.0, tau=12.0):
        self.fill = E.TCDMasker(em_fill)      # β=0
        self.carve = E.TCDMasker(em_carve)    # β=0.5
        self.s0, self.tau = float(s0), float(tau)
        self.s = self.fill.s                  # shared grid stride (8px)
        assert self.fill.s == self.carve.s, "maskers must share stride for idx alignment"

    def project(self, feat):
        self._zf = self.fill.project(feat)
        self._zc = self.carve.project(feat)
        return self._zf                        # geometry carrier; box_mask uses stored zf/zc

    def box_mask(self, zn, g, box, contrast=True):
        i_f, p_f = self.fill.box_mask(self._zf, g, box)
        i_c, p_c = self.carve.box_mask(self._zc, g, box)
        assert np.array_equal(i_f, i_c)        # same stride+grid -> identical in-box cells
        size = float(np.sqrt(max(box[2] - box[0], 1) * max(box[3] - box[1], 1)))
        gamma = 1.0 / (1.0 + np.exp(-(size - self.s0) / self.tau))
        return i_f, (1.0 - gamma) * p_f + gamma * p_c


@torch.no_grad()
def eval_4p_blend(ckpt_path, feat4p_test_dir, test_gt_path, em_fill, em_carve,
                  out_json, s0=65.0, tau=12.0, device="cuda"):
    """Single-scale box+mask eval with the resolution-gated BlendMasker (self-mask, both
    maskers read the same 4-phase L24 features as the detector). Same detector/box path as
    eval_4p_selfmask; only the box→mask masker changes. Compare to β=0 self-mask (0.579)."""
    device = pick_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    op_thr = cfg["score_thr"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                           ).to(device).eval()
    model.load_state_dict(ck["state"])
    masker = BlendMasker(em_fill, em_carve, s0=s0, tau=tau)
    gt = json.load(open(test_gt_path))
    tiles = [t for t in sorted(gt)
             if os.path.exists(os.path.join(feat4p_test_dir, t + ".npy"))]
    print(f"[eval_blend] {len(tiles)} tiles (op_thr={op_thr}, s0={s0}px tau={tau}, "
          f"s={masker.s})", flush=True)

    P_masks, P_scores, P_boxes, G_masks, G_boxes = [], [], [], [], []
    Ign_mask, Ign_box = [], []
    sem_tp = sem_fp = sem_fn = 0
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(feat4p_test_dir, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
        zn = masker.project(feat)
        pm = E.pred_instance_masks(masker, zn, g, bx)
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        pbox = bx / E.SCALE
        P_masks.append(pm); P_scores.append(sc); P_boxes.append(pbox)
        G_masks.append(gm)
        G_boxes.append(np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                                  *(np.asarray(t).reshape(-1, 2).max(0))]
                                 for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                       / E.SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        Ign_mask.append(ign_m); Ign_box.append(ign_b)
        op = pm[sc >= op_thr] if len(pm) else pm
        pf = (op.any(0) if len(op) else np.zeros((E.RES, E.RES), bool)) & ~can
        gf = (gm.any(0) if len(gm) else np.zeros((E.RES, E.RES), bool)) & ~can
        sem_tp += int((pf & gf).sum()); sem_fp += int((pf & ~gf).sum())
        sem_fn += int((~pf & gf).sum())
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles", flush=True)

    det, seg = _full_metrics(P_boxes, P_scores, G_boxes, Ign_box, P_masks, G_masks,
                             Ign_mask, op_thr, box_ious=(0.4, 0.5), mask_ious=(0.5,))
    prec = sem_tp / (sem_tp + sem_fp + 1e-9); rec = sem_tp / (sem_tp + sem_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    res = {"variant": "4phase_L24_gamma_blend", "scale": "single", "op_thr": op_thr,
           "s0_px": s0, "tau": tau, "em_fill": os.path.basename(em_fill),
           "em_carve": os.path.basename(em_carve),
           "n_tiles": len(tiles), "n_gt_trees": int(sum(len(x) for x in G_masks)),
           # flat headline keys (back-compat; box are detector-only, masker-invariant)
           "mask_mAP50": seg["mask_mAP50"], "mask_mAP50_95": seg["mask_mAP50_95"],
           "box_mAP50": det["box_mAP50"], "box_mAP50_95": det["box_mAP50_95"],
           "box_minus_mask_mAP50": round(det["box_mAP50"] - seg["mask_mAP50"], 4),
           # the two tables: Detection (box, IoU 0.4 + 0.5, op + best-F1) and
           # Instance-seg (mask, IoU 0.5). semantic = pixel-level (kept for continuity).
           "detection": det, "instance_seg": seg,
           "semantic_F1": round(f1, 4), "semantic_P": round(prec, 4),
           "semantic_R": round(rec, 4), "best_epoch": cfg.get("best_epoch")}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    d40, d50 = det["iou0.40"], det["iou0.50"]
    print(f"[eval_blend] DETECTION box mAP50={det['box_mAP50']} mAP40={det['box_mAP40']} | "
          f"bestF1@0.5={d50['best']['F1']} (P{d50['best']['P']}/R{d50['best']['R']}) "
          f"bestF1@0.4={d40['best']['F1']} (P{d40['best']['P']}/R{d40['best']['R']})",
          flush=True)
    print(f"[eval_blend] INSTANCE-SEG mask mAP50={seg['mask_mAP50']} "
          f"mAP50-95={seg['mask_mAP50_95']} (β=0 self-mask 0.579/0.1948)", flush=True)
    return res

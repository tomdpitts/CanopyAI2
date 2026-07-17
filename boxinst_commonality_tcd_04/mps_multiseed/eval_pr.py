"""Instance-level precision/recall @ IoU 0.5 for masks AND boxes, at each seed's
val-picked operating threshold — the P/R the AP-only eval jsons don't contain.

Reuses evaluate.py's EXACT prediction pipeline (detector -> multiscale decode ->
EM masks -> canopy-ignore) so the boxes/masks scored here are identical to the ones
behind eval_s*.json; only the scoring differs (single-threshold TP/FP/FN instead of
101-pt AP). Fully isolated: loads det_t8_s{seed}.pt from _out/artifacts, the vault EM,
and _out/test_gt.json (symlink); writes only pr_s{seed}.json under mps_multiseed/.

Operating point = the checkpoint's val-picked score_thr (cfg score_thr), the same
threshold semantic F1 uses. Canopy-ignore matches evaluate.py: an unmatched pred
sitting >50% in canopy is dropped (not a false positive), identically for box & mask.

Usage:  .venv/bin/python -m boxinst_commonality_tcd_04.mps_multiseed.eval_pr --seed 0
"""
import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.decode import decode
from dapt.eval import iou_matrix
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.cache_test import cache_dir
from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8

HERE = os.path.abspath(os.path.dirname(__file__))
OUTDIR = os.path.join(HERE, "_out")
ARTDIR = os.path.join(OUTDIR, "artifacts")
VAULT_EM = os.path.abspath(os.path.join(HERE, "..", "vault", "em_model.npz"))


def pr_counts(iou, ign, iou_thr):
    """Greedy match preds (rows, already score-desc-sorted) to GT (cols) at iou_thr.
    Returns (tp, fp, matched_gt_mask). Unmatched preds flagged ign are dropped."""
    ng = iou.shape[1]
    matched = np.zeros(ng, bool)
    tp = fp = 0
    for i in range(iou.shape[0]):
        j = int(np.argmax(iou[i])) if ng else -1
        if j >= 0 and iou[i, j] >= iou_thr and not matched[j]:
            matched[j] = True
            tp += 1
        elif ign[i]:
            continue                     # unmatched, in canopy -> ignore (not FP)
        else:
            fp += 1
    return tp, fp, matched


@torch.no_grad()
def run(seed, iou_thr=0.5):
    device = pick_device("mps")
    ck = torch.load(os.path.join(ARTDIR, f"det_t8_s{seed}.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    op_thr = cfg["score_thr"]                       # val-picked operating threshold
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    masker = E.TCDMasker(VAULT_EM)
    gt = json.load(open(os.path.join(OUTDIR, "test_gt.json")))
    cdir = cache_dir("web")
    tiles = [t for t in sorted(gt)
             if os.path.exists(os.path.join(cdir, t + ".npy"))]

    # multiscale eval args (mirror the headline run) for add_downscale()
    dargs = argparse.Namespace(arm="web", eval_score_thr=0.05, topk=600,
                               ms_nms_iou=0.5)

    m_tp = m_fp = m_fn = 0
    b_tp = b_fp = b_fn = 0
    n_gt = 0
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(cdir, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=dargs.eval_score_thr, stride=STRIDE8,
                        topk=dargs.topk)
        bx, sc = bx.numpy(), sc.numpy()
        bx, sc = E.add_downscale(model, tid, dargs, bx, sc, device)    # multiscale
        # operating point: keep only preds at/above the val-picked threshold
        keep = sc >= op_thr
        bx, sc = bx[keep], sc[keep]
        order = np.argsort(-sc)                        # greedy needs score-desc
        bx, sc = bx[order], sc[order]

        zn = masker.project(feat)
        pm = E.pred_instance_masks(masker, zn, g, bx)  # (N,RES,RES) bool
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        pbox = bx / E.SCALE
        gbox = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                           *(np.asarray(t).reshape(-1, 2).max(0))]
                          for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                / E.SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ng = len(gm)
        n_gt += ng

        # canopy-ignore flags, identical rule to evaluate.py (mask & box)
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5

        m_iou = E.mask_iou(pm, gm) if len(pm) and ng else np.zeros((len(pm), ng))
        b_iou = iou_matrix(pbox, gbox) if len(pbox) and ng else np.zeros((len(pbox), ng))
        tp, fp, mm = pr_counts(m_iou, ign_m, iou_thr)
        m_tp += tp; m_fp += fp; m_fn += ng - int(mm.sum())
        tp, fp, bm = pr_counts(b_iou, ign_b, iou_thr)
        b_tp += tp; b_fp += fp; b_fn += ng - int(bm.sum())
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            print(f"  seed {seed}: {k+1}/{len(tiles)} tiles", flush=True)

    def pr(tp, fp, fn):
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        return round(p, 4), round(r, 4), round(f, 4)

    mp, mr, mf = pr(m_tp, m_fp, m_fn)
    bp, br, bf = pr(b_tp, b_fp, b_fn)
    res = {"seed": seed, "op_thr": op_thr, "iou_thr": iou_thr, "n_gt": n_gt,
           "mask_P50": mp, "mask_R50": mr, "mask_F1_50": mf,
           "box_P50": bp, "box_R50": br, "box_F1_50": bf,
           "mask_tp": m_tp, "mask_fp": m_fp, "mask_fn": m_fn,
           "box_tp": b_tp, "box_fp": b_fp, "box_fn": b_fn,
           "det": f"det_t8_s{seed}", "em": os.path.basename(VAULT_EM)}
    out = os.path.join(HERE, f"pr_s{seed}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"[seed {seed}] mask P/R@50 {mp}/{mr}  box P/R@50 {bp}/{br} "
          f"(op_thr={op_thr}) -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    run(a.seed)

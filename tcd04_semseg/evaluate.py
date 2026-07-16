"""Full-439 eval of the two-head model: ITC mAP50 (must be unchanged) + area F1.

Reports three things:
  1. ITC mask mAP50  — det head → EM masks, canopy-ignore (should EQUAL det_t8's
     0.499, since the trunk+det head are frozen copies of it — a sanity check).
  2. area F1 (canopy-EXCLUDED)  — the current metric (instances only) = 0.587.
  3. area F1 (canopy-INCLUDED)  — GT tree-cover = ITC ∪ canopy; pred = ITC instance
     masks ∪ sem-head tree-cover. This is the lever the semantic head unlocks.

Native features from the main cache; sem head runs on the same 128×128 grid.
Delete tcd04_semseg/ to discard.

Usage:
    .venv/bin/python -m tcd04_semseg.evaluate --mh mh_sem
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from dapt.backbone import pick_device
from dapt.decode import decode
from boxinst_commonality_tcd_04.cache_test import cache_dir
from boxinst_commonality_tcd_04.detector import STRIDE8
from boxinst_commonality_tcd_04.em import MODEL_PATH, TCDMasker
from boxinst_commonality_tcd_04.evaluate import (RES, mask_ap,
                                                 pred_instance_masks, raster)
from boxinst_commonality_tcd_04.prepare_test import OUT as MAIN
from tcd04_semseg.model import MultiHead8

HERE = os.path.abspath(os.path.dirname(__file__))


def f1(tp, fp, fn):
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9), p, r


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mh", default="mh_sem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    ck = torch.load(os.path.join(HERE, "artifacts", args.mh + ".pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = MultiHead8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                       ).to(dev).eval()
    model.load_state_dict(ck["state"])
    masker = TCDMasker(MODEL_PATH)
    gt = json.load(open(os.path.join(MAIN, "test_gt.json")))
    cdir = cache_dir("web")
    tiles = [t for t in sorted(gt) if os.path.exists(os.path.join(cdir, t + ".npy"))]
    if args.limit:
        tiles = tiles[:args.limit]
    op, sthr = cfg["det_score_thr"], cfg["sem_thr"]

    P, Psc, Ign, G = [], [], [], []
    exc = [0, 0, 0]                     # canopy-excluded area F1 (instances only)
    inc0 = [0, 0, 0]                    # canopy-INCLUDED GT, instances only (baseline)
    inc = [0, 0, 0]                     # canopy-included area F1 (instances ∪ sem)
    for k, tid in enumerate(tiles):
        feat = np.load(os.path.join(cdir, tid + ".npy")).astype(np.float32)
        det, sem = model(torch.from_numpy(feat)[None].to(dev))
        bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
        zn = masker.project(feat)
        pm = pred_instance_masks(masker, zn, feat.shape[-1], bx)
        gm = np.array(raster(gt[tid]["trees"]))
        can = np.array(raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
        ign = np.array([bool(x.sum()) and (x & can).sum() / x.sum() > 0.5
                        for x in pm]) if len(pm) else np.zeros(0, bool)
        P.append(pm); Psc.append(sc); Ign.append(ign); G.append(gm)
        # foregrounds at op threshold
        inst = pm[sc >= op] if len(pm) else pm
        pf = inst.any(0) if len(inst) else np.zeros((RES, RES), bool)
        semf = np.array(Image.fromarray(
            torch.sigmoid(sem)[0, 0].cpu().numpy()).resize((RES, RES),
            Image.BILINEAR)) >= sthr
        gtree = gm.any(0) if len(gm) else np.zeros((RES, RES), bool)
        # (2) canopy-excluded: instances vs GT-tree, canopy removed from both
        pe, ge = pf & ~can, gtree & ~can
        exc[0] += int((pe & ge).sum()); exc[1] += int((pe & ~ge).sum())
        exc[2] += int((~pe & ge).sum())
        gi = gtree | can
        # (3a) baseline: canopy-included GT, instances ONLY (no sem head)
        inc0[0] += int((pf & gi).sum()); inc0[1] += int((pf & ~gi).sum())
        inc0[2] += int((~pf & gi).sum())
        # (3b) canopy-included: (instances ∪ sem) vs (GT-tree ∪ canopy)
        pi = pf | semf
        inc[0] += int((pi & gi).sum()); inc[1] += int((pi & ~gi).sum())
        inc[2] += int((~pi & gi).sum())
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)}", flush=True)

    m50 = mask_ap(P, Psc, G, Ign, 0.5)
    fe, pe, re = f1(*exc); f0, p0, r0 = f1(*inc0); fi, pi, ri = f1(*inc)
    out = {"n_tiles": len(tiles), "ITC_mask_mAP50": round(m50, 4),
           "areaF1_canopy_excluded": round(fe, 4), "excl_P": round(pe, 3),
           "excl_R": round(re, 3),
           "areaF1_canopy_incl_instancesONLY": round(f0, 4), "inc0_P": round(p0, 3),
           "inc0_R": round(r0, 3),
           "areaF1_canopy_incl_plus_sem": round(fi, 4), "incl_P": round(pi, 3),
           "incl_R": round(ri, 3), "sem_thr": sthr}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(HERE, "artifacts", "eval_" + args.mh + ".json"),
                        "w"), indent=2)
    print(f"\nITC mask mAP50 = {m50:.3f} (det_t8 = 0.499, must match)")
    print(f"area F1 (canopy-included GT): instances-only {f0:.3f} -> "
          f"+ sem head {fi:.3f}   [the semantic head's contribution]")


if __name__ == "__main__":
    main()

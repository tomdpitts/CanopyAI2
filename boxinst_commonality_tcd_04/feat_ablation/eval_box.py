"""Box-only single-scale eval on the OAM-TCD 439 TEST benchmark.

Reproduces the BOX path of boxinst_commonality_tcd_04.evaluate EXACTLY (same
decode thresholds, same canopy-ignore rule, same _greedy_ap matching via its
box_ap) but skips the EM mask stage — the ablation question is purely whether
detection survives feature compression, and the parent EM model is fit on
4096-dim features anyway. Reads GT from the parent test_gt.json; writes results
only into feat_ablation/results/.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.feat_ablation.eval_box \
        --variant pca256 [--ckpt path.pt] [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch

from dapt.backbone import pick_device
from dapt.decode import decode
from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8
from boxinst_commonality_tcd_04.evaluate import (IOU_50_95, RES, SCALE, box_ap,
                                                 raster)
from boxinst_commonality_tcd_04.prepare_test import OUT
from boxinst_commonality_tcd_04.feat_ablation.variants import (VARIANTS,
                                                               load_feat)

HERE = os.path.abspath(os.path.dirname(__file__))


@torch.no_grad()
def run(args):
    device = pick_device(args.device)
    ckpt = args.ckpt or os.path.join(HERE, "artifacts", f"det_fa_{args.variant}.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector8(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]
                      ).to(device).eval()
    model.load_state_dict(ck["state"])
    gt = json.load(open(os.path.join(OUT, "test_gt.json")))
    tiles = sorted(gt)
    if args.limit:
        tiles = tiles[:args.limit]
    print(f"[eval_box] variant={args.variant} ckpt={os.path.basename(ckpt)} "
          f"in_dim={cfg['in_dim']} device={device} tiles={len(tiles)}", flush=True)

    P_boxes, P_scores, G_boxes, Ign_box = [], [], [], []
    for k, tid in enumerate(tiles):
        feat = load_feat(args.variant, "feat_test", tid).astype(np.float32)
        det = model(torch.from_numpy(feat)[None].to(device))
        bx, sc = decode(det.cpu(), score_thr=args.eval_score_thr, stride=STRIDE8,
                        topk=args.topk)
        bx, sc = bx.numpy(), sc.numpy()
        can = np.array(raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
        pbox = bx / SCALE
        P_boxes.append(pbox); P_scores.append(sc)
        G_boxes.append(np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                                  *(np.asarray(t).reshape(-1, 2).max(0))]
                                 for t in gt[tid]["trees"]], np.float32).reshape(-1, 4)
                       / SCALE if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            xs, ys = slice(max(0, int(x0)), int(np.ceil(x1))), \
                slice(max(0, int(y0)), int(np.ceil(y1)))
            sub = can[ys, xs]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        Ign_box.append(ign_b)
        if (k + 1) % 40 == 0 or k + 1 == len(tiles):
            print(f"  {k+1}/{len(tiles)} tiles", flush=True)

    box50 = box_ap(P_boxes, P_scores, G_boxes, Ign_box, 0.5)
    box5095 = float(np.nanmean([box_ap(P_boxes, P_scores, G_boxes, Ign_box, t)
                                for t in IOU_50_95]))
    res = {"variant": args.variant, "ckpt": ckpt, "in_dim": cfg["in_dim"],
           "seed": cfg.get("seed"), "best_epoch": cfg.get("best_epoch"),
           "val_boxAP50": cfg.get("val_boxAP50"),
           "n_tiles": len(tiles),
           "n_gt_trees": int(sum(len(g) for g in G_boxes)),
           "box_mAP50": round(box50, 4), "box_mAP50_95": round(box5095, 4),
           "eval_score_thr": args.eval_score_thr, "topk": args.topk,
           "single_scale": True}
    print(json.dumps(res, indent=2), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"eval_box_{args.tag or args.variant}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (default: feat_ablation artifacts "
                         "det_fa_<variant>.pt)")
    ap.add_argument("--tag", default=None, help="results filename suffix")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument("--eval_score_thr", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

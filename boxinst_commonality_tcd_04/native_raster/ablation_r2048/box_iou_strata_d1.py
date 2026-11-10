"""tab:strata-maskiou (Table S9) re-based to the current Table 1 row set (2026-11-04, Tom).

Changes vs box_iou_strata_res_seamfix.py (untouched):
  - SelvaBox -> SAM 3 dropped (removed from the paper's results 2026-11-04).
  - BoxTeacher Swin-B (Tier B B2) added.
  - LACE = headline LACE (D1 variant) + edge fix + R3, single seed (replaces the 3 CenterNet-variant seeds).
  - Restor uses the tree-only file that Table 1 scores (the old strata rows used the both-class file).
  - Every row passes Table 1's protocol filter first: score >= 0.05, then top 600 by score per tile.
    Previously uncapped; BoxTeacher's stitched file carries up to 2,217 dets/tile.
Same matcher, strata and common-subset rule (box_supervised_baselines/box_iou_strata.py, unmodified).

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.box_iou_strata_d1 \
      --out boxinst_commonality_tcd_04/ablation/results_r2048/box_iou_strata_d1_r2048.json
"""
import argparse, os, sys

import numpy as np

from boxinst_commonality_tcd_04.box_supervised_baselines import box_iou_strata as B

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = B.PKG
SF = os.path.join(NR, "seamfix")
SCORE_FLOOR, MAX_DETS = 0.05, 600

ROWS = {
    "Restor MRCNN rpn1000": f"{NR}/preds_restor_rpn1000_r2048_treeonly.json",
    "DetecTree2 s0": f"{SF}/preds_dt2_s0v2_r2048_seamfix.json",
    "Box2Mask Swin-L": f"{SF}/preds_box2mask_swinl_s0_r2048_seamfix.json",
    "Box2Mask R-50": f"{SF}/preds_box2mask_r50_s0_r2048_seamfix.json",
    "BoxTeacher Swin-B": f"{PKG}/box_supervised_baselines/tierb/results/20261101_192510/boxteacher/results/preds_boxteacher_sub_test_r2048.json",
    "LACE (D1) + edge fix + R3": f"{PKG}/dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json",
}

_pairs = B.pairs_for_row


def _capped_pairs(preds, gt, min_box_iou=0.5):
    out = {}
    for tid, rec in preds.items():
        sc = np.asarray(rec["scores"], np.float64)
        keep = np.flatnonzero(sc >= SCORE_FLOOR)
        keep = keep[np.argsort(-sc[keep], kind="mergesort")][:MAX_DETS]
        out[tid] = {"scores": [rec["scores"][i] for i in keep],
                    "boxes_2048": [rec["boxes_2048"][i] for i in keep],
                    "masks_rle": [rec["masks_rle"][i] for i in keep]}
    print(f"[strata_d1] capped: max dets/tile {max(len(r['scores']) for r in out.values())}", flush=True)
    return _pairs(out, gt, min_box_iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    B.RES, B.SCALE = 2048, 1.0
    B.ROWS = dict(ROWS)
    B.pairs_for_row = _capped_pairs
    missing = [k for k, v in B.ROWS.items() if not os.path.exists(v)]
    assert not missing, missing
    print(f"[strata_d1] rows={B.ROWS} floor={SCORE_FLOOR} max_dets={MAX_DETS}", flush=True)
    sys.argv = [sys.argv[0], "--out", a.out]
    B.main()


if __name__ == "__main__":
    main()

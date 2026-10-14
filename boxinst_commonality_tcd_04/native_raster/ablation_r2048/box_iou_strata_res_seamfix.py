"""SEAM-FIX copy of box_iou_strata_res_guided.py (untouched): tab:strata-maskiou at 2048 with the
four subtile-stitched baselines replaced by their seam-fixed files (seamfix/, 2026-10-14).
LACE and Restor rows are the same files as the guided version.

Gate: --rows guided_current reproduces ablation/results_r2048/box_iou_strata_guided_r2048.json.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.box_iou_strata_res_seamfix \
      --rows seamfix --out boxinst_commonality_tcd_04/ablation/results_r2048/box_iou_strata_seamfix_r2048.json
"""
import argparse, os, sys

from boxinst_commonality_tcd_04.box_supervised_baselines import box_iou_strata as B
from boxinst_commonality_tcd_04.native_raster.ablation_r2048.box_iou_strata_res_guided import R2048

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.path.join(NR, "seamfix")
SEAMFIX = dict(R2048)
SEAMFIX.update({
    "DetecTree2 s0": f"{SF}/preds_dt2_s0v2_r2048_seamfix.json",
    "SelvaBox -> SAM 3": f"{SF}/preds_selvabox_seamfix_on_r2048.json",
    "Box2Mask R-50": f"{SF}/preds_box2mask_r50_s0_r2048_seamfix.json",
    "Box2Mask Swin-L": f"{SF}/preds_box2mask_swinl_s0_r2048_seamfix.json",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", choices=("guided_current", "seamfix"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    B.RES, B.SCALE = 2048, 1.0
    B.ROWS = dict(R2048 if a.rows == "guided_current" else SEAMFIX)
    missing = [k for k, v in B.ROWS.items() if not os.path.exists(v)]
    assert not missing, missing
    print(f"[box_iou_strata_seamfix] rows={a.rows} {B.ROWS}", flush=True)
    sys.argv = [sys.argv[0], "--out", a.out]
    B.main()


if __name__ == "__main__":
    main()

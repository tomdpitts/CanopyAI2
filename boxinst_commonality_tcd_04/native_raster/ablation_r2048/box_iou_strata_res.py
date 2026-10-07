"""tab:strata-maskiou (box_supervised_baselines/box_iou_strata.py) at an arbitrary raster,
without editing it: overrides RES/SCALE and the ROWS dict on the imported module, then calls
its unmodified main().

--rows published   the published ROWS (512 files; gate: must equal results_439/box_iou_strata.json)
--rows no_selva    every row EXCEPT SelvaBox->SAM 3, whose 2048 masks were not produced
                   (GPU run stopped by budget). Run at 512 and 2048 so the common-subset
                   intersection is the same row set at both rasters.
"""
import argparse, os, sys

from boxinst_commonality_tcd_04.box_supervised_baselines import box_iou_strata as B

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = B.PKG

R2048 = {
    "LACE s0 (product)": f"{NR}/preds_lace_s0_r2048_tau040.json",
    "LACE s1 (product)": f"{NR}/preds_lace_s1_r2048_tau040.json",
    "LACE s2 (product)": f"{NR}/preds_lace_s2_r2048_tau040.json",
    "Restor MRCNN rpn1000": f"{NR}/preds_restor_rpn1000_r2048.json",
    "DetecTree2 s0": f"{NR}/preds_dt2_s0v2_r2048.json",
    "Box2Mask R-50": f"{NR}/preds_box2mask_r50_s0_r2048.json",
    "Box2Mask Swin-L": f"{NR}/preds_box2mask_swinl_s0_r2048.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--rows", choices=("published", "no_selva"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    B.RES, B.SCALE = a.res, 2048.0 / a.res
    if a.rows == "published":
        assert a.res == 512
    else:
        base = R2048 if a.res == 2048 else {k: v for k, v in B.ROWS.items() if "SelvaBox" not in k}
        B.ROWS = dict(base)
    missing = [k for k, v in B.ROWS.items() if not os.path.exists(v)]
    assert not missing, missing
    print(f"[box_iou_strata_res] res={a.res} rows={list(B.ROWS)}", flush=True)
    sys.argv = [sys.argv[0], "--out", a.out]
    B.main()


if __name__ == "__main__":
    main()

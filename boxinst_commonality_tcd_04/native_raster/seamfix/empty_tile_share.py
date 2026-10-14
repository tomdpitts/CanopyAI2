"""Appendix "Empty tiles": share of each row's detections (score >= its scoring floor) that fall on
the test tiles carrying no individual crown. No script was recorded for the published 4.0 / 2.1 /
4.7 / 6.6 %; the current-file rows are the gate for this definition.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.seamfix.empty_tile_share
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
OUT = os.path.join(HERE, "empty_tile_share.json")
ROWS = {  # name: (file, floor)
    "LACE s0": (f"{NR}/preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json", 0.0),
    "Restor rpn1000": (f"{NR}/preds_restor_rpn1000_r2048_treeonly.json", 0.05),
    "Detectree2 (current)": (f"{NR}/preds_dt2_s0v2_r2048.json", 0.05),
    "Box2Mask Swin-L (current)": (f"{NR}/preds_box2mask_swinl_s0_r2048.json", 0.05),
    "Box2Mask R-50 (current)": (f"{NR}/preds_box2mask_r50_s0_r2048.json", 0.05),
    "SelvaBox (current)": (f"{NR}/preds_selvabox_bench_r2048.json", 0.05),
    "Detectree2 (seamfix)": (f"{HERE}/preds_dt2_s0v2_r2048_seamfix.json", 0.05),
    "Box2Mask Swin-L (seamfix)": (f"{HERE}/preds_box2mask_swinl_s0_r2048_seamfix.json", 0.05),
    "Box2Mask R-50 (seamfix)": (f"{HERE}/preds_box2mask_r50_s0_r2048_seamfix.json", 0.05),
    "SelvaBox (seamfix)": (f"{HERE}/preds_selvabox_seamfix_on_r2048.json", 0.05),
}


def main():
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    empty = {t for t, g in gt.items() if not g["trees"]}
    out = {"n_tiles": len(gt), "n_no_crown": len(empty), "rows": {}}
    for name, (fp, floor) in ROWS.items():
        if not os.path.exists(fp):
            print(name, "MISSING", fp); continue
        P = json.load(open(fp))["preds"]
        n = ne = 0
        for t, r in P.items():
            k = sum(s >= floor for s in r["scores"])
            n += k; ne += k if t in empty else 0
        out["rows"][name] = {"file": os.path.relpath(fp, PKG), "floor": floor, "n_det": n,
                             "n_on_empty": ne, "pct": round(100 * ne / n, 2)}
        print(f"{name:28s} {100*ne/n:5.2f}%  ({ne}/{n})", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()

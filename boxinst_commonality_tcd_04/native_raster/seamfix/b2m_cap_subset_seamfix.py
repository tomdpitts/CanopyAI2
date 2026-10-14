"""SEAM-FIX copy of b2m_cap_subset.py (untouched): note d's cap delta at 2048 on the seam-fixed
Box2Mask Swin-L file. Same definition (1024^2 subtiles at {0,512,1024}^2, crown in subtile if GT
box centre inside, cap 100; 39 of 3,951 subtiles over cap, not PLAN.md's 46 -- see audit).
Gate: the current 2048 file must reproduce the recorded +0.007 (b2m_cap_subset.json res 2048).

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.seamfix.b2m_cap_subset_seamfix
"""
import json
import os
import tempfile

from boxinst_commonality_tcd_04 import score_coco as S

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
OUT = os.path.join(HERE, "b2m_cap_subset_seamfix.json")
PREDS = {"current_r2048 (gate)": os.path.join(NR, "preds_box2mask_swinl_s0_r2048.json"),
         "seamfix_r2048": os.path.join(HERE, "preds_box2mask_swinl_s0_r2048_seamfix.json")}


def main():
    assert not os.path.exists(OUT), f"REFUSING to overwrite {OUT}"
    rec = json.load(open(os.path.join(NR, "b2m_cap_subset.json")))
    flagged = set(rec["flagged_tiles"])
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    tids = sorted(gt); keep = [t for t in tids if t not in flagged]
    out = {"definition": rec["definition"], "n_over_cap": rec["n_over_cap"],
           "n_flagged_tiles": len(flagged), "rows": {}}
    for name, fp in PREDS.items():
        P = json.load(open(fp)); preds = P["preds"]; row = {}
        for sub_name, ts in (("all", tids), ("without_flagged", keep)):
            sub = {t: gt[t] for t in ts}
            gt_fp, tid2img, tree_rles, cu = S.build_gt(sub, 2048, with_canopy=False)
            gt_cr, _, _, _ = S.build_gt(sub, 2048, with_canopy=True)
            fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
            json.dump({"meta": P["meta"], "preds": {t: preds[t] for t in ts if t in preds}}, open(tmp, "w"))
            row[sub_name] = S._score_one(tmp, (gt_fp, gt_cr, tid2img, tree_rles, cu, ts), 2048, S.MAX_DETS,
                                         S.SCORE_FLOOR)["canopy_neutral_crowd"]["mask"]
            os.remove(tmp)
        row["delta_AP50"] = round(row["without_flagged"]["AP50"] - row["all"]["AP50"], 4)
        out["rows"][name] = row
        print(name, row["all"]["AP50"], row["without_flagged"]["AP50"], row["delta_AP50"], flush=True)
    out["gate_pass"] = out["rows"]["current_r2048 (gate)"]["delta_AP50"] == rec["res"]["2048"]["delta_AP50"]
    json.dump(out, open(OUT, "w"), indent=2)
    print("gate (current == recorded", rec["res"]["2048"]["delta_AP50"], "):", out["gate_pass"], "->", OUT)


if __name__ == "__main__":
    main()

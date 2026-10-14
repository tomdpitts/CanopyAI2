"""SEAM-FIX copy of seam_analysis.py (untouched), 2026-10-14: same measure on the seam-fixed files;
the seam-cut share should fall to the ~0.5% chance rate of the whole-tile models.

Original docstring: Read-only: how many stitched predictions are cut at an interior subtile seam, and what AP does
each subtile-stitched row lose to them? A prediction is 'seam-cut' if its mask bbox edge lies on an
interior subtile edge (x or y in {512, 1024, 1536}, not the tile frame) -- a straight cut the crown
itself would not produce. Sensitivity: AP with those predictions removed."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, "/Users/tompitts/dphil/CanopyAI2")
from boxinst_commonality_tcd_04 import score_coco as S
NR = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04/native_raster"
SF = os.path.join(NR, "seamfix")
ROWS = {"Box2Mask Swin-L (seamfix)": os.path.join(SF, "preds_box2mask_swinl_s0_r2048_seamfix.json"),
        "Box2Mask R-50 (seamfix)": os.path.join(SF, "preds_box2mask_r50_s0_r2048_seamfix.json"),
        "Detectree2 (seamfix)": os.path.join(SF, "preds_dt2_s0v2_r2048_seamfix.json"),
        "SelvaBox->SAM3 (seamfix)": os.path.join(SF, "preds_selvabox_seamfix_on_r2048.json"),
        "LACE s0 (control, whole tile)": "preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json"}
SEAMS = {512, 1024, 1536}
gt = json.load(open("/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04/test_gt.json")); tids = sorted(gt)
gt_fp, tid2img, tree_rles, cu = S.build_gt(gt, 2048, with_canopy=False)
gt_cr, _, _, _ = S.build_gt(gt, 2048, with_canopy=True)
ctx = (gt_fp, gt_cr, tid2img, tree_rles, cu, tids)
out = {}
for name, f in ROWS.items():
    P = json.load(open(os.path.join(NR, f))); preds = P["preds"]
    n = cut = 0; keep_preds = {}
    per_tile_cut = {}
    for t in tids:
        r = preds.get(t, {"boxes_2048": [], "scores": [], "masks_rle": []})
        flags = []
        for b in r["boxes_2048"]:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            flags.append(any(v in SEAMS for v in (x0, x1, y0, y1)))
        n += len(flags); cut += sum(flags); per_tile_cut[t] = int(sum(flags))
        idx = [i for i, fl in enumerate(flags) if not fl]
        keep_preds[t] = {k: [r[k][i] for i in idx] for k in ("boxes_2048", "scores", "masks_rle")}
        if "canopy_ignore" in r: keep_preds[t]["canopy_ignore"] = [r["canopy_ignore"][i] for i in idx]
    res = {"n_det": n, "n_seam_cut": cut, "frac": round(cut / max(n, 1), 4), "tile125_cut": per_tile_cut.get("tcd_val_tile_125")}
    floor = 0.0 if "LACE" in name else 0.05
    for tag, PP in (("all", preds), ("seam_cut_removed", keep_preds)):
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump({"meta": {"mask_res": 2048}, "preds": PP}, open(tmp, "w"))
        c = S._score_one(tmp, ctx, 2048, S.MAX_DETS, floor)["canopy_neutral_crowd"]; os.remove(tmp)
        res[tag] = {"AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"], "box_AP50": c["box"]["AP50"]}
    out[name] = res; print(name, json.dumps(res), flush=True)
OUTF = os.path.join(SF, "seam_analysis_seamfix_r2048.json")
assert not os.path.exists(OUTF), OUTF
json.dump(out, open(OUTF, "w"), indent=1)

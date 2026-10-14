"""Score Tier A rows (PLAN.md §13) with the frozen protocol: score_coco, res 2048, maxDets 600, canopy as
crowd. Floors: 0.05 when ranked by a detector score (LACE s or SelvaBox s, both already >= 0.05);
0 when ranked by a product (LACE s' or sqrt(s x SAM)), as for LACE's published product rows.
Assembles A1/A2 from their per-tile dirs (asserting completeness, finite scores); A3 files are read as
written. Each output: tierA/scored_<row>.json (refuses to overwrite). Summary: tierA/summary_tierA.json.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.score_tierA --rows a3 a2 a1
"""
import argparse, json, os
import numpy as np
from boxinst_commonality_tcd_04 import score_coco as SC

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
GT = os.path.join(PKG, "test_gt.json")


def assemble(tiles_dir, key_scores, name, extra_meta):
    fp = os.path.join(HERE, f"preds_{name}_r2048.json")
    if os.path.exists(fp):
        return fp
    tids = sorted(json.load(open(GT)))
    preds = {}
    for t in tids:
        r = json.load(open(os.path.join(tiles_dir, t + ".json")))
        sc = key_scores(r)
        assert len(sc) == len(r["masks_rle"]) and all(np.isfinite(sc)), (name, t)
        preds[t] = {"boxes_2048": r["boxes_2048"], "scores": [float(x) for x in sc], "masks_rle": r["masks_rle"]}
    json.dump({"meta": dict(extra_meta, mask_res=2048), "preds": preds}, open(fp, "w"))
    return fp


def score(fp, floor, tag, out):
    o = os.path.join(HERE, f"scored_{tag}.json")
    if os.path.exists(o):
        r = json.load(open(o))
    else:
        r = SC.score_many([fp], GT, res=2048, max_dets=600, score_floor=floor)[0]
        json.dump(r, open(o, "w"))
    c = r["canopy_neutral_crowd"]
    out[tag] = {"AP50": c["mask"]["AP50"], "AP50_95": c["mask"]["AP50_95"], "box_AP50": c["box"]["AP50"],
                "floor": floor, "file": os.path.relpath(fp, PKG)}
    print(tag, out[tag], flush=True)


def band(out, tags, name):
    a = np.array([[out[t]["AP50"], out[t]["AP50_95"]] for t in tags])
    out[name] = {"AP50_mean": round(float(a[:, 0].mean()), 4), "AP50_sd": round(float(a[:, 0].std(ddof=1)), 4),
                 "AP50_95_mean": round(float(a[:, 1].mean()), 4), "AP50_95_sd": round(float(a[:, 1].std(ddof=1)), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", nargs="+", default=["a3", "a2", "a1"]); a = ap.parse_args()
    sp = os.path.join(HERE, "summary_tierA.json")
    out = json.load(open(sp)) if os.path.exists(sp) else {}
    if "a3" in a.rows:
        for s in (0, 1, 2):
            score(os.path.join(HERE, f"preds_a3_ellipse_lace_s{s}_r2048.json"), 0.05, f"a3_ellipse_lace_s{s}", out)
        band(out, [f"a3_ellipse_lace_s{s}" for s in (0, 1, 2)], "a3_ellipse_lace_band")
        score(os.path.join(HERE, "preds_a3_ellipse_selvabox_r2048.json"), 0.05, "a3_ellipse_selvabox", out)
    if "a2" in a.rows:
        d = os.path.join(HERE, "a2_tiles")
        meta = {"model": "SelvaBox seam-fixed boxes + LACE masker (deployed, guided render), local CPU"}
        fp = assemble(d, lambda r: r["scores"], "a2_selvabox_lacemasker_product", dict(meta, rank="s*pbar*m"))
        score(fp, 0.0, "a2_selvabox_lacemasker_product", out)
        fp = assemble(d, lambda r: r["scores_det"], "a2_selvabox_lacemasker_det", dict(meta, rank="SelvaBox s"))
        score(fp, 0.05, "a2_selvabox_lacemasker_det", out)
    if "a1" in a.rows:
        for s in (0, 1, 2):
            d = os.path.join(HERE, f"a1_tiles_s{s}")
            meta = {"model": f"LACE detector s{s} + SAM 3 zero-shot, per-subtile placement, local MPS"}
            fp = assemble(d, lambda r: r["scores"], f"a1_lace_s{s}_sam3zs_det", dict(meta, rank="LACE s"))
            score(fp, 0.05, f"a1_lace_s{s}_sam3zs_det", out)
            fp = assemble(d, lambda r: np.sqrt(np.asarray(r["scores"], float) * np.clip(np.asarray(r["sam_scores"], float), 0, None)).round(6),
                          f"a1_lace_s{s}_sam3zs_gmean", dict(meta, rank="sqrt(s*SAM)"))
            score(fp, 0.0, f"a1_lace_s{s}_sam3zs_gmean", out)
        band(out, [f"a1_lace_s{s}_sam3zs_det" for s in (0, 1, 2)], "a1_det_band")
        band(out, [f"a1_lace_s{s}_sam3zs_gmean" for s in (0, 1, 2)], "a1_gmean_band")
    json.dump(out, open(sp, "w"), indent=1)
    print("->", sp)


if __name__ == "__main__":
    main()

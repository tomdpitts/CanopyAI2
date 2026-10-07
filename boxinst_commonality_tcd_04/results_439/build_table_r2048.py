"""512 | 2048 | delta table for every row of the native-raster campaign. Reads scored JSONs only;
writes results_439/r2048/table_r2048.{json,md}. Re-runnable; overwrites only its own outputs.
Canopy-neutral (crowd) arm throughout. Only compare within a raster column.

  .venv/bin/python -m boxinst_commonality_tcd_04.results_439.build_table_r2048
"""
import json, os
import numpy as np

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, NR, SP = (os.path.join(H, "results_439"), os.path.join(H, "native_raster"),
             os.path.join(H, "results_439", "sparse236"))
OUT = os.path.join(R, "r2048")

# (block, row label, 512 scored file, 2048 scored file, note)
ROWS = [
  ("439", "LACE s0 (tau 0.25@512 / 0.40@2048)", f"{NR}/lace_s0_r512_control_scored.json", f"{NR}/scored_lace_s0_r2048_tau040.json", "same detections; tau val-selected per raster"),
  ("439", "LACE s1", f"{R}/rerank_product_s1.json", f"{NR}/scored_lace_s1_r2048_tau040.json", ""),
  ("439", "LACE s2", f"{R}/rerank_product_s2.json", f"{NR}/scored_lace_s2_r2048_tau040.json", ""),
  ("439", "Restor rpn512 (released) tree-only", f"{R}/restor_coco512.json", f"{NR}/scored_restor_rpn512_r2048_tree.json", "same detections"),
  ("439", "Restor rpn512 (released) pooled", f"{R}/restor_allcls.json", f"{NR}/scored_restor_rpn512_r2048_pooled.json", "same detections"),
  ("439", "Restor rpn1000 tree-only", f"{R}/restor_rpn1000_tree.json", f"{NR}/scored_restor_rpn1000_r2048_tree.json", "same detections"),
  ("439", "Restor rpn1000 pooled", f"{R}/restor_rpn1000_pooled.json", f"{NR}/scored_restor_rpn1000_r2048_pooled.json", "same detections"),
  ("439", "Restor rpn2000 tree-only", f"{R}/restor_rpn2000.json", f"{NR}/scored_restor_rpn2000_r2048_tree.json", "same detections"),
  ("439", "DetecTree2 s0v2", f"{R}/dt2_s0v2_coco512.json", f"{NR}/scored_dt2_s0v2_r2048.json", "re-stitched: mask-space dedup drifts"),
  ("439", "Box2Mask-T Swin-L", f"{R}/box2mask_swinl_s0_coco512.json", f"{NR}/scored_box2mask_swinl_s0_r2048.json", "re-stitched: mask-space dedup drifts"),
  ("439", "Box2Mask-T R-50", f"{R}/box2mask_s0_coco512.json", f"{NR}/scored_box2mask_r50_s0_r2048.json", "re-stitched: mask-space dedup drifts"),
  ("439", "SelvaBox -> SAM 3 (bench, maxDets 600)", f"{R}/selvabox_bench_600.json", None, "NOT PRODUCED: GPU run stopped at 145/439 (budget)"),
  ("sparse236", "LACE s0 (tau 0.25@512 / 0.40@2048)", f"{SP}/lace_prod_s0.json", f"{NR}/scored_lace_sparse_s0_r2048_tau040.json", "same detections"),
  ("sparse236", "LACE s1", f"{SP}/lace_prod_s1.json", f"{NR}/scored_lace_sparse_s1_r2048_tau040.json", ""),
  ("sparse236", "LACE s2", f"{SP}/lace_prod_s2.json", f"{NR}/scored_lace_sparse_s2_r2048_tau040.json", ""),
  ("sparse236", "LACE s0 @ tau 0.25 both rasters", f"{SP}/lace_prod_s0.json", f"{NR}/scored_lace_sparse_s0_r2048_tau025.json", "same tau: pure raster effect"),
  ("sparse236", "DetecTree2 s0v2", f"{SP}/dt2_s0v2.json", f"{NR}/scored_dt2_sparse_s0v2_r2048.json", "re-stitched"),
]
MET = [("mask", "AP50"), ("mask", "AP75"), ("mask", "AP50_95"), ("box", "AP50")]


def get(fp):
    if not fp or not os.path.exists(fp):
        return None
    c = json.load(open(fp))["canopy_neutral_crowd"]
    return {f"{a}_{m}": c[a][m] for a, m in MET} | {"n_det": c["mask"]["n_det"],
            "res": json.load(open(fp))["protocol"]["res"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for blk, lab, f512, f2048, note in ROWS:
        a, b = get(f512), get(f2048)
        if a: assert a["res"] == 512, (f512, a["res"])
        if b: assert b["res"] == 2048, (f2048, b["res"])
        rows.append({"block": blk, "row": lab, "r512": a, "r2048": b, "note": note,
                     "src512": f512 and os.path.relpath(f512, H), "src2048": f2048 and os.path.relpath(f2048, H)})
    bands = {}
    for blk in ("439", "sparse236"):
        seeds = [r for r in rows if r["block"] == blk and r["row"].startswith("LACE s") and "tau 0.25 both" not in r["row"]]
        for ras in ("r512", "r2048"):
            v = [r[ras] for r in seeds if r[ras]]
            if len(v) == 3:
                bands[f"{blk}_{ras}"] = {k: [round(float(np.mean([x[k] for x in v])), 4),
                                             round(float(np.std([x[k] for x in v], ddof=1)), 4)]
                                         for k in ("mask_AP50", "mask_AP75", "mask_AP50_95", "box_AP50")}
    json.dump({"rows": rows, "lace_3seed_bands_ddof1": bands}, open(os.path.join(OUT, "table_r2048.json"), "w"), indent=2)
    L = ["# 512 vs native 2048 — every row (canopy-neutral crowd). Compare only within a raster column.\n"]
    for blk in ("439", "sparse236"):
        L += [f"\n## {blk}\n", "| row | mAP50 512 | mAP50 2048 | Δ | mAP75 512 | mAP75 2048 | mAP50:95 512 | mAP50:95 2048 | box50 512 | box50 2048 | dets 512→2048 | note |", "|" + "---|" * 12]
        for r in rows:
            if r["block"] != blk: continue
            a, b = r["r512"] or {}, r["r2048"] or {}
            f = lambda d, k: f"{d[k]:.4f}" if k in d else "—"
            d = f"{b['mask_AP50']-a['mask_AP50']:+.4f}" if a and b else "—"
            L.append(f"| {r['row']} | {f(a,'mask_AP50')} | {f(b,'mask_AP50')} | {d} | {f(a,'mask_AP75')} | {f(b,'mask_AP75')} | "
                     f"{f(a,'mask_AP50_95')} | {f(b,'mask_AP50_95')} | {f(a,'box_AP50')} | {f(b,'box_AP50')} | "
                     f"{a.get('n_det','—')}→{b.get('n_det','—')} | {r['note']} |")
        for ras in ("r512", "r2048"):
            k = f"{blk}_{ras}"
            if k in bands:
                L.append(f"\nLACE 3-seed {ras[1:]} (mean ± sd, ddof=1): " + ", ".join(f"{m} {v[0]:.4f} ± {v[1]:.4f}" for m, v in bands[k].items()))
    open(os.path.join(OUT, "table_r2048.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

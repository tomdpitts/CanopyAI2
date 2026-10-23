"""The pre-registered rule for "a Tier B baseline clearly beats LACE" (TIERB.md section 7, fixed
2026-10-14 before any Tier B number exists). If it fires, finalize sets the global STOP flag:
every running Modal job checkpoints and exits and nothing new starts.

LACE references, read from the stored frozen-scorer files (res 2048, maxDets 600, canopy crowd):
  test 439, product s'      AP50 0.6470 (0.6472/0.6444/0.6493)  AP50:95 0.2778
      native_raster/scored_lace_s{0,1,2}_r2048_cgf_r12_tau035_guidedrun.json
  test 439, detector s       AP50 0.6137 (0.6101/0.6152/0.6159)  AP50:95 0.2557
      native_raster/scored_lace_s{0,1,2}_r2048_cgf_r12_tau035_detscore.json
  sparse 236, product s'     AP50 0.6775 (0.6889/0.6671/0.6766)  AP50:95 0.3215
  sparse 236, detector s     AP50 0.6410 (0.6561/0.6297/0.6371)  AP50:95 0.2946
  GT boxes, 366 tiles        mean IoU 0.7477 (25,692 valid crowns; gtbox_lace_guided_r2048.jsonl)

MARGIN = 0.010 AP (about 4x LACE's seed SD at AP50 on test). "Clearly beats" means EITHER
  (a) AP50 >= LACE + 0.010, OR (b) AP50 >= LACE and AP50:95 >= LACE + 0.010,
evaluated on EVERY reported row (primary and sensitivity geometry; test and sparse), against the
LACE row with the SAME ranking rule (B1/B2 systems vs LACE's deployed product ranking; B3 masks on
LACE's boxes ranked by LACE's detector score vs LACE's own masks under the same ranking). For B3 on
GT boxes: paired per-crown mean IoU (B3 - LACE) >= +0.010 with the tile-bootstrap 95% CI above 0.
"""
from __future__ import annotations

MARGIN = 0.010
LACE = {
    ("test", "product"): {"AP50": 0.6470, "AP50_95": 0.2778},
    ("test", "detscore"): {"AP50": 0.6137, "AP50_95": 0.2557},
    ("test", "detscore_s0"): {"AP50": 0.6101, "AP50_95": 0.2579},   # scored_lace_s0_..._detscore.json
    ("sparse", "product"): {"AP50": 0.6775, "AP50_95": 0.3215},
    ("sparse", "detscore"): {"AP50": 0.6410, "AP50_95": 0.2946},
}
LACE_GTBOX_MEAN_IOU = 0.7477


def clearly_beats(row, ref, margin=MARGIN):
    a50, a5095 = row["AP50"], row["AP50_95"]
    if a50 >= ref["AP50"] + margin:
        return True, f"AP50 {a50:.4f} >= {ref['AP50']:.4f}+{margin}"
    if a50 >= ref["AP50"] and a5095 >= ref["AP50_95"] + margin:
        return True, f"AP50 {a50:.4f} >= {ref['AP50']:.4f} and AP50:95 {a5095:.4f} >= {ref['AP50_95']:.4f}+{margin}"
    return False, None


def evaluate(rows, gtbox=None):
    """rows: list of {"name", "split" (test|sparse), "ranking" (product|detscore), "AP50", "AP50_95"}.
    gtbox: optional {"mean_diff", "ci95": [lo, hi]} for B3 (B3 - LACE, paired)."""
    hits = []
    for r in rows:
        ok, why = clearly_beats(r, LACE[(r["split"], r["ranking"])])
        if ok:
            hits.append(f"{r['name']}: {why}")
    if gtbox is not None and gtbox["mean_diff"] >= MARGIN and gtbox["ci95"][0] > 0:
        hits.append(f"GT boxes: paired mean IoU diff {gtbox['mean_diff']:+.4f} CI {gtbox['ci95']}")
    return {"triggered": bool(hits), "why": "; ".join(hits) or None, "margin": MARGIN,
            "rows": rows, "gtbox": gtbox, "lace_refs": {f"{k[0]}/{k[1]}": v for k, v in LACE.items()}}

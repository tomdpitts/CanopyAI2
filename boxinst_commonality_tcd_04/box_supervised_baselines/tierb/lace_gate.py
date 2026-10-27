"""The pre-registered rule for "a Tier B baseline clearly beats LACE" (TIERB.md section 7, fixed
2026-10-14 before any Tier B number exists). If it fires, finalize sets the global STOP flag:
every running Modal job checkpoints and exits and nothing new starts.

LACE references (re-pointed 2026-10-27 to the adopted lace4px_simple masker, TIERB.md section 00; the
8 px guided-render values they replace are in git history). Stored frozen-scorer files (res 2048, maxDets
600, canopy crowd), native_raster/hunt/test/:
  test 439, own s' (product)   AP50 0.6545 (0.6556/0.6503/0.6576)  AP50:95 0.2803   scored_lace_s{s}_r2048_lace4px_simple.json
  test 439, detector s          AP50 0.6237                          AP50:95 0.2609   ..._lace4px_simple_detscore.json
  test seed 0, s' / s           AP50 0.6556 / 0.6206                 AP50:95 0.2844 / 0.2638
  sparse 236, s' / s            AP50 0.6785 / 0.6478                 AP50:95 0.3192 / 0.2965   scored_lace_sparse_s{s}_...
  GT boxes, 366 tiles           mean IoU 0.7861 (25,692 valid crowns; gtbox4_lace4px_simple_percrown.jsonl)

MARGIN = 0.010 AP (about 4x LACE's seed SD at AP50 on test). "Clearly beats" means EITHER
  (a) AP50 >= LACE + 0.010, OR (b) AP50 >= LACE and AP50:95 >= LACE + 0.010,
evaluated on EVERY reported row (primary and sensitivity geometry; test and sparse), always against
LACE's FULL row, i.e. LACE ranked by its own s' (Tom 2026-10-27: LACE is only ever reported with its
reranking). B1/B2 systems use their own scores; B3 masks on LACE's boxes are ranked by the detector
score s (MAL's labeler has no confidence output, so s is its native ranking), vs LACE s'. For B3 on
GT boxes: paired per-crown mean IoU (B3 - LACE) >= +0.010 with the tile-bootstrap 95% CI above 0.
"""
from __future__ import annotations

MARGIN = 0.010
LACE = {
    ("test", "product"): {"AP50": 0.6545, "AP50_95": 0.2803},       # 3-seed mean, own s'
    ("test", "product_s0"): {"AP50": 0.6556, "AP50_95": 0.2844},    # seed 0, own s' (B3 on LACE s0 boxes)
    ("test", "detscore"): {"AP50": 0.6237, "AP50_95": 0.2609},      # record only: never a gate reference
    ("test", "detscore_s0"): {"AP50": 0.6206, "AP50_95": 0.2638},   # record only
    ("sparse", "product"): {"AP50": 0.6785, "AP50_95": 0.3192},
    ("sparse", "product_s0"): {"AP50": 0.6903, "AP50_95": 0.3258},
    ("sparse", "detscore"): {"AP50": 0.6478, "AP50_95": 0.2965},    # record only
}
LACE_GTBOX_MEAN_IOU = 0.7861
LACE_GTBOX_PERCROWN = "native_raster/hunt/test/gtbox4_lace4px_simple_percrown.jsonl"   # relative to the package


def clearly_beats(row, ref, margin=MARGIN):
    a50, a5095 = row["AP50"], row["AP50_95"]
    if a50 >= ref["AP50"] + margin:
        return True, f"AP50 {a50:.4f} >= {ref['AP50']:.4f}+{margin}"
    if a50 >= ref["AP50"] and a5095 >= ref["AP50_95"] + margin:
        return True, f"AP50 {a50:.4f} >= {ref['AP50']:.4f} and AP50:95 {a5095:.4f} >= {ref['AP50_95']:.4f}+{margin}"
    return False, None


def evaluate(rows, gtbox=None):
    """rows: list of {"name", "split" (test|sparse), "AP50", "AP50_95", optional "lace_ref" (product|product_s0,
    default product)}. The reference is ALWAYS LACE's own-s' row; the row's own "ranking" is recorded only.
    gtbox: optional {"mean_diff", "ci95": [lo, hi]} for B3 (B3 - LACE, paired)."""
    hits = []
    for r in rows:
        ref = r.get("lace_ref", "product")
        assert ref.startswith("product"), f"{r['name']}: LACE is only compared with its full s' ranking"
        ok, why = clearly_beats(r, LACE[(r["split"], ref)])
        if ok:
            hits.append(f"{r['name']}: {why}")
    if gtbox is not None and gtbox["mean_diff"] >= MARGIN and gtbox["ci95"][0] > 0:
        hits.append(f"GT boxes: paired mean IoU diff {gtbox['mean_diff']:+.4f} CI {gtbox['ci95']}")
    return {"triggered": bool(hits), "why": "; ".join(hits) or None, "margin": MARGIN,
            "rows": rows, "gtbox": gtbox, "lace_refs": {f"{k[0]}/{k[1]}": v for k, v in LACE.items()}}

"""Box->mask module bake-off under the FROZEN Table-1 protocol: LACE vs SAM 3, same boxes.

The 2026-07 SAM row (`sam3_results_439.json`, mask mAP50 0.6326) was scored with the legacy
`_greedy_ap` core, against the UN-knobbed masker (0.6203), with SAM pinned to detector-score
ranking. This re-scores it through `score_coco.py` -- pycocotools COCOeval, 439 whole 2048^2
tiles, masks 512^2, maxDets 600, floor 0.05, canopy as `iscrowd` -- the same gate every other
Table-1 row passes through, and gives SAM its own confidence as a ranking option.

The comparison is genuinely apples-to-apples on the input side: the boxes and detector scores
in `preds_sam3_s0.json` are bit-identical to those in `phase4/preds/knobbed_s0.json` (asserted
below). Only the box->mask module differs.

Rows produced
    LACE, detector score            phase4/preds/knobbed_s0.json
    LACE, posterior product         confidence/preds_knobbed_s0_product.json   <- the system
    SAM 3, detector score           SAM masks ranked by LACE's detector score
    SAM 3, own score                SAM masks ranked by SAM's per-mask score
    SAM 3, det x SAM score          SAM's counterpart to LACE's posterior product

The last row is the point: LACE's product multiplies in a confidence its own masker emits,
so SAM is given the same opportunity rather than being held at detector-score ranking while
LACE is not.

Usage
    .venv/bin/modal volume get tcd04-phase4-vol out/preds_sam3_s0.json \
        boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4_sam/
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4_sam.score_sam_vs_lace
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import score_coco as SC

HERE = os.path.dirname(os.path.abspath(__file__))
MTS = os.path.dirname(HERE)
PKG = os.path.dirname(MTS)

SAM_PREDS = os.path.join(HERE, "preds_sam3_s0.json")
LACE_DET = os.path.join(MTS, "phase4", "preds", "knobbed_s0.json")
LACE_PROD = os.path.join(PKG, "confidence", "preds_knobbed_s0_product.json")
GT = os.path.join(PKG, "test_gt.json")
OUT_DIR = os.path.join(PKG, "results_439")


def _load(fp):
    d = json.load(open(fp))
    return d.get("preds", d), d.get("meta", {})


def assert_same_boxes(a, b, name_a, name_b, scores=True):
    """The whole ablation rests on this: same detector, same boxes, only the masker differs.

    `scores=False` for a reranked file, whose `scores` field is the reranked key by design --
    box identity and per-tile ORDER are what must still hold, since the rerank is grafted on
    positionally."""
    assert set(a) == set(b), f"{name_a} / {name_b}: tile sets differ"
    for t in sorted(a):
        ba = np.asarray(a[t]["boxes_2048"], np.float64).reshape(-1, 4)
        bb = np.asarray(b[t]["boxes_2048"], np.float64).reshape(-1, 4)
        assert ba.shape == bb.shape and np.allclose(ba, bb), f"boxes differ on {t}"
        if scores:
            sa = np.asarray(a[t]["scores"], np.float64)
            sb = np.asarray(b[t]["scores"], np.float64)
            assert np.allclose(sa, sb), f"detector scores differ on {t}"
    n = sum(len(a[t]["scores"]) for t in a)
    print(f"[bakeoff] {name_a} vs {name_b}: {len(a)} tiles, {n} boxes -- IDENTICAL", flush=True)


def rescored(preds, key):
    """SAM preds re-ranked by `key`, in the schema score_coco.build_dt reads.

    THE SCORE FLOOR IS APPLIED TO THE DETECTOR SCORE, NOT THE RERANKED ONE. The floor's job
    is to define WHICH DETECTIONS EXIST; ranking among the survivors is a separate question,
    and AP is invariant to any monotone rescaling of the ranking key. Flooring the reranked
    key instead would penalise every multiplicative rerank purely for living on a smaller
    scale -- `det x sam` is a product of two sub-1 numbers and sinks ~30k detections under
    0.05 that `det` alone keeps, which would be an artefact of arithmetic, not of quality.

    This is exactly the convention the published LACE product row uses (it was scored at
    `score_floor: 0.0`, with `s >= 0.05` already true for all 159,687 detections), so doing
    it explicitly here puts LACE and SAM under the identical rule. Callers must therefore
    pass `score_floor=0.0` to score_coco for these rows -- the filtering already happened."""
    out, dropped, total = {}, 0, 0
    for t, r in preds.items():
        det = np.asarray(r["scores"], np.float32)
        sam = np.asarray(r["sam_scores"], np.float32)
        sc = {"det": det, "sam": sam, "det_x_sam": det * sam}[key]
        keep = np.flatnonzero(det >= SC.SCORE_FLOOR)      # floor on the DETECTOR score
        total += len(det)
        dropped += len(det) - len(keep)
        out[t] = {"boxes_2048": np.asarray(r["boxes_2048"], np.float32)
                              .reshape(-1, 4)[keep].tolist(),
                  "scores": sc[keep].tolist(),
                  "masks_rle": [r["masks_rle"][i] for i in keep]}
    print(f"[bakeoff] rank={key}: {total - dropped}/{total} dets survive the "
          f"{SC.SCORE_FLOOR} detector-score floor", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_preds", default=SAM_PREDS)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "box2mask_bakeoff.json"))
    a = ap.parse_args()

    sam, sam_meta = _load(a.sam_preds)
    lace, _ = _load(LACE_DET)
    prod, _ = _load(LACE_PROD)
    assert_same_boxes(sam, lace, "SAM 3 preds", "LACE knobbed_s0")
    assert_same_boxes(prod, lace, "LACE product", "LACE knobbed_s0", scores=False)

    # LACE's product file stores the reranked score in `scores`, so graft the raw detector
    # score back in and let `rescored` apply the one floor rule to every row alike.
    lace_src = {t: dict(prod[t], sam_scores=prod[t]["scores"], scores=lace[t]["scores"])
                for t in prod}

    rows = [("LACE, detector score", lace_src, "det"),
            ("LACE, posterior product (the system)", lace_src, "sam"),
            ("SAM 3, detector score", sam, "det"),
            ("SAM 3, own score", sam, "sam"),
            ("SAM 3, det x SAM score", sam, "det_x_sam")]

    # Every row is written out and scored through the SAME entry point (score_coco.score_many)
    # -- no second scoring path -- at score_floor=0.0, because `rescored` has already applied
    # the 0.05 floor to the detector score for all of them.
    tmp = []
    for i, (label, src, key) in enumerate(rows):
        fp = os.path.join(HERE, f"_row{i}.json")
        json.dump({"meta": {"model": label}, "preds": rescored(src, key)}, open(fp, "w"))
        tmp.append(fp)
    results = SC.score_many(tmp, GT, res=a.res, score_floor=0.0)
    rows = [(label, fp) for (label, _, _), fp in zip(rows, tmp)]

    table = []
    for (label, fp), r in zip(rows, results):
        cn = r["canopy_neutral_crowd"]["mask"]
        table.append({"row": label, "preds_file": os.path.relpath(fp),
                      "CN_AP50": cn["AP50"], "CN_AP75": cn["AP75"],
                      "CN_AP50_95": cn["AP50_95"], "n_det": cn["n_det"]})

    out = {"protocol": results[0]["protocol"],
           "boxes": "seed-0 LACE detector, bit-identical across every row",
           "sam_meta": sam_meta, "table": table,
           "rows_full": {t["row"]: r for t, r in zip(table, results)}}
    for r in out["rows_full"].values():
        r.pop("areas", None)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    w = max(len(t["row"]) for t in table)
    print(f"\n{'row'.ljust(w)} | CN AP50 | CN AP75 | CN AP50:95 | dets")
    print(f"{'-'*w}-+---------+---------+------------+------")
    for t in table:
        print(f"{t['row'].ljust(w)} |  {t['CN_AP50']:.4f} |  {t['CN_AP75']:.4f} |     "
              f"{t['CN_AP50_95']:.4f} | {t['n_det']}")
    print(f"\n-> {a.out}", flush=True)
    for fp in tmp:
        os.remove(fp)


if __name__ == "__main__":
    main()

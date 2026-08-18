"""Score ANY predictions in our schema over the sparse slice's cuts — one scorer, all models.

Both our pipeline and DetecTree2 serialise predictions the same way (`boxes_2048` +
`masks_rle`@512 + `scores`), so the identical scorer grades them: evaluate._greedy_ap
(COCO-101pt) + the >50%-in-canopy ignore rule + phase4_lib._instance_pr. That is the ONLY
way the numbers are comparable to the 0.630 / 0.545 records.

Streams tile-by-tile, keeping just the (Npred,Ngt) IoU matrices and discarding the masks —
DetecTree2 emits ~80k masks over 236 tiles (~20 GB if held), which is why score_detectree2.py
uses the same pattern. Cuts are then pure index selections over those small matrices, so all
subsets cost one pass.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.compare_subsets \
        --preds <preds.json> [--preds <seed1.json> ...] --label "DetecTree2" --out <out.json>
"""
import argparse
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from dapt.eval import iou_matrix
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.tile_index import BIOME, REPO

GT = os.path.join(REPO, "data/tcd_sparse/sparse_gt.json")
_HERE = os.path.dirname(os.path.abspath(__file__))
# the tracked copy first: data/** is gitignored, so only this one survives a fresh clone
SLICE = (os.path.join(_HERE, "slice_manifest.json")
         if os.path.exists(os.path.join(_HERE, "slice_manifest.json"))
         else os.path.join(REPO, "data/tcd_sparse/slice_manifest.json"))


def _decode(rle):
    c = rle["counts"]
    return maskUtils.decode({"size": rle["size"],
                             "counts": c.encode("ascii") if isinstance(c, str) else c}
                            ).astype(bool)


def per_tile(preds_path, gt):
    """{tid: (mask_iou, box_iou, scores, ign_mask, ign_box, n_gt)} — masks freed per tile."""
    P = json.load(open(preds_path))
    preds = P["preds"]
    out = {}
    # EVERY GT tile is scored. A tile absent from `preds` is a tile the model produced
    # nothing for -- that is zero recall on its crowns, not a tile to skip. Dropping it
    # would silently shrink the GT denominator and flatter a partial prediction run.
    tids = sorted(gt)
    missing = [t for t in tids if t not in preds]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(tids)} GT tiles have NO predictions; "
              f"scored as zero-prediction (recall penalised): {missing[:5]}", flush=True)
    empty = {"masks_rle": [], "scores": [], "boxes_2048": []}
    for k, tid in enumerate(tids):
        rec = preds.get(tid, empty)
        pm = (np.stack([_decode(r) for r in rec["masks_rle"]])
              if rec["masks_rle"] else np.zeros((0, E.RES, E.RES), bool))
        sc = np.array(rec["scores"], np.float32)
        pbox = np.array(rec["boxes_2048"], np.float32).reshape(-1, 4) / E.SCALE
        gm = np.array(E.raster(gt[tid]["trees"]))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        gbox = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                           *(np.asarray(t).reshape(-1, 2).max(0))]
                          for t in gt[tid]["trees"]], np.float32).reshape(-1, 4) / E.SCALE
                if gt[tid]["trees"] else np.zeros((0, 4), np.float32))
        ign_m = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5
                          for m in pm]) if len(pm) else np.zeros(0, bool)
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            sub = can[slice(max(0, int(y0)), int(np.ceil(y1))),
                      slice(max(0, int(x0)), int(np.ceil(x1)))]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5
        out[tid] = (E.mask_iou(pm, gm), iou_matrix(pbox, gbox), sc, ign_m, ign_b, len(gm))
        del pm, gm, can
        if (k + 1) % 50 == 0 or k + 1 == len(tids):
            print(f"  {os.path.basename(os.path.dirname(preds_path))} {k+1}/{len(tids)}",
                  flush=True)
    return out, P.get("meta", {})


def score(per, tids):
    tids = [t for t in tids if t in per]
    if not tids:
        return None
    mI = [per[t][0] for t in tids]; bI = [per[t][1] for t in tids]
    sc = [per[t][2] for t in tids]
    im = [per[t][3] for t in tids]; ib = [per[t][4] for t in tids]
    n = sum(per[t][5] for t in tids)
    return {
        "n_tiles": len(tids), "n_gt": n,
        "mask_mAP50": round(E._greedy_ap(mI, sc, im, n, 0.5), 4),
        "mask_mAP50_95": round(float(np.nanmean(
            [E._greedy_ap(mI, sc, im, n, t) for t in E.IOU_50_95])), 4),
        "box_mAP50": round(E._greedy_ap(bI, sc, ib, n, 0.5), 4),
        "box_mAP40": round(E._greedy_ap(bI, sc, ib, n, 0.4), 4),
        "n_pred": int(sum(len(s) for s in sc)),
    }


def cuts_of(sm, tids):
    c = {"all": list(tids),
         "scene_clean": [t for t in tids if sm[t]["scene_clean"]],
         "train_pool": [t for t in tids if sm[t]["source"] == "train_pool"],
         "scene_clean_and_train_pool": [t for t in tids if sm[t]["scene_clean"]
                                        and sm[t]["source"] == "train_pool"]}
    for b in sorted({sm[t]["biome"] for t in tids}):
        c[f"biome_{b}"] = [t for t in tids if sm[t]["biome"] == b]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="repeat for a multi-seed band")
    ap.add_argument("--label", required=True)
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--slice", dest="slice_manifest", default=SLICE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    sm = json.load(open(a.slice_manifest))["tiles"]
    runs, metas = [], []
    for p in a.preds:
        per, meta = per_tile(p, gt)
        metas.append(meta)
        runs.append({k: score(per, v) for k, v in cuts_of(sm, sorted(per)).items()})
        del per

    band = {}
    for cut in runs[0]:
        vals = [r[cut] for r in runs if r[cut]]
        if vals:
            band[cut] = {"n_tiles": vals[0]["n_tiles"], "n_gt": vals[0]["n_gt"], **{
                m: [round(float(np.mean([v[m] for v in vals])), 4),
                    round(float(np.std([v[m] for v in vals])), 4)]
                for m in ("mask_mAP50", "mask_mAP50_95", "box_mAP50", "box_mAP40")}}

    print(f"\n=== {a.label} ({len(runs)} run{'s' if len(runs) > 1 else ''}) ===")
    print(f"{'cut':<32}{'n':>5}{'n_gt':>7}{'mask50':>18}{'mask50-95':>17}{'box50':>18}")
    for k, v in band.items():
        lab = k if not k.startswith("biome_") else \
            f"{k} {BIOME[int(k.split('_')[1])][:18]}"
        print(f"{lab:<32}{v['n_tiles']:>5}{v['n_gt']:>7}"
              f"{v['mask_mAP50'][0]:>11.4f} +-{v['mask_mAP50'][1]:.4f}"
              f"{v['mask_mAP50_95'][0]:>10.4f} +-{v['mask_mAP50_95'][1]:.4f}"
              f"{v['box_mAP50'][0]:>11.4f} +-{v['box_mAP50'][1]:.4f}")
    out = {"label": a.label, "preds": a.preds, "pred_meta": metas,
           "per_run": runs, "band_mean_std": band}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")
    return out


if __name__ == "__main__":
    main()

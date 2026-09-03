"""How much of the OURS-vs-SelvaMask gap is AP *aggregation* rather than detection quality?

Our published 439 numbers are POOLED AP: per-tile greedy matching, then every tile's
TP/FP flags concatenated into ONE globally score-sorted PR curve against a global
n_gt (25,705). SelvaMask/CanopyRS instead compute AP INSIDE each subtile and average
those APs over tiles. Aggregation sits below the IoU type and below the IoU sweep, so
the choice moves mask AP and box AP, AP50 and AP50-95, identically -- it is not a
box-only footnote.

This script isolates the aggregation term (and two protocol terms that interact with it)
by rescoring the IDENTICAL saved predictions under a full cross of:

    aggregation  pooled | per_tile        (per_tile = mean of per-tile APs, NaN-skip)
    tiles        all439 | nonempty        (nonempty drops the 73 zero-GT tiles)
    ignore       on | off                 (the >50%-in-canopy ignore rule)
    maxdets      all | 600 | 400          (uncapped, our detector's topk, SelvaMask's cap)

plus a `per_tile_empty0` variant that scores a zero-GT tile as AP=0 when it emitted any
prediction, instead of dropping it -- the strict reading of per-tile averaging.

SENSITIVITY ONLY. The published numbers remain pooled / all439 / ignore-on / uncapped. This
does NOT reproduce SelvaMask's protocol: their 1024px @0.5-overlap subtiling and their
>80%-black tile drop are not simulated here, so a residual tiling term survives.

A reproduction gate checks the pooled/all439/ignore-on/uncapped cell against each run's
recorded mask AND box figures before any other cell is reported, and records the delta.
IoU here is the EXACT integer RLE IoU rather than evaluate.mask_iou's float32 SGEMM, which
is knife-edge on tied IoUs and not even stable run to run (see per_tile) -- so the gate
measures a small documented drift instead of asserting equality. Every cell shares the one
IoU basis, so the deltas BETWEEN cells, which is the actual measurement, are clean.

Usage (from repo root, with the repo venv):
    .venv/bin/python -m boxinst_commonality_tcd_04.aggregation_sensitivity.aggregation_sensitivity
"""
import argparse
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E
from dapt.eval import iou_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
GT = os.path.join(PKG, "test_gt.json")
P4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "preds")
DT2 = os.path.join(PKG, "detectree2_baseline")
OUT = os.path.join(HERE, "results_aggregation.json")

# None = uncapped, the published protocol for BOTH runs: our detector emits topk 600 per
# tile so 600 is a no-op for it, but DetecTree2 has 80 tiles above 600 preds (max 968), so
# capping at 600 would silently restate its published figures.
MAXDETS = (None, 600, 400)
AGGS = ("pooled", "per_tile", "per_tile_empty0")
TILESETS = ("all439", "nonempty")


def _mdkey(md):
    return "mdall" if md is None else f"md{md}"

# (label, preds path, recorded-reference path or None)
RUNS = [(f"ours_s{s}", os.path.join(P4, f"knobbed_s{s}.json"),
         os.path.join(P4, f"ref_knobbed_s{s}.json")) for s in (0, 1, 2)] + \
       [("detectree2_s0", os.path.join(DT2, "preds_dt2_s0v2.json"),
         os.path.join(DT2, "results_dt2_s0v2.json"))]


def _rle(r):
    """Saved RLE dict -> pycocotools RLE (bytes counts)."""
    c = r["counts"]
    return {"size": r["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}


def _enc(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def per_tile(preds_path, gt):
    """Stream tiles -> per-tile (mask IoU, box IoU, scores, mask-ignore, box-ignore, n_gt).

    Everything downstream works off the IoU matrices only, so nothing here is ever held at
    full resolution: ~160k predicted 512x512 masks would be ~40 GB.

    Every RULE is copied from the recorded scoring path (phase4_lib_tcd.eval_4p_selfmask ->
    _full_metrics): boxes divided by E.SCALE, GT boxes as polygon extents, mask-ignore =
    >50% of the pred mask inside canopy, box-ignore = canopy mean over the box crop > 0.5.

    The ARITHMETIC deliberately differs in one place. `evaluate.mask_iou` computes the
    intersection with a float32 SGEMM; on Apple Accelerate that path raises overflow/invalid
    warnings and lands up to 3e-8 off the true value. Mask IoU is a ratio of small integers,
    so exact ties with the sweep thresholds (0.65, 0.75, ...) are common, and `iou >= thr` is
    knife-edge there: ~150 predictions per 250 tiles flip. That moves pooled AP by up to
    0.007 at IoU 0.75 and ~0.001 on AP50-95 -- and it is not even stable run to run, since
    the SGEMM's last ulp depends on the process's allocation state.

    A protocol study cannot sit on an unstable scorer, so IoU here is the exact integer RLE
    IoU (pycocotools, C integer arithmetic). Every cell of the cross shares that one basis,
    so the deltas between cells -- the actual measurement -- are clean. The cost is that the
    published cell reproduces the recorded figure only to ~1e-3 at IoU >= 0.65; `gate()`
    measures that gap rather than assuming it.
    """
    preds = json.load(open(preds_path))["preds"]
    T = []
    for k, tid in enumerate(sorted(gt)):
        p = preds[tid]
        prle = [_rle(r) for r in p["masks_rle"]]
        sc = np.array(p["scores"], np.float32)
        pbox = np.array(p["boxes_2048"], np.float64).reshape(-1, 4) / E.SCALE
        gm = np.array(E.raster(gt[tid]["trees"]))
        grle = [_enc(g) for g in gm]
        gbox = (np.array([[*(np.asarray(t).reshape(-1, 2).min(0)),
                           *(np.asarray(t).reshape(-1, 2).max(0))]
                          for t in gt[tid]["trees"]], np.float64).reshape(-1, 4) / E.SCALE
                if gt[tid]["trees"] else np.zeros((0, 4), np.float64))
        can = np.array(E.raster(gt[tid]["canopy"]))
        can = can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)
        canrle = _enc(can)

        mI = (np.asarray(maskUtils.iou(prle, grle, [0] * len(grle))).reshape(len(prle),
                                                                            len(grle))
              if prle and grle else np.zeros((len(prle), len(grle))))
        # mask-ignore, same rule as upstream but on exact RLE areas rather than dense ops
        ign_m = np.zeros(len(prle), bool)
        for i, r in enumerate(prle):
            a = int(maskUtils.area(r))
            ign_m[i] = a > 0 and int(maskUtils.area(maskUtils.merge([r, canrle], 1))) / a > 0.5
        ign_b = np.zeros(len(pbox), bool)
        for i, (x0, y0, x1, y1) in enumerate(pbox):
            sub = can[slice(max(0, int(y0)), int(np.ceil(y1))),
                      slice(max(0, int(x0)), int(np.ceil(x1)))]
            ign_b[i] = sub.size > 0 and sub.mean() > 0.5

        T.append({"tid": tid, "mI": mI, "bI": iou_matrix(pbox, gbox),
                  "sc": sc, "ign_m": ign_m, "ign_b": ign_b, "n_gt": len(gm)})
        del gm, can, grle, prle
        if (k + 1) % 100 == 0 or k + 1 == len(gt):
            print(f"    {k + 1}/{len(gt)}", flush=True)
    return T


def _topk(t, key, maxdets):
    """Per-tile detection cap, applied by descending score like every COCO maxDets."""
    iou, sc = t[key], t["sc"]
    ign = t["ign_m"] if key == "mI" else t["ign_b"]
    if maxdets is None or len(sc) <= maxdets:
        return iou, sc, ign
    keep = np.argsort(-sc)[:maxdets]
    return iou[keep], sc[keep], ign[keep]


def match(T, key, iou_thr, maxdets):
    """Greedy one-to-one matching, byte-identical in logic to evaluate._greedy_ap's inner
    loop. Split out because matching depends ONLY on (IoU, score order, iou_thr, maxdets) --
    not on aggregation or tile subset -- so one pass feeds every cell of the cross.

    Note the ignore rule is applied INSIDE the loop exactly as upstream: an ignored pred is
    dropped only when it failed to match, so it never steals a GT from a later prediction.
    Returns per-tile (kept scores, kept tp flags, n_gt, n_kept_preds).
    """
    out = []
    for t in T:
        iou, sc, ign = _topk(t, key, maxdets)
        if len(sc) == 0:
            out.append((np.zeros(0, np.float32), np.zeros(0, bool),
                        np.zeros(0, bool), t["n_gt"]))
            continue
        _, sc, tp, keep, _ = E.match_tile(iou, sc, ign, iou_thr)   # THE shared matcher
        drop = ~keep                    # unmatched + in canopy -> neither TP nor FP
        out.append((sc, tp, drop, t["n_gt"]))
    return out


def _ap101(scores, tp, n_gt):
    """COCO 101-point interpolated AP, identical to evaluate._greedy_ap's tail."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores)
    tp = tp[o]
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n_gt, tpc / (tpc + fpc + 1e-9)
    return float(sum((prec[rec >= r].max() if np.any(rec >= r) else 0.0)
                     for r in np.linspace(0, 1, 101)) / 101)


def aggregate(matched, agg, tileset, ignore):
    """Reduce one matching pass to a single AP under (aggregation, tile subset, ignore rule).

    `ignore=False` forces every ignore flag off, so a canopy prediction becomes a plain FP.
    Because ignore never affected the matching itself, dropping the flag is exactly a
    re-inclusion of those predictions as false positives -- no rematch needed.
    """
    sel = [m for m in matched if not (tileset == "nonempty" and m[3] == 0)]

    def keep(sc, tp, drop):
        k = ~drop if ignore else np.ones(len(sc), bool)
        return sc[k], tp[k]

    if agg == "pooled":
        n_gt = sum(m[3] for m in sel)
        parts = [keep(sc, tp, dr) for sc, tp, dr, _ in sel if len(sc)]
        if n_gt == 0:
            return float("nan")
        if not parts:
            return 0.0
        return _ap101(np.concatenate([p[0] for p in parts]),
                      np.concatenate([p[1] for p in parts]), n_gt)

    aps = []
    for sc, tp, dr, n_gt in sel:
        s, t = keep(sc, tp, dr)
        if n_gt == 0:
            # NaN-skip is what per-tile averaging does by construction: a zero-GT tile has
            # no defined AP, so its false positives are UNCHARGEABLE. `per_tile_empty0`
            # charges them the harshest defensible way instead -- AP=0 if it fired at all.
            if agg == "per_tile_empty0" and len(s):
                aps.append(0.0)
            continue
        aps.append(_ap101(s, t, n_gt))
    return float(np.nanmean(aps)) if aps else float("nan")


def score_run(T):
    """Every cell of the cross for one run: {metric: {cell_key: {AP50, AP50_95, per_iou}}}."""
    res = {}
    for metric, key in (("mask", "mI"), ("box", "bI")):
        cells = {}
        for md in MAXDETS:
            passes = {float(t): match(T, key, float(t), md) for t in E.IOU_50_95}
            print(f"    [{metric} maxdets={_mdkey(md)}] matched {len(passes)} IoU "
                  f"thresholds",
                  flush=True)
            for agg in AGGS:
                for ts in TILESETS:
                    for ig in (True, False):
                        per_iou = [aggregate(passes[float(t)], agg, ts, ig)
                                   for t in E.IOU_50_95]
                        cells[f"{agg}|{ts}|{'ign' if ig else 'noign'}|{_mdkey(md)}"] = {
                            "AP50": round(float(per_iou[0]), 4),
                            "AP50_95": round(float(np.nanmean(per_iou)), 4),
                            "per_iou": [round(float(x), 4) for x in per_iou]}
        res[metric] = cells
    return res


PUB = "pooled|all439|ign|mdall"          # the published protocol cell


GATE_TOL = 0.005          # see gate(): exact-IoU basis + 2dp saved boxes, both sub-1e-2


def gate(label, res, ref_path):
    """Measure -- not assume -- how close the published cell lands to the recorded figure.

    Two known, documented sources of drift stop this from being an equality check:
      * IoU basis. We score on exact integer RLE IoU; the recorded runs used the float32
        SGEMM in `evaluate.mask_iou`, which ties-break differently at IoU >= 0.65 (see
        per_tile). Affects mask AP only.
      * Box precision. Saved boxes were rounded to 2dp at dump time (`np.round(bx, 2)`)
        while the recorded eval scored the unrounded tensors. Affects box AP only.
    Anything beyond GATE_TOL is a real scoring-path bug, not drift, and hard-fails.
    """
    ref = json.load(open(ref_path))
    checks = [("mask", "AP50", "mask_mAP50"), ("mask", "AP50_95", "mask_mAP50_95"),
              ("box", "AP50", "box_mAP50"), ("box", "AP50_95", "box_mAP50_95")]
    report = {}
    for metric, field, refkey in checks:
        if refkey not in ref:
            print(f"  gate: {label} has no recorded {refkey} -- SKIPPED", flush=True)
            continue
        got, want = res[metric][PUB][field], ref[refkey]
        report[refkey] = {"recomputed": got, "recorded": want, "delta": round(got - want, 4)}
        assert abs(got - want) < GATE_TOL, \
            f"GATE FAILED {label} {refkey}: recomputed {got} vs recorded {want}"
    line = "  ".join(f"{k} {v['recomputed']} vs {v['recorded']} ({v['delta']:+.4f})"
                     for k, v in report.items())
    print(f"  gate OK ({label}) {line}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-runs", type=int, default=None)
    a = ap.parse_args()

    gt = json.load(open(GT))
    n_empty = sum(1 for t in gt if not gt[t]["trees"])
    print(f"[gt] {len(gt)} tiles, {n_empty} with zero GT crowns", flush=True)

    runs = {}
    for label, pp, rp in RUNS[:a.limit_runs]:
        print(f"[{label}] {os.path.basename(pp)}", flush=True)
        T = per_tile(pp, gt)
        n_pred = int(sum(len(t["sc"]) for t in T))
        empty_pred = int(sum(len(t["sc"]) for t in T if t["n_gt"] == 0))
        res = score_run(T)
        rep = gate(label, res, rp)
        # sanity: dropping the ignore rule can only add FPs, never help
        for metric in ("mask", "box"):
            for k, v in res[metric].items():
                if "|ign|" not in k:
                    continue
                nk = k.replace("|ign|", "|noign|")
                assert res[metric][nk]["AP50_95"] <= v["AP50_95"] + 1e-9, \
                    f"{label} {metric} {k}: no-ignore beat with-ignore -- impossible"
        # per-tile averaging cannot see zero-GT tiles: all439 == nonempty by construction
        for metric in ("mask", "box"):
            for md in MAXDETS:
                for ig in ("ign", "noign"):
                    x = res[metric][f"per_tile|all439|{ig}|{_mdkey(md)}"]
                    y = res[metric][f"per_tile|nonempty|{ig}|{_mdkey(md)}"]
                    assert x == y, f"{label} {metric}: per_tile all439 != nonempty"
        runs[label] = {"preds": os.path.relpath(pp, os.path.dirname(PKG)),
                       "n_pred": n_pred, "n_pred_in_empty_tiles": empty_pred,
                       "reproduction_vs_recorded": rep, "cells": res}
        print(f"  {n_pred} preds, {empty_pred} of them in zero-GT tiles "
              f"({100 * empty_pred / n_pred:.1f}%)", flush=True)
        del T

    ours = [k for k in runs if k.startswith("ours_")]
    band = {}
    if ours:
        for metric in ("mask", "box"):
            band[metric] = {}
            for cell in runs[ours[0]]["cells"][metric]:
                for f in ("AP50", "AP50_95"):
                    v = [runs[k]["cells"][metric][cell][f] for k in ours]
                    band[metric].setdefault(cell, {})[f] = [
                        round(float(np.mean(v)), 4),
                        round(float(np.std(v, ddof=1)), 4)]   # ddof=1: repo seed-band convention

    out = {"label": "AP aggregation sensitivity, OAM-TCD 439",
           "note": ("SENSITIVITY ONLY. Published numbers are the "
                    f"'{PUB}' cell. Cell key = aggregation|tileset|ignore|maxdets. "
                    "per_tile = mean of per-tile APs (zero-GT tiles NaN-skipped, so their "
                    "false positives are uncharged); per_tile_empty0 = same but a zero-GT "
                    "tile that fired scores 0. This does NOT reproduce SelvaMask's protocol: "
                    "their 1024px @0.5-overlap subtiling and >80%-black tile drop are not "
                    "simulated, so a residual tiling term survives."),
           "n_tiles": len(gt), "n_empty_tiles": n_empty,
           "axes": {"aggregation": list(AGGS), "tileset": list(TILESETS),
                    "ignore": ["ign", "noign"],
                    "maxdets": [_mdkey(m) for m in MAXDETS]},
           "per_run": runs, "ours_band_mean_std": band}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n-> {os.path.relpath(a.out, os.path.dirname(PKG))}", flush=True)


if __name__ == "__main__":
    main()

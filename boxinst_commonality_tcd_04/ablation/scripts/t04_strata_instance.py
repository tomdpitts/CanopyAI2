"""T0.4 -- Does commonality break down when the box is NOT mostly one crown?

THE ASSUMPTION UNDER TEST
    LACE models foreground/background commonality WITHIN a predicted box. That presumes the
    box contains mostly one crown against separable background. Where crowns interlock, a
    box contains a neighbour's canopy that looks exactly like the target -- the commonality
    signal should degrade. This is the method's most obvious failure mode and the one a
    reviewer will name.

DEFINITION (ported verbatim from masker_lab/failure_analysis.py:106)
    touching = this GT box has box-IoU > 0 with at least one OTHER GT box.

WHAT COULD AND COULD NOT BE GATED
    The plan intended to gate this port against failure_analysis_results.json's recorded
    group means. That turns out to be impossible and the reason is worth recording: that
    run used a DIFFERENT model and cohort (200 tiles, 16px native-4096 beta=0 masker, live
    det_t8 detector, box-IoU matching), so its 0.316/0.4614 touching profile is not
    reproducible from the deployed 8px knobbed predictions. What IS gated:
      - the AP re-score reproduces the published per-seed numbers (lib/gate.py)
      - box_iou_mat agrees exactly with a brute-force reference (see --selftest)
      - the GT-level touching rate is reported, which is model-independent
    Faking a gate against an incomparable number would be worse than saying this.

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t04_strata_instance
    .venv/bin/python -m ...t04_strata_instance --selftest
"""
import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04 import evaluate as E
from ..lib import boundary as B
from ..lib import gate, io


def gt_boxes(polys):
    """Tight box per GT polygon, in 2048 tile coords. Ported from failure_analysis.py:38."""
    if not polys:
        return np.zeros((0, 4), np.float32)
    return np.array([[*np.asarray(p, float).reshape(-1, 2).min(0),
                      *np.asarray(p, float).reshape(-1, 2).max(0)] for p in polys], np.float32)


def box_iou_mat(a, b):
    """Ported from failure_analysis.py:41. --selftest checks it against brute force."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, 0][:, None], b[:, 0][None]); iy0 = np.maximum(a[:, 1][:, None], b[:, 1][None])
    ix1 = np.minimum(a[:, 2][:, None], b[:, 2][None]); iy1 = np.minimum(a[:, 3][:, None], b[:, 3][None])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def touching_flags(gb):
    """GT box overlaps another GT box. failure_analysis.py:106."""
    if len(gb) == 0:
        return np.zeros(0, bool)
    return (box_iou_mat(gb, gb) > 0).sum(1) > 1


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 8)
        a = rng.uniform(0, 100, (n, 2)); w = rng.uniform(1, 40, (n, 2))
        bx = np.hstack([a, a + w]).astype(np.float32)
        M = box_iou_mat(bx, bx)
        for i in range(n):
            for j in range(n):
                ix = max(0.0, min(bx[i, 2], bx[j, 2]) - max(bx[i, 0], bx[j, 0]))
                iy = max(0.0, min(bx[i, 3], bx[j, 3]) - max(bx[i, 1], bx[j, 1]))
                inter = ix * iy
                ai = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1])
                aj = (bx[j, 2] - bx[j, 0]) * (bx[j, 3] - bx[j, 1])
                assert abs(M[i, j] - inter / (ai + aj - inter + 1e-9)) < 1e-6
    print("selftest OK: box_iou_mat == brute force on 200 random configurations")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--selftest", action="store_true")
    ap_.add_argument("--out", default=os.path.join(io.RESULTS, "strata_instance.json"))
    a = ap_.parse_args()
    if a.selftest:
        return selftest()
    selftest()
    seeds = [int(s) for s in a.seeds.split(",")]
    gt = io.load_gt()

    # model-independent: how crowded is the GT itself?
    gtouch = {t: touching_flags(gt_boxes(gt[t]["trees"])) for t in sorted(gt)}
    n_gt_all = sum(len(v) for v in gtouch.values())
    gt_rate = float(np.mean(np.concatenate([v for v in gtouch.values() if len(v)])))

    per_seed = {}
    for s in seeds:
        print(f"[s{s}]", flush=True)
        rows, box_rows = [], []
        mI, S, IGN, n_gt = [], [], [], 0
        for tl in io.per_tile(io.knobbed(s), gt, progress=200):
            iou = E.mask_iou(tl["pm"], tl["gm"])
            mI.append(iou); S.append(tl["scores"]); IGN.append(tl["ignore"])
            n_gt += len(tl["gm"])
            tf = gtouch[tl["tid"]]
            gb = gt_boxes(gt[tl["tid"]]["trees"])
            # BOX-IoU matching, as failure_analysis.py:91-98 does. This is the arm that can
            # SEE a mask failure: a crown whose box was found but whose mask missed. Matching
            # on mask IoU (below) defines mask-IoU>=0.5 into existence and so reports a mask
            # fail rate of exactly zero -- true but vacuous.
            bmat = box_iou_mat(tl["boxes"], gb)
            for pi, gj in B.match_tps(bmat, tl["scores"], tl["ignore"], 0.5):
                box_rows.append({"tid": tl["tid"],
                                 "mask_iou": round(float(iou[pi, gj]), 4),
                                 "box_iou": round(float(bmat[pi, gj]), 4),
                                 "touching": bool(tf[gj]),
                                 "gt_size_px": round(float(np.sqrt(
                                     max(gb[gj, 2] - gb[gj, 0], 1) *
                                     max(gb[gj, 3] - gb[gj, 1], 1))), 2)})
            for pi, gj in B.match_tps(iou, tl["scores"], tl["ignore"], 0.5):
                gsum = float(tl["gm"][gj].sum()); psum = float(tl["pm"][pi].sum())
                # float() matters: gb is float32, and np.float32 survives round() and then
                # fails json.dump
                ga = float(max(float((gb[gj, 2] - gb[gj, 0]) * (gb[gj, 3] - gb[gj, 1])), 1.0)
                           / E.SCALE ** 2)
                pb = tl["boxes"][pi]
                rows.append({
                    "tid": tl["tid"], "mask_iou": round(float(iou[pi, gj]), 4),
                    "touching": bool(tf[gj]),
                    "gt_size_px": round(float(np.sqrt(max(gb[gj, 2] - gb[gj, 0], 1) *
                                                      max(gb[gj, 3] - gb[gj, 1], 1))), 2),
                    "box_oversize": round(float(float((pb[2]-pb[0])*(pb[3]-pb[1]))/E.SCALE**2 / ga), 4),
                    "gt_fill": round(float(gsum / ga), 4),
                    "cover_ratio": round(float(psum / (gsum + 1e-9)), 4),
                    "score": round(float(tl["scores"][pi]), 4)})
        ap = gate.score_ap(mI, S, IGN, n_gt)
        gate.check(ap, io.ref_knobbed(s), label=f"s{s}")

        arr = {k: np.array([r[k] for r in rows]) for k in
               ("mask_iou", "gt_size_px", "box_oversize", "gt_fill", "cover_ratio")}
        tch = np.array([r["touching"] for r in rows])
        grp = {}
        for lab, sel in (("touching", tch), ("isolated", ~tch)):
            if sel.sum() == 0:
                continue
            grp[lab] = {"n": int(sel.sum()),
                        "frac_of_tps": round(float(sel.mean()), 4),
                        "mean_mask_iou": round(float(arr["mask_iou"][sel].mean()), 4),
                        "median_mask_iou": round(float(np.median(arr["mask_iou"][sel])), 4),
                        "mask_fail_rate_iou_lt_0.5": round(float((arr["mask_iou"][sel] < 0.5).mean()), 4),
                        **{f"mean_{k}": round(float(arr[k][sel].mean()), 4)
                           for k in ("gt_size_px", "box_oversize", "gt_fill", "cover_ratio")}}
        # the informative arm: box-TP crowns, split by touching, with a real mask-fail rate
        bt = np.array([r["touching"] for r in box_rows])
        bmi = np.array([r["mask_iou"] for r in box_rows])
        bgrp = {}
        for lab, sel in (("touching", bt), ("isolated", ~bt)):
            if sel.sum():
                bgrp[lab] = {
                    "n": int(sel.sum()),
                    "frac_of_box_tps": round(float(sel.mean()), 4),
                    "mean_mask_iou": round(float(bmi[sel].mean()), 4),
                    "mask_fail_rate_iou_lt_0.5": round(float((bmi[sel] < 0.5).mean()), 4),
                    "mean_gt_size_px": round(float(np.mean(
                        [r["gt_size_px"] for r, k in zip(box_rows, sel) if k])), 2)}
        per_seed[s] = {"n_matched_tp": len(rows), "n_box_tp": len(box_rows), "ap": ap,
                       "by_touching_boxmatched": bgrp,
                       "overall_mask_fail_rate_among_box_tps": round(float((bmi < 0.5).mean()), 4),
                       "by_touching": grp,
                       "delta_mean_mask_iou_isolated_minus_touching": round(
                           grp["isolated"]["mean_mask_iou"] - grp["touching"]["mean_mask_iou"], 4)
                       if {"isolated", "touching"} <= set(grp) else None}
        # per-instance table, one seed only -- 20k rows x 3 seeds is not worth the file size
        if s == seeds[0]:
            json.dump({"note": "per-instance TP table, seed %d only" % s, "rows": rows},
                      open(os.path.join(io.RESULTS, f"strata_instance_rows_s{s}.json"), "w"))

    band, bband = {}, {}
    for lab in ("touching", "isolated"):
        ks = [s for s in seeds if lab in per_seed[s]["by_touching"]]
        if ks:
            band[lab] = {m: io.band([per_seed[s]["by_touching"][lab][m] for s in ks])
                         for m in ("mean_mask_iou", "mask_fail_rate_iou_lt_0.5", "frac_of_tps")}
        kb = [s for s in seeds if lab in per_seed[s]["by_touching_boxmatched"]]
        if kb:
            bband[lab] = {m: io.band([per_seed[s]["by_touching_boxmatched"][lab][m] for s in kb])
                          for m in ("mean_mask_iou", "mask_fail_rate_iou_lt_0.5",
                                    "frac_of_box_tps")}

    out = {"label": "Instance-level crown-overlap stratification, 439 OAM-TCD, knobbed, 3 seeds",
           "definition": "touching = GT box has box-IoU > 0 with another GT box "
                         "(masker_lab/failure_analysis.py:106)",
           "gating": {"ap": "reproduces published per-seed mask AP (lib/gate.py)",
                      "box_iou_mat": "exact vs brute force, 200 random configs",
                      "not_gated": ("failure_analysis_results.json's 0.316/0.4614 profile is "
                                    "from a different model and cohort (200 tiles, 16px "
                                    "native-4096 beta=0, live det_t8, box-IoU matching) and is "
                                    "not reproducible from the deployed 8px knobbed preds")},
           "gt_level": {"n_gt_crowns": n_gt_all, "touching_rate": round(gt_rate, 4),
                        "note": "model-independent property of the OAM-TCD test annotations"},
           "matching_arms": {
               "boxmatched": ("preds matched to GT by BOX IoU>=0.5, then mask IoU measured. "
                              "This is failure_analysis.py's convention and the only arm that "
                              "can observe a mask failure. USE THIS ONE."),
               "maskmatched": ("preds matched by MASK IoU>=0.5. Its mask-fail rate is zero by "
                               "construction and is reported only for completeness; its mean "
                               "mask IoU is conditioned on the mask already having succeeded.")},
           "band_mean_std_boxmatched": bband,
           "band_mean_std_maskmatched": band, "per_seed": per_seed}
    json.dump(out, open(a.out, "w"), indent=2)
    io.record_inputs([io.GT] + [io.knobbed(s) for s in seeds])

    print(f"\n=== GT crowds: {100*gt_rate:.1f}% of {n_gt_all} GT crowns touch another ===")
    print(f"\n--- BOX-matched (can see mask failures; failure_analysis.py convention) ---")
    print(f"{'group':<16}{'n':>8}{'%boxTP':>9}{'mask IoU':>10}{'fail<0.5':>10}{'gt size':>9}")
    for lab in ("isolated", "touching"):
        if lab in bband:
            g = per_seed[seeds[0]]["by_touching_boxmatched"][lab]
            print(f"{lab+' (s0)':<16}{g['n']:>8}{100*g['frac_of_box_tps']:>8.1f}%"
                  f"{g['mean_mask_iou']:>10.4f}{g['mask_fail_rate_iou_lt_0.5']:>10.4f}"
                  f"{g['mean_gt_size_px']:>9.1f}")
    for lab in ("isolated", "touching"):
        if lab in bband:
            b, f = bband[lab]["mean_mask_iou"], bband[lab]["mask_fail_rate_iou_lt_0.5"]
            print(f"{lab+' band':<16}{'':>8}{'':>9}{b[0]:>10.4f}{f[0]:>10.4f}"
                  f"   (+-{b[1]:.4f} / +-{f[1]:.4f})")
    print(f"\n--- MASK-matched (fail rate vacuous by construction) ---")
    print(f"{'group':<12}{'n':>8}{'%TPs':>8}{'mask IoU':>10}{'fail<0.5':>10}{'gt size':>9}{'cover':>8}")
    for lab in ("isolated", "touching"):
        if lab not in per_seed[seeds[0]]["by_touching"]:
            continue
        g = per_seed[seeds[0]]["by_touching"][lab]
        print(f"{lab+' (s0)':<12}{g['n']:>8}{100*g['frac_of_tps']:>7.1f}%"
              f"{g['mean_mask_iou']:>10.4f}{g['mask_fail_rate_iou_lt_0.5']:>10.4f}"
              f"{g['mean_gt_size_px']:>9.1f}{g['mean_cover_ratio']:>8.4f}")
    for lab in ("isolated", "touching"):
        if lab in band:
            b = band[lab]["mean_mask_iou"]
            print(f"{lab+' band':<12}{'':>8}{'':>8}{b[0]:>10.4f} +-{b[1]:.4f}")
    print(f"\n-> {os.path.relpath(a.out, io.REPO)}")


if __name__ == "__main__":
    main()

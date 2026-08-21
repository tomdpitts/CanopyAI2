"""T0.6 -- Qualitative figure for the DEPLOYED masker, including a failure case.

WHY NOT REUSE THE EXISTING PANELS
    claude_outputs/mask_compare_tpfpfn/ and beta_disagree_*/ are excellent, but both derive
    from phase4_research/viz_mask_compare.py, which loads em_model_4p.npz and
    em_model_4p_b0.npz -- the PRE-registration-fix maskers, on the 50-tile prediction set.
    They illustrate the beta mechanism, not the shipped model. This script renders from
    preds/knobbed_s0.json, i.e. the exact predictions behind the published 0.630 band.
    No feature extraction, no re-inference: the RLE masks are read straight off disk.

TILE SELECTION IS MEASURED, NOT EYEBALLED
    Tiles are chosen from results/strata_instance_rows_s0.json (T0.4):
      best    highest mean mask IoU among tiles with >=25 matched TPs
      crowded highest `touching` fraction among tiles with >=25 matched TPs
      worst   lowest mean mask IoU among tiles with >=25 matched TPs  <- the failure case
    Colours follow the repo convention (viz_mask_compare.py:2-4):
      GREEN = TP (mask IoU >= 0.5)   RED = FP   TEAL = FN (GT missed entirely)

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.ablation.scripts.t06_figures
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.prepare_test import TCD_TEST
from ..lib import boundary as B
from ..lib import io, render, strata as S

Image.MAX_IMAGE_PIXELS = None
DISP = 1024
TP_C, FP_C, FN_C = (60, 200, 90), (225, 60, 55), (60, 200, 205)


def overlay(tid, pm, scores, tp_pairs, gm, matched_gt, op_thr=0.4, alpha=0.45):
    """RGB tile with predicted masks tinted by outcome and missed GT outlined.

    Only predictions at or above the OPERATING POINT are drawn (repo convention, see
    viz_mask_compare.py:9). The saved preds file keeps every detection down to a very low
    score; drawing all of them buries the true positives under a carpet of low-confidence
    false positives and makes the panel unreadable. Matching itself still uses ALL preds, so
    the drawn TP/FP labels stay consistent with the AP computation.
    """
    im = Image.open(os.path.join(TCD_TEST, tid + ".tif")).convert("RGB").resize((DISP, DISP))
    base = np.asarray(im).astype(np.float32)
    tp_pred = {p for p, _ in tp_pairs}
    lay = np.zeros_like(base)
    hit = np.zeros(base.shape[:2], bool)
    keep = [i for i in range(len(scores)) if scores[i] >= op_thr]
    # FPs first, TPs painted over them: overlapping instances otherwise hide the matches
    for i in sorted(keep, key=lambda k: k in tp_pred):
        m = np.asarray(Image.fromarray(pm[i].astype(np.uint8) * 255).resize((DISP, DISP),
                                                                           Image.NEAREST)) > 127
        if not m.any():
            continue
        lay[m] = TP_C if i in tp_pred else FP_C
        hit |= m
    out = base.copy()
    out[hit] = (1 - alpha) * base[hit] + alpha * lay[hit]
    img = Image.fromarray(out.astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for j in range(len(gm)):                       # missed GT -> teal outline
        if j in matched_gt:
            continue
        ys, xs = np.nonzero(np.asarray(Image.fromarray(gm[j].astype(np.uint8) * 255)
                                       .resize((DISP, DISP), Image.NEAREST)) > 127)
        if len(ys):
            dr.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=FN_C, width=2)
    return np.asarray(img)


def pick(rows_path, min_tp=25):
    rows = json.load(open(rows_path))["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["tid"], []).append(r)
    cand = {t: v for t, v in by.items() if len(v) >= min_tp}
    if not cand:
        cand = by
    miou = {t: float(np.mean([r["mask_iou"] for r in v])) for t, v in cand.items()}
    touch = {t: float(np.mean([r["touching"] for r in v])) for t, v in cand.items()}
    best = max(miou, key=miou.get)
    worst = min(miou, key=miou.get)
    crowded = max(touch, key=touch.get)
    return [("best", best, miou[best], touch[best], len(cand[best])),
            ("crowded", crowded, miou[crowded], touch[crowded], len(cand[crowded])),
            ("worst (failure case)", worst, miou[worst], touch[worst], len(cand[worst]))]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--out", default=os.path.join(io.FIGURES, "posteriors.pdf"))
    a = ap_.parse_args()
    rows_path = os.path.join(io.RESULTS, f"strata_instance_rows_s{a.seed}.json")
    assert os.path.exists(rows_path), f"run t04_strata_instance first ({rows_path} missing)"

    gt = io.load_gt()
    op_thr = float(json.load(open(io.knobbed(a.seed)))["meta"].get("op_thr", 0.4))
    sel = pick(rows_path)
    want = {t: lab for lab, t, *_ in sel}
    panels, meta = {}, {}
    for tl in io.per_tile(io.knobbed(a.seed), gt, progress=0):
        if tl["tid"] not in want:
            continue
        iou = E.mask_iou(tl["pm"], tl["gm"])
        pairs = B.match_tps(iou, tl["scores"], tl["ignore"], 0.5)
        img = overlay(tl["tid"], tl["pm"], tl["scores"], pairs, tl["gm"],
                      {g for _, g in pairs}, op_thr=op_thr)
        n_drawn = int((tl["scores"] >= op_thr).sum())
        n_tp_drawn = sum(1 for pi, _ in pairs if tl["scores"][pi] >= op_thr)
        panels[tl["tid"]] = img
        tf = S.touching_flags(S.gt_boxes(gt[tl["tid"]]["trees"]))
        meta[tl["tid"]] = {"n_pred": len(tl["scores"]), "n_gt": len(tl["gm"]),
                           "n_tp": len(pairs), "n_drawn_at_op": n_drawn,
                           "n_tp_at_op": n_tp_drawn, "op_thr": round(op_thr, 3),
                           "gt_touching_frac": round(float(tf.mean()) if len(tf) else 0.0, 3)}
        if len(panels) == len(want):
            break

    ordered = []
    for lab, tid, mi, tc, ntp in sel:
        m = meta[tid]
        ordered.append((panels[tid],
                        f"{lab} — {tid}\n"
                        f"mask IoU {mi:.3f} · {m['n_tp_at_op']}/{m['n_drawn_at_op']} drawn are TP\n"
                        f"{m['n_tp']}/{m['n_gt']} GT matched · {100*m['gt_touching_frac']:.0f}% touching"))
    render.grid(ordered, a.out, ncol=3,
                title=f"LACE self-masks, OAM-TCD (deployed knobbed masker, seed {a.seed}) — "
                      f"green TP · red FP · teal missed GT · drawn at score ≥ {op_thr:.2f}")

    json.dump({"label": "qualitative panels, deployed masker",
               "provenance": ("rendered from preds/knobbed_s0.json -- the predictions behind "
                              "the published 0.630 band. Existing panels in claude_outputs/ "
                              "depict the PRE-registration-fix maskers (em_model_4p.npz / "
                              "em_model_4p_b0.npz) and were not reused."),
               "selection": "measured from results/strata_instance_rows_s0.json, >=25 matched TPs",
               "not_included": ("a soft EM posterior heatmap: saved preds carry only RLE masks "
                                "thresholded at 0.25, and recomputing the posterior locally "
                                "would use DINOv3 features that differ from the Modal ones "
                                "(cos 0.86), so it would not depict the deployed model"),
               "panels": [{"role": lab, "tid": t, "mean_mask_iou": round(mi, 4),
                           "touching_frac": round(tc, 4), "n_matched_tp": ntp, **meta[t]}
                          for lab, t, mi, tc, ntp in sel]},
              open(os.path.join(io.RESULTS, "figure_panels.json"), "w"), indent=2)

    for lab, t, mi, tc, ntp in sel:
        print(f"  {lab:22s} {t:20s} meanIoU {mi:.3f}  touching {tc:.2f}  TPs {ntp}")
    print(f"-> {os.path.relpath(a.out, io.REPO)} (+ .svg)")


if __name__ == "__main__":
    main()

"""Mask-vs-box AP50 decomposition (diagnostic; no tuning, no selection). Where does mask AP50 lose against box AP50 on the
SAME detections and the SAME ranking? score_lite crowd arm (canopy = iscrowd), IoU 0.5, maxDets 600, each row's floor.

Each detection gets a box status and a mask status from COCOeval (TP / FP / IGN = ignored via a crowd region). The
transition classes (box -> mask) are applied one at a time to the box statuses and AP50 is recomputed, giving each class's
AP50 contribution (an approximation: classes interact, the sum is reported against the true mask - box). Key classes:
  TP->FP  the mask fails on a box that is correct at IoU 0.5 (masker cost of box error)
  IGN->FP the box is ignored by the canopy crowd box (IoA on the canopy BBOX) but its mask is not (crowd mask is exact)
  FP->IGN / FP->TP / IGN->TP  the reverse gains.
For TP->FP: box IoU with the matched GT, the mask IoU with that same GT, and GT size (sqrt box area).
Val (108) is the primary split; test (439) rows are record only (already-run test predictions, no decision taken from them).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_maskbox [--split val|test]
"""
import argparse, json, os, time
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead")
FCC = "/Users/tompitts/dphil/feat_cache"
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json"); TEST_GT = os.path.join(PKG, "test_gt.json")
BT = os.path.join(PKG, "box_supervised_baselines", "tierb", "results", "20261101_192510", "boxteacher", "results")
HUNT = os.path.join(PKG, "native_raster", "hunt")
GATES = {"val": {"centernet_s0|sprime": 0.6752}}   # FROZEN_simple val s' mask AP50


def statuses(gd, dets, iou_type):
    """-> AP50 (COCOeval), {det_index: (status, gt_id)}, ious {(img, det_index, gt_id): iou}, gt boxes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, iou_type); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]; ap = float(p[p > -1].mean())
    st, best_iou = {}, {}
    for e in ev.evalImgs:
        if e is None:
            continue
        I = ev.ious[(e["image_id"], 1)]
        for k, did in enumerate(e["dtIds"]):
            m = int(e["dtMatches"][0][k]); ig = bool(e["dtIgnore"][0][k])
            st[did - 1] = ("IGN" if ig else ("TP" if m > 0 else "FP"), m)
        # ious: rows = dts sorted by score (truncated), cols = gts in ev._gts order
        dts = sorted(ev._dts[e["image_id"], 1], key=lambda d: -d["score"])[:600]
        gids = [g["id"] for g in ev._gts[e["image_id"], 1]]
        if len(I):
            I = np.asarray(I)
            for r, d in enumerate(dts):
                best_iou[d["id"] - 1] = {gids[c]: float(I[r, c]) for c in np.flatnonzero(I[r] > 0.1)}
    return ap, st, best_iou


def ap_from(scores, stat, npig):
    """COCOeval.accumulate's AP at one IoU, from per-det statuses (dets already score-sorted, stable)."""
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    keep = stat != "IGN"; s = stat[keep]
    tp = np.cumsum(s == "TP").astype(float); fp = np.cumsum(s == "FP").astype(float)
    rc = tp / npig; pr = tp / np.maximum(tp + fp, np.spacing(1))
    pr = pr.tolist()
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, SL.REC_THRS, side="left"); q = np.zeros(len(SL.REC_THRS))
    for ri, pi in enumerate(inds):
        if pi < len(pr):
            q[ri] = pr[pi]
    return float(q.mean())


def analyse(name, preds, gd, t2i, floor):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dm = SL.build_dt(preds, t2i, 2048, floor, with_masks=True)
    db = [{k: v for k, v in d.items() if k not in ("segmentation", "area")} for d in dm]
    apb, sb, ib = statuses(gd, db, "bbox"); apm, sm, im = statuses(gd, dm, "segm")
    # global order exactly as accumulate: per image top-600 by score, then a stable sort of the concatenation by -score
    per_img = {}
    for i, d in enumerate(dm):
        per_img.setdefault(d["image_id"], []).append(i)
    idx = []
    for img in sorted(per_img):
        L = per_img[img]; o = np.argsort([-dm[i]["score"] for i in L], kind="mergesort")[:600]; idx += [L[j] for j in o]
    sc = np.array([dm[i]["score"] for i in idx]); o = np.argsort(-sc, kind="mergesort"); idx = [idx[j] for j in o]
    B = np.array([sb[i][0] for i in idx]); M = np.array([sm[i][0] for i in idx])
    npig = sum(1 for a in gd["annotations"] if not a["iscrowd"])
    apb2, apm2 = ap_from(None, B, npig), ap_from(None, M, npig)
    rec = {"box_AP50": round(apb, 4), "mask_AP50": round(apm, 4), "diff": round(apm - apb, 4),
           "parity_box": abs(apb2 - apb) < 1e-6, "parity_mask": abs(apm2 - apm) < 1e-6, "n_det": len(idx), "n_gt": npig}
    trans = {}
    for a in ("TP", "FP", "IGN"):
        for b in ("TP", "FP", "IGN"):
            if a == b:
                continue
            sel = (B == a) & (M == b); n = int(sel.sum())
            if not n:
                continue
            H = B.copy(); H[sel] = b
            trans[f"{a}->{b}"] = {"n": n, "dAP50": round(ap_from(None, H, npig) - apb, 4)}
    rec["transitions"] = trans; rec["sum_dAP50"] = round(sum(v["dAP50"] for v in trans.values()), 4)
    # rank-weighted view: transitions among the dets ranked above the box row's recall-0.6 point
    keepB = B != "IGN"; tpc = np.cumsum(B == "TP"); cut = int(np.searchsorted(tpc, 0.6 * npig)) if tpc[-1] >= 0.6 * npig else len(B)
    top = np.zeros(len(B), bool); top[:cut + 1] = True
    rec["above_box_recall0.6"] = {f"{a}->{b}": int(((B == a) & (M == b) & top).sum()) for a in ("TP", "FP", "IGN") for b in ("TP", "FP", "IGN") if a != b}
    # TP->FP detail
    gtb = {a["id"]: a["bbox"] for a in gd["annotations"]}
    bi, mi, sz = [], [], []
    for j, i in enumerate(idx):
        if B[j] == "TP" and M[j] == "FP":
            g = sb[i][1]; bi.append(ib.get(i, {}).get(g, np.nan)); mi.append(im.get(i, {}).get(g, 0.0))
            w, h = gtb[g][2], gtb[g][3]; sz.append(np.sqrt(w * h))
    bi, mi, sz = np.array(bi), np.array(mi), np.array(sz)
    if len(bi):
        rec["TP->FP_detail"] = {
            "box_iou_bins": {f"{lo}-{hi}": int(((bi >= lo) & (bi < hi)).sum()) for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))},
            "mask_iou_same_gt_bins": {f"{lo}-{hi}": int(((mi >= lo) & (mi < hi)).sum()) for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.4), (0.4, 0.5))},
            "gt_size_px_bins": {f"{lo}-{hi}": int(((sz >= lo) & (sz < hi)).sum()) for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 1e9))},
            "median_box_iou": round(float(np.nanmedian(bi)), 3), "median_mask_iou": round(float(np.median(mi)), 3)}
        # all box-TPs, same bins, for the failure rate per box-IoU bin
        allb = np.array([ib.get(i, {}).get(sb[i][1], np.nan) for j, i in enumerate(idx) if B[j] == "TP"])
        rec["TP->FP_rate_by_box_iou"] = {f"{lo}-{hi}": round(float(((bi >= lo) & (bi < hi)).sum() / max(((allb >= lo) & (allb < hi)).sum(), 1)), 3)
                                         for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))}
        rec["box_TP_box_iou_bins"] = {f"{lo}-{hi}": int(((allb >= lo) & (allb < hi)).sum()) for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))}
    print(f"[{name}] {json.dumps({k: rec[k] for k in ('box_AP50', 'mask_AP50', 'diff', 'parity_box', 'parity_mask', 'sum_dAP50')})}", flush=True)
    print(f"[{name}] transitions {json.dumps(trans)}", flush=True)
    return rec


def sources(split, tids):
    """-> {row: (preds, floor)}. LACE rows use s' with floor 0 (LACE protocol); others their own score, floor 0.05."""
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    S = {}
    bt = json.load(open(os.path.join(BT, f"preds_boxteacher_sub_{split}_r2048.json")))["preds"]
    S["boxteacher_native"] = ({t: bt[t] for t in tids}, 0.05)
    B = BM.bt_top(split)
    S["bt_boxes+lace_masker|sprime"] = (BM.preds_for(split, B, tids, {"e": 0.0, "c": 0.33}, "sprime"), 0.0)
    S["bt_boxes+lace_masker|btscore"] = (BM.preds_for(split, B, tids, {"e": 0.0, "c": 0.33}, "bt"), 0.05)
    if split == "val":
        from boxinst_commonality_tcd_04.dethead.diag_val import render
        cn = {}
        for t in tids:
            d = json.load(open(f"{FCC}/boxfix_ap_val_tiles/{t}.json")); cn[t] = (d["boxes_2048"], d["scores_s"])
        R = render({"cn": cn}, {"cn": [{"e": 0.0, "c": 0.33}]})["cn"]["e0.00_c0.33"]
        S["centernet_s0|sprime"] = ({t: {"boxes_2048": cn[t][0], "scores": R[t]["s_prime"], "masks_rle": R[t]["masks_rle"]} for t in tids}, 0.0)
    else:
        P = json.load(open(os.path.join(HUNT, "test", "preds_lace_s0_r2048_lace4px_simple.json")))["preds"]
        S["centernet_s0|sprime"] = ({t: P[t] for t in tids}, 0.0)
        for tag in ("fpn_s0_nms05_detscore", "fcos_s0_nms05_detscore", "cn_local_s0_sprime"):
            P = json.load(open(os.path.join(HERE, "results", f"preds_{tag}_r2048.json"))); P = P.get("preds", P)
            S[tag] = ({t: P[t] for t in tids}, 0.0 if "sprime" in tag else 0.05)
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); a = ap.parse_args()
    out = os.path.join(HERE, "results", f"diag_maskbox_{a.split}.json"); assert not os.path.exists(out), out
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    g = json.load(open(VAL_GT if a.split == "val" else TEST_GT)); tids = sorted(g)
    assert len(tids) == (108 if a.split == "val" else 439), len(tids)
    gd, t2i = SL.build_gt_crowd(g, 2048)
    rows = {}
    for name, (preds, floor) in sources(a.split, tids).items():
        rows[name] = analyse(name, preds, gd, t2i, floor)
        if name in GATES.get(a.split, {}):
            rows[name]["gate"] = {"expect_mask_AP50": GATES[a.split][name], "pass": rows[name]["mask_AP50"] == GATES[a.split][name]}
            print(f"[{name}] gate {rows[name]['gate']}", flush=True)
    rec = {"split": a.split, "protocol": "score_lite crowd arm, IoU 0.5, maxDets 600; LACE s' floor 0, others 0.05",
           "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

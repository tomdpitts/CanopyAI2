"""VAL-only box AP50 decomposition (diagnostic; no tuning). 108 val tiles, score_lite crowd-arm GT (canopy = iscrowd),
COCOeval bbox, IoU 0.5, maxDets 600, floor as each row's protocol. Rows: LACE CenterNet s0 (s' ranking, as deployed),
FCOS NMS 0.6 and 0.5 (detector score), BoxTeacher (its score; stitched, mask-derived boxes).

Per row: AP50 (must equal score_lite box AP50), max recall, precision at recall 0.5/0.6/0.7, recall by GT size
(sqrt box area px) and touching vs isolated, missed-GT causes (best IoU with any detection: <0.1 none / 0.1-0.5 poor
localisation / >=0.5 but taken by another GT), and the causes of FPs ranked above the recall-0.6 operating score
(background IoU<0.1 / localisation 0.1-0.5 / duplicate of an already-matched GT).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.diag_box
"""
import json, os, time
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead")
FCC = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "results", "diag_box_val.json")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
BT_VAL = os.path.join(PKG, "box_supervised_baselines", "tierb", "results", "20261101_192510", "boxteacher", "results", "preds_boxteacher_sub_val_r2048.json")
SIZES = [(0, 24), (24, 48), (48, 96), (96, 1e9)]


def iou_m(A, B):
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    x0 = np.maximum(A[:, None, 0], B[None, :, 0]); y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2]); y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return i / np.maximum(aa[:, None] + ab[None] - i, 1e-9)


def analyse(name, S, gd, t2i, floor):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    i2t = {v: k for k, v in t2i.items()}
    dets = []
    for t, (B, sc) in S.items():
        for b, s in zip(B, sc):
            if s >= floor:
                dets.append({"image_id": t2i[t], "category_id": 1, "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    cg = COCO(); cg.dataset = gd; cg.createIndex(); cd = cg.loadRes([dict(d) for d in dets])
    ev = COCOeval(cg, cd, "bbox"); ev.params.iouThrs = np.array([0.5]); ev.params.recThrs = SL.REC_THRS
    ev.params.maxDets = [600]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]
    ap = float(p[p > -1].mean())
    rec = {"AP50": round(ap, 4), "n_det": len(dets)}
    # per-GT / per-det outcomes
    gts_all, det_rows = [], []
    for e in ev.evalImgs:
        if e is None:
            continue
        t = i2t[e["image_id"]]
        G = cg.loadAnns(e["gtIds"]); gign = np.array(e["gtIgnore"], bool)
        gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in G]).reshape(-1, 4)
        D = cd.loadAnns(e["dtIds"]); db = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in D]).reshape(-1, 4)
        dm = np.array(e["dtMatches"][0]); dig = np.array(e["dtIgnore"][0], bool); gm = np.array(e["gtMatches"][0])
        tree = ~gign; tb = gb[tree]
        I = iou_m(tb, tb); np.fill_diagonal(I, 0)
        touching = (I > 0).any(1) if len(tb) else np.zeros(0, bool)
        Igd = iou_m(tb, db)                                             # tree GT x det
        matched_ids = set(dm[dm > 0].astype(int).tolist())
        for k, gi in enumerate(np.nonzero(tree)[0]):
            best = Igd[k].max() if Igd.shape[1] else 0.0
            gts_all.append({"matched": gm[gi] > 0, "size": float(np.sqrt(max(tb[k, 2] - tb[k, 0], 0) * max(tb[k, 3] - tb[k, 1], 0))),
                            "touching": bool(touching[k]), "best_iou": float(best)})
        Idt = iou_m(db, tb)
        for j, d in enumerate(D):
            if dig[j]:
                continue
            tp = dm[j] > 0
            if tp:
                cause = "tp"
            else:
                bi = Idt[j].max() if Idt.shape[1] else 0.0
                # duplicate: IoU >= 0.5 with a tree GT that some (higher) det already took
                dup = Idt.shape[1] and any(Idt[j, k] >= 0.5 and G_id in matched_ids for k, G_id in enumerate(np.array(e["gtIds"])[tree]))
                cause = "dup" if dup else ("loc" if bi >= 0.1 else "bg")
            det_rows.append((d["score"], cause))
    n_gt = len(gts_all); R = np.array([g["matched"] for g in gts_all])
    rec["n_gt"] = n_gt; rec["max_recall"] = round(float(R.mean()), 4)
    rt = SL.REC_THRS
    rec["precision_at_recall"] = {str(r): round(float(p[int(np.argmin(np.abs(rt - r)))]), 4) for r in (0.5, 0.6, 0.7)}
    sz = np.array([g["size"] for g in gts_all]); tc = np.array([g["touching"] for g in gts_all])
    rec["recall_by_size_px"] = {f"{int(a)}-{'inf' if b > 1e8 else int(b)}": [round(float(R[(sz >= a) & (sz < b)].mean()), 4), int(((sz >= a) & (sz < b)).sum())] for a, b in SIZES}
    rec["recall_touching_isolated"] = {"touching": [round(float(R[tc].mean()), 4), int(tc.sum())], "isolated": [round(float(R[~tc].mean()), 4), int((~tc).sum())]}
    bi = np.array([g["best_iou"] for g in gts_all])[~R]
    rec["missed_causes"] = {"n_missed": int((~R).sum()), "no_det_iou<0.1": round(float((bi < 0.1).mean()), 4),
                            "poor_loc_0.1-0.5": round(float(((bi >= 0.1) & (bi < 0.5)).mean()), 4), "taken_iou>=0.5": round(float((bi >= 0.5).mean()), 4)}
    # FPs above the recall-0.6 operating score
    det_rows.sort(key=lambda x: -x[0]); tps = np.cumsum([c == "tp" for _, c in det_rows]) / n_gt
    k = int(np.searchsorted(tps, 0.6)) if tps[-1] >= 0.6 else len(det_rows)
    top = [c for _, c in det_rows[:k] if c != "tp"]
    rec["fp_above_recall0.6"] = {"n_dets_to_recall0.6": k, "n_fp": len(top), **{c: round(top.count(c) / max(len(top), 1), 4) for c in ("bg", "loc", "dup")}}
    print(f"[{name}] {json.dumps(rec)}", flush=True)
    return rec


def main():
    assert not os.path.exists(OUT), OUT
    import torch
    from boxinst_commonality_tcd_04.dethead import fcos as FC
    from boxinst_commonality_tcd_04.dethead.test import load_model
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(VAL_GT)); tids = sorted(vg); gd, t2i = SL.build_gt_crowd(vg, 2048)
    m = load_model(os.path.join(HERE, "runs", "fcos_s0", "best.pt"), "fcos"); f6, f5 = {}, {}
    with torch.no_grad():
        for t in tids:
            out = m(torch.from_numpy(np.load(f"{FCC}/feat_4p_val/{t}.npy")).float()[None])
            for nms_iou, D in ((0.6, f6), (0.5, f5)):
                FC.NMS_IOU = nms_iou; b, s = FC.decode(out); D[t] = (b.numpy().astype(float).tolist(), s.numpy().astype(float).tolist())
    FC.NMS_IOU = 0.6
    cn = {t: (lambda d: (d["boxes_2048"], d["scores_product"]))(json.load(open(f"{FCC}/boxfix_ap_val_tiles/{t}.json"))) for t in tids}
    bt = json.load(open(BT_VAL))["preds"]; btS = {t: (bt[t]["boxes_2048"], bt[t]["scores"]) for t in tids}
    rows = {"lace_centernet_s0_sprime": analyse("lace_cn", cn, gd, t2i, 0.0),
            "fcos_nms06_detscore": analyse("fcos06", f6, gd, t2i, 0.05),
            "fcos_nms05_detscore": analyse("fcos05", f5, gd, t2i, 0.05),
            "boxteacher": analyse("bt", btS, gd, t2i, 0.05)}
    json.dump({"rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

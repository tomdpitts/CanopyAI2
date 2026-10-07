"""Does the retrain pilot's val gain survive the DEPLOYED ranking (s * pbar * m), and what are
its extra ~10% detections? Val only (108 tiles), 2048, tau 0.40, local features.

Per detector (published s0 and pilot v2 s0), one pass per tile:
  - decode boxes (score floor 0.05, topk 600), render masks with E.pred_instance_masks;
  - posterior statistics from masker.box_mask with the deployed knobs (alpha 0.3, kappa 1.6):
    bimod = mean|2 pfg - 1|, pfg_mean = mean pfg -- the two factors of the product
    (confidence/apply_product.py, confidence/em_diag_multi.py);
  - score with score_coco twice: ranking s (floor 0) and ranking s*bimod*pfg_mean (floor 0;
    the detection set is frozen at decode, as in apply_product).

GATES
  (1) code path: on the first 2 tiles, box_mask's pfg equals the em_diag formula re-derived
      from the npz arrays (max |diff| < 1e-6) -- the product factors are the published ones;
  (2) s-ranked mask AP50 reproduces det_val_eval_v2_L24_s0_r2048.json: 0.6045 (published)
      and 0.6132 (pilot), and the det counts 40,058 / 44,103.
ANATOMY (fp_anatomy_test.py rule, val GT): max recall, precision and FP kinds at fixed recall,
and the score distribution, for each detector x ranking.

Checkpoints after each detector; refuses to overwrite finished outputs.

    VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u -m \
        boxinst_commonality_tcd_04.native_raster.det_val_product
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import torch
from pycocotools import mask as maskUtils
from scipy.special import logsumexp

from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04 import score_coco as S
from boxinst_commonality_tcd_04.detector import STRIDE8
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import (
    DET_PATH, EM_PATH, FEAT_DIR, HERE, KAPPA_SCALE, PRIOR_WEIGHT, VAL_GT)
from dapt.decode import decode

RES, TAU = 2048, 0.40
DETS = {"published_s0": (DET_PATH, 0.6045, 40058),
        "pilot_v2_s0": ("/Users/tompitts/dphil/feat_cache/det_v2_L24_s0.pt", 0.6132, 44103)}
OUT = os.path.join(HERE, "det_val_product_r2048.json")
RECALLS = (0.5, 0.6, 0.7)
SCORE_BINS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01)


def em_diag_pfg(masker, zn, g, b):
    """The em_diag_multi / val_diag_modal formula, re-derived from the npz arrays."""
    d = np.load(EM_PATH, allow_pickle=False)
    C, Gbg, wbg, pi = d["C"], d["Gbg"], d["wbg"], d["pi"]
    k = float(d["kappa"]) * KAPPA_SCALE
    s, NB = int(d["s_px"]), pi.shape[2]
    origin = float(d["cell_origin"]) if "cell_origin" in d else s / 2.0
    cy, cx = np.mgrid[0:g, 0:g] * s + origin
    cyr, cxr = cy.ravel(), cx.ravel()
    x0, y0, x1, y1 = b
    pad = s / 2.0
    idx = np.flatnonzero((cxr >= x0 - pad) & (cxr < x1 + pad) & (cyr >= y0 - pad) & (cyr < y1 + pad))
    u = np.clip((cxr[idx] - x0) / max(x1 - x0, 1), 0, 1)
    v = np.clip((cyr[idx] - y0) / max(y1 - y0, 1), 0, 1)
    bu = np.minimum((u * NB).astype(int), NB - 1); bv = np.minimum((v * NB).astype(int), NB - 1)
    sb = int(np.searchsorted(d["size_edges"], np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1))))
    zi = zn[idx]
    bgll = logsumexp(np.log(wbg)[None] + k * (zi @ Gbg.T), 1)
    pis = pi[sb][:, bv, bu]
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    A = logsumexp(k * (zi @ C.T) + np.log((pis / psum).T + 1e-9), 1) - bgll
    Ac = A - A.mean() if len(A) > 1 else A
    return idx, 1.0 / (1.0 + np.exp(-(Ac + PRIOR_WEIGHT * np.log(psum / (1 - psum)))))


def anatomy(preds, key, G, tids):
    """fp_anatomy_test.py's rule on val, matching in the order of preds[t][key].
    -> (ranking key, detector score s, kind) per kept prediction, and n_gt."""
    n_gt, R, Sd, KIND = 0, [], [], []
    for t in tids:
        tr, cu = G[t]
        n_gt += len(tr)
        r = preds[t]
        if not r["scores"]:
            continue
        pr = [{"size": x["size"], "counts": x["counts"].encode()} for x in r["masks_rle"]]
        key_s = np.asarray(r[key], float)
        M = (np.asarray(maskUtils.iou(pr, tr, [0] * len(tr))).reshape(len(pr), len(tr))
             if tr else np.zeros((len(pr), 0)))
        area = np.array([maskUtils.area(x) for x in pr], float)
        inc = (np.array([maskUtils.area(maskUtils.merge([x, cu], intersect=1)) for x in pr], float)
               / np.maximum(area, 1) if cu is not None else np.zeros(len(pr)))
        taken = np.zeros(M.shape[1], bool)
        for i in np.argsort(-key_s, kind="mergesort"):
            row = M[i] if M.shape[1] else np.zeros(0)
            c = np.where(taken, -1, row); j = int(c.argmax()) if len(c) else -1
            if j >= 0 and c[j] >= 0.5:
                taken[j] = True; k_ = "tp"
            elif inc[i] > 0.5:
                continue
            elif len(row) and row.max() >= 0.5:
                k_ = "dup"
            elif len(row) and row.max() >= 0.1:
                k_ = "loc"
            else:
                k_ = "bg"
            R.append(key_s[i]); Sd.append(r["scores"][i]); KIND.append(k_)
    return np.array(R), np.array(Sd), np.array(KIND), n_gt


def summarise(order_scores, S_det, KIND, n_gt):
    o = np.argsort(-order_scores, kind="mergesort"); K = KIND[o]
    tp = np.cumsum(K == "tp"); rec = tp / n_gt; prec = tp / np.arange(1, len(tp) + 1)
    ap = float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0
                        for r in np.linspace(0, 1, 101)]))
    row = {"ap50_check": round(ap, 4), "max_recall": round(float(rec[-1]), 4),
           "n_kept": int(len(K)), "n_tp": int(tp[-1]), "at_recall": {}}
    for R in RECALLS:
        if rec[-1] < R:
            row["at_recall"][str(R)] = None; continue
        k = int(np.argmax(rec >= R)) + 1; top = K[:k]; nfp = int((top != "tp").sum())
        row["at_recall"][str(R)] = {"precision": round(float(prec[k - 1]), 4), "n_dets": k,
                                    "n_fp": nfp, **{f"fp_{c}": int((top == c).sum())
                                                    for c in ("dup", "loc", "bg")}}
    bins = {}
    for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
        m = (S_det >= lo) & (S_det < hi)
        bins[f"s[{lo},{min(hi, 1.0)})"] = {"n": int(m.sum()), "tp": int((KIND[m] == "tp").sum()),
                                            **{c: int((KIND[m] == c).sum()) for c in ("loc", "bg", "dup")}}
    row["by_det_score"] = bins
    return row


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        "split": "val (108 tiles)", "res": RES, "tau": TAU, "prior_weight": PRIOR_WEIGHT,
        "kappa_scale": KAPPA_SCALE, "detectors": {}}
    assert "done" not in out, f"{OUT} complete -- refusing to overwrite"
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tids) == 108
    gt = {t: gt[t] for t in tids}
    gt_fp, tid2img, tree_rles, canopy_union = S.build_gt(gt, RES, with_canopy=False)
    gt_cr, _, _, _ = S.build_gt(gt, RES, with_canopy=True)
    ctx = (gt_fp, gt_cr, tid2img, tree_rles, canopy_union, tids)
    masker = E.TCDMasker(EM_PATH)
    enc = lambda ms: [maskUtils.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]
    G = {}
    for t in tids:
        cn = enc(E.raster(gt[t]["canopy"], res=RES, scale=1.0))
        G[t] = (enc(E.raster(gt[t]["trees"], res=RES, scale=1.0)), maskUtils.merge(cn) if cn else None)

    for name, (path, ref_ap, ref_n) in DETS.items():
        if name in out["detectors"]:
            print(f"[{name}] cached", flush=True); continue
        preds_fp = os.path.join(HERE, f"preds_val_{name}_r2048_tau040_withpost.json")
        assert not os.path.exists(preds_fp), f"{preds_fp} exists -- refusing to overwrite"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = Detector4Phase(ck["cfg"]["in_dim"], width=ck["cfg"]["width"],
                               tower=ck["cfg"]["tower"]).eval()
        model.load_state_dict(ck["state"])
        preds, t0, gate1 = {}, time.time(), []
        for n, tid in enumerate(tids):
            feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
            with torch.no_grad():
                det = model(torch.from_numpy(feat)[None])
            bx, sc = decode(det, score_thr=0.05, stride=STRIDE8, topk=600)
            bx, sc = bx.numpy(), sc.numpy()
            zn, g = masker.project(feat), feat.shape[-1]
            bim, pfm = [], []
            for b in bx:
                idx, r = masker.box_mask(zn, g, b, prior_weight=PRIOR_WEIGHT,
                                         kappa_scale=KAPPA_SCALE)
                bim.append(float(np.abs(2 * r - 1).mean())); pfm.append(float(r.mean()))
                if n < 2 and len(gate1) < 400:
                    idx2, r2 = em_diag_pfg(masker, zn, g, b)
                    gate1.append(float(np.max(np.abs(r - r2))) if np.array_equal(idx, idx2)
                                 else np.inf)
            pm = (E.pred_instance_masks(masker, zn, g, bx, res=RES, scale=2048.0 / RES,
                                        mask_thr=TAU, prior_weight=PRIOR_WEIGHT,
                                        kappa_scale=KAPPA_SCALE) if len(bx) else [])
            s = np.asarray(sc, np.float64)
            preds[tid] = {"boxes_2048": bx.tolist(), "scores": sc.tolist(),
                          "bimod": bim, "pfg_mean": pfm,
                          "product": [round(float(v), 8) for v in
                                      s * np.asarray(bim) * np.asarray(pfm)] if len(bx) else [],
                          "canopy_ignore": [False] * len(bx),
                          "masks_rle": [{"size": [RES, RES], "counts": maskUtils.encode(
                              np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}
                              for m in pm]}
            if n == 1:
                ok1 = bool(gate1) and max(gate1) < 1e-6
                print(f"[{name}] GATE1 code path: {len(gate1)} boxes, max|dpfg| "
                      f"{max(gate1):.2e} -> {'PASS' if ok1 else 'FAIL'}", flush=True)
                if not ok1:
                    raise SystemExit("gate 1 failed")
            if (n + 1) % 20 == 0 or n + 1 == len(tids):
                print(f"  [{name}] {n+1}/108 {time.time()-t0:.0f}s", flush=True)
        n_det = sum(len(p["scores"]) for p in preds.values())
        with open(preds_fp, "w") as f:
            json.dump({"meta": {"mask_res": RES, "det": path, "tau": TAU,
                                "product": "s*bimod*pfg_mean, alpha 0.3 kappa 1.6"},
                       "preds": preds}, f)
        res = {}
        tmp = os.path.join(tempfile.gettempdir(), "det_val_product_cell.json")
        for rank in ("scores", "product"):
            pp = {t: {**{k: v for k, v in p.items() if k not in ("bimod", "pfg_mean", "product")},
                      "scores": p[rank]} for t, p in preds.items()}
            with open(tmp, "w") as f:
                json.dump({"meta": {"mask_res": RES}, "preds": pp}, f)
            c = S._score_one(tmp, ctx, RES, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
            res["s" if rank == "scores" else "product"] = {"mask": c["mask"], "box": c["box"]}
        ok2 = abs(res["s"]["mask"]["AP50"] - ref_ap) <= 1e-4 and n_det == ref_n
        print(f"[{name}] GATE2 s-ranked AP50 {res['s']['mask']['AP50']} vs {ref_ap}, dets "
              f"{n_det} vs {ref_n} -> {'PASS' if ok2 else 'FAIL'}", flush=True)
        if not ok2:
            raise SystemExit("gate 2 failed")
        anat = {}
        for rank, key in (("s", "scores"), ("product", "product")):
            Rk, Sd, KIND, n_gt = anatomy(preds, key, G, tids)
            anat[rank] = summarise(Rk, Sd, KIND, n_gt)
            print(f"[{name}] anatomy {rank}: {json.dumps({k: v for k, v in anat[rank].items() if k != 'by_det_score'})}", flush=True)
        out["detectors"][name] = {"det": path, "n_det": n_det, "gate1_max_abs_dpfg": max(gate1),
                                  "gate2_pass": ok2, **res, "anatomy": anat, "preds_file": os.path.basename(preds_fp),
                                  "secs": round(time.time() - t0, 1)}
        json.dump(out, open(OUT, "w"), indent=2)
        print(f"[{name}] s: {res['s']['mask']['AP50']} product: {res['product']['mask']['AP50']} "
              f"[saved]", flush=True)
    out["done"] = True
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()

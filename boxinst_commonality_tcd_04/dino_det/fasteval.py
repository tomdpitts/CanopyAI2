"""Fast re-scoring evaluator (2026-11-04): mask AP at IoU 0.50:0.95 for ANY score vector over FIXED masks, identical to
score_lite.score (pycocotools COCOeval, crowd arm, areaRng all, maxDets 600, 101 recall points) but with the det-vs-GT
mask IoUs precomputed once. Used for the R3 streamlining and the AP50:95 side-checks, where only scores change.

  build(P, gt)   -> per-tile sparse candidates (det -> non-crowd GTs with IoU >= 0.5), max crowd IoA, n GT
  ap(E, scores)  -> np.array of 10 APs (IoU 0.50 ... 0.95); scores = {tid: list} or a flat vector in sorted-tid order
  parity         : must equal score_lite to 4 dp on s', s and a reversed-order vector
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.fasteval parity
"""
import json, os, pickle, sys, time
import numpy as np

THR = np.linspace(0.5, 0.95, 10); REC = np.linspace(0.0, 1.0, 101)


def _tile(args):
    from pycocotools import mask as M
    rles, gts, crowds = args; n = len(rles)
    if n == 0:
        return [], np.zeros(0), len(gts)
    iou = M.iou(rles, gts, [0] * len(gts)) if gts else np.zeros((n, 0))
    ioa = M.iou(rles, crowds, [1] * len(crowds)).max(1) if crowds else np.zeros(n)
    iou = np.asarray(iou).reshape(n, len(gts))
    cand = [(np.flatnonzero(r >= 0.5), r[r >= 0.5]) for r in iou]
    return cand, np.asarray(ioa, np.float64).reshape(n), len(gts)


def build(P, gt, cache=None, procs=12):
    """P: {tid: {"masks_rle": [...]}}; gt: the raw GT json ({tid: {"trees", "canopy"}}). Tiles in sorted order."""
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    tids = sorted(gt); assert sorted(P) == tids
    gd, t2i = SL.build_gt_crowd(gt, 2048); by = {i: ([], []) for i in t2i.values()}
    for a in gd["annotations"]:
        by[a["image_id"]][a["iscrowd"]].append(a["segmentation"])
    jobs = [([SL._as_bytes(r) for r in P[t]["masks_rle"]], by[t2i[t]][0], by[t2i[t]][1]) for t in tids]
    with mp.get_context("spawn").Pool(procs) as pool:
        tiles = pool.map(_tile, jobs)
    E = {"tids": tids, "tiles": tiles}
    if cache:
        pickle.dump(E, open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    return E


def _match(cand, ioa, order, t):
    """COCOeval.evaluateImg for one tile, one threshold: dets in score order -> (tp, ignore) flags in that order."""
    used = set(); tp = np.zeros(len(order), bool); ig = np.zeros(len(order), bool); thr = min(t, 1 - 1e-10)
    for k, d in enumerate(order):
        gi, gv = cand[d]; m = -1; best = thr
        for g, v in zip(gi, gv):
            if g in used or v < best:
                continue
            best = v; m = g
        if m >= 0:
            used.add(m); tp[k] = True
        elif ioa[d] >= thr:
            ig[k] = True
    return tp, ig


def ap(E, scores, thrs=THR):
    if not isinstance(scores, dict):
        v = np.asarray(scores); o = 0; s = {}
        for t, (c, _, _) in zip(E["tids"], E["tiles"]):
            s[t] = v[o:o + len(c)]; o += len(c)
        assert o == len(v); scores = s
    out = []
    for t_ in thrs:
        S, TP, IG, npig = [], [], [], 0
        for t, (cand, ioa, ng) in zip(E["tids"], E["tiles"]):
            npig += ng
            if not len(cand):
                continue
            sc = np.asarray(scores[t], np.float32).astype(np.float64); order = np.argsort(-sc, kind="mergesort")[:600]
            tp, ig = _match(cand, ioa, order, t_); S.append(sc[order]); TP.append(tp); IG.append(ig)
        sc = np.concatenate(S); tp = np.concatenate(TP); ig = np.concatenate(IG)
        o = np.argsort(-sc, kind="mergesort"); tp, ig = tp[o], ig[o]
        tpc = np.cumsum(tp & ~ig).astype(float); fpc = np.cumsum(~tp & ~ig).astype(float)
        rc = tpc / npig; pr = tpc / (fpc + tpc + np.spacing(1)); pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC, side="left"); q = np.zeros(len(REC)); ok = idx < len(pr); q[ok] = pr[idx[ok]]
        out.append(q.mean())
    return np.asarray(out)


def summary(a):
    return {"AP50": round(float(a[0]), 4), "AP75": round(float(a[5]), 4), "AP50_95": round(float(a.mean()), 4)}


def parity():
    import contextlib, io
    from boxinst_commonality_tcd_04.dino_det import rerank as R
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    t0 = time.time(); E = build(P, vg, os.path.join(R.FC_, "fasteval_val.pkl")); print(f"[fasteval] build {time.time()-t0:.0f}s", flush=True)
    gd, t2i = SL.build_gt_crowd(vg, 2048); rec = {}
    for name, f in {"s_prime": lambda r: r["s_prime"], "s": lambda r: r["scores"],
                    "reverse": lambda r: list(np.arange(len(r["scores"]))[::-1] / 600.0)}.items():
        sc = {t: f(P[t]) for t in sorted(P)}; t0 = time.time(); a = summary(ap(E, sc)); tf = time.time() - t0
        pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in P}
        with contextlib.redirect_stdout(io.StringIO()):
            ref = SL.score(pr, gd, t2i, masks=True, floor=0.0)["mask"]
        ref = {k: ref[k] for k in a}; rec[name] = {"fast": a, "score_lite": ref, "match": a == ref, "fast_s": round(tf, 2)}
        print(name, rec[name], flush=True)
    ok = all(r["match"] for r in rec.values()); rec["pass"] = ok
    json.dump(rec, open(os.path.join(R.OUT, "fasteval_parity_val.json"), "w"), indent=1); print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"parity": parity}[sys.argv[1]]()

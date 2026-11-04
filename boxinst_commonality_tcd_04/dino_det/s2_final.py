"""Pre-registration S2-final (README.md, 2026-11-04): ONE look at the val-selected S2 changes on the test 439 and sparse-236.

Rows (all on frozen D1 boxes, LACE masks; masks either as is or Gaussian-smoothed sigma 2 px at 2048, threshold 0.5):
  blur2_sprime   : smoothed masks, s' ranking
  blur2_R3       : smoothed masks, R3 scores (the frozen R3 25-model mean already applied in rerankR3_final)
  closed         : masks as is, s * (s'/s)^0.625 * (1 - max_iou_higher)^0.5
  blur2_closed   : smoothed masks, the same closed form
Headline numbers: frozen score_coco (2048, maxDets 600, floor 0, crowd arm). CIs: paired tile bootstrap (2000, seed 0) on AP50 AND
AP50:95, computed with fasteval (its point estimates are asserted equal to score_coco's to 4 dp).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.s2_final
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det import side2 as S2

OUT = os.path.join(R.OUT, "s2_final.json"); SPARSE_GT = os.path.join(R.REPO, "data", "tcd_sparse", "sparse_gt.json")
GTS = {"test": R.TEST_GT, "sparse": SPARSE_GT}; A, B, SIG = 0.625, 0.5, "blur2"


def pf(split, row):
    return os.path.join(R.OUT, f"preds_{R.TAG}_{split}_{row}_r2048.json")


def blurred(split, St, tids):
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    fp = os.path.join(R.FC_, f"{R.TAG}_{split}_masks_{SIG}.json")
    if os.path.exists(fp):
        return json.load(open(fp))
    t0 = time.time()
    with mp.get_context("spawn").Pool(S2.NPROC) as pool:
        M2 = pool.map(S2._post, [([SL._as_bytes(r) for r in St[t]["masks_rle"]], np.asarray(St[t]["boxes_2048"]).reshape(-1, 4), SIG) for t in tids])
    out = dict(zip(tids, M2)); json.dump(out, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    print(f"[s2f] {split}: smoothed {sum(len(m) for m in M2)} masks {time.time()-t0:.0f}s", flush=True)
    return out


def per_tile(E, scores):
    """Per tile, per IoU threshold: (sorted scores, tp, ignore) -- matching depends only on within-tile order."""
    D = {}
    for t, (cand, ioa, ng) in zip(E["tids"], E["tiles"]):
        if not len(cand):
            D[t] = (ng, None, None); continue
        sc = np.asarray(scores[t], np.float32).astype(np.float64); o = np.argsort(-sc, kind="mergesort")[:600]
        D[t] = (ng, sc[o], [FE._match(cand, ioa, o, th) for th in FE.THR])
    return D


def ap_from(D, tiles):
    out = []; npig = sum(D[t][0] for t in tiles); live = [t for t in tiles if D[t][1] is not None]
    o = np.argsort(-np.concatenate([D[t][1] for t in live]), kind="mergesort")   # one sort serves all 10 thresholds
    for k in range(len(FE.THR)):
        tp = np.concatenate([D[t][2][k][0] for t in live])[o]; ig = np.concatenate([D[t][2][k][1] for t in live])[o]
        tpc = np.cumsum(tp & ~ig).astype(float); fpc = np.cumsum(~tp & ~ig).astype(float)
        rc = tpc / npig; pr = np.maximum.accumulate((tpc / (fpc + tpc + np.spacing(1)))[::-1])[::-1]
        idx = np.searchsorted(rc, FE.REC, side="left"); q = np.zeros(len(FE.REC)); ok = idx < len(pr); q[ok] = pr[idx[ok]]; out.append(q.mean())
    return np.asarray(out)


def boot(Dn, Db, tids):
    """Paired tile bootstrap, AP50 and AP50:95. Image order within a resample is the sorted-id order (as COCO) with repeats."""
    rng = np.random.default_rng(R.SEED); idx = np.arange(len(tids)); d50, d95 = [], []
    for _ in range(R.NBOOT):
        s = [tids[i] for i in np.sort(rng.integers(0, len(idx), len(idx)))]; a, b = ap_from(Dn, s), ap_from(Db, s)
        d50.append(a[0] - b[0]); d95.append(a.mean() - b.mean())
    a, b = ap_from(Dn, tids), ap_from(Db, tids); r = lambda v: round(float(v), 4)
    return {"AP50": {"diff": r(a[0] - b[0]), "ci95": [r(np.percentile(d50, 2.5)), r(np.percentile(d50, 97.5))], "p_le_0": r(np.mean(np.asarray(d50) <= 0))},
            "AP50_95": {"diff": r(a.mean() - b.mean()), "ci95": [r(np.percentile(d95, 2.5)), r(np.percentile(d95, 97.5))], "p_le_0": r(np.mean(np.asarray(d95) <= 0))}}


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
    for split, gtp in GTS.items():
        if split in rec:
            continue
        G = json.load(open(gtp)); St = json.load(open(pf(split, "sprime")))["preds"]; tids = sorted(St); assert tids == sorted(G)
        R3 = json.load(open(pf(split, "rerankR3")))["preds"]
        Z = np.load(os.path.join(R.FC_, f"rerankR1_{split}.npz")); assert list(Z["tids"]) == tids
        X = Z["X"]; assert np.allclose(np.concatenate([St[t]["scores"] for t in tids]), X[:, 1]) and not np.allclose(X[:, 0], X[:, 1])
        cf = R.split_scores(tids, Z["tile"], X[:, 0] * (X[:, 1] / np.maximum(X[:, 0], 1e-9)) ** A * (1 - X[:, 12]) ** B)
        Mb = blurred(split, St, tids)
        rows = {"blur2_sprime": (Mb, {t: St[t]["scores"] for t in tids}), "blur2_R3": (Mb, {t: R3[t]["scores"] for t in tids}),
                "closed": ({t: St[t]["masks_rle"] for t in tids}, cf), "blur2_closed": (Mb, cf)}
        for k, (M_, sc) in rows.items():
            if not os.path.exists(pf(split, k)):
                json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker, S2-final row {k} (POST-HOC, val-selected; sigma 2 px smoothing / closed form a={A} b={B})"},
                           "preds": {t: {"boxes_2048": St[t]["boxes_2048"], "scores": list(map(float, sc[t])), "masks_rle": M_[t]} for t in tids}},
                          open(pf(split, k) + ".tmp", "w")); os.replace(pf(split, k) + ".tmp", pf(split, k))
        t0 = time.time(); cwd = os.getcwd(); os.chdir(R.REPO)
        scored = SC.score_many([pf(split, k) for k in rows], gtp, res=2048, max_dets=600, score_floor=0.0); os.chdir(cwd)
        print(f"[s2f] {split}: score_coco {time.time()-t0:.0f}s", flush=True)
        res = {}
        for k, s in zip(rows, scored):
            json.dump(s, open(pf(split, k).replace("preds_", "scored_"), "w")); cr = s["canopy_neutral_crowd"]
            res[k] = {"mask": {m: cr["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {m: cr["box"][m] for m in ("AP50", "AP75")}}
        ref = {k: json.load(open(pf(split, k).replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"] for k in ("sprime", "rerankR3")}
        # bootstrap via fasteval, after asserting it reproduces score_coco on every row
        E0 = FE.build(St, G, os.path.join(R.FC_, f"fasteval_{split}.pkl")); E2 = FE.build({t: {"masks_rle": Mb[t]} for t in tids}, G, os.path.join(R.FC_, f"fasteval_{split}_{SIG}.pkl"))
        sc_ = {"sprime": (E0, {t: St[t]["scores"] for t in tids}), "rerankR3": (E0, {t: R3[t]["scores"] for t in tids}),
               **{k: ({"closed": E0}.get(k, E2), rows[k][1]) for k in rows}}
        D = {}
        for k, (E, sc) in sc_.items():
            D[k] = per_tile(E, sc); a = FE.summary(ap_from(D[k], tids)); want = res[k]["mask"] if k in res else ref[k]
            assert all(abs(a[m] - want[m]) < 1.5e-4 for m in a), (split, k, a, want)
        print(f"[s2f] {split}: fasteval == score_coco on all 6 rows", flush=True)
        bt = {f"{k}_vs_sprime": boot(D[k], D["sprime"], tids) for k in rows}
        bt["blur2_R3_vs_rerankR3"] = boot(D["blur2_R3"], D["rerankR3"], tids); bt["blur2_closed_vs_closed"] = boot(D["blur2_closed"], D["closed"], tids)
        rec[split] = {"rows": res, "ref": {k: {m: v[m] for m in ("AP50", "AP75", "AP50_95")} for k, v in ref.items()}, "boot": bt}
        print(f"[s2f] {split}: {json.dumps(rec[split])}", flush=True)
        json.dump(rec, open(OUT, "w"), indent=1)
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(rec, open(OUT, "w"), indent=1); print("[s2f] DONE", flush=True)


if __name__ == "__main__":
    main()

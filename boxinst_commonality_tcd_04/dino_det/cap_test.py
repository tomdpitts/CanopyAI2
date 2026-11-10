"""Pre-registration CAP-test (README.md, 2026-11-04): does the 600-per-tile stitch cap bind for LACE (D1 + edge fix + R3)? ONE look, test 439.
Uncapped stitch of the raw D1 subtile dets (results/modal/raw/raw_test.jsonl; dino_modal.stitch with keep=inf), lace4px_simple + edge fix
masks on the uncapped list (gate: first 600 masks and s' == published edge-fixed preds), R3 refit as rerank3.final (gate: refit scores on
the published set == published R3 scores). Frozen score_coco (crowd, floor 0) at maxDets 1200 and 600 vs published R3 + edge fix;
paired tile bootstrap (2000, seed 0) on AP50 and AP50:95. -> results/cap/cap_test.json; masks feat_cache/cap_test/ (resume).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.cap_test
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"; RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "cap")
RAW = os.path.join(RES, "modal", "raw", "raw_test.jsonl"); TDIR = f"{FC}/cap_test"; N_BOOT, SEED = 2000, 0


def stitch_fn():
    """dino_modal.stitch / _edge_cut, exec'd from source so modal need not be importable here."""
    src = open(os.path.join(HERE, "dino_modal.py")).read(); ns = {}
    exec(src[src.index("def _edge_cut"):src.index("@app.function", src.index("def stitch("))], ns); return ns["stitch"]


def per_tile(preds, gd, t2i, md):
    """sam3_on_d1.per_tile (floor 0) with maxDets md instead of the hard-coded 600."""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    dets = SL.build_dt(preds, t2i, 2048, 0.0, with_masks=True)
    cg = COCO(); cg.dataset = gd; cg.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, "segm")
        ev.params.maxDets = [md]; ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]; ev.evaluate()
    return {e["image_id"]: (np.asarray(e["dtScores"]), np.asarray(e["dtMatches"]) > 0, ~np.asarray(e["dtIgnore"], bool),
                            int((~np.asarray(e["gtIgnore"], bool)).sum())) for e in ev.evalImgs if e is not None}


def _job(args):
    from boxinst_commonality_tcd_04.dino_det import edgefix_test as ET
    return ET._job(args)


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
    from boxinst_commonality_tcd_04.dino_det import rerank as R
    from boxinst_commonality_tcd_04.dino_det import rerank3 as R3
    os.makedirs(OUT, exist_ok=True); os.makedirs(TDIR, exist_ok=True); t0 = time.time()
    gtp = S.SPLIT["test"]["gt"]; G = json.load(open(gtp)); tids = sorted(G)
    Pub = json.load(open(os.path.join(RES, "preds_dino_swinl_s0_test_R3_ef_r2048.json")))["preds"]
    Ef = json.load(open(os.path.join(RES, "edgefix", "preds_d1_edgefix_test_sprime_r2048.json")))["preds"]
    Dp = S.det("test")

    # 1. uncapped stitch; capped prefix must equal the published boxes / detector scores
    stitch = stitch_fn(); U = {}
    for ln in open(RAW):
        r = json.loads(ln); raw = [(w["oy"], w["ox"], w["boxes"], w["scores"]) for w in r["windows"]]
        b, s = stitch(raw, keep=10 ** 9); n = len(Dp[r["tid"]]["scores"])
        assert s[:n] == Dp[r["tid"]]["scores"] and np.allclose(b[:n], Dp[r["tid"]]["boxes_2048"]), r["tid"]
        assert n == min(len(s), 600), r["tid"]; U[r["tid"]] = (b, s)
    assert sorted(U) == tids
    over = [t for t in tids if len(U[t][1]) > 600]
    print(f"[cap] uncapped: {len(over)} tiles > 600, max {max(len(U[t][1]) for t in tids)}, +{sum(len(U[t][1]) - 600 for t in over)} dets", flush=True)

    # 2. masks on the uncapped list for tiles over the cap (others reuse published)
    todo = [t for t in over if not os.path.exists(f"{TDIR}/{t}.json")]
    print(f"[cap] render {len(todo)}/{len(over)}", flush=True)
    with mp.get_context("spawn").Pool(12) as pool:
        for k, (t, r) in enumerate(pool.imap_unordered(_job, [("test", t, U[t][0], U[t][1]) for t in todo]), 1):
            json.dump(r, open(f"{TDIR}/{t}.json.tmp", "w")); os.replace(f"{TDIR}/{t}.json.tmp", f"{TDIR}/{t}.json")
            if k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    M = {}
    for t in tids:
        if t in over:
            r = json.load(open(f"{TDIR}/{t}.json"))
            assert [m["counts"] for m in r["masks_rle"][:600]] == [m["counts"] for m in Ef[t]["masks_rle"]], f"mask gate {t}"
            assert np.abs(np.asarray(r["s_prime"][:600]) - np.asarray(Ef[t]["scores"])).max() < 1e-6, f"s' gate {t}"
            M[t] = (r["masks_rle"], r["s_prime"])
        else:
            M[t] = (Ef[t]["masks_rle"], Ef[t]["scores"])
    print("[cap] mask + s' gate PASS", flush=True)

    # 3. R3 refit exactly as rerank3.final; gate on the published set
    ch = json.load(open(os.path.join(RES, "rerankR3_final.json")))["chosen"]; c = {**ch, "cols": tuple(ch["cols"])}
    R3._init(); W = R3._W
    fits = [f for s in range(5) for f in R3.oof_fits(c, s, "rr", W["X"], W["y"], W["soft"], W["tile"], W["F"])[1]]; assert len(fits) == 25
    cols = [R.NAMES.index(x) for x in c["cols"]]

    def r3(b, s, sp):
        b = np.asarray(b, np.float64).reshape(-1, 4); s = np.asarray(s, np.float64); sp = np.asarray(sp, np.float64)
        if not len(s):
            return np.zeros(0)
        X = np.zeros((len(s), R.NF)); h = np.maximum(b[:, 3] - b[:, 1], 1)
        X[:, 0], X[:, 1], X[:, 2], X[:, 4] = s, sp, np.log(np.maximum(sp, 1e-6) / np.maximum(s, 1e-6)), np.log(h)
        X[:, 12] = np.where(s[None, :] > s[:, None], R._box_iou(b, b), 0).max(1)
        return np.mean([f(X[:, cols]) for f in fits], 0)
    dev = max(np.abs(r3(Dp[t]["boxes_2048"], Dp[t]["scores"], Ef[t]["scores"]) - np.asarray(Pub[t]["scores"])).max(initial=0) for t in tids)
    print(f"[cap] R3 gate max |d| = {dev:.2e}", flush=True); assert dev < 1e-6, dev

    # 4. uncapped preds, score at maxDets 1200 and 600
    fp = os.path.join(OUT, "preds_d1_R3_ef_uncapped_test_r2048.json")
    if not os.path.exists(fp):
        P = {t: {"boxes_2048": U[t][0], "scores": r3(U[t][0], U[t][1], M[t][1]).tolist(), "masks_rle": M[t][0]} for t in tids}
        json.dump({"meta": {"mask_res": 2048, "model": "LACE (D1) + edge fix + R3, stitch cap removed (CAP-test)"}, "preds": P},
                  open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    P = json.load(open(fp))["preds"]; pub_fp = os.path.join(RES, "preds_dino_swinl_s0_test_R3_ef_r2048.json"); rows = {}
    for k, f, md in (("published_600", pub_fp, 600), ("uncapped_md1200", fp, 1200), ("uncapped_md600", fp, 600)):
        sp_ = os.path.join(OUT, f"scored_{k}.json")
        if not os.path.exists(sp_):
            cwd = os.getcwd(); os.chdir(REPO); json.dump(SC.score_many([f], gtp, res=2048, max_dets=md, score_floor=0.0)[0], open(sp_, "w")); os.chdir(cwd)
        cr = json.load(open(sp_))["canopy_neutral_crowd"]; rows[k] = {"mask": {m: cr["mask"][m] for m in ("AP50", "AP75", "AP50_95")},
                                                                       "box": {m: cr["box"][m] for m in ("AP50", "AP75")}}
        print(f"[cap] {k}: {rows[k]}", flush=True)
    assert abs(rows["published_600"]["mask"]["AP50"] - 0.7141) < 5e-5 and abs(rows["published_600"]["mask"]["AP50_95"] - 0.3424) < 5e-5

    # 5. paired tile bootstrap (AP at 10 IoU thresholds per tile, as edgefix_test), per-tile top-k by score
    gd, t2i = SL.build_gt_crowd(G, 2048); ids = np.array([t2i[t] for t in tids]); D = {}
    Pp = json.load(open(pub_fp))["preds"]
    for k, Q, md in (("published_600", Pp, 600), ("uncapped_md1200", P, 1200), ("uncapped_md600", P, 600)):
        Qk = {}
        for t in tids:
            o = np.argsort(-np.asarray(Q[t]["scores"]), kind="stable")[:md]
            Qk[t] = {kk: [Q[t][kk][i] for i in o] for kk in ("boxes_2048", "scores", "masks_rle")}
        D[k] = per_tile(Qk, gd, t2i, md)
        for i in ids:
            if i not in D[k]:
                ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
    full = {k: [S.ap_at(ids, D[k], SL.REC_THRS, j) for j in range(10)] for k in D}
    for k in D:
        assert abs(full[k][0] - rows[k]["mask"]["AP50"]) < 5e-4, (k, full[k][0], rows[k]["mask"]["AP50"])
    rng = np.random.default_rng(SEED); dd = {k: [] for k in ("uncapped_md1200", "uncapped_md600")}
    for b in range(N_BOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]; A = {k: [S.ap_at(s, D[k], SL.REC_THRS, j) for j in range(10)] for k in D}
        for k in dd:
            dd[k].append((A[k][0] - A["published_600"][0], np.mean(A[k]) - np.mean(A["published_600"])))
        if (b + 1) % 500 == 0:
            print(f"  boot {b+1}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)
    boot = {}
    for k, v in dd.items():
        v = np.asarray(v)
        boot[f"{k}_minus_published_600"] = {m: {"diff": round(rows[k]["mask"][m] - rows["published_600"]["mask"][m], 4),
                                                "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)]}
                                            for j, m in enumerate(("AP50", "AP50_95"))}
    n = np.array([len(U[t][1]) for t in tids])
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "n_tiles_over_600": int((n > 600).sum()), "max_dets_uncapped": int(n.max()),
           "extra_dets": int((n - np.minimum(n, 600)).sum()), "n_dets_uncapped": int(n.sum()), "r3_gate_max_abs": float(dev),
           "rows": rows, "boot": boot, "note": "ONE look per pre-registration CAP-test; single detector seed"}
    json.dump(rec, open(os.path.join(OUT, "cap_test.json"), "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()

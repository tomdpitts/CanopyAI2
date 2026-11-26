"""BTMASK-EF (pre-registration in btmask/README.md, 2026-11-26): the CURRENT deployed LACE masker (lace4px_simple post4,
c_peak 0.33, guided r 12 eps 1e-4, alpha 0.3, kappa 1.6, 4 px DINOv3, GA.EDGE_FILL True) on BoxTeacher B2's top-600 boxes.
  A = BoxTeacher masks + BT score; B = LACE masks + BT score (primary); C = LACE masks + hat-s (R3 recipe refit on BT val).
Test 439 primary, sparse 236 secondary. ONE look. Resumable: per-tile masks feat_cache/btmask_ef_<split>/, features
feat_cache/btmask_ef_feats_<split>.npz, fasteval caches feat_cache/btmask_ef_fasteval_<split>_<A|B>.pkl, gates gate_ef.json,
results compare_ef.json (per split). Never overwrites the 11-02 files.
  cd ~/dphil/CanopyAI2 && PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.btmask.btmask_ef \
      > boxinst_commonality_tcd_04/native_raster/logs/btmask_ef_2026-11-26.log 2>&1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM

HERE, FC, PKG, REPO = BM.HERE, BM.FC, BM.PKG, BM.REPO
DD = os.path.join(PKG, "dino_det")
GATES = os.path.join(HERE, "gate_ef.json"); OUT = os.path.join(HERE, "compare_ef.json")
BT_SRC = {"val": os.path.join(BM.BT, "preds_boxteacher_sub_val_r2048.json"),
          "test": os.path.join(BM.BT, "preds_boxteacher_sub_test_r2048.json"),
          "sparse": os.path.join(PKG, "box_supervised_baselines", "tierb", "results_sparse", "preds_boxteacher_sub_sparse_r2048.json")}
BT_SCORED = {"test": os.path.join(BM.BT, "scored_boxteacher_sub_test_r2048.json"),
             "sparse": os.path.join(PKG, "box_supervised_baselines", "tierb", "results_sparse", "scored_boxteacher_sub_sparse_r2048.json")}
NPROC = 12; CFG = BM.FROZEN; CN = BM.cfg_name(CFG)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bt_top(split):
    """= btmask.bt_top (top 600 per tile by BT score, stable), but from BT_SRC so sparse works too."""
    P = json.load(open(BT_SRC[split]))["preds"]; out = {}
    for t, r in P.items():
        o = np.argsort(-np.asarray(r["scores"], np.float64), kind="stable")[:BM.TOPK]
        out[t] = {k: [r[k][i] for i in o] for k in ("boxes_2048", "scores", "masks_rle")}
    return out


def _render(split, t, boxes, scores):
    if "m4" not in BM._W or BM._W.get("split") != split:
        BM._init(split)
    GA = BM._W["GA"]; assert GA.EDGE_FILL is True, "EDGE_FILL must be True (deployed)"
    return BM.render_boxes(split, t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [CFG])[CN]


def _job(a):
    split, t, boxes, scores = a
    fp = f"{FC}/btmask_ef_{split}/{t}.json"
    if not os.path.exists(fp):
        r = _render(split, t, boxes, scores)
        json.dump(r, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def render(split, B, pool):
    d = f"{FC}/btmask_ef_{split}"; os.makedirs(d, exist_ok=True); tids = sorted(B)
    todo = [t for t in tids if not os.path.exists(f"{d}/{t}.json")]; t0 = time.time()
    log(f"[render {split}] {len(todo)}/{len(tids)} tiles to render")
    for k, _ in enumerate(pool.imap_unordered(_job, [(split, t, B[t]["boxes_2048"], B[t]["scores"]) for t in todo]), 1):
        if k % 10 == 0 or k == len(todo):
            log(f"  [render {split}] {k}/{len(todo)} {time.time()-t0:.0f}s")
    return {t: json.load(open(f"{d}/{t}.json")) for t in tids}


def _feat_job(a):
    from boxinst_commonality_tcd_04.dino_det import rerank as R
    return R._feats(a)


def features(split, P, gt, pool):
    """rerank._feats per tile (R1's 24 features; y = box50 labels when gt given) -> own cache (never rerankR1_*)."""
    from boxinst_commonality_tcd_04.dino_det import rerank as R
    fp = f"{FC}/btmask_ef_feats_{split}.npz"
    if os.path.exists(fp):
        Z = np.load(fp); return list(Z["tids"]), Z["tile"], Z["X"], (Z["y"] if "y" in Z else None)
    tids = sorted(P); Rr = {}; t0 = time.time()
    for k, (t, X, y) in enumerate(pool.imap_unordered(_feat_job, [(split, t, P[t], gt[t] if gt else None) for t in tids]), 1):
        Rr[t] = (X, y)
        if k % 20 == 0 or k == len(tids):
            log(f"  [feats {split}] {k}/{len(tids)} {time.time()-t0:.0f}s")
    tile = np.concatenate([np.full(len(Rr[t][0]), i) for i, t in enumerate(tids)]); X = np.concatenate([Rr[t][0] for t in tids])
    d = {"tids": np.asarray(tids), "tile": tile, "X": X}
    if gt:
        d["y"] = np.concatenate([Rr[t][1] for t in tids])
    np.savez_compressed(fp + ".tmp.npz", **d); os.replace(fp + ".tmp.npz", fp)
    return tids, tile, X, d.get("y")


def r3_chosen():
    c = json.load(open(os.path.join(DD, "results", "rerankR3_final.json")))["chosen"]
    assert c == {"model": "mlp1x64", "epochs": 30, "dropout": 0.1, "wd": 0.001, "label": "box50",
                 "cols": ["s", "log_sp_s", "log_h", "max_iou_higher"]}, c
    return {**c, "cols": tuple(c["cols"])}


def r3_fits(X, y, tile, n_tiles):
    """Frozen R3 recipe: 25 models = seeds 0-4 x round-robin 5 tile folds; returns (fits, col indices)."""
    from boxinst_commonality_tcd_04.dino_det import rerank as R, rerank3 as R3
    c = r3_chosen(); F = R3.fold_sets(n_tiles)
    fits = [f for s in range(5) for f in R3.oof_fits(c, s, "rr", X, y, None, tile, F)[1]]; assert len(fits) == 25
    return fits, [R.NAMES.index(x) for x in c["cols"]]


# ------------------------------------------------------------------ gates
def gates(pool):
    g = json.load(open(GATES)) if os.path.exists(GATES) else {}
    from boxinst_commonality_tcd_04.dino_det import rerank as R, s2_final as F
    if "g1" not in g:                     # gate 1: masker on D1 boxes == deployed edge-fixed D1 masks (3 test tiles)
        Dp = json.load(open(os.path.join(DD, "results", "preds_dino_swinl_s0_test_detscore_r2048.json")))["preds"]
        Pe = json.load(open(os.path.join(DD, "results", "edgefix", "preds_d1_edgefix_test_sprime_r2048.json")))["preds"]
        tiles = [t for t in sorted(Pe) if Pe[t]["scores"]][:3]; per = {}
        for t in tiles:
            assert Dp[t]["boxes_2048"] == Pe[t]["boxes_2048"], t
            r = pool.apply(_render, ("test", t, Dp[t]["boxes_2048"], Dp[t]["scores"]))
            eq = sum(a["counts"] == b["counts"] for a, b in zip(r["masks_rle"], Pe[t]["masks_rle"]))
            n = len(Pe[t]["masks_rle"]); assert len(r["masks_rle"]) == n
            dsp = float(np.max(np.abs(np.asarray(r["s_prime"]) - np.asarray(Pe[t]["scores"]))))
            per[t] = {"masks_identical": f"{eq}/{n}", "s_prime_max_abs": dsp}; log(f"[gate1] {t} {per[t]}")
        ok = all(v["masks_identical"].split("/")[0] == v["masks_identical"].split("/")[1] and v["s_prime_max_abs"] < 1e-6 for v in per.values())
        g["g1"] = {"desc": "render_boxes(D1 test boxes, EDGE_FILL True) == preds_d1_edgefix_test_sprime masks byte-identical + s' < 1e-6",
                   "tiles": per, "pass": bool(ok), "date": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(g, open(GATES, "w"), indent=1)
    log(f"[gate1] pass={g['g1']['pass']}"); assert g["g1"]["pass"], "GATE 1 FAILED"
    if "g3" not in g:                     # gate 3 (arm C only): feature code + R3 recipe code reproduce published
        vg = json.load(open(R.VAL_GT)); Pv0 = json.load(open(os.path.join(FC, f"{R.TAG}_val_mask.json")))
        Z = np.load(os.path.join(FC, "rerankR1_val.npz")); zt = list(Z["tids"]); fe = True
        for t in zt[:3]:
            _, X, y = R._feats(("val", t, Pv0[t], vg[t])); i = zt.index(t)
            fe &= bool(np.allclose(X, Z["X"][Z["tile"] == i]) and np.array_equal(y, Z["y"][Z["tile"] == i]))
        fits, cols = r3_fits(Z["X"], Z["y"], Z["tile"], len(zt))
        Zt = np.load(os.path.join(FC, "rerankR1_test.npz")); tt = list(Zt["tids"])
        sc = R.split_scores(tt, Zt["tile"], np.mean([f(Zt["X"][:, cols]) for f in fits], 0))
        R3p = json.load(open(F.pf("test", "R3_ef")))["preds"]
        d = max(float(np.max(np.abs(np.asarray(sc[t]) - np.asarray(R3p[t]["scores"])))) if len(sc[t]) else 0.0 for t in tt)
        g["g3"] = {"desc": "rerank._feats reproduces rerankR1_val.npz (3 val tiles); R3 recipe refit reproduces published R3 test scores",
                   "feats_match": fe, "r3_refit_max_abs": d, "pass": bool(fe and d < 1e-6), "date": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(g, open(GATES, "w"), indent=1)
    log(f"[gate3] {g['g3']}")
    return g


# ------------------------------------------------------------------ main
def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.dino_det import rerank as R, s2_final as F, fasteval as FE
    log("[btmask_ef] start")
    with mp.get_context("spawn").Pool(NPROC) as pool:
        g = gates(pool); do_C = g["g3"]["pass"]
        if not do_C:
            log("[C] SKIPPED: gate 3 failed")
        fits = cols = None
        if do_C:                          # fit hat-s on BoxTeacher VAL detections (no test number involved)
            vg = json.load(open(R.VAL_GT)); Bv = bt_top("val"); assert sorted(Bv) == sorted(vg) and len(Bv) == 108
            Mv = render("val", Bv, pool)
            Pv = {t: {"boxes_2048": Bv[t]["boxes_2048"], "scores": Bv[t]["scores"], "s_prime": Mv[t]["s_prime"],
                      "masks_rle": Mv[t]["masks_rle"]} for t in sorted(Bv)}
            vt, vtile, vX, vy = features("val", Pv, vg, pool)
            log(f"[C] val feats: n={len(vX)} pos={int((vy == 1).sum())} neg={int((vy == 0).sum())} crowd={int((vy == -1).sum())}")
            fits, cols = r3_fits(vX, vy, vtile, len(vt)); log("[C] 25 R3-recipe models fitted on BoxTeacher val")
        rec = json.load(open(OUT)) if os.path.exists(OUT) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
        for split in ("test", "sparse"):
            if split in rec:
                log(f"[{split}] already done"); continue
            gtp = BM.SPLIT[split]["gt"]; G = json.load(open(gtp)); B = bt_top(split); tids = sorted(B)
            assert tids == sorted(G) and len(tids) == BM.SPLIT[split]["n"], split
            M = render(split, B, pool)
            arms = {"A": ({t: B[t]["masks_rle"] for t in tids}, {t: B[t]["scores"] for t in tids}, 0.05,
                          "BoxTeacher B2, top 600/tile by its score, its own masks (arm A)"),
                    "B": ({t: M[t]["masks_rle"] for t in tids}, {t: B[t]["scores"] for t in tids}, 0.05,
                          "deployed LACE masker (lace4px_simple, EDGE_FILL True) on BoxTeacher boxes, BoxTeacher score (arm B)")}
            if do_C:
                P = {t: {"boxes_2048": B[t]["boxes_2048"], "scores": B[t]["scores"], "s_prime": M[t]["s_prime"],
                         "masks_rle": M[t]["masks_rle"]} for t in tids}
                st, stile, sX, _ = features(split, P, None, pool); assert st == tids
                assert np.allclose(sX[:, 0], np.concatenate([B[t]["scores"] for t in tids]))
                arms["C"] = ({t: M[t]["masks_rle"] for t in tids}, R.split_scores(tids, stile, np.mean([f(sX[:, cols]) for f in fits], 0)), 0.0,
                             "deployed LACE masker on BoxTeacher boxes, hat-s (R3 recipe refit on BoxTeacher val) (arm C)")
            files = {}
            for a, (mk, sc, floor, model) in arms.items():
                fp = os.path.join(HERE, f"preds_{a}_ef_{split}_r2048.json"); files[a] = fp
                if not os.path.exists(fp):
                    json.dump({"meta": {"mask_res": 2048, "model": model, "score_floor": floor},
                               "preds": {t: {"boxes_2048": B[t]["boxes_2048"], "scores": list(map(float, sc[t])), "masks_rle": mk[t]} for t in tids}},
                              open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
            cwd = os.getcwd(); os.chdir(REPO); met = {}
            for a, fp in files.items():
                sp = fp.replace("preds_", "scored_")
                if not os.path.exists(sp):
                    t0 = time.time(); json.dump(SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=arms[a][2])[0], open(sp, "w"))
                    log(f"[{split}] scored {a} {time.time()-t0:.0f}s")
                r = json.load(open(sp))["canopy_neutral_crowd"]
                met[a] = {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")} | {"box_AP50": r["box"]["AP50"], "floor": arms[a][2]}
            os.chdir(cwd)
            pub = json.load(open(BT_SCORED[split]))["canopy_neutral_crowd"]["mask"]
            gate2 = all(met["A"][k] == pub[k] for k in ("AP50", "AP75", "AP50_95"))
            log(f"[{split}] gate 2 (A == published {pub}): {gate2}")
            if not gate2:
                rec[split] = {"gate2_A_reproduces_published": False, "published": pub, "A": met["A"]}
                json.dump(rec, open(OUT, "w"), indent=1); raise SystemExit(f"GATE 2 FAILED on {split}")
            # paired tile bootstrap on fasteval (gate 4: fasteval == score_coco)
            E = {a: FE.build({t: {"masks_rle": arms[a][0][t]} for t in tids}, G, f"{FC}/btmask_ef_fasteval_{split}_{a}.pkl") for a in ("A", "B")}
            D = {a: F.per_tile(E["A" if a == "A" else "B"], arms[a][1]) for a in arms}
            g4 = {}
            for a, d in D.items():
                s_ = FE.summary(F.ap_from(d, tids)); g4[a] = {k: s_[k] for k in ("AP50", "AP50_95")}
                assert all(abs(s_[k] - met[a][k]) < 1.5e-4 for k in ("AP50", "AP50_95")), ("gate 4", split, a, s_, met[a])
            log(f"[{split}] gate 4: fasteval == score_coco on {list(D)}")
            boot = {f"{a}_minus_A": F.boot(D[a], D["A"], tids) for a in arms if a != "A"}
            rec[split] = {"gate2_A_reproduces_published": True, "published_A": {k: pub[k] for k in ("AP50", "AP75", "AP50_95")},
                          "gate4_fasteval": g4, "metrics": met, "boot": boot, "C_run": do_C,
                          "files": {a: os.path.relpath(fp, PKG) for a, fp in files.items()}, "date": time.strftime("%Y-%m-%d %H:%M")}
            json.dump(rec, open(OUT, "w"), indent=1); log(f"[{split}] {json.dumps(rec[split])}")
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(rec, open(OUT, "w"), indent=1); log("[btmask_ef] DONE")


if __name__ == "__main__":
    main()

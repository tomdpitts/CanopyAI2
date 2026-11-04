"""Pre-registration R2 (README.md, POST-HOC): R1's MLP without appearance features. Val robustness gate, then test once.

  cv  : 5 seeds x 3 fold assignments OOF on val (the R1 features cache) + bootstrap -> results/rerankR2_cv.json; gate.
  test: ONLY if the gate passed. The mean of 25 models (5 seeds x 5 round-robin folds) on the test 439 -> frozen score_coco
        + paired bootstrap vs s'.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.rerank2 cv|test
"""
import argparse, contextlib, io, json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R

COLS = R.SCORE + R.GEOM + R.CTX; SEEDS = [0, 1, 2, 3, 4]


def fold_sets(n):
    rr = [i % R.K for i in range(n)]
    out = {"rr": rr}
    for s in (1, 2):
        p = np.random.default_rng(s).permutation(n); f = np.zeros(n, int); f[p] = np.arange(n) % R.K; out[f"perm{s}"] = f.tolist()
    return out


def cv():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(R.OUT, "rerankR2_cv.json"); assert not os.path.exists(out), out
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids, tile, X, y = R.features("val", P, vg); gd, t2i = SL.build_gt_crowd(vg, 2048)

    def val_ap(v):
        sc = R.split_scores(tids, tile, v)
        pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in tids}
        with contextlib.redirect_stdout(io.StringIO()):
            return round(SL.score(pr, gd, t2i, masks=True, floor=0.0)["mask"]["AP50"], 4), pr
    base, Pb = val_ap(np.concatenate([np.asarray(P[t]["s_prime"]) for t in tids])); assert abs(base - 0.7384) < 5e-4, base
    F = fold_sets(len(tids)); runs = []; t0 = time.time(); pr00 = None
    for fname, tf in F.items():
        for seed in SEEDS:
            v, _ = R.oof(tids, tile, X, y, "mlp", COLS, seed, tf); a, pr = val_ap(v)
            if fname == "rr" and seed == 0:
                pr00 = pr
            runs.append({"folds": fname, "seed": seed, "AP50": a, "gain": round(a - base, 4)})
            print(f"[R2] {fname} seed{seed}: {a} ({a - base:+.4f}) {time.time()-t0:.0f}s", flush=True)
    lr, _ = R.oof(tids, tile, X, y, "logreg", COLS); lra, _ = val_ap(lr)
    g = np.asarray([r["gain"] for r in runs]); bt = R.boot(pr00, Pb, vg, 0.0, 0.0)
    gate = bool(g.mean() >= 0.005 and (g > 0).sum() >= 13 and bt["ci95"][0] > 0)
    rec = {"base_sprime_AP50": base, "runs": runs, "gain_mean": round(float(g.mean()), 4), "gain_sd": round(float(g.std(ddof=1)), 4),
           "gain_min_max": [round(float(g.min()), 4), round(float(g.max()), 4)], "n_pos": int((g > 0).sum()),
           "boot_rr_seed0_vs_sprime": bt, "logreg_no_app": {"AP50": lra, "gain": round(lra - base, 4)},
           "gate_pass": gate, "cols": [R.NAMES[c] for c in COLS], "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: v for k, v in rec.items() if k != "runs"}, indent=1), flush=True)


def test():
    from boxinst_commonality_tcd_04 import score_coco as SC
    cvr = json.load(open(os.path.join(R.OUT, "rerankR2_cv.json"))); assert cvr["gate_pass"], "gate failed: test must not be touched"
    out = os.path.join(R.OUT, "rerankR2_test.json"); assert not os.path.exists(out), out
    vg = json.load(open(R.VAL_GT)); Pv = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids_v, tile_v, Xv, yv = R.features("val", Pv, vg)
    fits = [f for s in SEEDS for f in R.oof(tids_v, tile_v, Xv, yv, "mlp", COLS, s)[1]]; assert len(fits) == 25
    St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")))["preds"]
    Sd = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_detscore_r2048.json")))["preds"]  # detector s (the sprime file stores s' in "scores")
    assert all(Sd[t]["boxes_2048"] == St[t]["boxes_2048"] for t in St)
    M = {t: {**St[t], "scores": Sd[t]["scores"], "s_prime": St[t]["scores"]} for t in St}
    tids, tile, X, _ = R.features("test", M, None)
    v = np.mean([f(X[:, COLS]) for f in fits], 0); sc = R.split_scores(tids, tile, v)
    preds = {t: {"boxes_2048": St[t]["boxes_2048"], "scores": sc[t], "masks_rle": St[t]["masks_rle"]} for t in tids}
    fp = os.path.join(R.OUT, f"preds_{R.TAG}_test_rerankR2_r2048.json")
    json.dump({"meta": {"mask_res": 2048, "model": "D1 + LACE masker, R2 reranker (POST-HOC; MLP no-appearance, val-trained, box labels, 25-model mean)"},
               "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    os.chdir(R.REPO)
    c = SC.score_many([fp], R.TEST_GT, res=2048, max_dets=600, score_floor=0.0)[0]
    json.dump(c, open(fp.replace("preds_", "scored_"), "w")); cr = c["canopy_neutral_crowd"]
    bt = R.boot(preds, St, json.load(open(R.TEST_GT)), 0.0, 0.0)
    rec = {"mask": {k: cr["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: cr["box"][k] for k in ("AP50", "AP75")},
           "sprime_ref": {"AP50": 0.7035, "AP75": 0.2615, "AP50_95": 0.3347}, "boot_vs_sprime": bt, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


def sparse():
    """Pre-registration R2-sparse: the frozen R2 models, once, on the sparse-236 holdout."""
    from boxinst_commonality_tcd_04 import score_coco as SC
    out = os.path.join(R.OUT, "rerankR2_sparse.json"); assert not os.path.exists(out), out
    gt_path = os.path.join(R.REPO, "data", "tcd_sparse", "sparse_gt.json"); G = json.load(open(gt_path))
    vg = json.load(open(R.VAL_GT)); Pv = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); assert not set(vg) & set(G)
    tids_v, tile_v, Xv, yv = R.features("val", Pv, vg)
    fits = [f for s in SEEDS for f in R.oof(tids_v, tile_v, Xv, yv, "mlp", COLS, s)[1]]; assert len(fits) == 25
    St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_sparse_sprime_r2048.json")))["preds"]
    Sd = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_sparse_detscore_r2048.json")))["preds"]
    assert sorted(St) == sorted(G) and all(Sd[t]["boxes_2048"] == St[t]["boxes_2048"] for t in St)
    M = {t: {**St[t], "scores": Sd[t]["scores"], "s_prime": St[t]["scores"]} for t in St}
    tids, tile, X, _ = R.features("sparse", M, None); assert not np.allclose(X[:, 0], X[:, 1])
    v = np.mean([f(X[:, COLS]) for f in fits], 0); sc = R.split_scores(tids, tile, v)
    preds = {t: {"boxes_2048": St[t]["boxes_2048"], "scores": sc[t], "masks_rle": St[t]["masks_rle"]} for t in tids}
    fp = os.path.join(R.OUT, f"preds_{R.TAG}_sparse_rerankR2_r2048.json")
    json.dump({"meta": {"mask_res": 2048, "model": "D1 + LACE masker, R2 reranker (POST-HOC; frozen 25 models), sparse-236"}, "preds": preds},
              open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    os.chdir(R.REPO)
    c = SC.score_many([fp], gt_path, res=2048, max_dets=600, score_floor=0.0)[0]
    json.dump(c, open(fp.replace("preds_", "scored_"), "w")); cr = c["canopy_neutral_crowd"]
    bt = R.boot(preds, St, G, 0.0, 0.0)
    t439 = set(json.load(open(R.TEST_GT))); keep = [t for t in tids if t not in t439]
    bt220 = R.boot({t: preds[t] for t in keep}, {t: St[t] for t in keep}, {t: G[t] for t in keep}, 0.0, 0.0)
    rec = {"mask": {k: cr["mask"][k] for k in ("AP50", "AP75", "AP50_95")}, "box": {k: cr["box"][k] for k in ("AP50", "AP75")},
           "sprime_ref": {"AP50": 0.7349, "AP75": 0.390, "AP50_95": 0.396}, "boot_vs_sprime_236": bt,
           "boot_vs_sprime_220_not_in_439": {**bt220, "n": len(keep)},
           "confirmed": bool(bt["diff"] > 0 and bt["ci95"][0] > 0), "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["cv", "test", "sparse"]); a = a.parse_args()
    {"cv": cv, "test": test, "sparse": sparse}[a.cmd]()

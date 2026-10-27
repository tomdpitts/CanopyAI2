"""CONTROL: does local (CPU) recomputation of the ranking s' change AP vs the published (Modal) s'?

Published deployed test rows (preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json) carry Modal-computed s' = s*bimod*pfg_mean.
test_eval's own-posterior ranking recomputes s' locally (max |diff| 1.53e-6 on the gate tiles). Here: the PUBLISHED masks,
re-ranked by s' recomputed locally from the 8 px posterior (same code path as test_eval rank "own8"), 439 tiles x 3 seeds,
scored with score_coco (2048, maxDets 600, floor 0, canopy crowd). Pass = AP50 and AP50:95 equal to the published scored
rows to 4 dp. Outputs (refuse to overwrite): hunt/test/{preds,scored}_lace_s{s}_r2048_deployed_localrank8.json,
hunt/test/rank_precision_control.json. Per-tile checkpoints in feat_cache/hunt_test_localrank8_s{s}/.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.rank_precision_control
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.native_raster.hunt import test_eval as TE

_W = {}


def _init():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    _W.update(m=E.TCDMasker(f"{TE.FC}/em_model_4p_fix.npz"), GA=GA)


def _job(args):
    t, boxes, s_raw, d = args
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        m, GA = _W["m"], _W["GA"]
        bx = np.asarray(boxes, np.float32).reshape(-1, 4); sc = []
        if len(bx):
            feat = np.load(f"{TE.FC}/feat_4p_test_dl/feat_4p_test/{t}.npy").astype(np.float32)
            g = feat.shape[-1]; zn = m.project(feat); del feat
            for i, b in enumerate(bx):
                _, _, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
                sc.append(round(float(s_raw[i] * np.abs(2 * r - 1).mean() * r.mean()), 8))
        json.dump({"scores": sc}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def main():
    out = os.path.join(TE.OUTD, "rank_precision_control.json"); assert not os.path.exists(out), out
    paths = []
    for s in (0, 1, 2):
        P = json.load(open(TE.pub(s)))["preds"]; D = json.load(open(TE.detscore(s)))["preds"]; tids = sorted(P)
        assert len(tids) == 439 and all(D[t]["boxes_2048"] == P[t]["boxes_2048"] for t in tids)
        d = f"{TE.FC}/hunt_test_localrank8_s{s}"; os.makedirs(d, exist_ok=True)
        todo = [(t, P[t]["boxes_2048"], D[t]["scores"], d) for t in tids if not os.path.exists(os.path.join(d, t + ".json"))]
        print(f"[ctrl s{s}] {len(todo)}/439 tiles", flush=True); t0 = time.time()
        with mp.get_context("spawn").Pool(12, initializer=_init) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, todo), 1):
                if k % 40 == 0 or k == len(todo):
                    print(f"  s{s} {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        fp = os.path.join(TE.OUTD, f"preds_lace_s{s}_r2048_deployed_localrank8.json")
        if not os.path.exists(fp):
            preds, dmax, nswap = {}, 0.0, 0
            for t in tids:
                sc = json.load(open(os.path.join(d, t + ".json")))["scores"]
                pub_sc = np.asarray(P[t]["scores"]); loc = np.asarray(sc)
                if len(loc):
                    dmax = max(dmax, float(np.abs(loc - pub_sc).max()))
                    nswap += int((np.argsort(-loc, kind="stable") != np.argsort(-pub_sc, kind="stable")).sum())
                preds[t] = {"boxes_2048": P[t]["boxes_2048"], "scores": sc, "masks_rle": P[t]["masks_rle"]}
            json.dump({"meta": {"mask_res": 2048, "model": f"LACE s{s} published deployed masks, s' recomputed locally (8 px)",
                                "max_abs_score_diff_vs_published": dmax, "within_tile_rank_positions_changed": nswap},
                       "preds": preds}, open(fp, "w"))
        paths.append(fp)
    os.chdir(TE.REPO)
    from boxinst_commonality_tcd_04 import score_coco as SC
    todo_sc = [p for p in paths if not os.path.exists(p.replace("preds_", "scored_"))]
    for p, r in zip(todo_sc, SC.score_many(todo_sc, os.path.join(TE.PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=0.0) if todo_sc else []):
        json.dump(r, open(p.replace("preds_", "scored_"), "w"))
    res = {}
    for s, fp in zip((0, 1, 2), paths):
        loc = json.load(open(fp.replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]
        pub = json.load(open(os.path.join(TE.NR, f"scored_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))["canopy_neutral_crowd"]["mask"]
        meta = json.load(open(fp))["meta"]
        res[s] = {"local": loc, "published": pub, "dAP50": round(loc["AP50"] - pub["AP50"], 6), "dAP50_95": round(loc["AP50_95"] - pub["AP50_95"], 6),
                  "max_abs_score_diff": meta["max_abs_score_diff_vs_published"], "rank_positions_changed": meta["within_tile_rank_positions_changed"]}
        print(f"[ctrl s{s}] AP50 local {loc['AP50']:.4f} vs published {pub['AP50']:.4f} | AP50:95 {loc['AP50_95']:.4f} vs {pub['AP50_95']:.4f} | "
              f"max|ds| {meta['max_abs_score_diff_vs_published']:.2e}, rank positions changed {meta['within_tile_rank_positions_changed']}", flush=True)
    res["pass_4dp"] = all(abs(res[s]["dAP50"]) < 5e-5 and abs(res[s]["dAP50_95"]) < 5e-5 for s in (0, 1, 2))
    json.dump(res, open(out, "w"), indent=1); print(f"[ctrl] pass (4 dp): {res['pass_4dp']}", flush=True)


if __name__ == "__main__":
    main()

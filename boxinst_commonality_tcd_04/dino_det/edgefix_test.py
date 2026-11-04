"""Top/left tile-border fix on TEST 439 (primary) and sparse 236 (secondary), run ONCE per pre-registration in
dino_det/SAM3_D1.md. Fix and wrapper exactly as edgefix_val.py (edge-replicate the 6 px band _prob zero-fills).
Baseline = published D1 preds (results/preds_dino_swinl_s0_<split>_sprime_r2048.json); fixed masks re-rendered from the
same boxes + raw detector scores; s' asserted identical. Masks -> feat_cache/edgefix_<split>/ (resume).
Outputs results/edgefix/{preds,scored}_d1_edgefix_<split>_sprime_r2048.json, edgefix_<split>.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.edgefix_test
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"; RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "edgefix")
CFG = {"e": 0.0, "c": 0.33}; N_BOOT, SEED = 2000, 0


def _job(args):
    split, t, boxes, scores = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    from boxinst_commonality_tcd_04.dino_det import edgefix_val as EV
    if "m4" not in BM._W:
        BM._init(split)
    EV._install(True)
    return t, BM.render_boxes(split, t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [CFG])[BM.cfg_name(CFG)]


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
    os.makedirs(OUT, exist_ok=True)
    for split in ("test", "sparse"):
        base_fp = os.path.join(RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json")
        Bp = json.load(open(base_fp))["preds"]; Dp = S.det(split); tids = sorted(Bp); assert len(tids) == S.SPLIT[split]["n"]
        tdir = f"{FC}/edgefix_{split}"; os.makedirs(tdir, exist_ok=True)
        todo = [t for t in tids if not os.path.exists(f"{tdir}/{t}.json")]; t0 = time.time()
        print(f"[edgefix {split}] {len(todo)}/{len(tids)} to render", flush=True)
        with mp.get_context("spawn").Pool(12) as pool:
            for n, (t, r) in enumerate(pool.imap_unordered(_job, [(split, t, Dp[t]["boxes_2048"], Dp[t]["scores"]) for t in todo]), 1):
                json.dump(r, open(f"{tdir}/{t}.json.tmp", "w")); os.replace(f"{tdir}/{t}.json.tmp", f"{tdir}/{t}.json")
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        F = {t: json.load(open(f"{tdir}/{t}.json")) for t in tids}
        for t in tids:
            assert Bp[t]["boxes_2048"] == Dp[t]["boxes_2048"], t
            assert np.abs(np.asarray(F[t]["s_prime"], float) - np.asarray(Bp[t]["scores"], float)).max(initial=0) < 1e-6, t
        n_ch = sum(a["counts"] != b["counts"] for t in tids for a, b in zip(F[t]["masks_rle"], Bp[t]["masks_rle"]))
        fp = os.path.join(OUT, f"preds_d1_edgefix_{split}_sprime_r2048.json")
        if not os.path.exists(fp):
            json.dump({"meta": {"mask_res": 2048, "model": "LACE (D1 variant) + top/left border fix (edge-replicated _prob band)", "rank": "s'"},
                       "preds": {t: {"boxes_2048": Bp[t]["boxes_2048"], "scores": Bp[t]["scores"], "masks_rle": F[t]["masks_rle"]} for t in tids}},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_"); gtp = S.SPLIT[split]["gt"]
        if not os.path.exists(sp):
            cwd = os.getcwd(); os.chdir(REPO); json.dump(SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=0.0)[0], open(sp, "w")); os.chdir(cwd)
        rows = {}
        for k, f in (("baseline", base_fp), ("edgefix", fp), ("sam3zs_gmean", os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json"))):
            c = json.load(open(f.replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]; rows[k] = {m: c[m] for m in ("AP50", "AP75", "AP50_95")}
            print(f"[{split} {k}] {rows[k]}", flush=True)
        g = json.load(open(gtp)); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in sorted(g)]); D = {}
        for k, f in (("baseline", base_fp), ("edgefix", fp), ("sam3zs_gmean", os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json"))):
            P = json.load(open(f))["preds"]; D[k] = S.per_tile({t: P[t] for t in sorted(g)}, gd, t2i, 0.0)
            for i in ids:
                if i not in D[k]:
                    ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                    D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
        rng = np.random.default_rng(SEED); dd = {"edgefix_minus_baseline": [], "edgefix_minus_sam3zs_gmean": []}; t0 = time.time()
        for b in range(N_BOOT):
            s = ids[rng.integers(0, len(ids), len(ids))]
            A = {k: [S.ap_at(s, D[k], SL.REC_THRS, j) for j in range(10)] for k in D}
            for name, o in (("edgefix_minus_baseline", "baseline"), ("edgefix_minus_sam3zs_gmean", "sam3zs_gmean")):
                dd[name].append((A["edgefix"][0] - A[o][0], A["edgefix"][5] - A[o][5], np.mean(A["edgefix"]) - np.mean(A[o])))
            if (b + 1) % 500 == 0:
                print(f"  boot {b+1}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)
        boot = {}
        for name, o in (("edgefix_minus_baseline", "baseline"), ("edgefix_minus_sam3zs_gmean", "sam3zs_gmean")):
            v = np.asarray(dd[name])
            boot[name] = {m: {"diff": round(rows["edgefix"][m] - rows[o][m], 4), "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)]}
                          for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
        rec = {"split": split, "date": time.strftime("%Y-%m-%d %H:%M"), "n_masks_changed": int(n_ch),
               "n_masks": int(sum(len(Bp[t]["masks_rle"]) for t in tids)), "mask_AP": rows, "boot": boot,
               "note": "run once per pre-registration; test-set sampling only; single detector seed"}
        json.dump(rec, open(os.path.join(OUT, f"edgefix_{split}.json"), "w"), indent=1); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

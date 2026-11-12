"""Pre-registration T3 (README.md, 2026-11-10, Tom: "ok please run 1 end to end including table update"). ONE look, test 439 + sparse-236.
Fills paper Table 3 (tab:factors): factor rows s*m, s*pbar and the hat-s (repo R3) input-drop rows, on the frozen D1 detection set with
the edge-fixed masks; only the ranking changes.
  Gate 1: s * pbar * m (recomputed from GA._prob, as abl_lib.render) == cached s' on every box.
  Gate 2: the 4-input refit == published R3_ef scores.        Gate 3: fasteval == score_coco on every row.
-> results/t3_abl.json; preds/scored results/{preds,scored}_dino_swinl_s0_<split>_t3_<row>_r2048.json; per-tile pbar/m feat_cache/t3_pm_<split>/;
log native_raster/logs/t3_abl_2026-11-10.log
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.t3_abl
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det import s2_final as F

OUT = os.path.join(R.OUT, "t3_abl.json"); EF = os.path.join(R.OUT, "edgefix"); NPROC = 12
FULL = ("s", "log_sp_s", "log_h", "max_iou_higher")
DROPS = {"no_s": "s", "no_logpm": "log_sp_s", "no_logh": "log_h", "no_iou": "max_iou_higher"}


def _pm(a):
    """Per tile: pbar = mean r, m = mean |2r - 1| over the in-box 4 px posterior r (GA._prob, deployed masker, alpha 0.3, kappa x1.6)."""
    split, t, boxes = a
    fp = os.path.join(R.FC_, f"t3_pm_{split}", t + ".json")
    if os.path.exists(fp):
        return t
    from boxinst_commonality_tcd_04.dino_det import abl_lib as AL
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    GA, _ = AL._mods(); m = AL.masker(AL.DEPLOYED, True)
    zn4 = np.load(f"{BM.SPLIT[split]['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128); P, M = [], []
    for b in np.asarray(boxes, np.float32).reshape(-1, 4):
        r = GA._prob(m, zn4, 512, b, 0.3, 1.6)[2]; P.append(float(r.mean())); M.append(float(np.abs(2 * r - 1).mean()))
    json.dump({"pbar": P, "m": M}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def pm(split, St, tids):
    import multiprocessing as mp
    os.makedirs(os.path.join(R.FC_, f"t3_pm_{split}"), exist_ok=True); t0 = time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for k, _ in enumerate(pool.imap_unordered(_pm, [(split, t, St[t]["boxes_2048"]) for t in tids]), 1):
            if k % 25 == 0 or k == len(tids):
                print(f"  [t3] {split} pbar/m {k}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    return {t: json.load(open(os.path.join(R.FC_, f"t3_pm_{split}", t + ".json"))) for t in tids}


def heads():
    from boxinst_commonality_tcd_04.dino_det import rerank3 as R3
    ch = json.load(open(os.path.join(R.OUT, "rerankR3_final.json")))["chosen"]; assert tuple(ch["cols"]) == FULL, ch
    R3._init(); W = R3._W; H = {}
    for name, cols in {"full": FULL, **{k: tuple(c for c in FULL if c != d) for k, d in DROPS.items()}}.items():
        c = {**ch, "cols": cols}
        fits = [f for s in range(5) for f in R3.oof_fits(c, s, "rr", W["X"], W["y"], W["soft"], W["tile"], W["F"])[1]]; assert len(fits) == 25
        H[name] = (fits, [R.NAMES.index(x) for x in cols]); print(f"[t3] fitted {name} on {list(cols)}", flush=True)
    return H


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
    H = None
    for split, gtp in F.GTS.items():
        if split in rec:
            continue
        G = json.load(open(gtp)); St = json.load(open(F.pf(split, "sprime")))["preds"]; tids = sorted(St)
        Pe = json.load(open(os.path.join(EF, f"preds_d1_edgefix_{split}_sprime_r2048.json")))["preds"]
        R3p = json.load(open(F.pf(split, "R3_ef")))["preds"]; assert sorted(Pe) == tids == sorted(G) == sorted(R3p)
        for t in tids:
            assert Pe[t]["boxes_2048"] == St[t]["boxes_2048"] == R3p[t]["boxes_2048"], t
        Z = np.load(os.path.join(R.FC_, f"rerankR1_{split}.npz")); X = Z["X"]; tile = Z["tile"]; assert list(Z["tids"]) == tids
        assert np.allclose(np.concatenate([St[t]["scores"] for t in tids]), X[:, 1])
        s = R.split_scores(tids, tile, X[:, 0])
        # gate 1
        PM = pm(split, St, tids); g1 = max(float(np.max(np.abs(np.asarray(s[t]) * np.asarray(PM[t]["pbar"]) * np.asarray(PM[t]["m"]) - np.asarray(St[t]["scores"]))))
                                           if len(s[t]) else 0.0 for t in tids)
        print(f"[t3] {split} gate 1: max |s*pbar*m - s'| = {g1:.2e}", flush=True); assert g1 < 1e-7, g1
        # gate 2
        H = H or heads(); sc = {k: R.split_scores(tids, tile, np.mean([f(X[:, c]) for f in fits], 0)) for k, (fits, c) in H.items()}
        g2 = max(float(np.max(np.abs(np.asarray(sc["full"][t]) - np.asarray(R3p[t]["scores"])))) if len(sc["full"][t]) else 0.0 for t in tids)
        print(f"[t3] {split} gate 2: max |refit - published R3| = {g2:.2e}", flush=True); assert g2 < 1e-6, g2
        rows = {"sm": {t: list(np.asarray(s[t]) * np.asarray(PM[t]["m"])) for t in tids},
                "sp": {t: list(np.asarray(s[t]) * np.asarray(PM[t]["pbar"])) for t in tids}, **{k: sc[k] for k in DROPS}}
        for k, v in rows.items():
            fp = F.pf(split, f"t3_{k}")
            if not os.path.exists(fp):
                json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker (edge fix), Table 3 ablation row t3_{k} (pre-registration T3)"},
                           "preds": {t: {"boxes_2048": St[t]["boxes_2048"], "scores": list(map(float, v[t])), "masks_rle": Pe[t]["masks_rle"]} for t in tids}},
                          open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        t0 = time.time(); cwd = os.getcwd(); os.chdir(R.REPO)
        scored = SC.score_many([F.pf(split, f"t3_{k}") for k in rows], gtp, res=2048, max_dets=600, score_floor=0.0); os.chdir(cwd)
        print(f"[t3] {split}: score_coco {time.time()-t0:.0f}s", flush=True); res = {}
        for k, sc_ in zip(rows, scored):
            json.dump(sc_, open(F.pf(split, f"t3_{k}").replace("preds_", "scored_"), "w")); cr = sc_["canopy_neutral_crowd"]
            res[k] = {"mask": {m: cr["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {m: cr["box"][m] for m in ("AP50", "AP75")}}
        ref = {"sprime_ef": json.load(open(os.path.join(EF, f"scored_d1_edgefix_{split}_sprime_r2048.json")))["canopy_neutral_crowd"]["mask"],
               "shat": json.load(open(F.pf(split, "R3_ef").replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]}
        E = FE.build(Pe, G, os.path.join(R.FC_, f"fasteval_{split}_edgefix.pkl"))
        D = {"sprime_ef": F.per_tile(E, {t: St[t]["scores"] for t in tids}), "shat": F.per_tile(E, {t: R3p[t]["scores"] for t in tids}),
             **{k: F.per_tile(E, {t: list(map(float, v[t])) for t in tids}) for k, v in rows.items()}}
        for k, d in D.items():   # gate 3
            a = FE.summary(F.ap_from(d, tids)); want = res[k]["mask"] if k in res else ref[k]
            assert all(abs(a[m] - want[m]) < 1.5e-4 for m in a), (split, k, a, want)
        print(f"[t3] {split} gate 3: fasteval == score_coco on all {len(D)} rows", flush=True)
        bt = {**{f"{k}_vs_sprime_ef": F.boot(D[k], D["sprime_ef"], tids) for k in ("sm", "sp")},
              **{f"{k}_vs_shat": F.boot(D[k], D["shat"], tids) for k in DROPS}}
        rec[split] = {"gates": {"g1_max_abs": g1, "g2_max_abs": g2, "g3": "pass"}, "rows": res,
                      "ref": {k: {m: v[m] for m in ("AP50", "AP75", "AP50_95")} for k, v in ref.items()}, "boot": bt}
        print(f"[t3] {split}: {json.dumps(rec[split])}", flush=True); json.dump(rec, open(OUT, "w"), indent=1)
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(rec, open(OUT, "w"), indent=1); print("[t3] DONE", flush=True)


if __name__ == "__main__":
    main()

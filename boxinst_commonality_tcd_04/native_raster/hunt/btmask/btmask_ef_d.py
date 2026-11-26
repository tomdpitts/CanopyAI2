"""BTMASK-EF arm D (pre-registration in btmask/README.md, 2026-11-26): deployed LACE masks on BoxTeacher's boxes, ranked by the
DEPLOYED OAM-TCD hat-s (R3, 25 models fitted on D1 val), applied unchanged with s = BoxTeacher's score. Nothing fitted on BT data.
Reuses btmask_ef.py outputs (masks, features, fasteval caches). -> compare_ef_d.json, preds/scored_D_ef_<split>_r2048.json.
  cd ~/dphil/CanopyAI2 && PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.btmask.btmask_ef_d \
      > boxinst_commonality_tcd_04/native_raster/logs/btmask_ef_d_2026-11-26.log 2>&1
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM, btmask_ef as EF

HERE, FC, REPO = BM.HERE, BM.FC, BM.REPO
OUT = os.path.join(HERE, "compare_ef_d.json")
log = EF.log


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.dino_det import rerank as R, s2_final as F, fasteval as FE
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
    Zv = np.load(os.path.join(FC, "rerankR1_val.npz")); fits, cols = EF.r3_fits(Zv["X"], Zv["y"], Zv["tile"], len(Zv["tids"]))
    if "gateD1" not in rec:                    # deployed hat-s models == published R3
        Zt = np.load(os.path.join(FC, "rerankR1_test.npz")); tt = list(Zt["tids"])
        sc = R.split_scores(tt, Zt["tile"], np.mean([f(Zt["X"][:, cols]) for f in fits], 0))
        R3p = json.load(open(F.pf("test", "R3_ef")))["preds"]
        d = max(float(np.max(np.abs(np.asarray(sc[t]) - np.asarray(R3p[t]["scores"])))) if len(sc[t]) else 0.0 for t in tt)
        rec["gateD1"] = {"r3_refit_max_abs_vs_published": d, "pass": bool(d < 1e-6)}; json.dump(rec, open(OUT, "w"), indent=1)
    log(f"[gateD1] {rec['gateD1']}"); assert rec["gateD1"]["pass"], "GATE D1 FAILED"
    if "input_shift" not in rec:               # diagnostic: hat-s inputs on D1 val vs BT test
        Zb = np.load(f"{FC}/btmask_ef_feats_test.npz"); q = [0.05, 0.25, 0.5, 0.75, 0.95]
        rec["input_shift"] = {R.NAMES[c]: {"D1_val": [round(float(v), 4) for v in np.quantile(Zv["X"][:, c], q)],
                                           "BT_test": [round(float(v), 4) for v in np.quantile(Zb["X"][:, c], q)]} for c in cols}
        rec["input_shift"]["quantiles"] = q; json.dump(rec, open(OUT, "w"), indent=1)
    log(f"[shift] {json.dumps(rec['input_shift'])}")
    for split in ("test", "sparse"):
        if split in rec:
            log(f"[{split}] already done"); continue
        gtp = BM.SPLIT[split]["gt"]; G = json.load(open(gtp))
        PB = json.load(open(os.path.join(HERE, f"preds_B_ef_{split}_r2048.json")))["preds"]; tids = sorted(PB); assert tids == sorted(G)
        Z = np.load(f"{FC}/btmask_ef_feats_{split}.npz"); assert list(Z["tids"]) == tids
        assert np.allclose(Z["X"][:, 0], np.concatenate([PB[t]["scores"] for t in tids])), "feature s != BT score"
        sD = R.split_scores(tids, Z["tile"], np.mean([f(Z["X"][:, cols]) for f in fits], 0))
        fp = os.path.join(HERE, f"preds_D_ef_{split}_r2048.json")
        if not os.path.exists(fp):
            json.dump({"meta": {"mask_res": 2048, "score_floor": 0.0,
                                "model": "deployed LACE masker on BoxTeacher boxes, DEPLOYED OAM-TCD hat-s (R3, D1-val fit) unchanged, s = BT score (arm D)"},
                       "preds": {t: {"boxes_2048": PB[t]["boxes_2048"], "scores": list(map(float, sD[t])), "masks_rle": PB[t]["masks_rle"]} for t in tids}},
                      open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        sp = fp.replace("preds_", "scored_"); cwd = os.getcwd(); os.chdir(REPO)
        if not os.path.exists(sp):
            t0 = time.time(); json.dump(SC.score_many([fp], gtp, res=2048, max_dets=600, score_floor=0.0)[0], open(sp, "w")); log(f"[{split}] scored D {time.time()-t0:.0f}s")
        os.chdir(cwd); r = json.load(open(sp))["canopy_neutral_crowd"]
        met = {k: r["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")} | {"box_AP50": r["box"]["AP50"], "floor": 0.0}
        PA = json.load(open(os.path.join(HERE, f"preds_A_ef_{split}_r2048.json")))["preds"]
        PC = json.load(open(os.path.join(HERE, f"preds_C_ef_{split}_r2048.json")))["preds"]
        EA = FE.build(None, None, f"{FC}/btmask_ef_fasteval_{split}_A.pkl"); EB = FE.build(None, None, f"{FC}/btmask_ef_fasteval_{split}_B.pkl")
        assert EA["tids"] == EB["tids"] == tids
        D = {"A": F.per_tile(EA, {t: PA[t]["scores"] for t in tids}), "C": F.per_tile(EB, {t: PC[t]["scores"] for t in tids}),
             "D": F.per_tile(EB, sD)}
        s_ = FE.summary(F.ap_from(D["D"], tids)); assert all(abs(s_[k] - met[k]) < 1.5e-4 for k in ("AP50", "AP50_95")), ("gate D2", s_, met)
        log(f"[{split}] gate D2: fasteval == score_coco")
        rec[split] = {"metrics_D": met, "gateD2_fasteval": s_, "boot": {"D_minus_A": F.boot(D["D"], D["A"], tids), "D_minus_C": F.boot(D["D"], D["C"], tids)},
                      "file": os.path.relpath(fp, BM.PKG), "date": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(rec, open(OUT, "w"), indent=1); log(f"[{split}] {json.dumps(rec[split])}")
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(rec, open(OUT, "w"), indent=1); log("[btmask_ef_d] DONE")


if __name__ == "__main__":
    main()

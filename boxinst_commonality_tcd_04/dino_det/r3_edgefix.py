"""Pre-registration R3-edgefix (README.md, 2026-11-04, Tom: "re run R3 and R3 + smoothing using EDGE_FILL = True"). ONE look, test 439
and sparse-236. Masks: the edge-fixed D1 masks (GA.EDGE_FILL = True; results/edgefix/preds_d1_edgefix_<split>_sprime_r2048.json, made by
edgefix_test.py). s' is unchanged by the fix and R3 uses no mask feature, so the R3 scores are those already applied in rerankR3_final.
Rows: R3_ef (edge-fixed masks + R3 scores); R3_ef_blur2 (the same masks Gaussian-smoothed sigma 2 px, threshold 0.5, + R3 scores).
Frozen score_coco; paired tile bootstrap (2000, seed 0) on AP50 and AP50:95 via fasteval (asserted equal to score_coco), vs the
edge-fixed s' row, vs the original s' row, and R3_ef_blur2 vs R3_ef. -> results/r3_edgefix.json; log native_raster/logs/r3_edgefix_2026-11-04.log
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.r3_edgefix
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det import side2 as S2
from boxinst_commonality_tcd_04.dino_det import s2_final as F

OUT = os.path.join(R.OUT, "r3_edgefix.json"); EF = os.path.join(R.OUT, "edgefix")


def blurred(split, Pe, tids):
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    fp = os.path.join(R.FC_, f"{R.TAG}_{split}_masks_edgefix_blur2.json")
    if os.path.exists(fp):
        return json.load(open(fp))
    t0 = time.time()
    with mp.get_context("spawn").Pool(S2.NPROC) as pool:
        M2 = pool.map(S2._post, [([SL._as_bytes(r) for r in Pe[t]["masks_rle"]], np.asarray(Pe[t]["boxes_2048"]).reshape(-1, 4), "blur2") for t in tids])
    out = dict(zip(tids, M2)); json.dump(out, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    print(f"[r3ef] {split}: smoothed {sum(len(m) for m in M2)} masks {time.time()-t0:.0f}s", flush=True); return out


def main():
    from boxinst_commonality_tcd_04 import score_coco as SC
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {"date_start": time.strftime("%Y-%m-%d %H:%M")}
    for split, gtp in F.GTS.items():
        if split in rec:
            continue
        G = json.load(open(gtp)); St = json.load(open(F.pf(split, "sprime")))["preds"]; tids = sorted(St)
        Pe = json.load(open(os.path.join(EF, f"preds_d1_edgefix_{split}_sprime_r2048.json")))["preds"]
        R3 = json.load(open(F.pf(split, "rerankR3")))["preds"]; assert sorted(Pe) == tids == sorted(R3) == sorted(G)
        for t in tids:
            assert Pe[t]["boxes_2048"] == St[t]["boxes_2048"] == R3[t]["boxes_2048"], t
            assert np.allclose(Pe[t]["scores"], St[t]["scores"]), t                      # s' unchanged by the fix
        Mb = blurred(split, Pe, tids); r3 = {t: R3[t]["scores"] for t in tids}
        rows = {"R3_ef": ({t: Pe[t]["masks_rle"] for t in tids}, r3), "R3_ef_blur2": (Mb, r3)}
        for k, (M_, sc) in rows.items():
            if not os.path.exists(F.pf(split, k)):
                json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker with top/left border fix (EDGE_FILL), R3 scores; row {k} (POST-HOC)"},
                           "preds": {t: {"boxes_2048": St[t]["boxes_2048"], "scores": list(map(float, sc[t])), "masks_rle": M_[t]} for t in tids}},
                          open(F.pf(split, k) + ".tmp", "w")); os.replace(F.pf(split, k) + ".tmp", F.pf(split, k))
        t0 = time.time(); cwd = os.getcwd(); os.chdir(R.REPO)
        scored = SC.score_many([F.pf(split, k) for k in rows], gtp, res=2048, max_dets=600, score_floor=0.0); os.chdir(cwd)
        print(f"[r3ef] {split}: score_coco {time.time()-t0:.0f}s", flush=True)
        res = {}
        for k, s in zip(rows, scored):
            json.dump(s, open(F.pf(split, k).replace("preds_", "scored_"), "w")); cr = s["canopy_neutral_crowd"]
            res[k] = {"mask": {m: cr["mask"][m] for m in ("AP50", "AP75", "AP50_95")}, "box": {m: cr["box"][m] for m in ("AP50", "AP75")}}
        ref = {"sprime": json.load(open(F.pf(split, "sprime").replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"],
               "sprime_ef": json.load(open(os.path.join(EF, f"scored_d1_edgefix_{split}_sprime_r2048.json")))["canopy_neutral_crowd"]["mask"],
               "R3": json.load(open(F.pf(split, "rerankR3").replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]}
        E0 = FE.build(St, G, os.path.join(R.FC_, f"fasteval_{split}.pkl"))
        Ee = FE.build(Pe, G, os.path.join(R.FC_, f"fasteval_{split}_edgefix.pkl"))
        Eb = FE.build({t: {"masks_rle": Mb[t]} for t in tids}, G, os.path.join(R.FC_, f"fasteval_{split}_edgefix_blur2.pkl"))
        sp = {t: St[t]["scores"] for t in tids}
        D = {"sprime": F.per_tile(E0, sp), "sprime_ef": F.per_tile(Ee, sp), "R3": F.per_tile(E0, r3), "R3_ef": F.per_tile(Ee, r3), "R3_ef_blur2": F.per_tile(Eb, r3)}
        for k, d in D.items():
            a = FE.summary(F.ap_from(d, tids)); want = res[k]["mask"] if k in res else ref[k]
            assert all(abs(a[m] - want[m]) < 1.5e-4 for m in a), (split, k, a, want)
        print(f"[r3ef] {split}: fasteval == score_coco on all 5 rows", flush=True)
        bt = {"R3_ef_vs_sprime_ef": F.boot(D["R3_ef"], D["sprime_ef"], tids), "R3_ef_blur2_vs_sprime_ef": F.boot(D["R3_ef_blur2"], D["sprime_ef"], tids),
              "R3_ef_vs_sprime": F.boot(D["R3_ef"], D["sprime"], tids), "R3_ef_blur2_vs_sprime": F.boot(D["R3_ef_blur2"], D["sprime"], tids),
              "R3_ef_blur2_vs_R3_ef": F.boot(D["R3_ef_blur2"], D["R3_ef"], tids), "R3_ef_vs_R3": F.boot(D["R3_ef"], D["R3"], tids)}
        rec[split] = {"rows": res, "ref": {k: {m: v[m] for m in ("AP50", "AP75", "AP50_95")} for k, v in ref.items()}, "boot": bt}
        print(f"[r3ef] {split}: {json.dumps(rec[split])}", flush=True); json.dump(rec, open(OUT, "w"), indent=1)
    rec["date_end"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(rec, open(OUT, "w"), indent=1); print("[r3ef] DONE", flush=True)


if __name__ == "__main__":
    main()

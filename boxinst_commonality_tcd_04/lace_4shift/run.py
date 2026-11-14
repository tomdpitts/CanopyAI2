"""4-shift LACE (IL-test arm interp8) at the learned confidence s_hat (R3), test 439 + sparse 236. Pre-registration: README.md."""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
DR = os.path.join(PKG, "dino_det", "results"); FC = "/Users/tompitts/dphil/feat_cache"; OUT = os.path.join(HERE, "results")
GT = {"test": os.path.join(PKG, "test_gt.json"), "sparse": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
F4P = {"test": f"{FC}/feat_4p_test_dl/feat_4p_test", "sparse": f"{FC}/feat_4p_sparse_dl/feat_4p_sparse"}
M4 = {"test": f"{FC}/interlace_test_interp8", "sparse": f"{FC}/shift4_sparse_interp8"}
N_BOOT, SEED = 2000, 0


def _job(args):
    """interlace_test._job for the sparse split (module constants set inside the worker)."""
    from boxinst_commonality_tcd_04.dino_det import interlace_test as IT
    IT.SPLIT = "sparse"; IT.F4P = F4P["sparse"]
    t, out, st = IT._job(args)
    fp = os.path.join(M4["sparse"], t + ".json"); json.dump({**out["interp8"], "stats": st}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t, st


def render_sparse(Dt, tids):
    os.makedirs(M4["sparse"], exist_ok=True)
    todo = [(t, Dt[t]["boxes_2048"], Dt[t]["scores"], ("interp8",)) for t in tids if not os.path.exists(os.path.join(M4["sparse"], t + ".json"))]
    print(f"[sparse] rendering {len(todo)} tiles", flush=True); t0 = time.time()
    with mp.get_context("spawn").Pool(12) as pool:
        for k, (t, st) in enumerate(pool.imap_unordered(_job, todo)):
            assert st["gate_cos_mean"] >= 0.999 and st["gate_cos_p1"] >= 0.99, (t, st)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


def main():
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S3, rerank as R, rerank3 as R3
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    fo = os.path.join(OUT, "shift4.json"); rec = json.load(open(fo)) if os.path.exists(fo) else {}
    # frozen R3 refit exactly as cap_test.py
    ch = json.load(open(os.path.join(DR, "rerankR3_final.json")))["chosen"]; c = {**ch, "cols": tuple(ch["cols"])}
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

    ap3 = lambda s, D: (S3.ap_at(s, D, SL.REC_THRS, 0), S3.ap_at(s, D, SL.REC_THRS, 5), float(np.mean([S3.ap_at(s, D, SL.REC_THRS, k) for k in range(10)])))
    for split in ("test", "sparse"):
        if split in rec:
            continue
        t0 = time.time(); g = json.load(open(GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
        Dt = json.load(open(os.path.join(DR, f"preds_dino_swinl_s0_{split}_detscore_r2048.json")))["preds"]
        Pub = json.load(open(os.path.join(DR, f"preds_dino_swinl_s0_{split}_R3_ef_r2048.json")))["preds"]
        Ef = json.load(open(os.path.join(DR, "edgefix", f"preds_d1_edgefix_{split}_sprime_r2048.json")))["preds"]
        dev = max(np.abs(r3(Dt[t]["boxes_2048"], Dt[t]["scores"], Ef[t]["scores"]) - np.asarray(Pub[t]["scores"])).max(initial=0) for t in tids)
        print(f"[{split}] R3 gate max |d| = {dev:.2e}", flush=True); assert dev < 1e-6, dev
        if split == "sparse":
            render_sparse(Dt, tids)
        A = {"s16_shat": {}, "s4_shat": {}, "s16_sprime": {}, "s4_sprime": {}}; nsame = n = 0
        for t in tids:
            m = json.load(open(os.path.join(M4[split], t + ".json"))); b = Dt[t]["boxes_2048"]; assert len(m["masks_rle"]) == len(b) == len(Pub[t]["masks_rle"]), t
            assert Pub[t]["boxes_2048"] == b == Ef[t]["boxes_2048"] and Pub[t]["masks_rle"] == Ef[t]["masks_rle"], t
            nsame += sum(x["counts"] == y["counts"] for x, y in zip(m["masks_rle"], Pub[t]["masks_rle"])); n += len(b)
            mk = lambda masks, sc: {"boxes_2048": b, "scores": [float(x) for x in sc], "masks_rle": masks}
            A["s16_shat"][t] = mk(Pub[t]["masks_rle"], Pub[t]["scores"]); A["s4_shat"][t] = mk(m["masks_rle"], r3(b, Dt[t]["scores"], m["s_prime"]))
            A["s16_sprime"][t] = mk(Ef[t]["masks_rle"], Ef[t]["scores"]); A["s4_sprime"][t] = mk(m["masks_rle"], m["s_prime"])
        json.dump({"meta": {"mask_res": 2048, "model": "LACE D1, 4 shifts (IL-test interp8) + frozen R3 applied to 4-shift s' (lace_4shift/README.md)"}, "preds": A["s4_shat"]},
                  open(os.path.join(OUT, f"preds_d1_shift4_R3_{split}_r2048.json"), "w"))
        D, r = {}, {"n_tiles": len(tids), "n_dets": n, "masks_identical_to_16shift": nsame, "r3_gate_maxabs": float(dev), "arms": {}, "paired": {}}
        for a in A:
            D[a] = S3.per_tile(A[a], gd, t2i, 0.0)
            for i in ids:
                if i not in D[a]:
                    D[a][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), sum(1 for x in gd["annotations"] if x["image_id"] == i and not x["iscrowd"]))
            r["arms"][a] = dict(zip(("AP50", "AP75", "AP50_95"), (round(float(x), 4) for x in ap3(ids, D[a])))); print(f"[{split} {a}] {r['arms'][a]} [{time.time()-t0:.0f}s]", flush=True)
        pub = json.load(open(os.path.join(DR, f"scored_dino_swinl_s0_{split}_R3_ef_r2048.json")))["canopy_neutral_crowd"]["mask"]
        assert all(abs(r["arms"]["s16_shat"][m] - pub[m]) < 1e-4 for m in ("AP50", "AP75", "AP50_95")), (r["arms"]["s16_shat"], pub)
        rng = np.random.default_rng(SEED); S = [ids[rng.integers(0, len(ids), len(ids))] for _ in range(N_BOOT)]
        boot = {a: np.array([ap3(s, D[a]) for s in S]) for a in A}
        for a, b_ in (("s4_shat", "s16_shat"), ("s4_sprime", "s16_sprime")):
            v = boot[a] - boot[b_]
            r["paired"][f"{a}_minus_{b_}"] = {m: {"diff": round(r["arms"][a][m] - r["arms"][b_][m], 4), "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)],
                                                   "p_le_0": round(float(np.mean(v[:, j] <= 0)), 4)} for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
        rec[split] = r; json.dump(rec, open(fo, "w"), indent=1); print(f"[{split}] {json.dumps(r['paired'])}", flush=True)
    print("[shift4] DONE", flush=True)


if __name__ == "__main__":
    main()

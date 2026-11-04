"""Pre-registration S2b (README.md, 2026-11-04): follow-up to S2 boundary smoothing. Val only, local.

  morph : Gaussian smoothing sigma 1.5 / 2.5; size-gated sigma (2 if equiv diam >= 24 px else 1); 1 px erosion of masks < 24 px only
  stack : mask variant (none, blur2, blur-best) x ranking (s', closed form a=0.625/0.75 b=0.5, R3 15-run OOF) -> AP50 / AP75 / AP50:95
Results -> results/side2b_val.json; log native_raster/logs/side2b_2026-11-04.log.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.side2b
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det import side2 as S2

OUT = os.path.join(R.OUT, "side2b_val.json")


def E_for(op, P, vg, tids):
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    if op == "none":
        return FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl"))
    cache = os.path.join(R.FC_, f"fasteval_val_post_{op}.pkl")
    if os.path.exists(cache):
        return FE.build(None, None, cache)
    with mp.get_context("spawn").Pool(S2.NPROC) as pool:
        M2 = pool.map(S2._post, [([SL._as_bytes(r) for r in P[t]["masks_rle"]], np.asarray(P[t]["boxes_2048"]).reshape(-1, 4), op) for t in tids])
    return FE.build({t: {"masks_rle": m} for t, m in zip(tids, M2)}, vg, cache)


def main():
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.dino_det import rerank3 as R3
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}; rec.setdefault("morph", {})
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); tids = sorted(P)
    sp = {t: P[t]["s_prime"] for t in tids}
    for op in ("blur1.5", "blur2.5", "blurbig2", "erodesmall"):
        if op in rec["morph"]:
            continue
        t0 = time.time(); rec["morph"][op] = FE.summary(FE.ap(E_for(op, P, vg, tids), sp))
        print(f"[morph] {op}: {rec['morph'][op]} {time.time()-t0:.0f}s", flush=True)
        json.dump(rec, open(OUT, "w"), indent=1)
    s2 = json.load(open(S2.OUT))["morph"]; allm = {**s2, **rec["morph"]}
    best = max((k for k in allm if k.startswith("blur")), key=lambda k: (allm[k]["AP50_95"], allm[k]["AP50"]))
    rec["blur_best"] = best; print(f"[stack] best blur by val AP50:95: {best} {allm[best]}", flush=True)
    # rankings
    Z = np.load(os.path.join(R.FC_, "rerankR1_val.npz")); X = Z["X"]; assert list(Z["tids"]) == tids
    s, spv, mih = X[:, 0], X[:, 1], X[:, 12]
    ranks = {"sprime": spv, "closed_a0.625_b0.5": s * (spv / np.maximum(s, 1e-9)) ** 0.625 * (1 - mih) ** 0.5,
             "closed_a0.75_b0.5": s * (spv / np.maximum(s, 1e-9)) ** 0.75 * (1 - mih) ** 0.5}
    chosen = json.load(open(os.path.join(R.OUT, "rerankR3_confirm.json")))["chosen"]; c = {**chosen, "cols": tuple(chosen["cols"])}
    R3._init(); W = R3._W
    r3 = [R3.oof_fits(c, sd, f, W["X"], W["y"], W["soft"], W["tile"], W["F"])[0] for f, sd in R3.RUNS]
    rec.setdefault("stack", {})
    for op in ("none", "blur2", best):
        E = E_for(op, P, vg, tids); row = {k: FE.summary(FE.ap(E, v)) for k, v in ranks.items()}
        a = np.asarray([FE.ap(E, v) for v in r3]); row["R3_oof15_mean"] = {"AP50": round(float(a[:, 0].mean()), 4),
                                                                         "AP75": round(float(a[:, 5].mean()), 4), "AP50_95": round(float(a.mean(1).mean()), 4)}
        rec["stack"][op] = row; print(f"[stack] masks={op}: {json.dumps(row)}", flush=True)
        json.dump(rec, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

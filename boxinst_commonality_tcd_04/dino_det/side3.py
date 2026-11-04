"""Pre-registration S3 (README.md, 2026-11-04): scale-aware boundary smoothing, sigma = c x mask equivalent diameter. Val only, local.
Motivation: mask_diag.py -- LACE outlines are rough mainly on large crowns, where a fixed sigma = 2 px under-smooths.
  c in {0.02, 0.03, 0.04, 0.06, 0.08, 0.10}, s' ranking and R3 15-run OOF -> results/side3_val.json; log native_raster/logs/side3_2026-11-04.log
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.side3
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det.side2b import E_for

OUT = os.path.join(R.OUT, "side3_val.json"); CS = ("0.02", "0.03", "0.04", "0.06", "0.08", "0.10")


def main():
    from boxinst_commonality_tcd_04.dino_det import rerank3 as R3
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}; rec.setdefault("sprime", {})
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); tids = sorted(P)
    sp = {t: P[t]["s_prime"] for t in tids}
    for op in ["none", "blur2"] + [f"blurprop{c}" for c in CS]:
        if op in rec["sprime"]:
            continue
        t0 = time.time(); rec["sprime"][op] = FE.summary(FE.ap(E_for(op, P, vg, tids), sp))
        print(f"[s3] {op}: {rec['sprime'][op]} {time.time()-t0:.0f}s", flush=True); json.dump(rec, open(OUT, "w"), indent=1)
    best = max((k for k in rec["sprime"] if k.startswith("blurprop") and rec["sprime"][k]["AP50"] >= 0.7384 - 0.001),
               key=lambda k: rec["sprime"][k]["AP50_95"])
    rec["best"] = best; print(f"[s3] best by val AP50:95 (AP50 >= s' - 0.001): {best} {rec['sprime'][best]}", flush=True)
    chosen = json.load(open(os.path.join(R.OUT, "rerankR3_confirm.json")))["chosen"]; c = {**chosen, "cols": tuple(chosen["cols"])}
    R3._init(); W = R3._W; r3 = [R3.oof_fits(c, sd, f, W["X"], W["y"], W["soft"], W["tile"], W["F"])[0] for f, sd in R3.RUNS]
    rec["R3"] = {}
    for op in ("none", "blur2", best):
        a = np.asarray([FE.ap(E_for(op, P, vg, tids), v) for v in r3])
        rec["R3"][op] = {"AP50": round(float(a[:, 0].mean()), 4), "AP75": round(float(a[:, 5].mean()), 4), "AP50_95": round(float(a.mean(1).mean()), 4)}
        print(f"[s3] R3 + {op}: {rec['R3'][op]}", flush=True)
    json.dump(rec, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

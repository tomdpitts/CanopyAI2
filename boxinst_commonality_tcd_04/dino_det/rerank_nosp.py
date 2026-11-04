"""Val-only diagnostic (2026-11-03): does R2's gain need s'? It never goes to test.
R2's MLP (score + geometry + context) vs the same MLP with every s'-derived feature removed (sp, log_sp_s, rank_frac;
rank_frac is the s' rank within the tile). 5 seeds x 3 fold assignments, as in R2. Also: s' alone and detector s alone.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.rerank_nosp
"""
import contextlib, io, json, os, time
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det.rerank2 import COLS, SEEDS, fold_sets

NOSP = [c for c in COLS if R.NAMES[c] not in ("sp", "log_sp_s", "rank_frac")]


def main():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids, tile, X, y = R.features("val", P, vg); gd, t2i = SL.build_gt_crowd(vg, 2048)

    def val_ap(v):
        sc = R.split_scores(tids, tile, v)
        pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in tids}
        with contextlib.redirect_stdout(io.StringIO()):
            return round(SL.score(pr, gd, t2i, masks=True, floor=0.0)["mask"]["AP50"], 4)
    base = val_ap(X[:, 1]); det = val_ap(X[:, 0]); assert abs(base - 0.7384) < 5e-4, base
    print(f"[nosp] s' {base}  detector s {det}  cols {[R.NAMES[c] for c in NOSP]}", flush=True)
    F = fold_sets(len(tids)); rows = {"r2": [], "nosp": []}; t0 = time.time()
    for fname, tf in F.items():
        for seed in SEEDS:
            for name, cols in (("r2", COLS), ("nosp", NOSP)):
                a = val_ap(R.oof(tids, tile, X, y, "mlp", cols, seed, tf)[0]); rows[name].append(a)
            print(f"  {fname} s{seed}: r2 {rows['r2'][-1]} nosp {rows['nosp'][-1]} {time.time()-t0:.0f}s", flush=True)
    g = {k: np.asarray(v) - base for k, v in rows.items()}; d = np.asarray(rows["nosp"]) - np.asarray(rows["r2"])
    rec = {"sprime": base, "detector_s": det, "nosp_cols": [R.NAMES[c] for c in NOSP],
           "gain_vs_sprime": {k: {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)} for k, v in g.items()},
           "nosp_minus_r2": {"mean": round(float(d.mean()), 4), "sd": round(float(d.std(ddof=1)), 4), "n_pos": int((d > 0).sum())},
           "runs": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(os.path.join(R.OUT, "rerank_nosp_val.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "runs"}, indent=1), flush=True)


if __name__ == "__main__":
    main()

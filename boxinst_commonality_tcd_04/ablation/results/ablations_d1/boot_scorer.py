"""tab:scorer on LACE (D1 variant): paired tile bootstrap of s' - s (mask AP50), 439 test and sparse-236.
Same frozen detections and masks per split; only the ranking differs. Reuses dino_det/boot_vs_bt.py's per-tile COCOeval
statuses and AP (which reproduce the full-set AP exactly). 2,000 resamples, seed 0. Test-set sampling only (one detector seed).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.ablation.results.ablations_d1.boot_scorer
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..", "..", "..")); REPO = os.path.dirname(PKG)
DR = os.path.join(PKG, "dino_det", "results")
SPLITS = {"test439": os.path.join(PKG, "test_gt.json"), "sparse236": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
N, SEED = 2000, 0


def main():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det.boot_vs_bt import per_tile, ap
    out = os.path.join(HERE, "results", "boot_scorer.json"); assert not os.path.exists(out), out
    rec = {"n_boot": N, "seed": SEED, "note": "paired tile bootstrap; test-set sampling only, single detector seed", "splits": {}}
    for sp, gtp in SPLITS.items():
        tag = "test" if sp == "test439" else "sparse"
        g = json.load(open(gtp)); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
        D = {}
        for rank, floor in (("detscore", 0.05), ("sprime", 0.0)):
            P = json.load(open(os.path.join(DR, f"preds_dino_swinl_s0_{tag}_{rank}_r2048.json")))["preds"]
            D[rank] = per_tile({t: P[t] for t in tids}, gd, t2i, floor)
            for i in ids:
                if i not in D[rank]:
                    D[rank][i] = (np.zeros(0), np.zeros(0, bool), sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
        full = {k: ap(ids, D[k], SL.REC_THRS) for k in D}
        print(f"[{sp}] full {full}", flush=True)
        rng = np.random.default_rng(SEED); d = []; t0 = time.time()
        for b in range(N):
            s = ids[rng.integers(0, len(ids), len(ids))]
            d.append(ap(s, D["sprime"], SL.REC_THRS) - ap(s, D["detscore"], SL.REC_THRS))
            if (b + 1) % 500 == 0:
                print(f"  [{sp}] {b+1}/{N} {time.time()-t0:.0f}s", flush=True)
        rec["splits"][sp] = {"full_AP50": {k: round(v, 4) for k, v in full.items()}, "sprime_minus_s": round(full["sprime"] - full["detscore"], 4),
                             "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                             "sd": round(float(np.std(d, ddof=1)), 4), "p_le_0": round(float(np.mean(np.asarray(d) <= 0)), 4)}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()

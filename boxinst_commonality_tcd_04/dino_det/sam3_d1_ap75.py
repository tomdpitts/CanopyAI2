"""AP75 add-on to sam3_on_d1.score: unrounded full-set mask AP75 and paired tile bootstrap (2,000, seed 0) of LACE (D1) s'
minus SAM 3 zs (both rankings), same per-tile COCOeval statuses as sam3_on_d1 (IoU index 5 = 0.75). -> results/sam3_d1/ap75_{split}.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_d1_ap75
"""
import json, os, time
import numpy as np
from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S

K = 5  # iouThrs[5] == 0.75


def main():
    for split in ("test", "sparse"):
        files = {"lace_d1_sprime": (os.path.join(S.RES, f"preds_dino_swinl_s0_{split}_sprime_r2048.json"), 0.0),
                 "sam3zs_gmean": (os.path.join(S.OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json"), 0.0),
                 "sam3zs_det": (os.path.join(S.OUT, f"preds_d1_sam3zs_det_{split}_r2048.json"), 0.05)}
        g = json.load(open(S.SPLIT[split]["gt"])); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in sorted(g)])
        D, full = {}, {}
        for k, (fp, floor) in files.items():
            P = json.load(open(fp))["preds"]; D[k] = S.per_tile({t: P[t] for t in sorted(g)}, gd, t2i, floor)
            for i in ids:
                if i not in D[k]:
                    ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                    D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
            full[k] = S.ap_at(ids, D[k], SL.REC_THRS, K)
            sc = json.load(open(fp.replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]["AP75"]
            assert abs(full[k] - sc) < 6e-5, (k, full[k], sc); print(f"[{split} {k}] AP75 {full[k]:.6f} (scorer {sc})", flush=True)
        rng = np.random.default_rng(S.SEED); diffs = {k: [] for k in files if k != "lace_d1_sprime"}; t0 = time.time()
        for b in range(S.N_BOOT):
            s = ids[rng.integers(0, len(ids), len(ids))]; L = S.ap_at(s, D["lace_d1_sprime"], SL.REC_THRS, K)
            for k in diffs:
                diffs[k].append(L - S.ap_at(s, D[k], SL.REC_THRS, K))
            if (b + 1) % 500 == 0:
                print(f"  boot {b+1}/{S.N_BOOT} {time.time()-t0:.0f}s", flush=True)
        rec = {"split": split, "n_boot": S.N_BOOT, "seed": S.SEED, "AP75": {k: round(v, 6) for k, v in full.items()},
               "lace_d1_minus": {k: {"diff": round(full["lace_d1_sprime"] - full[k], 4), "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                                     "p_le_0": round(float(np.mean(np.asarray(v) <= 0)), 4)} for k, v in diffs.items()},
               "note": "test-set sampling only, single detector seed; LACE = frozen D1 s'"}
        json.dump(rec, open(os.path.join(S.OUT, f"ap75_{split}.json"), "w"), indent=1); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

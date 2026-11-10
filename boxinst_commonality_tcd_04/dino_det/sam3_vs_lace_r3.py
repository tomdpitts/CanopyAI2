"""Paired tile bootstrap: current LACE (D1 + edge fix + R3, the TABLE1_CURRENT row) vs SAM 3 zero-shot on the same D1 boxes.

Each pipeline at its own ranking: LACE = R3 scores (floor 0), SAM 3 = sqrt(s*SAM) (floor 0), as in SAM3_D1.md. Reuses
sam3_on_d1.per_tile / aps verbatim (AP50 and AP50:95, crowd arm, maxDets 600). 2,000 resamples, seed 0; test-set sampling only.
Writes results/sam3_d1/summary_lace_r3ef_vs_sam3_<split>.json (refuses to overwrite).
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sam3_vs_lace_r3 --split test|sparse
"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "sam3_d1")
GT = {"test": os.path.join(PKG, "test_gt.json"), "sparse": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
N_BOOT, SEED = 2000, 0


def main(split):
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S3
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(OUT, f"summary_lace_r3ef_vs_sam3_{split}.json"); assert not os.path.exists(out), out
    files = {"lace_r3ef": os.path.join(RES, f"preds_dino_swinl_s0_{split}_R3_ef_r2048.json"),
             "sam3zs_gmean": os.path.join(OUT, f"preds_d1_sam3zs_gmean_{split}_r2048.json")}
    g = json.load(open(GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
    D, rec = {}, {"split": split, "n_tiles": len(ids), "n_boot": N_BOOT, "seed": SEED, "files": files, "full": {},
                  "note": "paired tile bootstrap, test-set sampling only, single detector seed; each pipeline at its own ranking (LACE R3, SAM sqrt(s*SAM)); LACE R3 is post-hoc"}
    boxes = {}
    for k, fp in files.items():
        P = json.load(open(fp)); P = P.get("preds", P)
        boxes[k] = {t: P[t]["boxes_2048"] for t in tids}
        D[k] = S3.per_tile({t: P[t] for t in tids}, gd, t2i, 0.0)
        for i in ids:
            if i not in D[k]:
                ng = sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"])
                D[k][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
        a50, a5095 = S3.aps(ids, D[k], SL.REC_THRS)
        sc = json.load(open(fp.replace("preds_", "scored_")))["canopy_neutral_crowd"]["mask"]
        rec["full"][k] = {"scorer": {m: sc[m] for m in ("AP50", "AP75", "AP50_95")}, "bootstrap_full": {"AP50": round(a50, 4), "AP50_95": round(a5095, 4)}}
        print(f"[{split} {k}] scorer {rec['full'][k]['scorer']} bootstrap-full AP50 {a50:.4f} AP50:95 {a5095:.4f}", flush=True)
    rec["same_boxes"] = all(sorted(map(tuple, boxes["lace_r3ef"][t])) == sorted(map(tuple, boxes["sam3zs_gmean"][t])) for t in tids)
    print(f"[{split}] same box set per tile: {rec['same_boxes']}", flush=True)
    rng = np.random.default_rng(SEED); diffs = []; t0 = time.time()
    full = {k: S3.aps(ids, D[k], SL.REC_THRS) for k in D}
    for b in range(N_BOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]
        L, O = S3.aps(s, D["lace_r3ef"], SL.REC_THRS), S3.aps(s, D["sam3zs_gmean"], SL.REC_THRS)
        diffs.append((L[0] - O[0], L[1] - O[1]))
        if (b + 1) % 200 == 0:
            print(f"  boot {b+1}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)
    v = np.asarray(diffs)
    rec["lace_minus_sam3"] = {m: {"diff": round(full["lace_r3ef"][j] - full["sam3zs_gmean"][j], 4),
                                  "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)],
                                  "p_le_0": round(float(np.mean(v[:, j] <= 0)), 4)} for j, m in enumerate(("AP50", "AP50_95"))}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["test", "sparse"], required=True)
    main(ap.parse_args().split)

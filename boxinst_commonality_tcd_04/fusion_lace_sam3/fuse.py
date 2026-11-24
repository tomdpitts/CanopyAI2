"""Fusion of LACE and SAM 3 zero-shot masks on identical D1 boxes. Pre-registration: README.md (arms, scoring, reading rule).

  cd boxinst_commonality_tcd_04/fusion_lace_sam3 && PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u fuse.py test|sparse
Out: results/fusion_<split>.json (refuses to overwrite), results/preds_<arm>_<split>_r2048.json for the fusion arms.
"""
import json, os, sys, time

import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
DR = os.path.join(PKG, "dino_det", "results"); FC = "/Users/tompitts/dphil/feat_cache"; OUT = os.path.join(HERE, "results")
GT = {"test": os.path.join(PKG, "test_gt.json"), "sparse": os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")}
N_BOOT, SEED = 2000, 0
FUSIONS = ("F1_sam_at_shat", "F2_sam_at_gmean_shat", "F3_sam_at_agree", "F4_intersection", "F5_union")


def _b(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]}


def _j(r):
    return {"size": [int(x) for x in r["size"]], "counts": r["counts"].decode("ascii") if isinstance(r["counts"], bytes) else r["counts"]}


def build(split, tids):
    L = json.load(open(os.path.join(DR, f"preds_dino_swinl_s0_{split}_R3_ef_r2048.json")))["preds"]
    Dt = json.load(open(os.path.join(DR, f"preds_dino_swinl_s0_{split}_detscore_r2048.json")))["preds"]
    A = {a: {} for a in ("lace", "sam") + FUSIONS}; agree = []
    for t in tids:
        l = L[t]; s = json.load(open(f"{FC}/sam3_d1_{split}/{t}.json")); n = len(l["boxes_2048"])
        assert l["boxes_2048"] == s["boxes_2048"] == Dt[t]["boxes_2048"], t
        shat = np.asarray(l["scores"], float); sd = np.asarray(Dt[t]["scores"], float); ss = np.clip(np.asarray(s["sam_scores"], float), 0, None)
        lm, sm = [_b(r) for r in l["masks_rle"]], [_b(r) for r in s["masks_rle"]]
        iou = np.array([mu.iou([a], [b], [0])[0, 0] for a, b in zip(lm, sm)]) if n else np.zeros(0)
        agree.extend(iou.tolist())
        inter = [_j(mu.merge([a, b], intersect=True)) for a, b in zip(lm, sm)]; union = [_j(mu.merge([a, b], intersect=False)) for a, b in zip(lm, sm)]
        mk = lambda masks, sc: {"boxes_2048": l["boxes_2048"], "scores": [float(x) for x in sc], "masks_rle": masks}
        A["lace"][t] = mk(l["masks_rle"], shat)
        A["sam"][t] = mk(s["masks_rle"], np.sqrt(sd * ss).round(6))                # = sam3_on_d1.score row sam3zs_gmean
        A["F1_sam_at_shat"][t] = mk(s["masks_rle"], shat)
        A["F2_sam_at_gmean_shat"][t] = mk(s["masks_rle"], np.sqrt(np.clip(shat, 0, None) * ss))
        A["F3_sam_at_agree"][t] = mk(s["masks_rle"], shat * iou)
        A["F4_intersection"][t] = mk(inter, shat)
        A["F5_union"][t] = mk(union, shat)
    return A, np.asarray(agree)


def main(split):
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S3
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(OUT, f"fusion_{split}.json"); assert not os.path.exists(out), out
    g = json.load(open(GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
    t0 = time.time(); A, agree = build(split, tids); print(f"[{split}] arms built {time.time()-t0:.0f}s; mean IoU(LACE, SAM) {agree.mean():.4f}", flush=True)
    ap3 = lambda s, D: (S3.ap_at(s, D, SL.REC_THRS, 0), S3.ap_at(s, D, SL.REC_THRS, 5), float(np.mean([S3.ap_at(s, D, SL.REC_THRS, k) for k in range(10)])))
    D, rec = {}, {"split": split, "n_tiles": len(ids), "n_dets": int(len(agree)), "mean_iou_lace_sam": round(float(agree.mean()), 4), "n_boot": N_BOOT, "seed": SEED, "arms": {}}
    for a in A:
        D[a] = S3.per_tile(A[a], gd, t2i, 0.0)
        for i in ids:
            if i not in D[a]:
                ng = sum(1 for x in gd["annotations"] if x["image_id"] == i and not x["iscrowd"])
                D[a][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), ng)
        rec["arms"][a] = dict(zip(("AP50", "AP75", "AP50_95"), (round(float(x), 4) for x in ap3(ids, D[a]))))
        print(f"[{split} {a}] {rec['arms'][a]}  [{time.time()-t0:.0f}s]", flush=True)
        if a in FUSIONS:
            json.dump({"meta": {"mask_res": 2048, "model": f"fusion arm {a} (fusion_lace_sam3/README.md)"}, "preds": A[a]}, open(os.path.join(OUT, f"preds_{a}_{split}_r2048.json"), "w"))
    # gate: reference arms reproduce the published scorer numbers
    pub = {"lace": json.load(open(os.path.join(DR, f"scored_dino_swinl_s0_{split}_R3_ef_r2048.json")))["canopy_neutral_crowd"]["mask"],
           "sam": json.load(open(os.path.join(DR, "sam3_d1", f"scored_d1_sam3zs_gmean_{split}_r2048.json")))["canopy_neutral_crowd"]["mask"]}
    rec["gate"] = {a: {"published": {m: pub[a][m] for m in ("AP50", "AP75", "AP50_95")},
                       "ok": all(abs(rec["arms"][a][m] - pub[a][m]) < 5e-5 + 5e-5 for m in ("AP50", "AP75", "AP50_95"))} for a in pub}
    print(f"[{split}] gate {json.dumps(rec['gate'])}", flush=True); assert all(v["ok"] for v in rec["gate"].values()), "GATE FAIL"
    rng = np.random.default_rng(SEED); S = [ids[rng.integers(0, len(ids), len(ids))] for _ in range(N_BOOT)]
    boot = {}
    for a in A:
        boot[a] = np.array([ap3(s, D[a]) for s in S]); print(f"  boot {a} done [{time.time()-t0:.0f}s]", flush=True)
    rec["paired"] = {}
    for a in FUSIONS:
        for ref in ("sam", "lace"):
            v = boot[a] - boot[ref]
            rec["paired"][f"{a}_minus_{ref}"] = {m: {"diff": round(rec["arms"][a][m] - rec["arms"][ref][m], 4),
                                                     "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)],
                                                     "p_le_0": round(float(np.mean(v[:, j] <= 0)), 4)} for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec["paired"], indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

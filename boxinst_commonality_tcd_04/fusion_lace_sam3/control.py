"""POST-HOC control (2026-11-14, after fusion_<split>.json): is the F2 gain due to LACE's mask, or to the learned
confidence's box-only inputs? F2c = SAM 3 masks ranked by sqrt(s_hat3 * SAM), s_hat3 = the 3-input learned confidence
WITHOUT the LACE mask term (T3 `no_logpm` scores, dino_det/results/preds_dino_swinl_s0_<split>_t3_no_logpm_r2048.json).
Paired tile bootstrap (2,000, seed 0) of F2 - F2c and F2c - sam. Same scorer as fuse.py.

  PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u control.py test|sparse  -> results/control_<split>.json
"""
import json, os, sys
import numpy as np
import fuse as F

def main(split):
    from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S3
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(F.OUT, f"control_{split}.json"); assert not os.path.exists(out)
    g = json.load(open(F.GT[split])); tids = sorted(g); gd, t2i = SL.build_gt_crowd(g, 2048); ids = np.array([t2i[t] for t in tids])
    A, _ = F.build(split, tids)
    N = json.load(open(os.path.join(F.DR, f"preds_dino_swinl_s0_{split}_t3_no_logpm_r2048.json")))["preds"]
    A = {"sam": A["sam"], "F2_sam_at_gmean_shat": A["F2_sam_at_gmean_shat"], "F2c_sam_at_gmean_shat3": {}}
    for t in tids:
        s = json.load(open(f"{F.FC}/sam3_d1_{split}/{t}.json")); assert N[t]["boxes_2048"] == s["boxes_2048"], t
        ss = np.clip(np.asarray(s["sam_scores"], float), 0, None); h3 = np.clip(np.asarray(N[t]["scores"], float), 0, None)
        A["F2c_sam_at_gmean_shat3"][t] = {"boxes_2048": s["boxes_2048"], "scores": [float(x) for x in np.sqrt(h3 * ss)], "masks_rle": s["masks_rle"]}
    ap3 = lambda s, D: (S3.ap_at(s, D, SL.REC_THRS, 0), S3.ap_at(s, D, SL.REC_THRS, 5), float(np.mean([S3.ap_at(s, D, SL.REC_THRS, k) for k in range(10)])))
    D, rec = {}, {"split": split, "post_hoc": True, "arms": {}, "paired": {}}
    for a in A:
        D[a] = S3.per_tile(A[a], gd, t2i, 0.0)
        for i in ids:
            if i not in D[a]:
                D[a][i] = (np.zeros(0), np.zeros((10, 0), bool), np.zeros((10, 0), bool), sum(1 for x in gd["annotations"] if x["image_id"] == i and not x["iscrowd"]))
        rec["arms"][a] = dict(zip(("AP50", "AP75", "AP50_95"), (round(float(x), 4) for x in ap3(ids, D[a])))); print(split, a, rec["arms"][a], flush=True)
    rng = np.random.default_rng(F.SEED); S = [ids[rng.integers(0, len(ids), len(ids))] for _ in range(F.N_BOOT)]
    boot = {a: np.array([ap3(s, D[a]) for s in S]) for a in A}
    for a, b in (("F2_sam_at_gmean_shat", "F2c_sam_at_gmean_shat3"), ("F2c_sam_at_gmean_shat3", "sam")):
        v = boot[a] - boot[b]
        rec["paired"][f"{a}_minus_{b}"] = {m: {"diff": round(rec["arms"][a][m] - rec["arms"][b][m], 4), "ci95": [round(float(np.percentile(v[:, j], 2.5)), 4), round(float(np.percentile(v[:, j], 97.5)), 4)]} for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec["paired"]), flush=True)

if __name__ == "__main__":
    main(sys.argv[1])

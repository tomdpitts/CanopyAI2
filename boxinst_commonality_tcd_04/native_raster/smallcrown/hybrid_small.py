"""Upper bound on what fixing SMALL-crown masks is worth in AP50: on LACE's own test detections, use the
zero-shot SAM 3 per-subtile mask (tierA A1) for detections whose box is small (w*h < 800 px, ~28x28, the
small-crown tercile edge in box terms) and LACE's deployed guided mask otherwise. Two rankings: detector
score s (compare LACE 0.6137, all-SAM 0.6329) and LACE's s' (compare 0.6470). 3 seeds. Box order is shared
(detscore_s*.json == guided preds == A1 inputs; asserted)."""
import json, os
from boxinst_commonality_tcd_04 import score_coco as SC
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
SMALL = 800.0
paths = []
for s in (0, 1, 2):
    L = json.load(open(os.path.join(NR, f"preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")))["preds"]
    A = json.load(open(os.path.join(NR, "tierA", f"preds_a1_lace_s{s}_sam3zs_det_r2048.json")))["preds"]
    for rank in ("s", "sprime"):
        fp = os.path.join(HERE, f"preds_hybrid_small_s{s}_rank{rank}.json"); paths.append(fp)
        if os.path.exists(fp):
            continue
        P, nsw = {}, 0
        for t in L:
            assert L[t]["boxes_2048"] == A[t]["boxes_2048"], t
            masks = []
            for i, b in enumerate(L[t]["boxes_2048"]):
                small = (b[2] - b[0]) * (b[3] - b[1]) < SMALL
                masks.append(A[t]["masks_rle"][i] if small else L[t]["masks_rle"][i]); nsw += small
            P[t] = {"boxes_2048": L[t]["boxes_2048"], "masks_rle": masks,
                    "scores": A[t]["scores"] if rank == "s" else L[t]["scores"]}
        json.dump({"meta": {"model": f"hybrid: SAM3-zs masks for box area < {SMALL} px, LACE otherwise; rank {rank}",
                            "n_swapped": nsw, "mask_res": 2048}, "preds": P}, open(fp, "w"))
        print(f"s{s} rank {rank}: swapped {nsw}", flush=True)
os.chdir("/Users/tompitts/dphil/CanopyAI2")
for md_floor in ((600, 0.0),):
    for p, r in zip(paths, SC.score_many(paths, "boxinst_commonality_tcd_04/test_gt.json", res=2048, max_dets=600, score_floor=0.0)):
        o = p.replace("preds_", "scored_"); assert not os.path.exists(o), o
        json.dump(r, open(o, "w")); c = r["canopy_neutral_crowd"]
        print(os.path.basename(p), "AP50", c["mask"]["AP50"], "AP50:95", c["mask"]["AP50_95"], flush=True)

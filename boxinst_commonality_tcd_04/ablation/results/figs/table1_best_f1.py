"""Best-F1 operating point (TP/FP/FN/F1) for the rows of Tables 1-3 and 5 (tab:oamtcd439, tab:sparse236, tab:samebox, tab:factors) (2026-11-25, Tom).

Matching = make_qualitative.best_f1_threshold: per tile, greedy score-ordered mask IoU >= 0.5, unmatched preds > 50 % inside
the canopy class ignored (as the COCO scorer's iscrowd), one global score threshold per row swept 0.05..0.95 step 0.01,
F1 maximised over the whole split. Descriptive only; AP in the tables is the frozen COCOeval number. Pred files are the
exact sources behind TABLE1_CURRENT.json (Tables 1-2) and boxshape_nm_{split}.json / sam3_d1 summaries (Table 3).

Usage: .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/table1_best_f1.py
Writes table1_best_f1.json next to this script (rows keyed "split|label"; a row whose split+source is already there is skipped).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_qualitative as MQ  # noqa: E402

NR = MQ.NR  # native_raster/; paths below are relative to it
GT = {"test": os.path.join(MQ.ROOT, "test_gt.json"), "sparse": os.path.join(MQ.REPO, "data", "tcd_sparse", "sparse_gt.json")}
DD = "../dino_det/results/"
ROWS = [  # (split, label, preds)
    # Table 1
    ("test", "Restor Mask R-CNN", "preds_restor_rpn1000_r2048_treeonly.json"),
    ("test", "Detectree2", "seamfix/preds_dt2_s0v2_r2048_seamfix.json"),
    ("test", "Box2Mask Swin-L", "seamfix/preds_box2mask_swinl_s0_r2048_seamfix.json"),
    ("test", "Box2Mask R-50", "seamfix/preds_box2mask_r50_s0_r2048_seamfix.json"),
    ("test", "BoxTeacher", "../box_supervised_baselines/tierb/results/20261101_192510/boxteacher/results/preds_boxteacher_sub_test_r2048.json"),
    ("test", "LACE (ours)", DD + "preds_dino_swinl_s0_test_R3_ef_r2048.json"),
    # Table 2
    ("sparse", "Detectree2", "seamfix/preds_dt2_sparse_s0v2_r2048_seamfix.json"),
    ("sparse", "Box2Mask Swin-L", "seamfix/preds_box2mask_swinl_sparse_s0_r2048_seamfix.json"),
    ("sparse", "BoxTeacher", "../box_supervised_baselines/tierb/results_sparse/preds_boxteacher_sub_sparse_r2048.json"),
    ("sparse", "LACE (ours)", DD + "preds_dino_swinl_s0_sparse_R3_ef_r2048.json"),
] + [r for sp in ("test", "sparse") for r in [  # Table 3 (identical D1 boxes); "LACE (ours)" own-ranking row = Table 1/2 LACE
    (sp, "Filled box (shared ranking)", DD + f"preds_boxshape_nm_box_{sp}_r2048.json"),
    (sp, "Ellipse (shared ranking)", DD + f"preds_boxshape_nm_ellipse_{sp}_r2048.json"),
    (sp, "LACE (shared ranking)", DD + f"preds_dino_swinl_s0_{sp}_t3_no_logpm_r2048.json"),
    (sp, "LACE detector + SAM 3 zs", DD + f"sam3_d1/preds_d1_sam3zs_gmean_{sp}_r2048.json")]] + [
    # Table 5 (tab:factors) scorer ablation, added 2026-11-25; s sources per paper.tex tab:factors source comment.
    # "hat s (4 inputs)" = LACE (ours); "- log(p D)" = LACE (shared ranking) above.
    ("test", "T5 s (detector score)", DD + "interlace/preds_d1_il_real4_dets_test_r2048.json"),
    ("sparse", "T5 s (detector score)", DD + "edgefix/preds_d1_edgefix_sparse_dets_r2048.json")] + [
    r for sp in ("test", "sparse") for r in [
    (sp, "T5 hat s - s", DD + f"preds_dino_swinl_s0_{sp}_t3_no_s_r2048.json"),
    (sp, "T5 hat s - log h", DD + f"preds_dino_swinl_s0_{sp}_t3_no_logh_r2048.json"),
    (sp, "T5 hat s - IoU_up", DD + f"preds_dino_swinl_s0_{sp}_t3_no_iou_r2048.json")]]


def main():
    fp_ = os.path.join(HERE, "table1_best_f1.json")
    out = json.load(open(fp_)) if os.path.exists(fp_) else {"rows": {}}
    out["rows"] = {(k if "|" in k else "test|" + k): v for k, v in out["rows"].items()}  # migrate pre-split keys (all test)
    out["doc"] = __doc__.strip().split("\n\n")[1]
    gts = {}
    for sp, lab, f in ROWS:
        key = f"{sp}|{lab}"
        if key in out["rows"] and out["rows"][key]["source"] == f:
            continue
        gt = gts.setdefault(sp, json.load(open(GT[sp])))
        p = json.load(open(os.path.join(NR, f))); p = p.get("preds", p)
        th, f1, c = MQ.best_f1_threshold(gt, p)
        out["rows"][key] = dict(split=sp, source=f, threshold=th, f1=round(f1, 4), **c,
                                precision=round(c["tp"] / max(c["tp"] + c["fp"], 1), 4), recall=round(c["tp"] / max(c["tp"] + c["fn"], 1), 4))
        print(f"{key:40s} thr={th:.2f} F1={f1:.4f} {c}", flush=True)
        json.dump(out, open(fp_, "w"), indent=1)


if __name__ == "__main__":
    main()

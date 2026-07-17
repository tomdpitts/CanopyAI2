"""Step 3 — DeepForest RGB prebuilt reproduction (scorer gatekeeper).

Runs the DeepForest prebuilt 'tree' model (RGB only, no CHM filter) over the 194
scored NEON tiles and scores it with OUR scorer.py at IoU 0.4. Confirms we land near
the maintained-package operating point (P~0.66 / R~0.79) — i.e. the paper's ~70%.
A large miss => diagnose snapshot / coord frame / matching / averaging BEFORE trusting
the scorer on our own model.

RGB-ONLY: reads only evaluation/RGB/*.tif. No LiDAR/CHM/HSI.

Run in the isolated DeepForest venv:
    .venv_df/bin/python eval_deepforest.py            # all 194
    .venv_df/bin/python eval_deepforest.py --limit 5  # smoke
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from scorer import evaluate, pr_curve  # noqa: E402

RGB_DIR = os.path.join(HERE, "NeonTreeEvaluation", "evaluation", "RGB")
GT_JSON = os.path.join(HERE, "neon_gt.json")
PRED_JSON = os.path.join(HERE, "preds_deepforest.json")
RESULT_JSON = os.path.join(HERE, "deepforest_repro.json")
DF_SCORE_THRESH = 0.01          # low, to keep the PR-curve recall tail
DF_DEFAULT_OP = 0.10            # DeepForest's default score_thresh (package op point)


def predict_all(limit=None):
    from deepforest import main
    m = main.deepforest()
    m.load_model("weecology/deepforest-tree")
    m.config["score_thresh"] = DF_SCORE_THRESH        # keep low-confidence tail
    gt = json.load(open(GT_JSON))
    plots = sorted(gt)[:limit] if limit else sorted(gt)
    preds = {}
    for i, plot in enumerate(plots):
        path = os.path.join(RGB_DIR, plot + ".tif")
        df = m.predict_image(path=path)
        if df is None or len(df) == 0:
            preds[plot] = {"boxes": [], "scores": []}
        else:
            preds[plot] = {
                "boxes": df[["xmin", "ymin", "xmax", "ymax"]].values.tolist(),
                "scores": df["score"].tolist()}
        if (i + 1) % 20 == 0 or i + 1 == len(plots):
            print(f"  predicted {i+1}/{len(plots)}", flush=True)
    return preds, {p: gt[p] for p in plots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse preds_deepforest.json instead of re-predicting")
    a = ap.parse_args()

    gt_full = json.load(open(GT_JSON))
    if a.reuse and os.path.exists(PRED_JSON):
        preds = json.load(open(PRED_JSON))
        gt = {p: gt_full[p] for p in preds}
        print(f"reusing {len(preds)} cached predictions", flush=True)
    else:
        preds, gt = predict_all(a.limit)
        json.dump(preds, open(PRED_JSON, "w"))
        print(f"wrote {PRED_JSON}", flush=True)

    gt_arr = {p: np.array(b, float).reshape(-1, 4) for p, b in gt.items()}
    pr_in = {p: {"boxes": np.array(v["boxes"], float).reshape(-1, 4),
                 "scores": np.array(v["scores"], float)} for p, v in preds.items()}

    # operating point = DeepForest default score_thresh 0.10 (the package op point)
    e = evaluate(pr_in, gt_arr, iou_thr=0.4, score_thr=DF_DEFAULT_OP,
                 nan_precision="zero")
    curve = pr_curve(pr_in, gt_arr, iou_thr=0.4,
                     thresholds=np.round(np.arange(0.05, 0.95, 0.05), 2))
    res = {
        "pinned_tag": "1.8.0", "n_plots": len(gt), "n_gt_boxes": int(
            sum(len(v) for v in gt_arr.values())),
        "iou_thr": 0.4, "operating_score_thr": DF_DEFAULT_OP,
        "mean_precision": round(e["mean_precision"], 4),
        "mean_recall": round(e["mean_recall"], 4),
        "micro_precision": round(e["micro_precision"], 4),
        "micro_recall": round(e["micro_recall"], 4),
        "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
        "target_paper_P": 0.76, "target_paper_R": 0.67,
        "target_package_P": 0.66, "target_package_R": 0.79,
        "pr_curve": curve,
    }
    json.dump(res, open(RESULT_JSON, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "pr_curve"}, indent=2))
    print(f"\n[repro] DeepForest @IoU0.4, score_thr={DF_DEFAULT_OP}: "
          f"macro P={e['mean_precision']:.3f} R={e['mean_recall']:.3f}  "
          f"(package target ~0.66/0.79; paper Table 3 0.76/0.67)")
    print(f"wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()

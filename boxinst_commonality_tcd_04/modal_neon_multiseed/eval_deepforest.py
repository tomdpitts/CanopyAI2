"""Step 3 — DeepForest RGB prebuilt reproduction.

Runs the DeepForest prebuilt 'tree' model (RGB only, no CHM filter) over the 194 scored
NEON tiles and scores it with the authors' benchmark scorer (df_scorer.py =
deepforest.evaluate_boxes) at IoU 0.4. Confirms we land near the maintained-package
operating point (P~0.66 / R~0.79). Same evaluation code used for every reported number.

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
import df_scorer  # noqa: E402  (the NEON benchmark authors' scorer, swept)

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

    # Score with the AUTHORS' scorer (deepforest.evaluate_boxes, swept) — same evaluation
    # code used for every reported number.
    res = df_scorer.score(PRED_JSON, GT_JSON, RESULT_JSON)
    curve = res["pr_curve"]
    p10 = min(curve, key=lambda x: abs(x["score_thr"] - DF_DEFAULT_OP))
    bf = res["best_f1_point"]
    print(f"\n[repro] DeepForest via deepforest.evaluate_boxes @IoU0.4: "
          f"@thr{DF_DEFAULT_OP} P={p10['mean_precision']:.3f} R={p10['mean_recall']:.3f} "
          f"| best-F1 P={bf['P']} R={bf['R']} @thr{bf['score_thr']}")
    print(f"(maintained-package regime ~P0.66/R0.79; paper Table 3 P0.659/R0.790)")
    print(f"wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()

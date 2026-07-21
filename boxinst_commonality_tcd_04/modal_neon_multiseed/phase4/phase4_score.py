"""Apples-to-apples NEON scoring: native seed-0 vs 4-phase seed-0, on the full 194-tile
test set AND the NIWO-12 subset, with the benchmark authors' scorer (df_scorer =
deepforest.evaluate_boxes, IoU 0.4, threshold-swept PR curve).

Run in the deepforest venv (has pandas/deepforest):
    .venv_df/bin/python phase4/phase4_score.py [phase4_preds.json]

Reports best-F1 P/R (the reported operating point) and maxR (recall at thr->0, the
CEILING the experiment targets), per arm per scope. Native NIWO should reproduce
0.534/0.343 (ours_persite.json) -> validates the scorer wiring.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEON = os.path.dirname(HERE)
sys.path.insert(0, NEON)
import df_scorer  # noqa: E402

GT = os.path.join(NEON, "neon_gt.json")
NATIVE = os.path.join(NEON, "preds", "preds_neon_s0.json")
PHASE4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    NEON, "phase4", "out", "preds_phase4_s0.json")
TMP = "/private/tmp/claude-501/-Users-tompitts-dphil-CanopyAI2/scratch_p4score"
os.makedirs(TMP, exist_ok=True)


def _subset(d, pref):
    return {k: v for k, v in d.items() if pref is None or pref in k}


def score(preds_path, pref, tag):
    gt = _subset(json.load(open(GT)), pref)
    preds = _subset(json.load(open(preds_path)), pref)
    gtf = os.path.join(TMP, f"gt_{tag}.json")
    prf = os.path.join(TMP, f"pr_{tag}.json")
    json.dump(gt, open(gtf, "w"))
    json.dump(preds, open(prf, "w"))
    r = df_scorer.score(prf, gtf, iou=0.4)
    maxR = max(c["mean_recall"] for c in r["pr_curve"])
    return {"P": r["mean_precision"], "R": r["mean_recall"], "maxR": round(maxR, 3),
            "thr": r["best_f1_point"]["score_thr"], "n": len(gt)}


def main():
    arms = [("native", NATIVE), ("4phase", PHASE4)]
    scopes = [("global", None), ("NIWO", "NIWO")]
    print(f"\nnative preds : {NATIVE}")
    print(f"4phase preds : {PHASE4}")
    print(f"\n{'scope':7} {'arm':8} {'n':>3}  {'P':>6} {'R':>6} {'maxR':>6} {'thr':>5}")
    print("-" * 46)
    rows = {}
    for sc_name, pref in scopes:
        for arm, path in arms:
            if not os.path.exists(path):
                print(f"{sc_name:7} {arm:8}  MISSING {path}")
                continue
            r = score(path, pref, f"{arm}_{sc_name}")
            rows[(sc_name, arm)] = r
            print(f"{sc_name:7} {arm:8} {r['n']:>3}  {r['P']:>6.3f} {r['R']:>6.3f} "
                  f"{r['maxR']:>6.3f} {r['thr']:>5.2f}")
    # deltas (paired seed-0)
    print("\nΔ (4phase − native), same seed 0:")
    for sc_name, _ in scopes:
        n = rows.get((sc_name, "native")); p = rows.get((sc_name, "4phase"))
        if n and p:
            print(f"  {sc_name:7} ΔP={p['P']-n['P']:+.3f} ΔR={p['R']-n['R']:+.3f} "
                  f"ΔmaxR={p['maxR']-n['maxR']:+.3f}")
    print("\nGO/NO-GO: 4phase clears native maxR (esp. NIWO 0.348) → real sampling win;"
          "\n         flat/negative → NIWO is window-limited (pivot to train-time upscale).")


if __name__ == "__main__":
    main()

"""Summarise results/{sam_p1,lace_p1,sam_p2} -> results/summary.json + print. Warm-up tile excluded.

  cd boxinst_commonality_tcd_04/masker_cost && ../../.venv/bin/python -u summarise.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")


def load(run, tiles):
    return [json.load(open(os.path.join(OUT, run, t + ".json"))) for t in tiles]


def main():
    sel = json.load(open(os.path.join(OUT, "selection.json"))); T = sel["tiles"]
    L, S1, S2 = load("lace_p1", T), load("sam_p1", T), load("sam_p2", T)
    f = lambda g: {k: round(float(fn(g)), 3) for k, fn in (("median", np.median), ("mean", np.mean), ("min", np.min), ("max", np.max))}
    tc = lambda R, k: np.array([r["time"][k] for r in R])
    nb = np.array([r["n_boxes"] for r in L]); assert [r["n_boxes"] for r in S1] == list(nb)
    sam = (tc(S1, "total_s") + tc(S2, "total_s")) / 2                       # per tile mean of the two bracketing passes
    out = {"n_tiles": len(T), "boxes_per_tile": f(nb), "boxes_total": int(nb.sum()),
           "lace_s": {k: f(tc(L, k)) for k in ("feat_s", "render_s", "total_s")},
           "sam3_p1_s": {k: f(tc(S1, k)) for k in ("encoder_s", "decode_post_s", "total_s")},
           "sam3_p2_s": {k: f(tc(S2, k)) for k in ("encoder_s", "decode_post_s", "total_s")},
           "sam3_n_encodes": f(tc(S1, "n_encodes")),
           "sam3_drift_p2_minus_p1_total_s": f(tc(S2, "total_s") - tc(S1, "total_s")),
           "ratio_lace_total_over_sam3_total": f(tc(L, "total_s") / sam),
           "ratio_lace_render_over_sam3_total": f(tc(L, "render_s") / sam),
           "ms_per_box": {"lace_render": round(1000 * float(tc(L, "render_s").sum() / nb.sum()), 2),
                          "sam3_decode_post_p1": round(1000 * float(tc(S1, "decode_post_s").sum() / nb.sum()), 2)},
           "gates": {"lace_feat_cos_mean_min": round(min(r["gate"]["feat_cos_mean"] for r in L), 7),
                     "lace_rle_identical": f"{sum(r['gate']['rle_identical'] for r in L)}/{int(nb.sum())}",
                     "sam_p1_iou_mean_min": round(min(r["gate"]["iou_mean"] for r in S1), 5),
                     "sam_p2_iou_mean_min": round(min(r["gate"]["iou_mean"] for r in S2), 5)},
           "env": {run: json.load(open(os.path.join(OUT, run, "_env.json"))) for run in ("lace_p1", "sam_p1", "sam_p2")}}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()

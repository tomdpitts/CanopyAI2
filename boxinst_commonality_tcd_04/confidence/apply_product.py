"""Apply the posterior-product confidence to a predictions file.

    score' = s * bimod * pfg_mean

Zero fitted parameters (confidence/README.md). The masker is training-free and never sees
the heatmap, so its posterior is an independent second opinion -- which is also what licenses
multiplying the two probabilities rather than fitting a weight between them.

SCORE SCALE: score' is a product of probabilities on a different scale from the heatmap
score, so re-applying the 0.05 decode floor to it would cut a DIFFERENT detection set and the
comparison would not be like-for-like. The detection SET is frozen (everything that passed
0.05 at decode, i.e. exactly what the input file holds); only the ORDERING changes. Score the
output with `--score_floor 0`. The baseline at floor 0 is identical to it at 0.05 because
nothing in the saved preds is below 0.05.

    .venv/bin/python -m boxinst_commonality_tcd_04.confidence.apply_product \
        --preds <preds.json> --post em_post_<tag>.json --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--post", required=True, help="em_post_<tag>.json from em_diag_multi.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = json.load(open(a.preds))
    preds = P.get("preds", P)
    post = json.load(open(a.post if os.path.isabs(a.post) else os.path.join(HERE, a.post)))

    n_box = n_tile = 0
    missing = []
    for t, rec in preds.items():
        if not rec["scores"]:
            continue
        if t not in post:
            missing.append(t)
            continue
        s = np.asarray(rec["scores"], np.float64)
        bm = np.asarray(post[t]["bimod"], np.float64)
        pm = np.asarray(post[t]["pfg_mean"], np.float64)
        if not (len(s) == len(bm) == len(pm)):
            raise ValueError(f"{t}: {len(s)} scores vs {len(bm)} bimod / {len(pm)} pfg_mean "
                             f"-- the posterior file was built from DIFFERENT boxes")
        rec["scores"] = [round(float(v), 8) for v in (s * bm * pm)]
        n_box += len(s); n_tile += 1
    if missing:
        raise ValueError(f"{len(missing)} tiles have predictions but no posterior stats "
                         f"(e.g. {missing[:3]}) -- extraction incomplete")

    meta = P.setdefault("meta", {})
    meta["model"] = meta.get("model", "") + " + posterior product (s x bimod x pfg_mean)"
    meta["confidence"] = {"form": "s * bimod * pfg_mean", "fitted_parameters": 0,
                          "post_file": os.path.basename(a.post),
                          "score_with": "--score_floor 0 (set frozen, ordering only)"}
    json.dump(P, open(a.out, "w"))
    print(f"[product] rescored {n_box} boxes over {n_tile} tiles -> {a.out}")


if __name__ == "__main__":
    main()

"""Re-rank LACE predictions by multiplying in the EM masker's own posterior.

    score' = s * bimod * pfg_mean

    s         sigmoid(hm_logit), the CenterNet heatmap peak LACE already ranks by
    pfg_mean  mean of the masker's foreground posterior over the box's cells
    bimod     mean |2*pfg - 1|, how far that posterior is from undecided

No fit, no weights, no supervision, no held-out split, no re-run of the masker: `estep`
(boxinst_commonality/em.py:193) RETURNS pfg, so both statistics are two reductions over a
value the masker already has in hand. Nothing is recovered from inside estep and nothing
about the pipeline changes -- the detection set is frozen and only the ordering moves.

Why a plain product is the right form: the masker is training-free and never sees the
heatmap, so its opinion is an independent second estimator, and multiplying independent
per-box probabilities is what independence licenses. Fitting a logistic regression on the
same evidence buys nothing (see ablation/RESULTS below).

SCORE SCALE: score' is a product of three probabilities, so it lives on a different scale
from s and the 0.05 decode floor would cut a different detection set. Score the output
with --score_floor 0; the baseline is identical at floor 0 and 0.05 because nothing in the
saved preds sits below 0.05.

    .venv/bin/python -m boxinst_commonality_tcd_04.confidence.rerank_product
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DIAG = os.path.join(HERE, "em_diag_s0.json")
PREDS = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "preds", "knobbed_s0.json")
OUT = os.path.join(HERE, "preds_knobbed_s0_product.json")


def main():
    diag = json.load(open(DIAG))
    P = json.load(open(PREDS))
    n = 0
    for t, r in P["preds"].items():
        d = diag[t]
        r["scores"] = [round(s * b * p, 8) for s, b, p
                       in zip(r["scores"], d["bimod"], d["pfg_mean"])]
        n += len(r["scores"])
    P["meta"]["model"] += " + posterior product rerank (s * bimod * pfg_mean)"
    P["meta"]["rerank"] = {"form": "s * bimod * pfg_mean", "fitted_parameters": 0,
                           "supervision": "none"}
    json.dump(P, open(OUT, "w"))
    print(f"[rerank] {n} boxes rescored -> {os.path.relpath(OUT)}")
    print("Score it with:  .venv/bin/python -m boxinst_commonality_tcd_04.score_coco \\")
    print(f"    --preds {os.path.relpath(OUT)} --res 512 --score_floor 0")


if __name__ == "__main__":
    main()

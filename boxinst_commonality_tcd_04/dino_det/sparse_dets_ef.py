"""Detector-score (s) ranking of the edge-fixed D1 masks on sparse-236, for tab:factors (2026-11-05).

The test-439 s row already exists (results/interlace/scored_d1_il_real4_dets_test_r2048.json). On sparse the only s row
(scored_dino_swinl_s0_sparse_detscore_r2048.json) predates the edge fix. This re-ranks the published edge-fixed masks
(results/edgefix/preds_d1_edgefix_sparse_sprime_r2048.json) by detector s taken from the detscore preds. Pure re-ordering of
a frozen set; no choice is made from it. Frozen score_coco, same protocol as every other row.

Run from repo root: PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.sparse_dets_ef
"""
import json, os

from boxinst_commonality_tcd_04 import score_coco as SC
from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EF = os.path.join(R, "edgefix", "preds_d1_edgefix_sparse_sprime_r2048.json")
DS = os.path.join(R, "preds_dino_swinl_s0_sparse_detscore_r2048.json")
OUTP = os.path.join(R, "edgefix", "preds_d1_edgefix_sparse_dets_r2048.json")
OUTS = OUTP.replace("preds_", "scored_")


def main():
    if os.path.exists(OUTS):
        print("[skip] exists", OUTS); return
    E = json.load(open(EF))["preds"]; D = json.load(open(DS))["preds"]
    assert sorted(E) == sorted(D) and len(E) == 236
    for t in E:
        assert E[t]["boxes_2048"] == D[t]["boxes_2048"], t
    print(f"[ok] 236 tiles, boxes identical, {sum(len(E[t]['boxes_2048']) for t in E)} dets", flush=True)
    json.dump({"meta": {"mask_res": 2048, "model": "LACE (D1 variant) + edge fix, sparse-236, ranking detector s", "rank": "dets"},
               "preds": {t: {"boxes_2048": E[t]["boxes_2048"], "scores": D[t]["scores"], "masks_rle": E[t]["masks_rle"]} for t in E}},
              open(OUTP + ".tmp", "w")); os.replace(OUTP + ".tmp", OUTP)
    s = SC.score_many([OUTP], S.SPLIT["sparse"]["gt"], res=2048, max_dets=600, score_floor=0.0)[0]
    json.dump(s, open(OUTS + ".tmp", "w")); os.replace(OUTS + ".tmp", OUTS)
    print("[score]", json.dumps(s["canopy_neutral_crowd"]), flush=True)


if __name__ == "__main__":
    main()

"""Diagnostic (2026-11-04, Tom: "how much AP does the edge / interior gap represent?"): test 439, s' ranking, D1 boxes; LACE masks with
SAM 3 masks swapped in for (a) tile-edge detections (box within 3 px of the border), (b) interior detections, (c) all. fasteval (parity
with score_coco asserted on test in s2_final). Diagnostic only. -> results/mask_swap_diag_test.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.mask_swap_diag
"""
import json, os
import numpy as np
from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det.mask_diag import SAMD

if __name__ == "__main__":
    G = json.load(open(R.TEST_GT)); St = json.load(open(os.path.join(R.OUT, f"preds_{R.TAG}_test_sprime_r2048.json")))["preds"]; tids = sorted(St)
    Sm = {t: json.load(open(os.path.join(SAMD, t + ".json")))["masks_rle"] for t in tids}; sp = {t: St[t]["scores"] for t in tids}
    edge = {t: [min(b[0], b[1], 2048 - b[2], 2048 - b[3]) <= 3 for b in St[t]["boxes_2048"]] for t in tids}
    rec = {"lace": FE.summary(FE.ap(FE.build(St, G, os.path.join(R.FC_, "fasteval_test.pkl")), sp))}
    for name, pick in (("sam_edge", lambda e: e), ("sam_interior", lambda e: not e), ("sam_all", lambda e: True)):
        P = {t: {"masks_rle": [Sm[t][i] if pick(e) else St[t]["masks_rle"][i] for i, e in enumerate(edge[t])]} for t in tids}
        rec[name] = FE.summary(FE.ap(FE.build(P, G, os.path.join(R.FC_, f"fasteval_test_swap_{name}.pkl")), sp)); print(name, rec[name], flush=True)
    rec["n_edge_dets"] = int(sum(sum(v) for v in edge.values())); rec["n_dets"] = int(sum(len(v) for v in edge.values()))
    json.dump(rec, open(os.path.join(R.OUT, "mask_swap_diag_test.json"), "w"), indent=1); print(json.dumps(rec, indent=1))

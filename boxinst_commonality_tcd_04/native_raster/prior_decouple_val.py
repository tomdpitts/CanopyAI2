"""Prior for RANKING only: masks from one alpha, posterior-product factors from another. VAL ONLY.

prior_sweep_val.py showed the spatial prior's value is in ranking (product AP50 moves 0.024 with
alpha; detector-ranked AP50 moves <= 0.003). This re-scores its per-tile intermediates with the
masks taken from config M and bimod/pfg_mean taken from config R, so ranking = s*bimod_R*pfg_R
while the mask is rendered at M. No new rendering: everything comes from
feat_cache/prior_sweep_val_r2048_tiles/.

Grid: every mask config (10) x tau (5), ranking factors from R in {a0.3_k1.6, a0.3_k2}.
GATE: (M=R=a0.3_k1.6, tau 0.40) reproduces 0.6427.

    .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.prior_decouple_val --workers 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from boxinst_commonality_tcd_04.native_raster.prior_sweep_val import (
    CONFIGS, RES, TAUS, TILE_DIR, _C, _init_score_worker)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prior_decouple_val_r2048.json")
RANK_CFGS = ("a0.3_k1.6", "a0.3_k2")
GATE = ("a0.3_k1.6", "a0.3_k1.6", 0.40, 0.6427)


def score_job(cell):
    mcfg, rcfg, tau = cell
    from boxinst_commonality_tcd_04 import score_coco as S
    P = {}
    for tid in _C["tids"]:
        T = pickle.load(open(os.path.join(TILE_DIR, tid + ".pkl"), "rb"))
        s = np.asarray(T["scores"], np.float64)
        r = T["cfg"][rcfg]
        prod = (s * np.asarray(r["bimod"]) * np.asarray(r["pfg_mean"])).tolist() if len(s) else []
        P[tid] = {"boxes_2048": T["boxes"], "canopy_ignore": [False] * len(s),
                  "masks_rle": [{"size": [RES, RES], "counts": x} for x in T["cfg"][mcfg]["rles"][tau]],
                  "scores": [round(float(v), 8) for v in prod]}
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    with open(tmp, "w") as f:
        json.dump({"meta": {"mask_res": RES}, "preds": P}, f)
    c = S._score_one(tmp, _C["ctx"], RES, S.MAX_DETS, 0.0)["canopy_neutral_crowd"]
    os.remove(tmp)
    return mcfg, rcfg, tau, {"mask": c["mask"], "box": c["box"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from boxinst_commonality_tcd_04.native_raster.oracle_boxes_val import FEAT_DIR, VAL_GT
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        "question": "prior for ranking only: masks at alpha_M, product factors at alpha_R",
        "split": "val (108 tiles)", "res": RES, "source": "prior_sweep_val_r2048 intermediates",
        "cells": {}}
    assert "done" not in out, f"{OUT} complete -- refusing to overwrite"
    gt = json.load(open(VAL_GT))
    tids = sorted(t for t in gt if os.path.exists(os.path.join(FEAT_DIR, t + ".npy")))
    assert len(tids) == 108 and all(os.path.exists(os.path.join(TILE_DIR, t + ".pkl")) for t in tids)
    cells = [(m[0], r, t) for r in RANK_CFGS for m in CONFIGS for t in TAUS]
    cells.sort(key=lambda c: c != GATE[:3])
    todo = [c for c in cells if f"M={c[0]}|R={c[1]}|{c[2]:.2f}" not in out["cells"]]
    print(f"[decouple] {len(todo)}/{len(cells)} cells", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_score_worker,
                                      initargs=(tids,)) as pool:
        for m, r, t, res in pool.imap_unordered(score_job, todo):
            key = f"M={m}|R={r}|{t:.2f}"
            out["cells"][key] = res
            if (m, r, t) == GATE[:3]:
                ok = abs(res["mask"]["AP50"] - GATE[3]) <= 1e-4
                out["gate"] = {"cell": key, "ref": GATE[3], "got": res["mask"]["AP50"], "pass": ok}
                print(f"GATE {key}: {res['mask']['AP50']} vs {GATE[3]} -> {'PASS' if ok else 'FAIL'}",
                      flush=True)
                if not ok:
                    json.dump(out, open(OUT, "w"), indent=2); pool.terminate()
                    raise SystemExit("gate failed")
            json.dump(out, open(OUT, "w"), indent=2)
    best = max(out["cells"].items(), key=lambda kv: kv[1]["mask"]["AP50"])
    out["best_by_AP50"] = {"cell": best[0], **best[1]["mask"]}
    out["done"] = True
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"BEST {best[0]} {best[1]['mask']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

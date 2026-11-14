"""GATE: the ported apply/render path (lace_core.Masker4 + prob4 + render_post4, non-square generalisation) must reproduce
the DEPLOYED LACE masks byte-for-byte on OAM-TCD before it is used on LIVECell.

Reference: feat_cache/refit4p_val_deployed/<tid>.json (masker_refit4p/eval4p.py val --tag deployed: em_model_4p_fix.npz,
D1 val boxes, lace4px_simple render, EDGE_FILL on; its val AP50 0.7414 passed the R4P parity gate 2026-11-04).
Inputs read in place (read-only): hunt_z4p_val/<tid>.npy (4 px cells already projected with the deployed whitening),
D1 val boxes + scores, RGB via btmask._rgb. Also checks s' = s * mean|2r-1| * mean(r) and _verbatim.verify().

  (copied into zeroshot_phenobench/code with lace_core.py / _verbatim.py; only this docstring differs)
  cd zeroshot_phenobench/code && PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../../.venv/bin/python -u gate_parity.py [n_tiles]
Out: ../results/gate_parity.json
"""
import json, os, sys, time

import numpy as np

import _verbatim as V
import lace_core as LC

FC = "/Users/tompitts/dphil/feat_cache"
PKG = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gate_parity.json")


def main(n=2):
    from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import EDGE_FILL, _rle
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask.btmask import _rgb
    assert EDGE_FILL is True
    assert V.verify() and LC.check_frozen()
    P = json.load(open(os.path.join(PKG, "dino_det", "results", "modal", "dino_swinl_s0_preds_val.json")))["preds"]
    m4 = LC.Masker4(np.load(LC.DEPLOYED_NPZ))
    tids = sorted(f[:-5] for f in os.listdir(f"{FC}/refit4p_val_deployed") if f.endswith(".json"))[:n]
    rec = {"reference": f"{FC}/refit4p_val_deployed", "tiles": {}}
    for t in tids:
        t0 = time.time()
        ref = json.load(open(f"{FC}/refit4p_val_deployed/{t}.json"))
        zn = np.load(f"{FC}/hunt_z4p_val/{t}.npy").astype(np.float32).reshape(-1, 128)
        rgbf = _rgb("val", t)
        boxes = np.asarray(P[t]["boxes_2048"], np.float32).reshape(-1, 4); sc = P[t]["scores"]
        same, sp_ok = 0, 0
        for i, b in enumerate(boxes):
            clip, prob, r = LC.prob4(m4, zn, 512, 512, b, 2048, 2048)
            m = LC.render_post4(rgbf, clip, prob, 2048, 2048)
            same += _rle(m) == ref["masks_rle"][i]
            sp_ok += round(float(sc[i] * np.abs(2 * r - 1).mean() * r.mean()), 8) == ref["s_prime"][i]
        rec["tiles"][t] = {"n_boxes": len(boxes), "rle_identical": int(same), "sprime_identical": int(sp_ok),
                           "sec": round(time.time() - t0, 1)}
        print(t, rec["tiles"][t], flush=True)
    rec["pass"] = all(v["rle_identical"] == v["n_boxes"] == v["sprime_identical"] for v in rec["tiles"].values())
    rec["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(rec, open(OUT, "w"), indent=1)
    print("GATE", "PASS" if rec["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

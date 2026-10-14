"""Local, read-only. (1) GATE: replaying selection from raw_dets with the edge band OFF must give
the stored boxes_2048/scores exactly, on every tile. (2) PLAN: with the band ON at interior
edges, which kept boxes were already prompted to SAM 3 (mask reusable) and which are new.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.selvabox_seam_plan \
      --tiles /Users/tompitts/dphil/feat_cache/selvabox_bench_r2048_tiles/tiles
"""
import argparse
import json
import os
import time

import numpy as np

from boxinst_commonality_tcd_04.native_raster import selvabox_seam as SB

NR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", default=os.path.join(NR, "seamfix", "selvabox_seam_plan.json"))
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    tids = sorted(f[:-5] for f in os.listdir(a.tiles) if f.endswith(".json"))
    assert len(tids) == 439, len(tids)
    per, gate_fail = {}, []
    tot = dict(raw=0, edge_dropped=0, old_kept=0, new_kept=0, reused=0, to_prompt=0,
               off_kept=0, off_only=0, stored_only=0)
    for t in tids:
        r = json.load(open(os.path.join(a.tiles, t + ".json")))
        old = {SB.key(b, s) for b, s in zip(r["boxes_2048"], r["scores"])}
        B0, S0, _ = SB.select(r["raw_dets"], None, old)
        # stored values are float32 (rounded) carried as Python floats; raw_dets hold the same
        # float32 values, so the replay must match them EXACTLY (no re-rounding).
        sb = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
        ss = np.asarray(r["scores"], np.float64)
        off = [SB.key(b, s) for b, s in zip(B0, S0)]
        if not (np.array_equal(B0, sb) and np.array_equal(S0, ss)):
            gate_fail.append(t)
        B1, S1, nd = SB.select(r["raw_dets"], SB.EDGE_FRAC, old)
        new = [SB.key(b, s) for b, s in zip(B1, S1)]
        reuse = sum(k in old for k in new)
        n_raw = sum(len(d["scores"]) for d in r["raw_dets"])
        off_extra = sum(k not in old for k in off)
        per[t] = dict(raw=n_raw, edge_dropped=nd, old_kept=len(old), new_kept=len(new),
                      reused=reuse, to_prompt=len(new) - reuse, off_kept=len(off),
                      off_only=off_extra, stored_only=len(old - set(off)))
        for k in tot:
            tot[k] += per[t][k]
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "tiles_dir": a.tiles,
           "rule": f"geodataset _filter_edge_polygons, margin {SB.EDGE_FRAC}*{SB.SUB} px, "
                   "interior subtile edges only, before floor/NMS",
           "gate": {"desc": "edge band OFF oracle replay == stored boxes_2048 & scores "
                            "(box level; exact match impossible from rounded raw_dets, see "
                            "selvabox_seam.py; AP-level gate follows the SAM refill)",
                    "n_fail": len(gate_fail), "fail": gate_fail, "pass": not gate_fail},
           "totals": tot, "per_tile": per}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("gate", "totals")}, indent=1))


if __name__ == "__main__":
    main()

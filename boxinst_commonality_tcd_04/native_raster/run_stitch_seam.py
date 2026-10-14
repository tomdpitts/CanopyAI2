"""Driver for `stitch_seam`: copy of run_stitch_b2m.py plus `--edge_px` (omit = filter off).

Pooled over tiles, JSONL checkpoint per tile with resume, refuses to overwrite. With the filter
off and `--gate_ref <published preds json>`, asserts the written "preds" payload is
byte-identical to the reference file's and writes a small gate record.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.run_stitch_seam \
      --pred_dir <raw subtile dir> --res 2048 --edge_px 10 --model "..." --out <preds.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

from boxinst_commonality_tcd_04.native_raster import stitch_res as ST
from boxinst_commonality_tcd_04.native_raster import stitch_seam as SS

_CFG = {}


def _init(pred_dir, res, iou_thr, cont_thr, min_score, edge_px):
    _CFG.update(pred_dir=pred_dir, res=res, iou_thr=iou_thr, cont_thr=cont_thr,
                min_score=min_score, edge_px=edge_px)


def _one(tid):
    c = _CFG
    crowns, n_drop = SS.load_tile_crowns(c["pred_dir"], tid, c["res"], c["min_score"],
                                         c["edge_px"])
    crowns = ST.dedup(crowns, c["iou_thr"], c["cont_thr"])
    return tid, ST.serialize_tile(crowns, c["res"]), n_drop


def _preds_payload(path):
    """Bytes of the "preds" value exactly as written by json.dump({"meta":..,"preds":..})."""
    b = open(path, "rb").read()
    k = b.index(b'"preds": ')
    assert b.endswith(b"}")
    return b[k + len(b'"preds": '):-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt", default="boxinst_commonality_tcd_04/test_gt.json")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--iou_thr", type=float, default=0.7)
    ap.add_argument("--cont_thr", type=float, default=0.85)
    ap.add_argument("--min_score", type=float, default=0.05)
    ap.add_argument("--edge_px", type=int, default=None,
                    help="interior-edge band in subtile px; omit = filter OFF (gate mode)")
    ap.add_argument("--model", required=True, help="meta.model string for the output file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="JSONL checkpoint (default: <out>.jsonl)")
    ap.add_argument("--gate_ref", default=None, help="published preds json to byte-compare")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    assert a.gate_ref is None or a.edge_px is None, "gate only with the filter OFF"
    ckpt = a.ckpt or a.out + ".jsonl"
    tids = sorted(json.load(open(a.gt)))

    done, drops = {}, {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["tid"]] = r["rec"]
            drops[r["tid"]] = r["n_edge_dropped"]
        print(f"resumed {len(done)} tiles from {ckpt}", flush=True)

    todo = [t for t in tids if t not in done]
    t0 = time.time()
    if todo:
        with open(ckpt, "a") as fh, Pool(a.workers, _init,
                                         (a.pred_dir, a.res, a.iou_thr, a.cont_thr,
                                          a.min_score, a.edge_px)) as pool:
            for k, (tid, rec, nd) in enumerate(pool.imap_unordered(_one, todo, chunksize=1)):
                done[tid] = rec
                drops[tid] = nd
                fh.write(json.dumps({"tid": tid, "rec": rec, "n_edge_dropped": nd}) + "\n")
                fh.flush()
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {k+1}/{len(todo)} tiles  {el:.0f}s  "
                          f"eta {el/(k+1)*(len(todo)-k-1):.0f}s", flush=True)

    preds = {t: done[t] for t in tids}
    n_drop = sum(drops[t] for t in tids)
    meta = {"model": a.model, "mask_res": a.res, "scale_box_to_mask": 2048 / a.res,
            "n_tiles": len(preds),
            "dedup": {"iou_thr": a.iou_thr, "cont_thr": a.cont_thr,
                      "min_score": a.min_score},
            "min_score": a.min_score,
            "seam_filter": None if a.edge_px is None else {
                "edge_px": a.edge_px, "where": "interior subtile edges only, before dedup",
                "rule": "detectree2 stitch_crowns(shift=1 m): keep iff mask within subtile "
                        "inset by edge_px", "n_subtile_preds_dropped": n_drop},
            "source_raw_preds": a.pred_dir,
            "stitcher": "native_raster/stitch_seam.py (stitch_res + pre-dedup interior-edge "
                        "filter; filter-off gated byte-identical)"}
    json.dump({"meta": meta, "preds": preds}, open(a.out, "w"))
    n = sum(len(v["scores"]) for v in preds.values())
    print(f"-> {a.out}  tiles={len(preds)} dets={n} edge_dropped={n_drop}  "
          f"{time.time()-t0:.0f}s", flush=True)

    if a.gate_ref:
        new, ref = _preds_payload(a.out), _preds_payload(a.gate_ref)
        g = {"gate": "filter OFF reproduces published preds payload byte-for-byte",
             "pass": new == ref, "new": a.out, "ref": a.gate_ref,
             "sha256_new": hashlib.sha256(new).hexdigest(),
             "sha256_ref": hashlib.sha256(ref).hexdigest(), "n_dets": n,
             "date": time.strftime("%Y-%m-%d %H:%M")}
        gp = a.out[:-5] + "_GATE.json"
        assert not os.path.exists(gp), f"REFUSING to overwrite {gp}"
        json.dump(g, open(gp, "w"), indent=1)
        print(f"GATE {'PASS' if g['pass'] else 'FAIL'}  -> {gp}", flush=True)


if __name__ == "__main__":
    main()

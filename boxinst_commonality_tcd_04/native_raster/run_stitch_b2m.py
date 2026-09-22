"""Re-stitch the persisted Box2Mask per-subtile predictions at an arbitrary raster.

Tiles are independent under `stitch_all` (the loop carries no cross-tile state), so this
shards them over a process pool and reassembles in `sorted(gt)` order. The per-tile result is
byte-identical to the serial loop; the 512 gate proves it against the published files.

Writes a JSONL checkpoint per tile as it goes and resumes from it, so a killed run costs only
the tiles in flight.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.run_stitch_b2m \
      --pred_dir <raw subtile dir> --res 512 --out <preds.json> --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

from boxinst_commonality_tcd_04.native_raster import stitch_res as ST

_CFG = {}


def _init(pred_dir, res, iou_thr, cont_thr, min_score):
    _CFG.update(pred_dir=pred_dir, res=res, iou_thr=iou_thr, cont_thr=cont_thr,
                min_score=min_score)


def _one(tid):
    c = _CFG
    crowns = ST.dedup(ST.load_tile_crowns(c["pred_dir"], tid, c["res"], c["min_score"]),
                      c["iou_thr"], c["cont_thr"])
    return tid, ST.serialize_tile(crowns, c["res"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt", default="boxinst_commonality_tcd_04/test_gt.json")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--iou_thr", type=float, default=0.7)
    ap.add_argument("--cont_thr", type=float, default=0.85)
    ap.add_argument("--min_score", type=float, default=0.05)
    ap.add_argument("--model", required=True, help="meta.model string for the output file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="JSONL checkpoint (default: <out>.jsonl)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    ckpt = a.ckpt or a.out + ".jsonl"
    tids = sorted(json.load(open(a.gt)))
    if a.limit:
        tids = tids[:a.limit]

    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["tid"]] = r["rec"]
        print(f"resumed {len(done)} tiles from {ckpt}", flush=True)

    todo = [t for t in tids if t not in done]
    t0 = time.time()
    if todo:
        with open(ckpt, "a") as fh, Pool(a.workers, _init,
                                         (a.pred_dir, a.res, a.iou_thr, a.cont_thr,
                                          a.min_score)) as pool:
            for k, (tid, rec) in enumerate(pool.imap_unordered(_one, todo, chunksize=1)):
                done[tid] = rec
                fh.write(json.dumps({"tid": tid, "rec": rec}) + "\n")
                fh.flush()
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {k+1}/{len(todo)} tiles  {el:.0f}s  "
                          f"eta {el/(k+1)*(len(todo)-k-1):.0f}s", flush=True)

    preds = {t: done[t] for t in tids}
    meta = {"model": a.model, "mask_res": a.res, "scale_box_to_mask": 2048 / a.res,
            "n_tiles": len(preds),
            "dedup": {"iou_thr": a.iou_thr, "cont_thr": a.cont_thr,
                      "min_score": a.min_score},
            "min_score": a.min_score,
            "source_raw_preds": a.pred_dir,
            "stitcher": "native_raster/stitch_res.py (RES-parametrised; gated against the "
                        "published 512 stitch)"}
    json.dump({"meta": meta, "preds": preds}, open(a.out, "w"))
    n = sum(len(v["scores"]) for v in preds.values())
    print(f"-> {a.out}  tiles={len(preds)} dets={n}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

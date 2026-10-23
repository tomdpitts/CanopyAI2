"""Sweep render variants on cached GT-box windows. Writes per-crown IoUs for each variant
(new file per call; never overwrites) plus a summary keyed by variant name.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.boxfix.sweep \
      --split val --variants boxfix/variants_v1.json --out boxfix/sweep_val_v1.json
"""
import argparse, json, multiprocessing as mp, os, time
import numpy as np
from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R

_V = {}


def _tile(fp):
    crowns = R.load_tile(fp)
    out = {"tile": os.path.basename(fp)[:-4], "valid": [c["valid"] for c in crowns],
           "area": [c["gt_area"] for c in crowns],
           "box": [[round(float(t), 2) for t in c["box"]] for c in crowns], "iou": {}}
    for name, v in _V.items():
        out["iou"][name] = [round(R.iou(c, R.render(c, v)), 5) for c in crowns]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"refusing to overwrite {a.out}"
    _V.update(json.load(open(a.variants)))
    files = R.tile_files(a.split)
    print(f"[sweep] {a.split}: {len(files)} tiles, {len(_V)} variants", flush=True)
    t0, recs = time.time(), []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_tile, files)):
            recs.append(r)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    recs.sort(key=lambda r: r["tile"])
    summ = {}
    for name in _V:
        x = np.concatenate([np.asarray(r["iou"][name])[np.asarray(r["valid"], bool)] for r in recs])
        summ[name] = {"mean_iou": round(float(x.mean()), 4), "n": int(len(x))}
    json.dump({"split": a.split, "variants": _V, "summary": summ, "tiles": recs}, open(a.out, "w"))
    for name, s in summ.items():
        print(f"  {name:40s} {s['mean_iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()

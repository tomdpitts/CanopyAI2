"""Matched-pair mask IoU on DETECTIONS, masker-only: pairs fixed by BOX IoU >= 0.5 (E.match_tile, greedy, descending
score of --pair_preds, no canopy ignore), so every arm is scored on the same (detection, crown) pairs and differences
are due to masks alone. GT masks: E.raster at 2048 (gtbox_lib.gt_rles_and_valid; invalid crowns dropped).
All arm preds must carry identical boxes_2048 per tile (asserted). Strata: GT area, fixed test tercile edges.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.boxfix.pair_iou --gt GT.json \
      --pair_preds P.json --arm name=preds.json [--arm ...] --out OUT.json
"""
import argparse, json, multiprocessing as mp, os
import numpy as np
from pycocotools import mask as mu

EDGES = (622.0, 2456.3)
_S = {}


def _tile(t):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G
    from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import box_iou_mat
    polys = _S["gt"][t]["trees"]; pp = _S["pair"][t]
    bx = np.asarray(pp["boxes_2048"], np.float64).reshape(-1, 4); sc = np.asarray(pp["scores"], np.float64)
    out = {"tile": t, "det": [], "crown": [], "area": [], "iou": {a: [] for a in _S["arms"]}}
    if not len(polys) or not len(bx):
        return out
    gb = G.gt_boxes_2048(polys).astype(np.float64)
    order, _, _, _, gidx = E.match_tile(box_iou_mat(bx, gb), sc, np.zeros(len(sc), bool), 0.5)
    pairs = [(int(order[k]), int(j)) for k, j in enumerate(gidx) if j >= 0]
    if not pairs:
        return out
    g_rles, valid = G.gt_rles_and_valid([polys[j] for _, j in pairs], res=2048)
    for (i, j), gr, v in zip(pairs, g_rles, valid):
        if not v:
            continue
        out["det"].append(i); out["crown"].append(j); out["area"].append(int(mu.area(gr)))
        for a, P in _S["arms"].items():
            r = P[t]["masks_rle"][i]; pr = {"size": r["size"], "counts": r["counts"].encode()}
            inter = float(mu.area(mu.merge([pr, gr], intersect=1))); u = float(mu.area(pr)) + float(mu.area(gr)) - inter
            out["iou"][a].append(round(inter / u, 5) if u > 0 else 0.0)
    return out


def boot(d_by_tile, n=2000, seed=0):
    rng = np.random.default_rng(seed); s = np.array([x.sum() for x in d_by_tile]); c = np.array([len(x) for x in d_by_tile])
    T = len(s); bs = [s[i].sum() / max(c[i].sum(), 1) for i in (rng.integers(0, T, T) for _ in range(n))]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pair_preds", required=True)
    ap.add_argument("--arm", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", default=None, help="optional JSON list of tile ids")
    a = ap.parse_args()
    assert not os.path.exists(a.out), a.out
    _S["gt"] = json.load(open(a.gt)); _S["pair"] = json.load(open(a.pair_preds))["preds"]
    _S["arms"] = {}
    for spec in a.arm:
        name, fp = spec.split("=", 1); _S["arms"][name] = json.load(open(fp))["preds"]
    tids = json.load(open(a.tiles)) if a.tiles else sorted(t for t in _S["gt"] if _S["gt"][t]["trees"])
    for t in tids:
        for name, P in _S["arms"].items():
            assert P[t]["boxes_2048"] == _S["pair"][t]["boxes_2048"], f"box mismatch {name} {t}"
    with mp.get_context("fork").Pool(8) as pool:
        recs = pool.map(_tile, tids)
    names = list(_S["arms"]); ref = names[0]
    summ = {"pair_preds": a.pair_preds, "arms": {n: f for n, f in (s.split("=", 1) for s in a.arm)},
            "n_tiles": len(recs), "n_pairs": int(sum(len(r["det"]) for r in recs)), "strata": {}}
    for st, lo, hi in (("all", 0, 1e12), ("small", 0, EDGES[0]), ("mid", EDGES[0], EDGES[1]), ("large", EDGES[1], 1e12)):
        ks = [(np.asarray(r["area"]) >= lo) & (np.asarray(r["area"]) < hi) for r in recs]
        row = {"n": int(sum(k.sum() for k in ks))}
        for n in names:
            x = np.concatenate([np.asarray(r["iou"][n])[k] for r, k in zip(recs, ks)])
            row[n] = {"mean_iou": round(float(x.mean()), 4), "ge50": round(float((x >= .5).mean()), 4),
                      "ge75": round(float((x >= .75).mean()), 4), "ge90": round(float((x >= .9).mean()), 4)}
            if n != ref:
                d = [np.asarray(r["iou"][n])[k] - np.asarray(r["iou"][ref])[k] for r, k in zip(recs, ks)]
                row[f"{n}_minus_{ref}"] = {"mean": round(float(np.concatenate(d).mean()), 4), "ci95_tile_boot": boot(d)}
        summ["strata"][st] = row
    json.dump({"summary": summ, "tiles": recs}, open(a.out, "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

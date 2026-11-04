"""Pre-registration S2 (README.md, 2026-11-04): val-only search for an AP50 or AP50:95 gain over s' with no learned model. Local compute.

  closed : score = s * (s'/s)^a * (1 - max_iou_higher)^b on a grid; full-val best AND 5-fold tile-CV of the grid choice (rr + perm1/perm2)
  morph  : mask post-processing under s' ranking (fill holes, largest component, convex hull, Gaussian boundary smoothing, clip to box)
  ratio  : matched pred/GT mask area ratio and IoU by GT size (diagnostic)
Results -> results/side2_val.json (resumable per item); log native_raster/logs/side2_2026-11-04.log.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.side2 all
"""
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE
from boxinst_commonality_tcd_04.dino_det.rerank3 import fold_sets

OUT = os.path.join(R.OUT, "side2_val.json"); NPROC = 14
A = np.round(np.arange(0, 1.51, 0.125), 3); B = np.round(np.arange(0, 3.01, 0.25), 2)


def save(rec):
    json.dump(rec, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)


def closed(rec):
    if "closed" in rec:
        return
    E = FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl")); Z = np.load(os.path.join(R.FC_, "rerankR1_val.npz"))
    X, tile = Z["X"], Z["tile"]; s, sp, mih = X[:, 0], X[:, 1], X[:, 12]; n = len(Z["tids"])
    sc = lambda a, b: s * (sp / np.maximum(s, 1e-9)) ** a * (1 - mih) ** b
    t0 = time.time(); G = {}
    for a in A:
        for b in B:
            G[(a, b)] = FE.ap(E, sc(a, b))
    print(f"[closed] grid {len(G)} {time.time()-t0:.0f}s", flush=True)
    full = max(G, key=lambda k: G[k][0]); base = G[(1.0, 0.0)]
    # tile-fold CV of the grid choice: choose (a, b) on 4/5 of tiles by AP50, score the held-out 1/5 with it; OOF AP50 of the
    # concatenated held-out scores is NOT comparable across folds' choices unless scores are comparable, so evaluate per fold.
    tids = list(Z["tids"]); cv = {}
    for fname, tf in {k: v for k, v in fold_sets(n).items() if k in ("rr", "perm1", "perm2")}.items():
        tf = np.asarray(tf); rows = []
        for k in range(R.K):
            tr = {tids[i] for i in range(n) if tf[i] != k}; te = {tids[i] for i in range(n) if tf[i] == k}
            Etr = {"tids": [t for t in E["tids"] if t in tr], "tiles": [x for t, x in zip(E["tids"], E["tiles"]) if t in tr]}
            Ete = {"tids": [t for t in E["tids"] if t in te], "tiles": [x for t, x in zip(E["tids"], E["tiles"]) if t in te]}
            mtr = np.isin(tile, [i for i in range(n) if tf[i] != k]); mte = ~mtr
            ch = max(G, key=lambda q: FE.ap(Etr, sc(*q)[mtr], FE.THR[:1])[0])
            v, v0 = FE.ap(Ete, sc(*ch)[mte]), FE.ap(Ete, sp[mte])
            rows.append({"fold": k, "choice": list(ch), "ap50_gain": round(float(v[0] - v0[0]), 4), "ap5095_gain": round(float(v.mean() - v0.mean()), 4)})
        cv[fname] = {"folds": rows, "mean_ap50_gain": round(float(np.mean([r["ap50_gain"] for r in rows])), 4),
                     "mean_ap5095_gain": round(float(np.mean([r["ap5095_gain"] for r in rows])), 4)}
        print(f"[closed] cv {fname}: {cv[fname]['mean_ap50_gain']:+.4f} AP50, {cv[fname]['mean_ap5095_gain']:+.4f} AP50:95, "
              f"choices {[r['choice'] for r in rows]}", flush=True)
    b0 = {f"{a}": FE.summary(G[(a, 0.0)]) for a in A}
    rec["closed"] = {"sprime": FE.summary(base), "full_val_best": {"a": full[0], "b": full[1], **FE.summary(G[full])},
                     "b0_curve": b0, "a1_curve": {f"{b}": FE.summary(G[(1.0, b)]) for b in B}, "cv": cv,
                     "grid_ap50": {f"{a},{b}": round(float(v[0]), 4) for (a, b), v in G.items()},
                     "grid_ap5095": {f"{a},{b}": round(float(v.mean()), 4) for (a, b), v in G.items()}}
    print(f"[closed] full-val best a={full[0]} b={full[1]}: {FE.summary(G[full])} vs s' {FE.summary(base)}", flush=True)
    save(rec)


def _post(a):
    from pycocotools import mask as M
    from scipy import ndimage as ndi
    rles, boxes, op = a; out = []
    for r, bx in zip(rles, boxes):
        m = M.decode(r); ys, xs = np.nonzero(m)
        if len(ys) == 0:
            out.append(r); continue
        p = 8; y0, y1, x0, x1 = max(0, ys.min() - p), min(2048, ys.max() + p + 1), max(0, xs.min() - p), min(2048, xs.max() + p + 1)
        c = m[y0:y1, x0:x1].astype(bool)
        if op == "fill":
            c = ndi.binary_fill_holes(c)
        elif op == "lcc":
            lab, nl = ndi.label(c)
            if nl > 1:
                c = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        elif op == "hull":
            from scipy.spatial import ConvexHull
            from matplotlib.path import Path
            yy, xx = np.nonzero(c); pts = np.concatenate([np.c_[xx + dx, yy + dy] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
            try:
                h = ConvexHull(pts); gy, gx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
                c = Path(pts[h.vertices]).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(c.shape) | c
            except Exception:
                pass
        elif op == "erodesmall":   # S2b: 1 px erosion only for masks of equivalent diameter < 24 px (the ratio diagnostic)
            if np.sqrt(c.sum() * 4 / np.pi) < 24:
                c2 = ndi.binary_erosion(c, ndi.generate_binary_structure(2, 1)); c = c2 if c2.any() else c
        elif op == "blurbig2":     # S2b: sigma 2 for masks >= 24 px equivalent diameter, sigma 1 below
            sg = 2.0 if np.sqrt(c.sum() * 4 / np.pi) >= 24 else 1.0
            c2 = ndi.gaussian_filter(c.astype(np.float32), sg) >= 0.5; c = c2 if c2.any() else c
        elif op.startswith("blurprop"):  # S3: scale-aware smoothing, sigma = c x equivalent diameter
            sg = float(op[8:]) * np.sqrt(c.sum() * 4 / np.pi); q = int(np.ceil(3 * sg)) + 1
            cp = np.pad(c, q); c2 = (ndi.gaussian_filter(cp.astype(np.float32), sg) >= 0.5)[q:-q, q:-q]; c = c2 if c2.any() else c
        elif op.startswith("blur"):
            c2 = ndi.gaussian_filter(c.astype(np.float32), float(op[4:])) >= 0.5
            c = c2 if c2.any() else c
        elif op.startswith("clip"):
            mg = int(op[4:]); X0, Y0, X1, Y1 = bx; k = np.zeros_like(c)
            k[max(0, int(np.floor(Y0)) - mg - y0):max(0, int(np.ceil(Y1)) + mg - y0), max(0, int(np.floor(X0)) - mg - x0):max(0, int(np.ceil(X1)) + mg - x0)] = True
            c2 = c & k; c = c2 if c2.any() else c
        m2 = np.zeros_like(m); m2[y0:y1, x0:x1] = c; e = M.encode(np.asfortranarray(m2)); e["counts"] = e["counts"].decode(); out.append(e)
    return out


def morph(rec):
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    rec.setdefault("morph", {}); vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))
    tids = sorted(P); sp = {t: P[t]["s_prime"] for t in tids}
    E0 = FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl")); rec["morph"]["none"] = FE.summary(FE.ap(E0, sp))
    for op in ("fill", "lcc", "hull", "blur1", "blur2", "blur3", "clip0", "clip2"):
        if op in rec["morph"]:
            continue
        t0 = time.time(); cache = os.path.join(R.FC_, f"fasteval_val_post_{op}.pkl")
        if not os.path.exists(cache):
            with mp.get_context("spawn").Pool(NPROC) as pool:
                M2 = pool.map(_post, [([SL._as_bytes(r) for r in P[t]["masks_rle"]], np.asarray(P[t]["boxes_2048"]).reshape(-1, 4), op) for t in tids])
            E = FE.build({t: {"masks_rle": m} for t, m in zip(tids, M2)}, vg, cache)
        else:
            E = FE.build(None, None, cache)
        rec["morph"][op] = FE.summary(FE.ap(E, sp)); print(f"[morph] {op}: {rec['morph'][op]} (none {rec['morph']['none']}) {time.time()-t0:.0f}s", flush=True)
        save(rec)


def _ratio_tile(a):
    from pycocotools import mask as M
    rles, gts = a
    if not rles or not gts:
        return []
    iou = np.asarray(M.iou(rles, gts, [0] * len(gts))).reshape(len(rles), len(gts)); j = iou.argmax(1); ga = np.array([M.area(g) for g in gts], float); pa = np.array([M.area(r) for r in rles], float)  # batch M.area overflows > 255 masks (numpy 2)
    return [(float(ga[j[i]]), float(pa[i] / ga[j[i]]), float(iou[i, j[i]])) for i in range(len(rles)) if iou[i, j[i]] >= 0.5]


def ratio(rec):
    if "ratio" in rec:
        return
    import multiprocessing as mp
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); tids = sorted(P)
    gd, t2i = SL.build_gt_crowd(vg, 2048); by = {i: [] for i in t2i.values()}
    for a in gd["annotations"]:
        if not a["iscrowd"]:
            by[a["image_id"]].append(a["segmentation"])
    with mp.get_context("spawn").Pool(NPROC) as pool:
        rows = sum(pool.map(_ratio_tile, [([SL._as_bytes(r) for r in P[t]["masks_rle"]], by[t2i[t]]) for t in tids]), [])
    a = np.asarray(rows); d = np.sqrt(a[:, 0] * 4 / np.pi); out = {}
    for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 1e9)):
        k = (d >= lo) & (d < hi)
        out[f"{lo}-{int(min(hi, 9999))}px"] = {"n": int(k.sum()), "area_ratio_median": round(float(np.median(a[k, 1])), 3),
                                               "iou_median": round(float(np.median(a[k, 2])), 3), "frac_iou_ge_075": round(float((a[k, 2] >= 0.75).mean()), 3)}
    rec["ratio"] = out; print("[ratio]", json.dumps(out), flush=True); save(rec)


if __name__ == "__main__":
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for f in (closed, ratio, morph):
        f(rec)

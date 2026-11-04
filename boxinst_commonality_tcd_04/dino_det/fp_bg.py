"""What are D1's background FPs? (2026-11-03) Diagnostic only; follows fp_breakdown.py.

Re-runs fp_breakdown's matching per tile and persists a per-det table for all 200,148 test dets
(results/fp_det_table_dino_swinl_s0_test.npz: tile, det idx, s', label, canopy fraction, mask area, box, max IoU).
Then, for bg FPs vs TPs: canopy-fraction histogram, area quantiles, and the same split for bg FPs with s' >= the median TP.
Contact sheet: the top 100 bg FPs by s'. Red = det mask, green = GT trees, yellow = canopy polygons.
Written to results/fp_bg_top100_dino_swinl_s0_test.jpg.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.fp_bg
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"
TIF = os.path.join(REPO, "data", "tcd", "test", "{}.tif")
NPZ = os.path.join(OUT, f"fp_det_table_{TAG}_test.npz")


def _tile(args):
    t, g, r = args
    from pycocotools import mask as M
    from boxinst_commonality_tcd_04 import score_coco as SC
    from boxinst_commonality_tcd_04.dino_det.fp_breakdown import _tile as T
    _, s, lab, _, _ = T(args)
    d = [SC._as_bytes(x) for x in r["masks_rle"]]; nd = len(d)
    if nd == 0:
        return t, np.zeros((0, 10))
    tr = SC._polys_to_rle(g["trees"], 2048); can = SC._union(SC._polys_to_rle(g["canopy"], 2048), 2048)
    frac = SC._frac_inside(d, can); area = SC._areas(d)
    m = np.asarray(M.iou(d, tr, [0] * len(tr)), np.float64).reshape(nd, len(tr)).max(1) if tr else np.zeros(nd)
    b = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4)
    return t, np.column_stack([np.arange(nd), s, lab, frac, area, b, m])


def build():
    t0 = time.time()
    gt = json.load(open(os.path.join(PKG, "test_gt.json"))); tids = sorted(gt)
    P = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")))["preds"]
    R = {}
    with mp.get_context("spawn").Pool(12) as pool:
        for k, (t, a) in enumerate(pool.imap_unordered(_tile, [(t, gt[t], P[t]) for t in tids]), 1):
            R[t] = a
            if k % 40 == 0 or k == len(tids):
                print(f"  tile {k}/439 {time.time()-t0:.0f}s", flush=True)
    ti = np.concatenate([np.full(len(R[t]), i) for i, t in enumerate(tids)]); A = np.concatenate([R[t] for t in tids])
    np.savez_compressed(NPZ + ".tmp.npz", tids=np.asarray(tids), tile=ti, det=A[:, 0].astype(int), s=A[:, 1], lab=A[:, 2].astype(int),
                        frac=A[:, 3], area=A[:, 4], box=A[:, 5:9], maxiou=A[:, 9])
    os.replace(NPZ + ".tmp.npz", NPZ)


def stats():
    Z = np.load(NPZ); s, lab, frac, area = Z["s"], Z["lab"], Z["frac"], Z["area"]
    tp = lab == -1; bg = lab == 5; med = float(np.median(s[tp])); hi = bg & (s >= med)
    bins = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    h = lambda m: np.histogram(frac[m], bins)[0].tolist()
    q = lambda m: np.percentile(np.sqrt(area[m]), [10, 25, 50, 75, 90]).round(1).tolist()
    tilecount = np.bincount(Z["tile"][hi], minlength=len(Z["tids"]))
    out = {"median_tp_sprime": round(med, 4), "canopy_frac_bins": bins,
           "frac_hist": {"tp": h(tp), "bg_all": h(bg), "bg_hi": h(hi)},
           "sqrt_area_px_q10_25_50_75_90": {"tp": q(tp), "bg_all": q(bg), "bg_hi": q(hi)},
           "bg_hi_n": int(hi.sum()), "bg_hi_tiles_nonzero": int((tilecount > 0).sum()),
           "bg_hi_top10_tiles": [[str(Z["tids"][i]), int(tilecount[i])] for i in np.argsort(-tilecount)[:10]],
           "bg_hi_maxiou_gt0": int((Z["maxiou"][hi] > 0).sum())}
    json.dump(out, open(os.path.join(OUT, f"fp_bg_stats_{TAG}_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def sheet(n=100, cell=200, cols=10):
    from PIL import Image, ImageDraw
    from pycocotools import mask as M
    Z = np.load(NPZ); gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    P = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")))["preds"]
    idx = np.flatnonzero(Z["lab"] == 5); idx = idx[np.argsort(-Z["s"][idx], kind="mergesort")][:n]
    S = Image.new("RGB", (cols * cell, ((n + cols - 1) // cols) * cell), "black"); rgb = {}
    for k, i in enumerate(idx):
        t = str(Z["tids"][Z["tile"][i]])
        if t not in rgb:
            rgb = {t: Image.open(TIF.format(t)).convert("RGB")}
        x0, y0, x1, y1 = Z["box"][i]; c = ((x0 + x1) / 2, (y0 + y1) / 2); h = max(x1 - x0, y1 - y0, 96)
        L, T_ = int(max(0, c[0] - h)), int(max(0, c[1] - h)); R_, B = int(min(2048, c[0] + h)), int(min(2048, c[1] + h))
        im = rgb[t].crop((L, T_, R_, B)).copy(); dr = ImageDraw.Draw(im)
        for key, col in (("canopy", (255, 230, 0)), ("trees", (0, 255, 0))):
            for p in gt[t][key]:
                if len(p) >= 6:
                    dr.line([(p[j] - L, p[j + 1] - T_) for j in range(0, len(p), 2)] + [(p[0] - L, p[1] - T_)], fill=col, width=2)
        m = M.decode({"size": [2048, 2048], "counts": P[t]["masks_rle"][Z["det"][i]]["counts"].encode()})[T_:B, L:R_]
        edge = m.astype(bool) & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)).astype(bool)
        a = np.asarray(im).copy(); a[edge] = (255, 0, 0); im = Image.fromarray(a).resize((cell, cell))
        ImageDraw.Draw(im).text((3, 3), f"{k} s'{Z['s'][i]:.2f} c{Z['frac'][i]:.2f}", fill=(255, 255, 255))
        S.paste(im, ((k % cols) * cell, (k // cols) * cell))
    fp = os.path.join(OUT, f"fp_bg_top{n}_{TAG}_test.jpg"); S.save(fp, quality=90); print(fp, flush=True)


if __name__ == "__main__":
    if not os.path.exists(NPZ):
        build()
    stats(); sheet()

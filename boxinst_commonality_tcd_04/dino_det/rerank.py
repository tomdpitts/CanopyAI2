"""Pre-registration R1 (README.md): a small learned reranker on frozen D1, trained on val only with BOX labels.

  cv  : val features + box labels -> 5-fold tile-grouped OOF -> val mask AP50 (score_lite), ablations, bootstrap CI, gate.
  test: ONLY if the gate passed. The mean of the 5 fold models is applied to the test 439 -> frozen score_coco + paired bootstrap vs s'.
Features are cached at feat_cache/rerankR1_<split>.npz. Results go to results/rerankR1_{cv,test}.json.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.rerank cv|test
"""
import argparse, contextlib, io, json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"; FC_ = "/Users/tompitts/dphil/feat_cache"
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json"); TEST_GT = os.path.join(PKG, "test_gt.json")
TIF = {"val": os.path.join(REPO, "data", "tcd", "train", "{}.tif"), "test": os.path.join(REPO, "data", "tcd", "test", "{}.tif"),
       "sparse": os.path.join(REPO, "data", "tcd_sparse", "test", "{}.tif")}
NAMES = ["s", "sp", "log_sp_s",
         "log_w", "log_h", "log_aspect", "log_marea", "fill", "border",
         "log_nhi", "tile_top50", "rank_frac", "max_iou_higher", "n_nbr",
         "R", "G", "B", "exg", "L_std", "L_grad",
         "ring_dL", "ring_dexg", "ring_quad_asym", "ring_quad_min"]
SCORE = [0, 1, 2]; GEOM = [3, 4, 5, 6, 7, 8]; CTX = [9, 10, 11, 12, 13]; APP = list(range(14, 24))
NF = len(NAMES); K, SEED, NBOOT = 5, 0, 2000


def _box_iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / np.maximum(aa[:, None] + bb[None, :] - i, 1e-9)


def _feats(args):
    """One tile -> (tid, X [n, NF], y [n] (1 / 0 / -1 crowd-ignored; None for test))."""
    split, t, r, g = args
    from PIL import Image
    from scipy import ndimage as ndi
    from pycocotools import mask as M
    from boxinst_commonality_tcd_04 import score_coco as SC
    b = np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4); s = np.asarray(r["scores"], np.float64)
    sp = np.asarray(r["s_prime"], np.float64); rles = [SC._as_bytes(x) for x in r["masks_rle"]]; n = len(s)
    X = np.zeros((n, NF))
    if n == 0:
        return t, X, (np.zeros(0) if g is not None else None)
    img = np.asarray(Image.open(TIF[split].format(t)).convert("RGB"), np.float64) / 255.0
    assert img.shape[:2] == (2048, 2048), img.shape
    L = img.mean(2); exg = (2 * img[..., 1] - img[..., 0] - img[..., 2]) / np.maximum(img.sum(2), 1e-3)
    gy, gx = np.gradient(L); G = np.hypot(gx, gy)
    w = np.maximum(b[:, 2] - b[:, 0], 1); h = np.maximum(b[:, 3] - b[:, 1], 1)
    X[:, 0], X[:, 1], X[:, 2] = s, sp, np.log(np.maximum(sp, 1e-6) / np.maximum(s, 1e-6))
    X[:, 3], X[:, 4], X[:, 5] = np.log(w), np.log(h), np.log(w / h)
    X[:, 8] = np.minimum.reduce([b[:, 0], b[:, 1], 2048 - b[:, 2], 2048 - b[:, 3]]) / 2048.0
    hi = s >= 0.3
    X[:, 9] = np.log1p(hi.sum()); X[:, 10] = np.sort(s)[::-1][:50].mean()
    X[:, 11] = np.argsort(np.argsort(-sp, kind="mergesort"), kind="mergesort") / n
    iou = _box_iou(b, b); higher = s[None, :] > s[:, None]
    X[:, 12] = np.where(higher, iou, 0).max(1)
    c = np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
    dist = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    X[:, 13] = ((dist < 1.5 * np.maximum(w, h)[:, None]) & hi[None, :] & ~np.eye(n, dtype=bool)).sum(1)
    for i, rle in enumerate(rles):
        area = float(M.area(rle))
        X[i, 6] = np.log1p(area); X[i, 7] = area / (w[i] * h[i])
        if area < 1:
            continue
        rad = max(3.0, 0.25 * np.sqrt(area)); x0, y0, bw, bh = M.toBbox(rle); p = int(rad) + 2
        X0, Y0 = max(0, int(x0) - p), max(0, int(y0) - p); X1, Y1 = min(2048, int(x0 + bw) + p + 1), min(2048, int(y0 + bh) + p + 1)
        m = M.decode(rle)[Y0:Y1, X0:X1].astype(bool)
        sl = (slice(Y0, Y1), slice(X0, X1)); Lc, Ec = L[sl], exg[sl]
        X[i, 14:17] = img[sl][m].mean(0); X[i, 17] = Ec[m].mean(); X[i, 18] = Lc[m].std(); X[i, 19] = G[sl][m].mean()
        ring = (ndi.distance_transform_edt(~m) <= rad) & ~m
        if ring.sum() < 4:
            continue
        X[i, 20] = Lc[ring].mean() - Lc[m].mean(); X[i, 21] = Ec[ring].mean() - Ec[m].mean()
        yy, xx = np.nonzero(m); cy, cx = yy.mean(), xx.mean(); ry, rx = np.nonzero(ring)
        q = (ry >= cy).astype(int) * 2 + (rx >= cx).astype(int); qm = [Lc[ry[q == k], rx[q == k]].mean() for k in range(4) if (q == k).any()]
        X[i, 22] = min(qm) - max(qm); X[i, 23] = min(qm)
    if g is None:
        return t, X, None
    gb = np.asarray([[min(p[0::2]), min(p[1::2]), max(p[0::2]), max(p[1::2])] for p in g["trees"] if len(p) >= 6], np.float64).reshape(-1, 4)
    can = SC._union(SC._polys_to_rle(g["canopy"], 2048), 2048); crowd = SC._frac_inside(rles, can) >= 0.5
    bi = _box_iou(b, gb) if len(gb) else np.zeros((n, 0)); taken = np.zeros(len(gb), bool); y = np.zeros(n)
    for i in np.argsort(-sp, kind="mergesort"):
        if len(gb):
            cc = np.where(~taken & (bi[i] >= 0.5), bi[i], -1.0); j = int(cc.argmax())
            if cc[j] >= 0.5:
                taken[j] = True; y[i] = 1; continue
        if crowd[i]:
            y[i] = -1
    return t, X, y


def features(split, P, gt):
    fp = os.path.join(FC_, f"rerankR1_{split}.npz")
    if os.path.exists(fp):
        Z = np.load(fp, allow_pickle=False); return list(Z["tids"]), Z["tile"], Z["X"], (Z["y"] if "y" in Z else None)
    t0 = time.time(); tids = sorted(P); R = {}
    with mp.get_context("spawn").Pool(12) as pool:
        for k, (t, X, y) in enumerate(pool.imap_unordered(_feats, [(split, t, P[t], gt[t] if gt else None) for t in tids]), 1):
            R[t] = (X, y)
            if k % 20 == 0 or k == len(tids):
                print(f"  feats {split} {k}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    tile = np.concatenate([np.full(len(R[t][0]), i) for i, t in enumerate(tids)]); X = np.concatenate([R[t][0] for t in tids])
    d = {"tids": np.asarray(tids), "tile": tile, "X": X}
    if gt:
        d["y"] = np.concatenate([R[t][1] for t in tids])
    np.savez_compressed(fp + ".tmp.npz", **d); os.replace(fp + ".tmp.npz", fp)
    return tids, tile, X, d.get("y")


def fit(X, y, kind, seed=SEED):
    import torch
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-6
    net = (torch.nn.Linear(X.shape[1], 1) if kind == "logreg" else
           torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed); lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(30):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(1024):
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
    net.eval()

    def pred(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(1)).numpy().astype(np.float64)
    return pred


def oof(tids, tile, X, y, kind, cols, seed=SEED, tile_fold=None):
    tf = np.asarray([i % K for i in range(len(tids))]) if tile_fold is None else np.asarray(tile_fold)
    fold = tf[tile]; out = np.zeros(len(X)); preds = []
    for k in range(K):
        tr = (fold != k) & (y >= 0); f = fit(X[tr][:, cols], y[tr], kind, seed); te = fold == k
        out[te] = f(X[te][:, cols]); preds.append(f)
    return out, preds


def split_scores(tids, tile, v):
    return {t: v[tile == i].tolist() for i, t in enumerate(tids)}


def boot(P_new, P_base, gt, floor_new, floor_base):
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    from boxinst_commonality_tcd_04.dino_det.boot_vs_bt import per_tile, ap
    tids = sorted(gt); gd, t2i = SL.build_gt_crowd(gt, 2048); ids = np.array([t2i[t] for t in tids]); D = {}
    for k, (P, fl) in {"new": (P_new, floor_new), "base": (P_base, floor_base)}.items():
        D[k] = per_tile(P, gd, t2i, fl)
        for i in ids:
            if i not in D[k]:
                D[k][i] = (np.zeros(0), np.zeros(0, bool), sum(1 for a in gd["annotations"] if a["image_id"] == i and not a["iscrowd"]))
    rng = np.random.default_rng(SEED); d = []
    for _ in range(NBOOT):
        s = ids[rng.integers(0, len(ids), len(ids))]; d.append(ap(s, D["new"], SL.REC_THRS) - ap(s, D["base"], SL.REC_THRS))
    full = ap(ids, D["new"], SL.REC_THRS) - ap(ids, D["base"], SL.REC_THRS)
    return {"diff": round(full, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_le_0": round(float(np.mean(np.asarray(d) <= 0)), 4)}


def cv():
    from boxinst_commonality_tcd_04.box_supervised_baselines.tierb import score_lite as SL
    out = os.path.join(OUT, "rerankR1_cv.json"); assert not os.path.exists(out), out
    vg = json.load(open(VAL_GT)); P = json.load(open(os.path.join(FC_, f"{TAG}_val_mask.json"))); assert sorted(P) == sorted(vg)
    tids, tile, X, y = features("val", P, vg)
    print(f"[R1] val dets {len(y)}: pos {int((y == 1).sum())} neg {int((y == 0).sum())} ignored {int((y < 0).sum())}", flush=True)
    gd, t2i = SL.build_gt_crowd(vg, 2048)

    def val_ap(sc):
        pr = {t: {"boxes_2048": P[t]["boxes_2048"], "scores": sc[t], "masks_rle": P[t]["masks_rle"]} for t in tids}
        with contextlib.redirect_stdout(io.StringIO()):
            r = SL.score(pr, gd, t2i, masks=True, floor=0.0)
        return round(r["mask"]["AP50"], 4), pr
    base, Pb = val_ap({t: P[t]["s_prime"] for t in tids}); print(f"[R1] val s' AP50 {base} (README 0.7384)", flush=True)
    assert abs(base - 0.7384) < 5e-4, base
    rows, prs = {}, {}
    allc = list(range(NF))
    for name, kind, cols in (("logreg", "logreg", allc), ("mlp", "mlp", allc),
                             ("mlp_no_app", "mlp", SCORE + GEOM + CTX), ("mlp_no_ctx", "mlp", SCORE + GEOM + APP),
                             ("mlp_score_geom", "mlp", SCORE + GEOM)):
        v, _ = oof(tids, tile, X, y, kind, cols); a, prs[name] = val_ap(split_scores(tids, tile, v))
        rows[name] = {"AP50": a, "gain": round(a - base, 4)}; print(f"[R1] {name}: {rows[name]}", flush=True)
    chosen = "mlp" if rows["mlp"]["AP50"] > rows["logreg"]["AP50"] + 0.002 else "logreg"
    bt = boot(prs[chosen], Pb, vg, 0.0, 0.0)
    gate = rows[chosen]["gain"] >= 0.005 and bt["ci95"][0] > 0
    rec = {"base_sprime_AP50": base, "rows": rows, "chosen": chosen, "boot_vs_sprime": bt, "gate_pass": bool(gate),
           "n": {"pos": int((y == 1).sum()), "neg": int((y == 0).sum()), "ign": int((y < 0).sum())}, "features": NAMES,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


def test():
    from boxinst_commonality_tcd_04 import score_coco as SC
    cvr = json.load(open(os.path.join(OUT, "rerankR1_cv.json"))); assert cvr["gate_pass"], "gate failed: test must not be touched"
    out = os.path.join(OUT, "rerankR1_test.json"); assert not os.path.exists(out), out
    vg = json.load(open(VAL_GT)); Pv = json.load(open(os.path.join(FC_, f"{TAG}_val_mask.json")))
    tids_v, tile_v, Xv, yv = features("val", Pv, vg)
    _, fits = oof(tids_v, tile_v, Xv, yv, cvr["chosen"], list(range(NF)))
    St = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")))["preds"]
    Sd = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_detscore_r2048.json")))["preds"]  # detector s (the sprime file stores s' in "scores")
    assert all(Sd[t]["boxes_2048"] == St[t]["boxes_2048"] for t in St)
    M = {t: {**St[t], "scores": Sd[t]["scores"], "s_prime": St[t]["scores"]} for t in St}
    tids, tile, X, _ = features("test", M, None)
    v = np.mean([f(X) for f in fits], 0); sc = split_scores(tids, tile, v)
    preds = {t: {"boxes_2048": St[t]["boxes_2048"], "scores": sc[t], "masks_rle": St[t]["masks_rle"]} for t in tids}
    fp = os.path.join(OUT, f"preds_{TAG}_test_rerankR1_r2048.json")
    json.dump({"meta": {"mask_res": 2048, "model": f"D1 + LACE masker, R1 reranker ({cvr['chosen']}, val-trained, box labels)"}, "preds": preds},
              open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    os.chdir(REPO)
    c = SC.score_many([fp], TEST_GT, res=2048, max_dets=600, score_floor=0.0)[0]
    json.dump(c, open(fp.replace("preds_", "scored_"), "w"))
    cr = c["canopy_neutral_crowd"]
    bt = boot(preds, St, json.load(open(TEST_GT)), 0.0, 0.0)
    rec = {"chosen": cvr["chosen"], "mask": {k: cr["mask"][k] for k in ("AP50", "AP75", "AP50_95")},
           "box": {k: cr["box"][k] for k in ("AP50", "AP75")}, "sprime_ref": {"AP50": 0.7035, "AP75": 0.2615, "AP50_95": 0.3347},
           "boot_vs_sprime": bt, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd", choices=["cv", "test"]); a = ap_.parse_args()
    {"cv": cv, "test": test}[a.cmd]()

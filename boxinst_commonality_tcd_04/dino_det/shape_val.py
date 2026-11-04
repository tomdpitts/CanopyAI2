"""Pre-registration C1 (README.md, 2026-11-04, Tom: "can we introduce a notion of shape or connectedness to the hard cut?"). VAL ONLY.

Re-renders the frozen lace4px_simple masker (4 px posterior -> bilinear -> colour-guided filter r 12 / eps 1e-4 -> cut at
0.33 x box peak, clip to box) on the frozen D1 val boxes, with the cut replaced by shape / connectedness variants. The posterior
r, hence s', is unchanged. GA.EDGE_FILL is forced False so the only change vs the cached D1 val masks is the cut (the top/left
border fix is a separate, parallel line of work).
  base       : q >= tau (must reproduce feat_cache/dino_swinl_s0_val_mask.json exactly)
  conn       : the 8-connected component of the cut containing the box's q peak (largest component if the peak is not in it)
  fill       : base with holes filled                      conn_fill : conn, holes filled
  hyst       : hysteresis -- components of the cut that contain a seed pixel q >= 0.6 x peak;   hyst_fill: + holes filled
  qs1/2/3    : cut a Gaussian-smoothed q (sigma 1/2/3 px), tau = 0.33 x the smoothed peak
  eps1e-3/1e-2 : guided-filter eps 1e-3 / 1e-2 (follows only stronger colour edges)
  qs2_conn_fill : qs2, then conn, then fill
Masks -> feat_cache/shape_val/<tid>.json; scores (fasteval, s' ranking; R3 15-run OOF for the best) -> results/shape_val.json;
log native_raster/logs/shape_val_2026-11-04.log.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.shape_val
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import rerank as R
from boxinst_commonality_tcd_04.dino_det import fasteval as FE

VARS = ["base", "conn", "fill", "conn_fill", "hyst", "hyst_fill", "qs1", "qs2", "qs3", "eps1e-3", "eps1e-2", "qs2_conn_fill"]
TDIR = os.path.join(R.FC_, "shape_val"); OUT = os.path.join(R.OUT, "shape_val.json"); C, SEED_C = 0.33, 0.6


def _cut(q, inb, c=C):
    qb = q[inb]; tau = float(np.clip(c * (float(qb.max()) if qb.size else 0.0), 0.0, 1.0)); return (q >= tau) & inb


def _conn(m, q, inb):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n <= 1:
        return m
    pk = np.unravel_index(np.argmax(np.where(inb, q, -np.inf)), q.shape)
    k = lab[pk] if lab[pk] else 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    return lab == k


def _hyst(m, q, inb):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n <= 1:
        return m
    seed = _cut(q, inb, SEED_C); keep = np.unique(lab[seed & m]); keep = keep[keep > 0]
    return np.isin(lab, keep) if len(keep) else m


def _fill(m, inb):
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(m) & inb


def _tile(t):
    from scipy import ndimage as ndi
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    from boxinst_commonality_tcd_04.native_raster.hunt import render as RD
    fp = os.path.join(TDIR, t + ".json")
    if os.path.exists(fp):
        return t
    if "m4" not in BM._W:
        BM._init("val")
    GA, m4 = BM._W["GA"], BM._W["m4"]; GA.EDGE_FILL = False
    P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json")))[t]
    rgb = BM._rgb("val", t); zn4 = np.load(f"{BM.SPLIT['val']['z4']}/{t}.npy").astype(np.float32).reshape(-1, 128)
    out = {v: [] for v in VARS}
    for b in np.asarray(P["boxes_2048"], np.float32).reshape(-1, 4):
        clip, prob, _ = GA._prob(m4, zn4, 512, b, 0.3, 1.6); win = RD._win(clip); wy0, wy1, wx0, wx1 = win; by0, by1, bx0, bx1 = clip
        I, Pp = rgb[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1]
        inb = np.zeros(Pp.shape, bool); inb[by0 - wy0:by1 - wy0, bx0 - wx0:bx1 - wx0] = True
        q = GA.color_guided_filter(I, Pp, 12, 1e-4); base = _cut(q, inb); conn = _conn(base, q, inb)
        M = {"base": base, "conn": conn, "fill": _fill(base, inb), "conn_fill": _fill(conn, inb)}
        h = _hyst(base, q, inb); M["hyst"] = h; M["hyst_fill"] = _fill(h, inb)
        for s in (1, 2, 3):
            qs = ndi.gaussian_filter(q, s); M[f"qs{s}"] = _cut(qs, inb)
            if s == 2:
                M["qs2_conn_fill"] = _fill(_conn(M["qs2"], qs, inb), inb)
        for e in (1e-3, 1e-2):
            M[f"eps{e:.0e}".replace("e-0", "e-")] = _cut(GA.color_guided_filter(I, Pp, 12, e), inb)
        for v in VARS:
            full = np.zeros((2048, 2048), bool); full[wy0:wy1, wx0:wx1] = M[v]; out[v].append(GA._rle(full))
    same = sum(a["counts"] == b["counts"] for a, b in zip(out["base"], P["masks_rle"]))
    out["_parity"] = {"n": len(P["masks_rle"]), "base_identical": int(same)}
    json.dump(out, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def main():
    import multiprocessing as mp
    os.makedirs(TDIR, exist_ok=True)
    vg = json.load(open(R.VAL_GT)); P = json.load(open(os.path.join(R.FC_, f"{R.TAG}_val_mask.json"))); tids = sorted(P)
    t0 = time.time()
    with mp.get_context("spawn").Pool(12) as pool:
        for k, t in enumerate(pool.imap_unordered(_tile, tids), 1):
            if k % 12 == 0 or k == len(tids):
                print(f"  render {k}/{len(tids)} {time.time()-t0:.0f}s", flush=True)
    T = {t: json.load(open(os.path.join(TDIR, t + ".json"))) for t in tids}
    par = sum(T[t]["_parity"]["base_identical"] for t in tids), sum(T[t]["_parity"]["n"] for t in tids)
    print(f"[c1] parity base == cached D1 val masks: {par[0]}/{par[1]}", flush=True); assert par[0] == par[1], par
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}; rec["parity"] = list(par); rec.setdefault("sprime", {})
    sp = {t: P[t]["s_prime"] for t in tids}
    for v in VARS:
        if v in rec["sprime"]:
            continue
        cache = os.path.join(R.FC_, "fasteval_val.pkl") if v == "base" else os.path.join(R.FC_, f"fasteval_val_c1_{v}.pkl")
        E = FE.build({t: {"masks_rle": T[t][v]} for t in tids}, vg, cache); rec["sprime"][v] = FE.summary(FE.ap(E, sp))
        print(f"[c1] {v:14s} {rec['sprime'][v]}", flush=True); json.dump(rec, open(OUT, "w"), indent=1)
    b = rec["sprime"]["base"]; ok = [v for v in VARS if v != "base" and rec["sprime"][v]["AP50"] >= b["AP50"] - 0.001]
    best = max(ok, key=lambda v: rec["sprime"][v]["AP50_95"]) if ok else None; rec["best"] = best
    print(f"[c1] best by val AP50:95 (AP50 >= base - 0.001): {best} {rec['sprime'].get(best)}", flush=True)
    if best:
        from boxinst_commonality_tcd_04.dino_det import rerank3 as R3
        ch = json.load(open(os.path.join(R.OUT, "rerankR3_confirm.json")))["chosen"]; c = {**ch, "cols": tuple(ch["cols"])}
        R3._init(); W = R3._W; r3 = [R3.oof_fits(c, sd, f, W["X"], W["y"], W["soft"], W["tile"], W["F"])[0] for f, sd in R3.RUNS]
        rec["R3"] = {}
        for v in ("base", best):
            E = FE.build(None, None, os.path.join(R.FC_, "fasteval_val.pkl") if v == "base" else os.path.join(R.FC_, f"fasteval_val_c1_{v}.pkl"))
            a = np.asarray([FE.ap(E, x) for x in r3])
            rec["R3"][v] = {"AP50": round(float(a[:, 0].mean()), 4), "AP75": round(float(a[:, 5].mean()), 4), "AP50_95": round(float(a.mean(1).mean()), 4)}
            print(f"[c1] R3 + {v}: {rec['R3'][v]}", flush=True)
    json.dump(rec, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

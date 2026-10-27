"""tab:factors for lace4px_simple: drop each factor of s' = s * m * pbar, detector seed 0, with m = mean|2r-1| and
pbar = mean(r) from the 4 px cell posterior (m4, alpha 0.3, kappa 1.6 -- the posterior that makes the adopted mask).

Same protocol as ablation_r2048/conf_ablation_res_guided.py -> confidence/ablation.py "forms": test = canopy-neutral
legacy greedy AP over a cached mask-IoU matrix (439 tiles, the frozen 159,687 seed-0 boxes; masks = the adopted
render, hunt/test/preds_lace_s0_r2048_lace4px_simple_detscore.json); val = BOX AP50 on the 108 val tiles against GT
polygon extents (confidence/em_diag_val_s0.json boxes + raw scores).

Per-box statistics are recomputed locally for BOTH lattices (8 px: m, feat_4p_*; 4 px: m4, hunt_z4p_*), unrounded,
checkpointed per tile in feat_cache/hunt_factors4_<split>_s0/ and assembled into hunt/test/factors4_diag_<split>_s0.json.
GATE: the 8 px statistics with the deployed guided masks (ablation_r2048/_ap_cache_s0_r2048_guided.pkl) must reproduce
confidence_ablation_results_r2048_guided.json forms (s, s*m, s*pbar, s*m*pbar; test AP50, AP50:95, val box AP50) to 4 dp.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.factors4
"""
import json, multiprocessing as mp, os, pickle, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR)
FC = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "test", "factors4_lace4px_simple_s0.json")
FEAT = {"test": (f"{FC}/feat_4p_test_dl/feat_4p_test", f"{FC}/hunt_z4p_test"), "val": (f"{FC}/feat_4p_val", f"{FC}/hunt_z4p_val")}
_W = {}


def _init():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4 = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4.s, m4.origin = 4, 8.0
    _W.update(m=m, m4=m4, GA=GA)


def _job(args):
    split, t, boxes, d = args
    fp = os.path.join(d, t + ".json")
    if os.path.exists(fp):
        return t
    GA, rec = _W["GA"], {}
    f8, f4 = FEAT[split]
    zn = _W["m"].project(np.load(f"{f8}/{t}.npy").astype(np.float32)); g = int(round(np.sqrt(zn.shape[0])))
    zn4 = np.load(f"{f4}/{t}.npy").astype(np.float32).reshape(-1, 128)
    for lab, mk, z, gg in (("8", _W["m"], zn, g), ("4", _W["m4"], zn4, 512)):
        bm, pm, nc = [], [], []
        for b in np.asarray(boxes, np.float32).reshape(-1, 4):
            r = GA._prob(mk, z, gg, b, GA.ALPHA, GA.KAPPA)[2]
            bm.append(float(np.abs(2 * r - 1).mean())); pm.append(float(r.mean())); nc.append(int(r.size))
        rec.update({f"bimod{lab}": bm, f"pfg{lab}": pm, f"n_cells{lab}": nc})
    json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def diag(split, src):
    """src: {tile: {"boxes_2048", "scores"}} -> {tile: {..., bimod8, pfg8, bimod4, pfg4}} (checkpointed)."""
    out_fp = os.path.join(HERE, "test", f"factors4_diag_{split}_s0.json")
    if os.path.exists(out_fp):
        return json.load(open(out_fp))
    d = f"{FC}/hunt_factors4_{split}_s0"; os.makedirs(d, exist_ok=True)
    tids = [t for t in sorted(src) if src[t]["scores"]]
    todo = [(split, t, src[t]["boxes_2048"], d) for t in tids if not os.path.exists(os.path.join(d, t + ".json"))]
    print(f"[factors4 {split}] {len(todo)}/{len(tids)} tiles to compute", flush=True); t0 = time.time()
    if todo:
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 8)), initializer=_init) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, todo), 1):
                if k % 20 == 0 or k == len(todo):
                    print(f"  {split} {k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    D = {t: {"boxes_2048": src[t]["boxes_2048"], "scores": src[t]["scores"], **json.load(open(os.path.join(d, t + ".json")))} for t in tids}
    json.dump(D, open(out_fp, "w")); return D


def forms(lab):
    """s, s*m, s*pbar, s*m*pbar as functions of a per-tile stats dict, statistics of lattice `lab`."""
    ls = lambda r: np.log(np.asarray(r["scores"], np.float64))
    lb = lambda r: np.log(np.asarray(r[f"bimod{lab}"], np.float64))
    lp = lambda r: np.log(np.asarray(r[f"pfg{lab}"], np.float64))
    return {"s": ls, "s*m": lambda r: ls(r) + lb(r), "s*pbar": lambda r: ls(r) + lp(r), "s*m*pbar": lambda r: ls(r) + lb(r) + lp(r)}


def test_ap(cache, Dt, fn, thr):
    from boxinst_commonality_tcd_04 import evaluate as E
    Iou, S, G = [], [], []
    for t in sorted(cache["D"]):
        c = cache["D"][t]; s = fn(Dt[t]); o = np.argsort(-s, kind="mergesort")
        Iou.append(c["iou"][o]); S.append(s[o]); G.append(c["ign"][o])
    return E._greedy_ap(Iou, S, G, cache["n_gt"], thr)


def val_box_ap(Dv, vg, fn, tiles):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.confidence import ablation as A
    Iou, S, G, n_gt = [], [], [], 0
    for t in sorted(tiles):
        g = A.gt_boxes(vg[t]["trees"]); n_gt += len(g)
        if t not in Dv:
            continue
        r = Dv[t]; s = fn(r); o = np.argsort(-s, kind="mergesort")
        iou = A.box_iou(np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4), g)
        Iou.append(iou[o]); S.append(s[o]); G.append(np.zeros(len(s), bool))
    return E._greedy_ap(Iou, S, G, n_gt, 0.5)


def table(cache, Dt, Dv, vg, lab, vt):
    out = {}
    for nm, fn in forms(lab).items():
        out[nm] = {"cn_ap50": round(test_ap(cache, Dt, fn, 0.5), 4),
                   "ap50_95": round(float(np.nanmean([test_ap(cache, Dt, fn, t) for t in np.linspace(0.5, 0.95, 10)])), 4),
                   "val_box_ap50": round(val_box_ap(Dv, vg, fn, vt), 4)}
        print(f"  [{lab} px] {nm:9} AP50 {out[nm]['cn_ap50']:.4f} AP50:95 {out[nm]['ap50_95']:.4f} val box AP50 {out[nm]['val_box_ap50']:.4f}", flush=True)
    return out


def main():
    assert not os.path.exists(OUT), OUT
    from boxinst_commonality_tcd_04.confidence import ablation as A
    from boxinst_commonality_tcd_04.native_raster.ablation_r2048.conf_ablation_res_guided import build_cache_res
    det = json.load(open(os.path.join(NR, "preds_lace_s0_r2048_cgf_r12_tau035_detscore.json")))["preds"]
    vd = json.load(open(os.path.join(PKG, "confidence", "em_diag_val_s0.json")))
    vg = json.load(open(os.path.join(A.P4, "val_gt.json")))
    Dt = diag("test", det); Dv = diag("val", {t: {"boxes_2048": r["boxes_2048"], "scores": r["scores"]} for t, r in vd.items()})
    assert sum(len(r["scores"]) for r in Dt.values()) == 159687
    # GATE: 8 px statistics + deployed guided masks reproduce the published table
    ref = json.load(open(os.path.join(PKG, "ablation", "results_r2048", "confidence_ablation_results_r2048_guided.json")))["forms"]
    refk = {"s": "s  (baseline)", "s*m": "s * bimod", "s*pbar": "s * pfg_mean", "s*m*pbar": "s * bimod * pfg_mean"}
    old = pickle.load(open(os.path.join(NR, "ablation_r2048", "_ap_cache_s0_r2048_guided.pkl"), "rb"))
    print("[factors4] GATE: 8 px statistics, deployed masks", flush=True)
    g8 = table(old, Dt, Dv, vg, "8", list(vd))
    bad = [(k, f, g8[k][f], ref[refk[k]][f]) for k in g8 for f in g8[k] if abs(g8[k][f] - ref[refk[k]][f]) > 1e-4 + 1e-9]
    print(f"[factors4] GATE {'PASS' if not bad else 'FAIL'} {bad}", flush=True)
    assert not bad, "GATE FAILED"
    # adopted: 4 px statistics + lace4px_simple masks
    cfp = f"{FC}/hunt_factors4_ap_cache_s0_r2048_lace4px_simple.pkl"
    preds = os.path.join(HERE, "test", "preds_lace_s0_r2048_lace4px_simple_detscore.json")
    new = pickle.load(open(cfp, "rb")) if os.path.exists(cfp) else (lambda D: dict(D=D[0], n_gt=D[1]))(build_cache_res(2048, preds, cfp))
    print("[factors4] ADOPTED: 4 px statistics, lace4px_simple masks", flush=True)
    t4 = table(new, Dt, Dv, vg, "4", list(vd))
    json.dump({"gate_8px_deployed": g8, "reference": {k: ref[refk[k]] for k in refk}, "lace4px_simple_4px": t4,
               "masks": preds, "n_boxes": 159687}, open(OUT, "w"), indent=1)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()

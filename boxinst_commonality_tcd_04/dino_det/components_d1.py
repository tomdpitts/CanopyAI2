"""Pre-registration ABL-D1 item 4 (README.md, 2026-11-10): tab:components (M-step repulsion x E-step recentring) on the deployed
4 px path, val 108, ONE look. Cells (npz, inference contrast flag); primary "fit+infer" applies the recentring flag at fit (npz)
and inference; secondary "fit-only" always recentres at inference, as every published components eval did.
Box sources: GT boxes (all val crowns) and D1 val boxes (box IoU >= 0.5 greedy by detector s, evaluate.match_tile).
Metric: mean per-crown mask IoU at 2048 (post4 mask vs rasterised GT polygon). Per-tile checkpoints feat_cache/abl_comp_val/.
-> results/abl_d1/components.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.components_d1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import abl_lib as L

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "abl_d1"); TDIR = f"{L.FC}/abl_comp_val"
COL = os.path.join(PKG, "modal_tcd_multiseed", "phase4_collapse", "results", "npz")
NPZ = {"b0_abs": os.path.join(PKG, "modal_tcd_multiseed", "phase4", "em_model_4p_b0_fix.npz"),
       "b0_rec": os.path.join(COL, "k16_s0_b0_c1.npz"), "b05_abs": os.path.join(COL, "k16_s0_b0.5_c0.npz"), "b05_rec": L.DEPLOYED}
# (cell name, npz key, inference contrast)
CELLS = [("b0_abs_infer0", "b0_abs", False), ("b0_abs_infer1", "b0_abs", True), ("b0_rec", "b0_rec", True),
         ("b05_abs_infer0", "b05_abs", False), ("b05_abs_infer1", "b05_abs", True), ("b05_rec", "b05_rec", True)]
DEFS = {"fit+infer": {"b0/absolute": "b0_abs_infer0", "b0/recentred": "b0_rec", "b05/absolute": "b05_abs_infer0", "b05/recentred": "b05_rec"},
        "fit-only": {"b0/absolute": "b0_abs_infer1", "b0/recentred": "b0_rec", "b05/absolute": "b05_abs_infer1", "b05/recentred": "b05_rec"}}


def _tile(args):
    t, polys, dbox, dsc = args
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster.guided_ablate_modal import box_iou_mat
    gb = np.array([[*np.asarray(p, float).reshape(-1, 2).min(0), *np.asarray(p, float).reshape(-1, 2).max(0)] for p in polys],
                  np.float32).reshape(-1, 4)
    db = np.asarray(dbox, np.float32).reshape(-1, 4); sc = np.asarray(dsc, np.float32)
    pairs = []
    if len(sc) and len(gb):
        order, _, _, _, gidx = E.match_tile(box_iou_mat(db, gb), sc, np.zeros(len(sc), bool), 0.5)
        pairs = [(int(order[rk]), int(j)) for rk, j in enumerate(gidx) if j >= 0]
    gm = E.raster(polys, res=2048, scale=1.0)
    out = {"tid": t, "n_gt": len(gb), "pred_pairs": pairs, "gt": {}, "pred": {}}
    for name, key, con in CELLS:
        Mg, _ = L.render("val", t, gb, None, npz=NPZ[key], contrast=con, as_rle=False)
        out["gt"][name] = [float((m & g).sum() / max((m | g).sum(), 1)) for m, g in zip(Mg, gm)]
        Mp, _ = L.render("val", t, db[[i for i, _ in pairs]], None, npz=NPZ[key], contrast=con, as_rle=False)
        out["pred"][name] = [float((m & gm[j]).sum() / max((m | gm[j]).sum(), 1)) for m, (_, j) in zip(Mp, pairs)]
    return t, out


def main():
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    os.makedirs(OUT, exist_ok=True); os.makedirs(TDIR, exist_ok=True); fout = os.path.join(OUT, "components.json")
    assert not os.path.exists(fout), f"REFUSING to overwrite {fout}"
    vg = json.load(open(BM.SPLIT["val"]["gt"])); Dv = json.load(open(f"{L.FC}/dino_swinl_s0_val_mask.json"))
    tids = sorted(vg); assert sorted(Dv) == tids and len(tids) == 108
    # whitening identity (zn4 = deployed projection is reused for every cell)
    ref = np.load(L.DEPLOYED)
    for k, f in NPZ.items():
        d = np.load(f); assert all(np.array_equal(d[x], ref[x]) for x in ("mu", "U", "scale")), f"whitening differs {k}"
    # gate: deployed cell on D1 val boxes == cached edge-fixed val masks (3 tiles)
    for t in tids[:3]:
        M, _ = L.render("val", t, Dv[t]["boxes_2048"], Dv[t]["scores"])
        assert [m["counts"] for m in M] == [m["counts"] for m in json.load(open(f"{L.FC}/edgefix_val/{t}.json"))["masks_rle"]], f"gate {t}"
    print("[comp] gates PASS: whitening identical (4 npz), deployed render == edgefix_val (3 tiles)", flush=True)
    todo = [t for t in tids if vg[t]["trees"] and not os.path.exists(f"{TDIR}/{t}.json")]; t0 = time.time()
    print(f"[comp] {len(todo)} tiles to run ({sum(1 for t in tids if vg[t]['trees'])} with crowns)", flush=True)
    with mp.get_context("spawn").Pool(12) as pool:
        for n, (t, r) in enumerate(pool.imap_unordered(_tile, [(t, vg[t]["trees"], Dv[t]["boxes_2048"], Dv[t]["scores"]) for t in todo]), 1):
            json.dump(r, open(f"{TDIR}/{t}.json.tmp", "w")); os.replace(f"{TDIR}/{t}.json.tmp", f"{TDIR}/{t}.json")
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    R = [json.load(open(f"{TDIR}/{t}.json")) for t in tids if vg[t]["trees"]]
    mean = {src: {c: round(float(np.mean(np.concatenate([r[src][c] for r in R]))), 4) for c, _, _ in CELLS} for src in ("gt", "pred")}
    n = {src: int(sum(len(r[src][CELLS[0][0]]) for r in R)) for src in ("gt", "pred")}
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "n_tiles": len(R), "n_crowns": n, "mean_crown_iou": mean, "tables": {}}
    for dname, d in DEFS.items():
        tab = {}
        for src in ("gt", "pred"):
            v = {k: mean[src][c] for k, c in d.items()}; base = v["b0/absolute"]
            rep, cen, both = v["b05/absolute"] - base, v["b0/recentred"] - base, v["b05/recentred"] - base
            tab[src] = {"cells": v, "repulsion_alone": round(rep, 4), "recentring_alone": round(cen, 4), "both": round(both, 4),
                        "interaction": round(both - rep - cen, 4), "additive_prediction": round(rep + cen, 4)}
        rec["tables"][dname] = tab
    rec["note"] = "ONE look per pre-registration ABL-D1 item 4; primary definition fit+infer; single detector run, EM seed 0"
    json.dump(rec, open(fout, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()

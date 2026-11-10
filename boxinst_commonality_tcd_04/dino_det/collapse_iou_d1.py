"""Pre-registration ABL-D1 item 5 (README.md, 2026-11-10): tab:collapse Crown IoU column on the deployed 4 px path, val 108,
GT boxes only, recentring on (contrast True), alpha 0.3 / kappa 16, EDGE_FILL. ONE look. Gate: k16 b0 / b0.5 == components.json GT.
Per-tile checkpoints feat_cache/abl_collapse_val/. -> results/abl_d1/collapse_iou.json
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.collapse_iou_d1
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

from boxinst_commonality_tcd_04.dino_det import abl_lib as L

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "abl_d1"); TDIR = f"{L.FC}/abl_collapse_val"
COL = os.path.join(PKG, "modal_tcd_multiseed", "phase4_collapse", "results", "npz")
NPZ = {"k16_b0": f"{COL}/k16_s0_b0_c1.npz", "k16_b0.1": f"{COL}/k16_s0_b0.1_c1.npz", "k16_b0.25": f"{COL}/k16_s0_b0.25_c1.npz",
       "k16_b0.5": L.DEPLOYED, "k2_b0": f"{COL}/k2_s0_b0_c1.npz", "k2_b0.5": f"{COL}/k2_s0_b0.5.npz"}


def _tile(args):
    t, polys = args
    from boxinst_commonality_tcd_04 import evaluate as E
    gb = np.array([[*np.asarray(p, float).reshape(-1, 2).min(0), *np.asarray(p, float).reshape(-1, 2).max(0)] for p in polys],
                  np.float32).reshape(-1, 4)
    gm = E.raster(polys, res=2048, scale=1.0); out = {"tid": t, "gt": {}}
    for name, f in NPZ.items():
        Mg, _ = L.render("val", t, gb, None, npz=f, contrast=True, as_rle=False)
        out["gt"][name] = [float((m & g).sum() / max((m | g).sum(), 1)) for m, g in zip(Mg, gm)]
    return t, out


def main():
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    os.makedirs(TDIR, exist_ok=True); fout = os.path.join(OUT, "collapse_iou.json")
    assert not os.path.exists(fout), f"REFUSING to overwrite {fout}"
    ref = np.load(L.DEPLOYED)
    for k, f in NPZ.items():
        d = np.load(f); assert all(np.array_equal(d[x], ref[x]) for x in ("mu", "U", "scale")), f"whitening differs {k}"
    vg = json.load(open(BM.SPLIT["val"]["gt"])); tids = sorted(vg); assert len(tids) == 108
    todo = [t for t in tids if vg[t]["trees"] and not os.path.exists(f"{TDIR}/{t}.json")]; t0 = time.time()
    print(f"[collapse] whitening PASS; {len(todo)} tiles to run", flush=True)
    with mp.get_context("spawn").Pool(12) as pool:
        for n, (t, r) in enumerate(pool.imap_unordered(_tile, [(t, vg[t]["trees"]) for t in todo]), 1):
            json.dump(r, open(f"{TDIR}/{t}.json.tmp", "w")); os.replace(f"{TDIR}/{t}.json.tmp", f"{TDIR}/{t}.json")
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    R = [json.load(open(f"{TDIR}/{t}.json")) for t in tids if vg[t]["trees"]]
    mean = {k: round(float(np.mean(np.concatenate([r["gt"][k] for r in R]))), 4) for k in NPZ}
    n = int(sum(len(r["gt"]["k16_b0"]) for r in R))
    C = json.load(open(os.path.join(OUT, "components.json")))["mean_crown_iou"]["gt"]
    assert mean["k16_b0"] == C["b0_rec"] and mean["k16_b0.5"] == C["b05_rec"], f"gate {mean} vs {C}"
    print(f"[collapse] gate PASS: k16 b0 {mean['k16_b0']} / b0.5 {mean['k16_b0.5']} == components.json", flush=True)
    k16 = [mean[k] for k in ("k16_b0", "k16_b0.1", "k16_b0.25", "k16_b0.5")]
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "n_tiles": len(R), "n_crowns": n, "crown_iou_gt": mean,
           "k16_span": round(max(k16) - min(k16), 4), "k16_b05_lowest": mean["k16_b0.5"] == min(k16),
           "note": "ONE look per pre-registration ABL-D1 item 5; EM seed 0; recentring on; deployed 4 px path"}
    json.dump(rec, open(fout, "w"), indent=1); print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()

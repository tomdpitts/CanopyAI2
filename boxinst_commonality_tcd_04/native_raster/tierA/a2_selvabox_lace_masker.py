"""Tier A2 (PLAN.md §13, approved 2026-10-14): SelvaBox detector boxes + LACE masker.

Boxes + detector scores s: seam-fixed SelvaBox selection (`seamfix/preds_selvabox_seamfix_on_r2048.json`,
307,500 boxes). Masker: deployed `em_model_4p_fix.npz` (local copy of tcd04-phase4-vol:out/, sha1
2f0eaa3f...), alpha 0.3, kappa 1.6, colour-guided render r 12 / eps 1e-4 / tau 0.35, box clip -- the
render functions are IMPORTED from guided_ablate_modal.py (the code that made the paper's LACE masks),
not re-implemented. Ranking s' = s * |2r-1|.mean() * r.mean() (LACE's product, s = SelvaBox score).
Features: the Modal-cached 4-phase test features (downloaded; NEVER re-extracted locally -- see
project_environment memory). No re-tuning of any LACE setting.

GATE before the run: on LACE s0's OWN boxes for the first 3 non-empty tiles, the local path must
reproduce the published product scores (max |diff| < 5e-5) and masks (mean IoU >= 0.999) of
preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json. Writes tierA/a2_gate.json.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a2_selvabox_lace_masker \
      --feat_dir /Users/tompitts/dphil/feat_cache/feat_4p_test_dl/feat_4p_test [--gate_only]
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
NPZ = "/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"
RGB_DIR = os.path.join(REPO, "data", "tcd", "test")
_W = {}


def _init(feat_dir):
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    assert (GA.GF_R, GA.GF_EPS, GA.GF_TAU) == (12, 1e-4, 0.35)
    _W.update(GA=GA, masker=m, feat_dir=feat_dir,
              man=json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"])


def render(tid, boxes, scores):
    GA, m = _W["GA"], _W["masker"]
    out = {"boxes_2048": boxes, "scores_det": scores, "scores": [], "masks_rle": [], "device": "cpu (local)"}
    if not boxes:
        return out
    rgb = GA._rgb(RGB_DIR, tid, _W["man"])
    feat = np.load(os.path.join(_W["feat_dir"], f"{tid}.npy")).astype(np.float32)
    g = feat.shape[-1]; zn = m.project(feat); del feat
    for b, s in zip(np.asarray(boxes, np.float32).reshape(-1, 4), scores):
        clip, prob, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        out["scores"].append(round(float(s * np.abs(2 * r - 1).mean() * r.mean()), 8))
        out["masks_rle"].append(GA._rle(GA._guided(rgb, clip, prob)))
    return out


def _job(args):
    tid, boxes, scores, out_dir = args
    rec = render(tid, boxes, scores)
    tmp = os.path.join(out_dir, tid + ".json.tmp"); json.dump(rec, open(tmp, "w"))
    os.replace(tmp, os.path.join(out_dir, tid + ".json"))
    return tid, len(boxes)


def gate(feat_dir):
    out = os.path.join(HERE, "a2_gate.json"); assert not os.path.exists(out), out
    _init(feat_dir)
    det = json.load(open(os.path.join(NR, "guided_ablate_inputs", "detscore_s0.json")))
    pub = json.load(open(os.path.join(NR, "preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json")))["preds"]
    tiles = [t for t in sorted(det) if det[t]["scores"]][:3]
    worst, ious, n_eq = 0.0, [], 0
    for t in tiles:
        assert det[t]["boxes_2048"] == pub[t]["boxes_2048"], t
        r = render(t, det[t]["boxes_2048"], det[t]["scores"])
        worst = max(worst, float(np.max(np.abs(np.asarray(r["scores"]) - np.asarray(pub[t]["scores"])))))
        for a, b in zip(r["masks_rle"], pub[t]["masks_rle"]):
            n_eq += a["counts"] == b["counts"]
            A = mu.decode({"size": a["size"], "counts": a["counts"].encode()}).astype(bool)
            Bm = mu.decode({"size": b["size"], "counts": b["counts"].encode()}).astype(bool)
            u = (A | Bm).sum()
            ious.append(1.0 if u == 0 else float((A & Bm).sum() / u))   # two empty masks agree (v0 bug: scored 0)
    res = {"tiles": tiles, "n_boxes": len(ious), "product_max_abs_diff": worst, "masks_identical": n_eq,
           "mask_mean_iou": float(np.mean(ious)), "mask_min_iou": float(np.min(ious)),
           "pass": worst < 5e-5 and float(np.mean(ious)) >= 0.999}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1), flush=True)
    return res["pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True); ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    gp = os.path.join(HERE, "a2_gate.json")
    ok = json.load(open(gp))["pass"] if os.path.exists(gp) else gate(a.feat_dir)
    assert ok, "A2 GATE FAILED -- not running"
    if a.gate_only:
        return
    sel = json.load(open(os.path.join(NR, "seamfix", "preds_selvabox_seamfix_on_r2048.json")))["preds"]
    out_dir = os.path.join(HERE, "a2_tiles"); os.makedirs(out_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(out_dir) if f.endswith(".json")}
    jobs = [(t, sel[t]["boxes_2048"], sel[t]["scores"], out_dir) for t in sorted(sel) if t not in done]
    print(f"[A2] {len(sel)} tiles, {len(done)} done, {len(jobs)} to do", flush=True)
    import platform, scipy, PIL
    envp = os.path.join(out_dir, "_env.json")
    if not os.path.exists(envp):
        json.dump({"device": "cpu (local)", "machine": platform.machine(), "platform": platform.platform(),
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "pillow": PIL.__version__, "features": a.feat_dir + " (Modal-cached, tcd04-phase4-vol:feat_4p_test)",
                   "masker_npz": NPZ, "started": time.strftime("%Y-%m-%d %H:%M")}, open(envp, "w"), indent=1)
    t0 = time.time()
    with Pool(a.workers, _init, (a.feat_dir,)) as pool:
        for k, (t, n) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            el = time.time() - t0
            print(f"  {len(done)+k+1}/{len(sel)} {t} {n} boxes  {el:.0f}s  eta {el/(k+1)*(len(jobs)-k-1):.0f}s", flush=True)
    print("[A2] done", flush=True)


if __name__ == "__main__":
    main()

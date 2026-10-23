"""Stage 2: the full render swap on the 439 TEST tiles, 3 seeds. LACE's published boxes + product scores verbatim
(preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json); masks re-rendered with the frozen boxfix_v1 render
(FROZEN_val_2026-10-15.json) from the same posterior (em_model_4p_fix.npz 0.3/1.6, Modal-cached test features).
Render functions imported from guided_ablate_modal.py / render_lib.py.

GATE before any render: on s0's first 3 non-empty tiles, the DEPLOYED variant through render_lib.render_full must
reproduce the published product scores (max |diff| < 5e-5) and masks (mean IoU >= 0.999) -> boxfix/ap_test_gate.json.
Per-tile JSON -> feat_cache/boxfix_ap_test_s{s}/ (resume). Assembled -> boxfix/preds_lace_s{s}_r2048_boxfix_v1.json
(scores = product). Scoring: score_test.py.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.boxfix.ap_test [--seeds 0 1 2]
"""
import argparse, json, os, time
from multiprocessing import Pool
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE); PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
FEAT = f"{FC}/feat_4p_test_dl/feat_4p_test"
NPZ = f"{FC}/em_model_4p_fix.npz"
RGB_DIR = os.path.join(REPO, "data", "tcd", "test")
FROZEN = json.load(open(os.path.join(HERE, "FROZEN_val_2026-10-15.json")))["variant"]
_W = {}


def pub_path(s):
    return os.path.join(NR, f"preds_lace_s{s}_r2048_cgf_r12_tau035_guidedrun.json")


def _init():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    assert (GA.GF_R, GA.GF_EPS, GA.GF_TAU) == (12, 1e-4, 0.35)
    _W.update(GA=GA, masker=m, man=json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"])


def render(tid, boxes, variant):
    from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R
    GA, m = _W["GA"], _W["masker"]
    out = {"tile": tid, "boxes_2048": boxes, "product": [], "masks_rle": []}
    if not boxes:
        return out
    rgbf = GA._rgb(RGB_DIR, tid, _W["man"])
    feat = np.load(os.path.join(FEAT, f"{tid}.npy")).astype(np.float32)
    g = feat.shape[-1]; zn = m.project(feat); del feat
    for b in np.asarray(boxes, np.float32).reshape(-1, 4):
        clip, prob, r = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        out["product"].append(float(np.abs(2 * r - 1).mean() * r.mean()))       # times s -> product
        out["masks_rle"].append(GA._rle(R.render_full(rgbf, prob, clip, b, variant)))
    return out


def gate():
    fp = os.path.join(HERE, "ap_test_gate.json")
    if os.path.exists(fp):
        return json.load(open(fp))["pass"]
    _init()
    det = json.load(open(os.path.join(NR, "guided_ablate_inputs", "detscore_s0.json")))
    pub = json.load(open(pub_path(0)))["preds"]
    tiles = [t for t in sorted(det) if det[t]["scores"]][:3]
    worst, ious, n_eq = 0.0, [], 0
    for t in tiles:
        assert det[t]["boxes_2048"] == pub[t]["boxes_2048"], t
        r = render(t, det[t]["boxes_2048"], {})
        prod = np.asarray(det[t]["scores"]) * np.asarray(r["product"])
        worst = max(worst, float(np.max(np.abs(prod - np.asarray(pub[t]["scores"])))))
        for a, b in zip(r["masks_rle"], pub[t]["masks_rle"]):
            n_eq += a["counts"] == b["counts"]
            A = mu.decode({"size": a["size"], "counts": a["counts"].encode()}).astype(bool)
            B = mu.decode({"size": b["size"], "counts": b["counts"].encode()}).astype(bool)
            u = (A | B).sum(); ious.append(1.0 if u == 0 else float((A & B).sum() / u))
    res = {"tiles": tiles, "n_boxes": len(ious), "product_max_abs_diff": worst, "masks_identical": n_eq,
           "mask_mean_iou": float(np.mean(ious)), "mask_min_iou": float(np.min(ious)),
           "pass": worst < 5e-5 and float(np.mean(ious)) >= 0.999}
    json.dump(res, open(fp, "w"), indent=1); print("[gate]", json.dumps(res), flush=True)
    return res["pass"]


def _job(args):
    tid, boxes, out_dir = args
    rec = render(tid, boxes, FROZEN)
    tmp = os.path.join(out_dir, tid + ".json.tmp"); json.dump(rec, open(tmp, "w"))
    os.replace(tmp, os.path.join(out_dir, tid + ".json"))
    return tid, len(boxes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    assert gate(), "GATE FAILED -- not running"
    for s in a.seeds:
        outp = os.path.join(HERE, f"preds_lace_s{s}_r2048_boxfix_v1.json")
        if os.path.exists(outp):
            print(f"[s{s}] {outp} exists, skip", flush=True); continue
        pub = json.load(open(pub_path(s)))
        det = json.load(open(os.path.join(NR, "guided_ablate_inputs", f"detscore_s{s}.json")))
        out_dir = f"{FC}/boxfix_ap_test_s{s}"; os.makedirs(out_dir, exist_ok=True)
        done = {f[:-5] for f in os.listdir(out_dir) if f.endswith(".json")}
        tids = sorted(pub["preds"])
        jobs = [(t, pub["preds"][t]["boxes_2048"], out_dir) for t in tids if t not in done]
        print(f"[s{s}] {len(tids)} tiles, {len(done)} done, {len(jobs)} to do", flush=True)
        t0 = time.time()
        with Pool(a.workers, _init) as pool:
            for k, (t, n) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
                el = time.time() - t0
                print(f"  s{s} {len(done)+k+1}/{len(tids)} {t} {n} boxes {el:.0f}s eta {el/(k+1)*(len(jobs)-k-1):.0f}s", flush=True)
        preds, worst = {}, 0.0
        for t in tids:
            r = json.load(open(os.path.join(out_dir, t + ".json"))); P = pub["preds"][t]
            assert r["boxes_2048"] == P["boxes_2048"] == det[t]["boxes_2048"], t
            prod = np.asarray(det[t]["scores"]) * np.asarray(r["product"]) if r["product"] else np.zeros(0)
            if len(prod):
                worst = max(worst, float(np.abs(prod - np.asarray(P["scores"])).max()))
            preds[t] = {"boxes_2048": P["boxes_2048"], "scores": P["scores"], "masks_rle": r["masks_rle"]}
        assert worst < 5e-5, f"s{s}: recomputed product differs from published by {worst}"
        json.dump({"meta": dict(pub["meta"], model=f"LACE s{s}, boxfix_v1 render", render="boxfix_v1", variant=FROZEN,
                                scores="published product verbatim (recomputed, max |diff| %.1e)" % worst,
                                masks="re-rendered locally CPU, render_lib.render_full"), "preds": preds}, open(outp, "w"))
        print(f"[s{s}] wrote {outp} (product check max |diff| {worst:.1e})", flush=True)
    print("[ap_test] done", flush=True)


if __name__ == "__main__":
    main()

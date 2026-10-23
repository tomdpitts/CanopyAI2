"""Per-crown window cache for the box->mask render sweep (GT boxes, tab:gtbox protocol).

For every GT crown of every tile with crowns: the LACE posterior `prob` (evaluate/_prob, as
guided_ablate_modal.py) cropped to the deployed render window (box clip + MARGIN 16), for a few
masker knob settings (alpha, kappa); the tile RGB in that window; the GT mask in that window and
the GT pixels outside it (so window IoU == full-raster IoU exactly); box, clip, valid.
Render variants are then swept on windows without recomputing posteriors.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.boxfix.cache_windows --split val
"""
import argparse, json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"
NPZ = f"{FC}/em_model_4p_fix.npz"
SPLITS = {
    "val": dict(feat=f"{FC}/feat_4p_val", rgb=f"{REPO}/data/tcd/train",
                gt=f"{PKG}/modal_tcd_multiseed/phase4/val_gt.json", out=f"{FC}/boxfix_val"),
    "test": dict(feat=f"{FC}/feat_4p_test_dl/feat_4p_test", rgb=f"{REPO}/data/tcd/test",
                 gt=f"{PKG}/test_gt.json", out=f"{FC}/boxfix_test"),
}
KNOBS = [(None, None), (0.0, 1.6), (0.15, 1.6), (0.3, 1.0)]   # first = npz knobs (0.3, 1.6)
MARGIN = 16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=SPLITS)
    a = ap.parse_args(); S = SPLITS[a.split]
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    assert GA.MARGIN == MARGIN
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    gt = json.load(open(S["gt"]))
    tids = [t for t in sorted(gt) if gt[t]["trees"] and os.path.exists(os.path.join(S["feat"], t + ".npy"))]
    print(f"[cache {a.split}] {len(tids)} tiles with crowns", flush=True)
    os.makedirs(S["out"], exist_ok=True)
    t0 = time.time()
    for n, t in enumerate(tids):
        fp = os.path.join(S["out"], t + ".npz")
        if os.path.exists(fp):
            continue
        polys = gt[t]["trees"]
        B = G.gt_boxes_2048(polys)
        g_rles, valid = G.gt_rles_and_valid(polys, res=2048)
        rgb = np.asarray(Image.open(os.path.join(S["rgb"], t + ".tif")).convert("RGB"))
        assert rgb.shape == (2048, 2048, 3)
        feat = np.load(os.path.join(S["feat"], t + ".npy")).astype(np.float32)
        gg = feat.shape[-1]; zn = m.project(feat); del feat
        rec = {"box": B, "valid": valid, "clip": [], "win": [], "gt_out": [], "gt_area": []}
        probs, rgbs, gts = {k: [] for k in range(len(KNOBS))}, [], []
        for i, b in enumerate(B):
            for k, (al, ka) in enumerate(KNOBS):
                clip, prob, _ = GA._prob(m, zn, gg, b, al, ka)
                if k == 0:
                    by0, by1, bx0, bx1 = clip
                    wy0, wy1 = max(0, by0 - MARGIN), min(2048, by1 + MARGIN)
                    wx0, wx1 = max(0, bx0 - MARGIN), min(2048, bx1 + MARGIN)
                    gm = mu.decode(g_rles[i]).astype(bool)
                    gw = gm[wy0:wy1, wx0:wx1]
                    rec["clip"].append(clip); rec["win"].append((wy0, wy1, wx0, wx1))
                    rec["gt_area"].append(int(gm.sum())); rec["gt_out"].append(int(gm.sum() - gw.sum()))
                    rgbs.append(rgb[wy0:wy1, wx0:wx1].copy()); gts.append(np.packbits(gw.ravel()))
                probs[k].append(prob[wy0:wy1, wx0:wx1].astype(np.float32).copy())
        arrs = {"box": B, "valid": valid, "clip": np.asarray(rec["clip"], np.int32),
                "win": np.asarray(rec["win"], np.int32), "gt_area": np.asarray(rec["gt_area"]),
                "gt_out": np.asarray(rec["gt_out"]), "knobs": np.asarray([[np.nan if v is None else v for v in kk] for kk in KNOBS])}
        for i in range(len(B)):
            arrs[f"rgb_{i}"] = rgbs[i]; arrs[f"gt_{i}"] = gts[i]
            for k in range(len(KNOBS)):
                arrs[f"p{k}_{i}"] = probs[k][i]
        np.savez_compressed(fp + ".tmp.npz", **arrs); os.replace(fp + ".tmp.npz", fp)
        print(f"  {n+1}/{len(tids)} {t} {len(B)} crowns {time.time()-t0:.0f}s", flush=True)
    print(f"[cache {a.split}] done", flush=True)


if __name__ == "__main__":
    main()

"""Small-crown diagnosis on the 108 VALIDATION tiles (test stays held out; any fix found here is val-derived).
GT boxes (gtbox_lib, as tab:gtbox). Per crown: GT area, LACE deployed mask (colour-guided r12/eps1e-4/tau0.35,
alpha/kappa from em_model_4p_fix.npz; render functions imported from guided_ablate_modal.py; Modal-cached val
features), LACE posterior stats inside the box (max, mean over the clip window, before the guided filter),
ellipse-in-box and box-fill IoU, SAM 3 zero-shot per-subtile IoU (local MPS, gtbox_sam_persub.arm).
LACE and SAM masks are stored as RLE for crowns < SMALL_PX (for the figure).
Small = GT area < 622 px at 2048 (the test-set tercile edge from tierA/a5_strata.json, fixed, not re-fit on val).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.smallcrown.diag_val
"""
import importlib.util, json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
FEAT = "/Users/tompitts/dphil/feat_cache/feat_4p_val"
NPZ = "/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"
RGB_DIR = os.path.join(REPO, "data", "tcd", "train")
VAL_GT = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "val_gt.json")
SMALL_PX = 622.0
OUT = os.path.join(HERE, "val_crowns")
spec = importlib.util.spec_from_file_location("gsp", os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub.py"))
GSP = importlib.util.module_from_spec(spec); spec.loader.exec_module(GSP)


def ellipse(b):
    x0, y0, x1, y1 = [float(v) for v in b]
    cx, cy, a, bb = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)
    m = np.zeros((2048, 2048), bool)
    X0, X1, Y0, Y1 = max(0, int(np.floor(x0))), min(2048, int(np.ceil(x1))), max(0, int(np.floor(y0))), min(2048, int(np.ceil(y1)))
    if X1 > X0 and Y1 > Y0:
        xs = np.arange(X0, X1) + .5; ys = np.arange(Y0, Y1) + .5
        m[Y0:Y1, X0:X1] = ((xs[None] - cx) / a) ** 2 + ((ys[:, None] - cy) / bb) ** 2 <= 1
    return m


def boxfill(b):
    x0, y0, x1, y1 = [float(v) for v in b]
    m = np.zeros((2048, 2048), bool)
    m[max(0, int(np.floor(y0))):min(2048, int(np.ceil(y1))), max(0, int(np.floor(x0))):min(2048, int(np.ceil(x1)))] = True
    return m


def rle(m):
    e = mu.encode(np.asfortranarray(m.astype(np.uint8))); return {"size": [2048, 2048], "counts": e["counts"].decode("ascii")}


def main():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    os.makedirs(OUT, exist_ok=True)
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    base = GSP.load_base()
    gt = json.load(open(VAL_GT))
    tids = [t for t in sorted(gt) if gt[t]["trees"] and os.path.exists(os.path.join(FEAT, t + ".npy"))]
    print(f"[diag] {len(tids)} val tiles with crowns", flush=True)
    t0 = time.time()
    for n, t in enumerate(tids):
        fp = os.path.join(OUT, t + ".json")
        if os.path.exists(fp):
            continue
        polys = gt[t]["trees"]
        B = G.gt_boxes_2048(polys).astype(np.float64)
        g, valid = G.gt_rles_and_valid(polys, res=2048)
        img = Image.open(os.path.join(RGB_DIR, t + ".tif")).convert("RGB"); assert img.size == (2048, 2048)
        rgbf = np.asarray(img).astype(np.float32) / 255.0
        feat = np.load(os.path.join(FEAT, t + ".npy")).astype(np.float32)
        gg = feat.shape[-1]; zn = m.project(feat); del feat
        lace, pmax, pmean = [], [], []
        for b in B.astype(np.float32):
            clip, prob, _ = GA._prob(m, zn, gg, b, None, None)
            y0, y1, x0, x1 = clip
            w = prob[y0:y1, x0:x1]
            pmax.append(float(w.max()) if w.size else 0.0); pmean.append(float(w.mean()) if w.size else 0.0)
            lace.append(GA._guided(rgbf, clip, prob))
        sam = GSP.arm(base[0], base[1], img, B, True, None)
        ell = [ellipse(b) for b in B]; bf = [boxfill(b) for b in B]
        iou = lambda ms: G.per_crown_iou_rle(G.to_rle(ms), g).round(5).tolist()
        areas = [int(mu.area(r)) for r in g]
        rec = {"valid": valid.tolist(), "area": areas, "box": B.round(2).tolist(),
               "lace": iou(lace), "sam_zs_sub": iou(sam), "ellipse": iou(ell), "boxfill": iou(bf),
               "lace_area": [int(x.sum()) for x in lace], "sam_area": [int(x.sum()) for x in sam],
               "pmax": [round(x, 4) for x in pmax], "pmean": [round(x, 4) for x in pmean],
               "small_masks": {str(i): {"lace": rle(lace[i]), "sam": rle(sam[i])}
                               for i in range(len(B)) if areas[i] < SMALL_PX}}
        json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        print(f"  {n+1}/{len(tids)} {t} {len(B)} crowns  {time.time()-t0:.0f}s", flush=True)
    print("[diag] done", flush=True)


if __name__ == "__main__":
    main()

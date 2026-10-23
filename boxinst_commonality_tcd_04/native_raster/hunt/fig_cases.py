"""Figure: val detections where SAM's mask is a TP and LACE's is not (or vice versa), from a diag_pair JSON.
Columns: RGB + det box (yellow) + GT crown outline (green); LACE cell posterior (bilinear prob); LACE deployed mask;
SAM mask. 64x64-px windows around the box (upscaled). -> hunt/diag/<name>.png
  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.hunt.fig_cases DIAG.json gain|loss small|mid|large N SEED OUT.png [LACE_DIR:KEY]"""
import json, sys, random
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
FC = F.FC; EDG = {"small": (0, 622), "mid": (622, 2456.3), "large": (2456.3, 1e12)}
diag, kind, size, N, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
LSPEC = sys.argv[7] if len(sys.argv) > 7 else "boxfix_ap_val_tiles:deployed"      # LACE mask source DIR:KEY ('-' = list)
rows = [r for r in json.load(open(diag))["rows"] if r["gain"] == (kind == "gain") and EDG[size][0] <= r["areaG"] < EDG[size][1]]
random.Random(seed).shuffle(rows); rows = rows[:N]
m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); gtc = F.gt_cache("val")
fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)))
for k, r in enumerate(rows):
    t, i = r["tile"], r["det"]
    L = json.load(open(f"{FC}/boxfix_ap_val_tiles/{t}.json")); S = json.load(open(f"{FC}/hunt_sam_val_dets/{t}.json"))
    b = np.asarray(L["boxes_2048"][i]); rgb = np.asarray(Image.open(f"/Users/tompitts/dphil/CanopyAI2/data/tcd/train/{t}.tif").convert("RGB"))
    feat = np.load(f"{FC}/feat_4p_val/{t}.npy").astype(np.float32); g = feat.shape[-1]; zn = m.project(feat); del feat
    _, prob, _ = GA._prob(m, zn, g, b.astype(np.float32), GA.ALPHA, GA.KAPPA)
    cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2); h = int(max(32, 0.8 * max(b[2] - b[0], b[3] - b[1])))
    y0, y1, x0, x1 = max(0, cy - h), min(2048, cy + h), max(0, cx - h), min(2048, cx + h)
    _d, _k = LSPEC.split(":"); _L = json.load(open(f"{FC}/{_d}/{t}.json")); _lm = _L["masks_rle"] if _k == "-" else _L["masks_rle"][_k]
    lm = mu.decode({"size": [2048, 2048], "counts": _lm[i]["counts"].encode()})[y0:y1, x0:x1]
    sm = mu.decode({"size": [2048, 2048], "counts": S["masks_rle"][i]["counts"].encode()})[y0:y1, x0:x1]
    # GT crown: best IoU crown with SAM (gain) / LACE (loss) mask
    tr = gtc[t]["trees"]; ref = S["masks_rle"][i] if kind == "gain" else _lm[i]
    ious = mu.iou([{"size": ref["size"], "counts": ref["counts"].encode()}], tr, [0] * len(tr))[0]
    gm = mu.decode(tr[int(np.argmax(ious))])[y0:y1, x0:x1]
    for c, (img, ttl) in enumerate([(rgb[y0:y1, x0:x1], f"{t[-8:]} d{i} s={r['score']:.3f}"), (prob[y0:y1, x0:x1], f"post max {prob[y0:y1, x0:x1].max():.2f}"),
                                    (lm, f"LACE iou {r['iouA']:.2f}"), (sm, f"SAM iou {r['iouB']:.2f}")]):
        a = ax[k, c]; a.imshow(rgb[y0:y1, x0:x1]) if c else a.imshow(img)
        if c == 1: a.imshow(img, cmap="magma", vmin=0, vmax=1, alpha=0.8)
        if c >= 2: a.imshow(np.ma.masked_where(img == 0, img), cmap="cool", alpha=0.55, vmin=0, vmax=1)
        a.contour(gm, levels=[0.5], colors="lime", linewidths=0.8)
        a.add_patch(plt.Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=0.8))
        a.set_title(ttl, fontsize=6); a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=130); print(out)

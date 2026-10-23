"""Labelled figure of LACE's small-crown failures on GROUND-TRUTH boxes (test, A4 per-crown data).
Rows: 4 crowns where LACE's mask is empty + 4 where it is non-empty but IoU < 0.5 (SAM 3 zero-shot >= 0.8),
small = GT area < 622 px, one crown per tile, ordered by (SAM - LACE) IoU gap.
Columns: image + GT box | GT mask | LACE posterior inside the box (before the guided filter / threshold) |
LACE mask (deployed guided render) | SAM 3 zero-shot per-subtile mask (recomputed, local MPS).
  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.smallcrown.figure_small_failures
"""
import importlib.util, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR); REPO = os.path.dirname(PKG)
FEAT = "/Users/tompitts/dphil/feat_cache/feat_4p_test_dl/feat_4p_test"
NPZ = "/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"
spec = importlib.util.spec_from_file_location("gsp", os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub.py"))
GSP = importlib.util.module_from_spec(spec); spec.loader.exec_module(GSP)
W = 40   # half window, px


def pick(gt, lace):
    emp, ne = [], []
    for t, L in lace.items():
        s = json.load(open(os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub_tiles", t + ".json")))
        g, _ = G.gt_rles_and_valid(gt[t]["trees"], res=2048)
        for i, v in enumerate(L["valid"]):
            if not v or mu.area(g[i]) >= 622:
                continue
            la = mu.area({"size": L["masks_rle"][i]["size"], "counts": L["masks_rle"][i]["counts"].encode()})
            rec = (s["base_sub"][i] - L["iou"][i], t, i, L["iou"][i], s["base_sub"][i], int(mu.area(g[i])))
            if la == 0:
                emp.append(rec)
            elif L["iou"][i] < 0.5 and s["base_sub"][i] >= 0.8:
                ne.append(rec)
    out = []
    for lst, k in ((emp, 4), (ne, 4)):
        lst.sort(reverse=True); seen = set()
        for r in lst:
            if r[1] in seen or r[1] in {o[1] for o in out}:
                continue
            out.append(r); seen.add(r[1])
            if len(seen) == k:
                break
    return out


def main():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    gt = json.load(open(os.path.join(PKG, "test_gt.json")))
    lace = {}
    for l in open(os.path.join(NR, "gtbox_guided_raw", "gtbox_lace_guided_r2048.jsonl")):
        r = json.loads(l); lace[r["tile"]] = r
    rows = pick(gt, lace)
    masker = E.TCDMasker(NPZ); base = GSP.load_base()
    fig, ax = plt.subplots(len(rows), 5, figsize=(13, 2.75 * len(rows)))
    heads = ["Image + GT box (yellow)", "Ground truth (green)", "LACE posterior in box\n(before threshold τ=0.35)",
             "LACE mask, given the GT box\n(magenta; empty = nothing ≥ τ)", "SAM 3 zero-shot, same box\n(cyan)"]
    for r, (gap, t, i, li, si, area) in enumerate(rows):
        polys = gt[t]["trees"]; b = G.gt_boxes_2048([polys[i]])[0].astype(np.float64)
        gm = mu.decode(G.gt_rles_and_valid([polys[i]], res=2048)[0][0]).astype(bool)
        lm = mu.decode({"size": lace[t]["masks_rle"][i]["size"], "counts": lace[t]["masks_rle"][i]["counts"].encode()}).astype(bool)
        img = Image.open(os.path.join(REPO, "data", "tcd", "test", t + ".tif")).convert("RGB")
        rgb = np.asarray(img)
        feat = np.load(os.path.join(FEAT, t + ".npy")).astype(np.float32)
        zn = masker.project(feat); g = feat.shape[-1]; del feat
        clip, prob, _ = GA._prob(masker, zn, g, b.astype(np.float32), None, None)
        sm = GSP.arm(base[0], base[1], img, b[None], True, None)[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(np.clip(cx - W, 0, 2048 - 2 * W)); y0 = int(np.clip(cy - W, 0, 2048 - 2 * W))
        sl = (slice(y0, y0 + 2 * W), slice(x0, x0 + 2 * W)); crop = rgb[sl]
        def over(m, col):
            im = crop.astype(float).copy(); mm = m[sl]; im[mm] = 0.45 * im[mm] + 0.55 * np.array(col); return im.astype(np.uint8)
        pm = np.zeros((2048, 2048), np.float32); y_0, y_1, x_0, x_1 = clip; pm[y_0:y_1, x_0:x_1] = prob[y_0:y_1, x_0:x_1]
        panels = [crop, over(gm, (0, 255, 0)), None, over(lm, (255, 0, 255)), over(sm, (0, 255, 255))]
        for c in range(5):
            a = ax[r, c]
            if c == 2:
                a.imshow(crop); hm = a.imshow(pm[sl], cmap="magma", vmin=0, vmax=1, alpha=0.75)
                a.text(2, 2 * W - 3, f"max {prob[y_0:y_1, x_0:x_1].max():.2f}", color="w", fontsize=8,
                       bbox=dict(fc="k", alpha=0.6, pad=1))
            else:
                a.imshow(panels[c], interpolation="nearest")
            a.add_patch(Rectangle((b[0] - x0 - .5, b[1] - y0 - .5), b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=1))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(heads[c], fontsize=9)
        kind = "EMPTY mask" if lm.sum() == 0 else "wrong mask"
        ax[r, 0].set_ylabel(f"{t.replace('tcd_val_tile_', 'tile ')}\n{kind}\nGT {area} px, box {b[2]-b[0]:.0f}×{b[3]-b[1]:.0f}\n"
                            f"IoU LACE {li:.2f} | SAM {si:.2f}", fontsize=8)
        print(t, i, kind, area, round(li, 3), round(si, 3), flush=True)
    fig.suptitle("LACE small-crown failures on ground-truth boxes (OAM-TCD test). Box-to-mask module only: "
                 "no detector involved. 80×80 px windows (~8 m), native 0.1 m/px.", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(HERE, "small_crown_failures_labelled.png")
    assert not os.path.exists(out), out
    fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__":
    main()

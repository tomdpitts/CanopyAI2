"""Real-data insets for the LACE architecture figure (../lace_architecture.tex).

Every inset comes from ONE OAM-TCD holdout crop (TID, CROP), run through the DEPLOYED seed-0
pipeline:
  - 4-phase interlaced DINOv3-web L24 features: the DEPLOYED ones, pulled from Modal
      .venv/bin/modal volume get tcd04-phase4-vol feat_4p_test/<TID>.npy _cache/<TID>_4p_L24.npy
    (local transformers 5.12 drifts to cos~0.86 vs Modal's 4.57; a local re-extraction of tile
    122 gave 204 boxes vs the published 445, so local features are only a last-resort fallback)
  - detector det_phase4_L24_s0 (copy at modal_savanna_multiseed/phase4/det_tcd_s0.pt)
  - masker em_model_4p_fix.npz (beta=0.5, K_fg=16, alpha=0.3, kappa 10x1.6)
Boxes, scores (product-reranked s') and native-2048 masks for the detection/output insets are
the PUBLISHED ones (native_raster/preds_lace_s0_r2048_tau040.json), shown at the paper's
qualitative-figure threshold s' >= 0.11 (qualitative_thresholds_tcd439.json). The local run
supplies the heatmap and the per-cell A / A-bar / p_fg of the zoomed box, and is checked against
the published boxes and s' on every run.

The fit-partition inset applies the fit's cell partition (in-box / ring / clear) to the SAME
crop using its GT boxes. This tile is holdout: the fit never saw it; the inset only illustrates
the partition rule.

Run from repo root:  .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/arch/make_arch_panels.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4"))

from boxinst_commonality.em import estep, logsumexp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T04 = os.path.join(REPO, "boxinst_commonality_tcd_04")
DET = os.path.join(T04, "modal_savanna_multiseed/phase4/det_tcd_s0.pt")
EM = os.path.join(T04, "modal_tcd_multiseed/phase4/em_model_4p_fix.npz")
PUB = os.path.join(T04, "native_raster/preds_lace_s0_r2048_tau040.json")
TID = os.environ.get("TID", "tcd_val_tile_147")
FEAT_CACHE = os.environ.get("FEAT_CACHE", os.path.join(HERE, "_cache"))
CROP = [int(v) for v in os.environ.get("CROP", "128,768,512").split(",")]  # x0,y0,size
TAU = 0.40          # native-2048 raster threshold of the published preds
SHOW = 0.11         # display cut on s' (LACE row of qualitative_thresholds_tcd439.json)
SPLO, SPHI = 0.15, 0.28   # s' colour range (shown masks span 0.17-0.26); TikZ bar is low..high
PHASE_RGB = [(59, 110, 168), (181, 71, 125), (42, 157, 143), (198, 146, 20)]  # (0,0),(0,8),(8,0),(8,8)


def features(tid, img):
    """(1024,256,256) fp16 interlaced L24, cached."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    fp = os.path.join(FEAT_CACHE, f"{tid}_4p_L24.npy")
    if os.path.exists(fp):
        return np.load(fp)
    print("WARNING: no Modal features cached; extracting locally (drifted, see docstring)")
    from dapt.backbone import FrozenDinoV3Features
    import phase4_features_tcd as p4
    net = FrozenDinoV3Features("web", layers=(24,))        # per-layer L2 -> == L24 slice
    arr = np.asarray(img, np.float32) / 255.0
    ph = {(dy, dx): p4._extract_2048(net, p4._shift(arr, dy, dx)) for dy, dx in p4.PHASES}
    asm = p4.interleave(ph).astype(np.float16)
    np.save(fp, asm)
    return asm


def box_terms(M, zn, g, box):
    """Replicates TCDMasker.box_mask, also returning A and A-bar for display."""
    s, origin = M["s"], M["origin"]
    cy, cx = np.mgrid[0:g, 0:g] * s + origin
    x0, y0, x1, y1 = box
    pad = s / 2.0
    m = (cx >= x0 - pad) & (cx < x1 + pad) & (cy >= y0 - pad) & (cy < y1 + pad)
    idx = np.flatnonzero(m.ravel())
    u = np.clip((cx.ravel()[idx] - x0) / max(x1 - x0, 1), 0, 1)
    v = np.clip((cy.ravel()[idx] - y0) / max(y1 - y0, 1), 0, 1)
    NB = M["pi"].shape[2]
    bu = np.minimum((u * NB).astype(int), NB - 1)
    bv = np.minimum((v * NB).astype(int), NB - 1)
    sb = int(np.searchsorted(M["size_edges"], np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1))))
    k = M["kappa"] * M["kappa_scale"]
    pis = M["pi"][sb][:, bv, bu]
    z = zn[idx]
    bgll = logsumexp(np.log(M["wbg"])[None] + k * (z @ M["Gbg"].T), 1)
    pfg, _ = estep(z, pis, k, M["C"], bgll, True, prior_weight=M["prior_weight"])
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    A = logsumexp(k * (z @ M["C"].T) + np.log((pis / psum).T + 1e-9), 1) - bgll
    return idx, A, A - A.mean(), pfg, sb


def colour(vals, cmap, lo, hi):
    import matplotlib
    c = matplotlib.colormaps[cmap]((np.clip(vals, lo, hi) - lo) / (hi - lo))
    return [f"{int(r*255)},{int(gg*255)},{int(b*255)}" for r, gg, b, _ in c]


def tikz_fill(c):
    r, g, b = c.split(",")
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def main():
    import matplotlib
    from pycocotools import mask as mu
    from scipy import ndimage
    from torchvision.ops import box_iou
    import phase4_lib_tcd as L
    from dapt.decode import decode
    from boxinst_commonality_tcd_04.detector import STRIDE8

    img = Image.open(os.path.join(REPO, f"data/tcd/test/{TID}.tif")).convert("RGB")
    feat = features(TID, img).astype(np.float32)
    g = feat.shape[-1]

    ck = torch.load(DET, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["tag"] == "phase4_L24_s0", cfg["tag"]
    det = L.Detector4Phase(cfg["in_dim"], width=cfg["width"], tower=cfg["tower"]).eval()
    det.load_state_dict(ck["state"])
    with torch.no_grad():
        out = det(torch.from_numpy(feat)[None])
    heat = torch.sigmoid(out[0, 0]).numpy()
    bx, sc = decode(out, score_thr=0.05, stride=STRIDE8, topk=600)
    bx, sc = bx.numpy(), sc.numpy()

    d = np.load(EM)
    M = {k: d[k] for k in d.files}
    M["s"], M["origin"] = int(d["s_px"]), float(d["cell_origin"])
    M["kappa"], M["kappa_scale"] = float(d["kappa"]), float(d["kappa_scale"])
    M["prior_weight"] = float(d["prior_weight"])
    zn = (feat.reshape(feat.shape[0], -1).T - M["mu"]) @ M["U"] / M["scale"]
    zn /= np.linalg.norm(zn, axis=1, keepdims=True) + 1e-8

    rows = []
    for b, s in zip(bx, sc):
        idx, A, Ab, pfg, sb = box_terms(M, zn, g, b)
        pbar, mdec = float(pfg.mean()), float(np.abs(2 * pfg - 1).mean())
        rows.append(dict(box=b, s=float(s), idx=idx, A=A, Ab=Ab, pfg=pfg, sb=sb,
                         pbar=pbar, m=mdec, s2=float(s) * pbar * mdec))

    # published boxes / s' / masks, and the reproduction check
    pub = json.load(open(PUB))["preds"][TID]
    pb = np.array(pub["boxes_2048"], np.float32).reshape(-1, 4)
    ps = np.array(pub["scores"])
    iou = box_iou(torch.from_numpy(bx), torch.from_numpy(pb)).numpy()
    j = iou.argmax(1)
    s2 = np.array([r["s2"] for r in rows])
    print(f"local {len(bx)} boxes, published {len(pb)}; matched IoU>0.99: "
          f"{(iou.max(1) > 0.99).mean():.3f}; max |s'_local - s'_pub| = "
          f"{np.abs(s2 - ps[j]).max():.2e}")

    x0, y0, S = CROP
    rgb = img.crop((x0, y0, x0 + S, y0 + S))
    rgb.save(os.path.join(HERE, "in_rgb.jpg"), quality=92)

    hc = heat[(y0 // 8):(y0 + S) // 8, (x0 // 8):(x0 + S) // 8]
    hm = (matplotlib.colormaps["inferno"](np.sqrt(np.clip(hc / 0.8, 0, 1)))[..., :3] * 255).astype(np.uint8)
    Image.fromarray(hm).resize((S, S), Image.NEAREST).save(os.path.join(HERE, "det_heat.png"))

    # shown set: published s' >= SHOW, box centre inside the crop
    cxy = (pb[:, :2] + pb[:, 2:]) / 2
    show = np.where((ps >= SHOW) & (cxy[:, 0] >= x0) & (cxy[:, 0] < x0 + S)
                    & (cxy[:, 1] >= y0) & (cxy[:, 1] < y0 + S))[0]
    show = show[np.argsort(-ps[show])]
    im = rgb.copy(); dr = ImageDraw.Draw(im)
    for i in show:
        dr.rectangle(list(pb[i] - [x0, y0, x0, y0]), outline=(217, 130, 43), width=7)
    im.save(os.path.join(HERE, "det_boxes.jpg"), quality=92)

    base = np.asarray(rgb).astype(np.float32)
    ov = base.copy(); lab = np.zeros((S, S), int)
    cols = (np.array(matplotlib.colormaps["viridis"]((np.clip(ps[show], SPLO, SPHI) - SPLO) / (SPHI - SPLO)))[:, :3]
            * 255).astype(np.float32)
    for k, i in list(enumerate(show))[::-1]:                  # highest s' painted last
        r = dict(pub["masks_rle"][i]); r["counts"] = r["counts"].encode()
        m = mu.decode(r)[y0:y0 + S, x0:x0 + S].astype(bool)
        lab[m] = k + 1
    for k in range(len(show)):
        ov[lab == k + 1] = 0.35 * base[lab == k + 1] + 0.65 * cols[k]
    edge = (lab > 0) & (ndimage.grey_dilation(lab, 3) != ndimage.grey_erosion(lab, 3))
    ov[edge] = 255
    Image.fromarray(ov.astype(np.uint8)).save(os.path.join(HERE, "out_masks.jpg"), quality=92)

    # zoom candidates = local rows matching the shown published boxes
    cand = []
    for k, i in enumerate(show):
        li = int(np.argmax(iou[:, i]))
        r = rows[li]
        w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        print(f"  [{k}] box={np.round(r['box']).astype(int).tolist()} {w:.0f}x{h:.0f}px s={r['s']:.2f} "
              f"pbar={r['pbar']:.2f} m={r['m']:.2f} s'={r['s2']:.3f} (pub {ps[i]:.3f}) band={r['sb']}")
        cand.append(r)
    z = cand[int(os.environ.get("ZOOM", "0"))]
    write_zoom(z, img, g, M)
    write_rho(M, z["sb"])
    write_partition()
    enc = write_encoding(img, z["box"])
    meta = dict(tile=TID, crop_xywh=CROP, det=os.path.relpath(DET, REPO), em=os.path.relpath(EM, REPO),
                preds=os.path.relpath(PUB, REPO), show_thr_sprime=SHOW, tau=TAU, n_shown=len(show),
                encoder_window_xywh=enc,
                zoom=dict(box=np.round(z["box"], 1).tolist(), s=round(z["s"], 3), pbar=round(z["pbar"], 3),
                          m=round(z["m"], 3), s2=round(z["s2"], 3), band=z["sb"]),
                partition="same crop, GT boxes of this holdout tile (illustrates the fit's partition rule)",
                note="deployed s0 weights + deployed Modal features; reproduces published boxes/s'")
    json.dump(meta, open(os.path.join(HERE, "arch_meta.json"), "w"), indent=1)


def write_zoom(z, img, g, M):
    """Zoom box: RGB of its cell block, and per-cell A, A-bar, p_fg, mask as TikZ fills."""
    s, origin = M["s"], M["origin"]
    ys, xs = np.divmod(z["idx"], g)
    X0, X1, Y0, Y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nx, ny = X1 - X0 + 1, Y1 - Y0 + 1
    px0, py0 = s * X0 + origin - s / 2, s * Y0 + origin - s / 2   # cell (Y,X) centred at s*X+origin
    crop = img.crop((int(px0), int(py0), int(px0 + nx * s), int(py0 + ny * s)))
    crop.resize((nx * 32, ny * 32), Image.LANCZOS).save(os.path.join(HERE, "zoom_rgb.jpg"), quality=95)
    lim = float(np.abs(z["A"]).max())
    blocks = {"A": colour(z["A"], "BrBG", -lim, lim), "Abar": colour(z["Ab"], "BrBG", -lim, lim),
              "pfg": colour(z["pfg"], "BrBG", 0, 1),
              "mask": [("1,102,94" if p >= TAU else "246,232,195") for p in z["pfg"]]}
    bx = z["box"]
    with open(os.path.join(HERE, "zoom_cells.tex"), "w") as f:
        f.write(f"% generated by make_arch_panels.py: {TID} box {np.round(bx, 1).tolist()}, "
                f"{nx}x{ny} cells, |A| colour limit {lim:.2f}\n")
        f.write(f"\\def\\zoomnx{{{nx}}}\\def\\zoomny{{{ny}}}\n")
        f.write("\\def\\zoombox{(%.3f,%.3f) rectangle (%.3f,%.3f)}\n" % (
            (bx[0] - px0) / s, ny - (bx[1] - py0) / s, (bx[2] - px0) / s, ny - (bx[3] - py0) / s))
        for name, cols in blocks.items():
            f.write(f"\\def\\zoom{name}{{%\n")
            for yy, xx, c in zip(ys - Y0, xs - X0, cols):
                f.write(f"\\fill[fill={tikz_fill(c)}] ({xx},{ny-1-yy}) rectangle ++(1,1);\n")
            f.write("}\n")
        f.write("\\def\\zooms{%.2f}\\def\\zoompbar{%.2f}\\def\\zoomm{%.2f}\\def\\zoomsp{%.2f}\n"
                % (z["s"], z["pbar"], z["m"], z["s2"]))


def write_rho(M, band):
    """Learned spatial prior sum_k rho_k for one size band, 8x8, as TikZ fills."""
    r = M["pi"][band].sum(0)
    cols = colour(r.ravel(), "Greens", 0, 1.35)            # 0.95 cap -> mid green
    NB = r.shape[0]
    with open(os.path.join(HERE, "rho_cells.tex"), "w") as f:
        f.write(f"% generated by make_arch_panels.py: sum_k rho_k, size band {band}, "
                f"mean mass {r.mean():.3f}\n\\def\\rhocells{{%\n")
        for i, c in enumerate(cols):
            yy, xx = divmod(i, NB)
            f.write(f"\\fill[fill={tikz_fill(c)}] ({xx},{NB-1-yy}) rectangle ++(1,1);\n")
        f.write("}\n")


def write_partition():
    """The fit's cell partition (in-box / ring / clear) applied to the example crop's GT boxes."""
    from phase4_fit_tcd import _cell_labels, _raster_canopy, _ring_cells
    gt = json.load(open(os.path.join(T04, "test_gt.json")))[TID]
    img = Image.open(os.path.join(REPO, f"data/tcd/test/{TID}.tif")).convert("RGB")
    bx = np.array([[*np.asarray(t).reshape(-1, 2).min(0), *np.asarray(t).reshape(-1, 2).max(0)]
                   for t in gt["trees"]], np.float32).reshape(-1, 4)
    can = _raster_canopy(gt.get("canopy", []), res=2048)
    g, s = 256, 8
    lab = _cell_labels(bx, can, g, s).reshape(g, g)        # the deployed fit's own helpers
    ring = _ring_cells(bx, can, g, s).reshape(g, g)
    x0, y0, S = CROP
    cell = np.zeros((g, g, 3), np.float32); a = np.zeros((g, g), np.float32)
    cell[lab == 1] = (94, 158, 78); a[lab == 1] = 0.45       # in-box: crown green
    cell[ring] = (217, 130, 43); a[ring] = 0.8               # ring: orange
    cell[lab == 0] = (59, 110, 168); a[lab == 0] = 0.35      # clear background: blue
    cs = slice(y0 // 8, (y0 + S) // 8), slice(x0 // 8, (x0 + S) // 8)
    up = lambda q: np.kron(q[cs], np.ones((8, 8) + ((1,) if q.ndim == 3 else ())))
    base = np.asarray(img.crop((x0, y0, x0 + S, y0 + S))).astype(np.float32)
    A = up(a)[..., None]
    Image.fromarray((base * (1 - A) + up(cell) * A).astype(np.uint8)).save(
        os.path.join(HERE, "fit_partition.jpg"), quality=92)


def write_encoding(img, box):
    """48x48 px window on the zoomed crown's left edge, aligned to the 16 px patch grid,
    for the encoder inset (TikZ draws the four offset patches and the 8 px centre lattice)."""
    ex = int(16 * np.floor((box[0] - 16) / 16))
    ey = int(16 * np.floor(((box[1] + box[3]) / 2 - 24) / 16))
    win = img.crop((ex, ey, ex + 48, ey + 48)).resize((384, 384), Image.NEAREST)
    win.save(os.path.join(HERE, "enc_rgb.png"))
    x0, y0, S = CROP
    with open(os.path.join(HERE, "enc_window.tex"), "w") as f:     # for the input-tile zoom box
        f.write(f"% generated by make_arch_panels.py: encoder window in crop px ({TID}, crop {CROP})\n"
                f"\\def\\encwx{{{ex - x0}}}\\def\\encwy{{{ey - y0}}}\\def\\encws{{48}}\\def\\cropS{{{S}}}\n")
    return [ex, ey, 48]


if __name__ == "__main__":
    main()

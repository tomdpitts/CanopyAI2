"""Real-data insets for the LACE architecture figure (../lace_architecture.tex), LACE (D1 variant), 2026-11-05.

Generator of the d1_* (PREFIX) assets that lace_architecture.tex inputs. A parallel session's competing
make_arch_panels_d1.py and its _d1-SUFFIX outputs are in ../_quarantine_parallel_session_2026-11-05/ (not used).

Replaces the CenterNet-variant insets of make_arch_panels.py for Fig 1 only. Writes d1_* files so the masker-fit
figure (lace_masker_fit.tex), which shares zoom_cells.tex / rho_cells.tex / fit_partition.jpg, is untouched.

Every inset comes from ONE OAM-TCD holdout crop (TID, CROP), run through the paper's headline pipeline:
  - detector: D1 DINO-Swin-L (dino_det/README.md), stitched boxes and raw detector scores s
      dino_det/results/preds_dino_swinl_s0_test_detscore_r2048.json
  - masker: lace4px_simple (em_model_4p_fix.npz at s 4 / origin 8, alpha 0.3, kappa x1.6) on the REAL 4 px
      16-offset features feat_cache/hunt_z4p_test/<TID>.npy (masker's 128-d projection, unit norm)
  - published masks + ranking: D1 + edge fix + R3 (paper Table 1 headline row)
      dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json
All three preds files share box order (asserted). The local E-step is checked against GA._prob (deployed posterior)
and the local s' against preds_..._sprime_r2048.json, on every box of the tile.
Display: mask colour = rank within the shown set (illustrative, not a score scale). Shown set: R3 score at its own best-F1 threshold over the 439 tiles (make_qualitative.best_f1_threshold),
cached in ../qualitative_thresholds_tcd439_d1r3.json. Zoomed box: the shown D1 box best matching the previous
figure's zoom box (continuity), else the top-ranked shown box. The cell mask in the zoom (p_fg >= tau) uses the
box-relative tau = 0.33 x max p_fg on the cells; the deployed mask applies it after RGB-guided upsampling.
Zoom cell size (Tom 2026-11-05: 4 px cells too fine to read): the zoom insets are drawn at ZOOM_S = 8 px, i.e. the
same masker (native s 8 / origin 8, the lattice it was fitted at) on the {0,8}^2 sub-lattice of the deployed 4 px
features (= the deployed 4-offset 8 px features). Illustrative; detection, ranking and output masks are the 4 px pipeline.

Run from repo root:  PYTHONPATH=. .venv/bin/python -W ignore::RuntimeWarning boxinst_commonality_tcd_04/ablation/results/figs/arch/make_arch_panels_d1_prefix.py
Log of the 2026-11-05 run: arch/logs/make_arch_panels_d1_2026-11-05.log
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, REPO)
T04 = os.path.join(REPO, "boxinst_commonality_tcd_04")
RES = os.path.join(T04, "dino_det", "results")
FC = "/Users/tompitts/dphil/feat_cache"
EM = f"{FC}/em_model_4p_fix.npz"
P_R3 = os.path.join(RES, "preds_dino_swinl_s0_test_R3_ef_r2048.json")
P_SP = os.path.join(RES, "preds_dino_swinl_s0_test_sprime_r2048.json")
P_S = os.path.join(RES, "preds_dino_swinl_s0_test_detscore_r2048.json")
TID = os.environ.get("TID", "tcd_val_tile_147")
CROP = [int(v) for v in os.environ.get("CROP", "128,768,512").split(",")]  # x0,y0,size (same crop as before)
OLD_ZOOM = [433.2, 764.8, 533.4, 875.6]     # arch_meta.json zoom box of the CenterNet-variant figure
ALPHA, KSC, C_PEAK = 0.3, 1.6, 0.33
ZOOM_S = int(os.environ.get("ZOOM_S", "8"))     # cell size of the zoom insets only (4 = deployed lattice)
SUB, SSTR = 1024, 512                        # D1 inference: 3 x 3 subtiles of 1024 at stride 512
THR_PATH = os.path.join(FIGS, "qualitative_thresholds_tcd439_d1r3.json")
DET_BOX_RGB = (230, 30, 45)                  # detection boxes: red (Tom 2026-11-07: was trn orange 217,130,43, reused elsewhere)

_mpb = importlib.util.spec_from_file_location("mpb", os.path.join(HERE, "make_arch_panels.py"))
MP = importlib.util.module_from_spec(_mpb); _mpb.loader.exec_module(MP)      # box_terms, colour, tikz_fill (unchanged)


def threshold(gt):
    if os.path.exists(THR_PATH):
        return json.load(open(THR_PATH))["LACE (ours)"]["threshold"]
    spec = importlib.util.spec_from_file_location("mq", os.path.join(FIGS, "make_qualitative.py"))
    mq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mq)
    preds = json.load(open(P_R3))["preds"]
    th, f1, c = mq.best_f1_threshold(gt, preds)
    json.dump({"LACE (ours)": dict(threshold=th, f1=f1, **c, preds=os.path.relpath(P_R3, REPO))},
              open(THR_PATH, "w"), indent=1)
    print(f"R3 best-F1 thr={th:.2f} F1={f1:.4f} {c}", flush=True)
    return th


def main():
    import matplotlib
    from pycocotools import mask as mu
    from scipy import ndimage
    from torchvision.ops import box_iou
    import torch
    from boxinst_commonality_tcd_04.em import TCDMasker
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA

    gt = json.load(open(os.path.join(T04, "test_gt.json")))
    SHOW = threshold(gt)
    pub = {k: json.load(open(p))["preds"][TID] for k, p in (("r3", P_R3), ("sp", P_SP), ("s", P_S))}
    pb = np.array(pub["r3"]["boxes_2048"], np.float32).reshape(-1, 4)
    for k in ("sp", "s"):
        assert np.array_equal(pb, np.array(pub[k]["boxes_2048"], np.float32).reshape(-1, 4)), k
    s_det, s_sp, s_r3 = (np.array(pub[k]["scores"]) for k in ("s", "sp", "r3"))

    img = Image.open(os.path.join(REPO, f"data/tcd/test/{TID}.tif")).convert("RGB")
    zn = np.load(f"{FC}/hunt_z4p_test/{TID}.npy").astype(np.float32).reshape(-1, 128)
    g = 512
    d = np.load(EM)
    M = {k: d[k] for k in d.files}
    M["s"], M["origin"] = 4, 8.0
    M["kappa"], M["kappa_scale"], M["prior_weight"] = float(d["kappa"]), KSC, ALPHA
    m4 = TCDMasker(EM); m4.s, m4.origin = 4, 8.0

    rows, dmax = [], 0.0
    for b, s in zip(pb, s_det):
        _, _, r = GA._prob(m4, zn, g, b, ALPHA, KSC)
        idx, A, Ab, pfg, sb = MP.box_terms(M, zn, g, b) if (b[2] - b[0]) * (b[3] - b[1]) > 0 else ([], [], [], [], -1)
        if len(idx) == 0:                     # no cell in the box: TCDMasker falls back to the nearest cell
            idx, A, Ab, pfg = None, None, None, r
        dmax = max(dmax, float(np.abs(r - pfg).max()))
        rows.append(dict(box=b, s=float(s), idx=idx, A=A, Ab=Ab, pfg=pfg, sb=sb, pbar=float(pfg.mean()),
                         m=float(np.abs(2 * pfg - 1).mean()), s2=float(s * np.abs(2 * pfg - 1).mean() * pfg.mean())))
    s2 = np.array([r["s2"] for r in rows])
    print(f"{len(pb)} boxes; max |p_fg local - GA._prob| = {dmax:.2e}; max |s'_local - s'_pub| = "
          f"{np.abs(s2 - s_sp).max():.2e}", flush=True)
    assert dmax < 1e-5 and np.abs(s2 - s_sp).max() < 1e-5

    x0, y0, S = CROP
    rgb = img.crop((x0, y0, x0 + S, y0 + S))
    rgb.save(os.path.join(HERE, "d1_in_rgb.jpg"), quality=92)

    # detection input: whole tile with the 3 x 3 subtile layout; the subtile holding the crop is drawn by TikZ
    img.resize((512, 512), Image.LANCZOS).save(os.path.join(HERE, "d1_tile.jpg"), quality=90)
    sx = min(max(0, SSTR * int(np.floor(x0 / SSTR))), 2048 - SUB)
    sy = min(max(0, SSTR * int(np.floor(y0 / SSTR))), 2048 - SUB)
    assert x0 >= sx and y0 >= sy and x0 + S <= sx + SUB and y0 + S <= sy + SUB, "crop not inside one subtile"
    # input panel = the full 2048 tile (d1_tile.jpg; Tom 2026-11-06); detection panel = the 1024 subtile holding the crop
    img.crop((sx, sy, sx + SUB, sy + SUB)).resize((512, 512), Image.LANCZOS).save(os.path.join(HERE, "d1_sub.jpg"), quality=90)

    cxy = (pb[:, :2] + pb[:, 2:]) / 2
    show = np.where((s_r3 >= SHOW) & (cxy[:, 0] >= x0) & (cxy[:, 0] < x0 + S)
                    & (cxy[:, 1] >= y0) & (cxy[:, 1] < y0 + S))[0]
    show = show[np.argsort(-s_r3[show])]
    im = rgb.copy(); dr = ImageDraw.Draw(im)
    for i in show:
        dr.rectangle(list(pb[i] - [x0, y0, x0, y0]), outline=DET_BOX_RGB, width=7)
    im.save(os.path.join(HERE, "d1_det_boxes.jpg"), quality=92)

    lo, hi = float(np.floor(s_r3[show].min() * 20) / 20), float(np.ceil(s_r3[show].max() * 20) / 20)
    base = np.asarray(rgb).astype(np.float32)
    ov = base.copy(); lab = np.zeros((S, S), int)
    # colour = RANK within the shown set (Tom 2026-11-05: "a visual"; R3 scores bunch at 0.97-0.99), top = yellow
    rank01 = 1.0 - np.arange(len(show)) / max(len(show) - 1, 1)          # show is sorted by descending R3
    cols = (np.array(matplotlib.colormaps["viridis"](rank01))[:, :3]
            * 255).astype(np.float32)
    for k, i in list(enumerate(show))[::-1]:                  # highest score painted last
        r = dict(pub["r3"]["masks_rle"][i]); r["counts"] = r["counts"].encode()
        lab[mu.decode(r)[y0:y0 + S, x0:x0 + S].astype(bool)] = k + 1
    for k in range(len(show)):
        ov[lab == k + 1] = 0.35 * base[lab == k + 1] + 0.65 * cols[k]
    edge = (lab > 0) & (ndimage.grey_dilation(lab, 3) != ndimage.grey_erosion(lab, 3))
    ov[edge] = 255
    Image.fromarray(ov.astype(np.uint8)).save(os.path.join(HERE, "d1_out_masks.jpg"), quality=92)

    iou_old = box_iou(torch.tensor([OLD_ZOOM]), torch.from_numpy(pb[show])).numpy()[0]
    for k, i in enumerate(show):
        r = rows[i]; w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        print(f"  [{k}] box={np.round(r['box']).astype(int).tolist()} {w:.0f}x{h:.0f}px s={r['s']:.2f} "
              f"pbar={r['pbar']:.2f} m={r['m']:.2f} s'={r['s2']:.3f} R3={s_r3[i]:.3f} IoU(old zoom)={iou_old[k]:.2f}")
    kz = int(os.environ["ZOOM"]) if "ZOOM" in os.environ else (int(iou_old.argmax()) if iou_old.max() >= 0.5 else 0)
    z = rows[show[kz]]
    if ZOOM_S == 8:                                    # same box, masker on the 8 px sub-lattice (see docstring)
        M8 = dict(M, s=8, origin=8.0); zn8 = zn.reshape(g, g, -1)[0::2, 0::2].reshape(-1, zn.shape[1])
        idx, A, Ab, pfg, sb = MP.box_terms(M8, zn8, g // 2, z["box"])
        z = dict(z, idx=idx, A=A, Ab=Ab, pfg=pfg, sb=sb, pbar=float(pfg.mean()), m=float(np.abs(2 * pfg - 1).mean()))
        write_zoom(z, img, g // 2, M8)
    else:
        write_zoom(z, img, g, M)
    enc = write_encoding(z["box"])
    meta = dict(tile=TID, crop_xywh=CROP, preds_masks_rank=os.path.relpath(P_R3, REPO),
                preds_sprime=os.path.relpath(P_SP, REPO), preds_detscore=os.path.relpath(P_S, REPO),
                feats=f"{FC}/hunt_z4p_test/{TID}.npy", em=EM, alpha=ALPHA, kappa_scale=KSC, c_peak=C_PEAK,
                show_thr_r3=SHOW, r3_range_shown=[lo, hi], mask_colour="rank within shown set (viridis, top = yellow)", n_shown=len(show), subtile_xy=[sx, sy, SUB],
                encoder_window_xywh=enc, zoom_index=kz, zoom_cell_px=ZOOM_S,
                zoom=dict(box=np.round(z["box"], 1).tolist(), s=round(z["s"], 3), pbar=round(z["pbar"], 3),
                          m=round(z["m"], 3), s2=round(z["s2"], 3), band=z["sb"]),
                checks=dict(max_abs_pfg_vs_GA_prob=dmax, max_abs_sprime_vs_pub=float(np.abs(s2 - s_sp).max())),
                generator="arch/make_arch_panels_d1_prefix.py",
                note="LACE (D1 variant) + edge fix + R3, the paper's headline pipeline; local posterior reproduces it")
    json.dump(meta, open(os.path.join(HERE, "d1_arch_meta.json"), "w"), indent=1)
    with open(os.path.join(HERE, "d1_window.tex"), "w") as f:
        f.write(f"% generated by make_arch_panels_d1_prefix.py ({TID}, crop {CROP})\n"
                f"\\def\\subx{{{sx}}}\\def\\suby{{{sy}}}\\def\\subS{{{SUB}}}\\def\\cropx{{{x0}}}\\def\\cropy{{{y0}}}\n"
                f"\\def\\scorelo{{{lo:.2f}}}\\def\\scorehi{{{hi:.2f}}}\n")


def write_zoom(z, img, g, M):
    """As make_arch_panels.write_zoom, box-relative tau, cell size M["s"]; -> d1_zoom_cells.tex (same macro names)."""
    s, origin = M["s"], M["origin"]
    ys, xs = np.divmod(z["idx"], g)
    X0, X1, Y0, Y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nx, ny = X1 - X0 + 1, Y1 - Y0 + 1
    px0, py0 = s * X0 + origin - s / 2, s * Y0 + origin - s / 2
    lim = float(np.abs(z["A"]).max())
    tau = C_PEAK * float(z["pfg"].max())
    blocks = {"A": MP.colour(z["A"], "BrBG", -lim, lim), "Abar": MP.colour(z["Ab"], "BrBG", -lim, lim),
              "pfg": MP.colour(z["pfg"], "BrBG", 0, 1),
              "mask": [(MP.colour(np.array([230 / 255]), "BrBG", 0, 1)[0] if p >= tau else "246,232,195")
                       for p in z["pfg"]]}
    bx = z["box"]
    with open(os.path.join(HERE, "d1_zoom_cells.tex"), "w") as f:
        f.write(f"% generated by make_arch_panels_d1_prefix.py: {TID} box {np.round(bx, 1).tolist()}, "
                f"{nx}x{ny} cells ({s} px), |A| colour limit {lim:.2f}, tau {tau:.3f}\n")
        f.write(f"\\def\\zoomnx{{{nx}}}\\def\\zoomny{{{ny}}}\n")
        f.write("\\def\\zoombox{(%.3f,%.3f) rectangle (%.3f,%.3f)}\n" % (
            (bx[0] - px0) / s, ny - (bx[1] - py0) / s, (bx[2] - px0) / s, ny - (bx[3] - py0) / s))
        for name, cols in blocks.items():
            f.write(f"\\def\\zoom{name}{{%\n")
            for yy, xx, c in zip(ys - Y0, xs - X0, cols):
                f.write(f"\\fill[fill={MP.tikz_fill(c)}] ({xx},{ny-1-yy}) rectangle ++(1,1);\n")
            f.write("}\n")
        f.write("\\def\\zooms{%.2f}\\def\\zoompbar{%.2f}\\def\\zoomm{%.2f}\\def\\zoomsp{%.2f}\n"
                % (z["s"], z["pbar"], z["m"], z["s2"]))


def write_encoding(box):
    """48 x 48 px window on the zoomed crown's left edge, aligned to the 16 px patch grid (as make_arch_panels)."""
    img = Image.open(os.path.join(REPO, f"data/tcd/test/{TID}.tif")).convert("RGB")
    ex = int(16 * np.floor((box[0] - 16) / 16))
    ey = int(16 * np.floor(((box[1] + box[3]) / 2 - 24) / 16))
    img.crop((ex, ey, ex + 48, ey + 48)).resize((384, 384), Image.NEAREST).save(os.path.join(HERE, "d1_enc_rgb.png"))
    x0, y0, S = CROP
    with open(os.path.join(HERE, "d1_enc_window.tex"), "w") as f:
        f.write(f"% generated by make_arch_panels_d1_prefix.py: encoder window in crop px ({TID}, crop {CROP})\n"
                f"\\def\\encwx{{{ex - x0}}}\\def\\encwy{{{ey - y0}}}\\def\\encws{{48}}\\def\\cropS{{{S}}}\n")
    return [ex, ey, 48]


if __name__ == "__main__":
    main()

"""GT-box-prompted proxy mask metrics for the EM posterior, vs the boxinst head.

Same protocol and zone formulas as boxinst/eval_masks.py (fill / corner / centre /
leak at the 128x128 mask grid, GT boxes on the test partition) so numbers are
directly comparable. No mask GT anywhere. Note: the EM mask is defined only on
cells overlapping its box (+half-cell pad), so `leak` is structurally ~0 for it —
read fill/corner/centre, not leak, when comparing against the head.

Usage:
    .venv/bin/python -m boxinst_commonality.evaluate \
        --baseline boxinst/artifacts/boxinst_s0.pt
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dapt.cache_features import cache_key
from dapt.data.cohort import REPO
from boxinst.commonality import load_split
from boxinst_commonality.em import (ART, G, MODEL_PATH, EMModel,
                                    draw_active_cells, draw_grid, out_dir)

MASK_RES, MASK_STRIDE = 128, 4
CMAP = [plt.get_cmap("tab20")(i)[:3] for i in range(20)]


def largest_cc_prob(prob, thr=0.5):
    """Zero every >=thr pixel outside the largest 4-connected component.

    One instance = one crown: a box straddling two trees gives a bimodal
    posterior whose thr level line pinches into a figure-8 at the saddle;
    keeping the largest lobe makes every instance mask a single simple blob.
    Sub-thr soft values are left untouched (calibration is unchanged).
    """
    from scipy import ndimage
    b = prob >= thr
    if b.any():
        lab, n = ndimage.label(b)                    # 4-connectivity
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            prob = np.where(b & (lab != sizes.argmax()), 0.0, prob)
    return prob


def resolve_overlaps(probs):
    """Make instance masks mutually exclusive (list of (H,W) soft maps).

    Two overlapping detections otherwise draw crossing boundaries that read as
    a figure-8. Each contested pixel (>=0.5 in several instances) is kept only
    by the instance with the highest posterior — pass the list in descending
    detection-score order so argmax ties break toward the stronger detection —
    then each mask is re-cleaned to its largest component.
    """
    if len(probs) < 2:
        return probs
    P = np.stack(probs)
    owner = P.argmax(0)
    out = []
    for j in range(len(probs)):
        pj = np.where((P[j] >= 0.5) & (owner != j), 0.0, P[j])
        out.append(largest_cc_prob(pj))
    return out


def prob_map_128(model, zn, box, H0, W0):
    """EM posterior for one box -> (128,128) soft prob map (0 outside box cells),
    cleaned to a single blob (see largest_cc_prob)."""
    idx, r = model.box_posterior(zn, box, H0, W0)
    g = model.grid
    grid = np.zeros(g * g, np.float32)
    grid[idx] = r
    prob = np.array(Image.fromarray(grid.reshape(g, g)).resize(
        (MASK_RES, MASK_RES), Image.BILINEAR))
    return largest_cc_prob(prob)


def proxy_stats(prob, box):
    """Same zones as boxinst/eval_masks.py; prob (128,128), box in image px."""
    yy, xx = np.mgrid[0:MASK_RES, 0:MASK_RES].astype(np.float32)
    x0, y0, x1, y1 = np.asarray(box, np.float32) / MASK_STRIDE
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    inb = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).astype(np.float32)
    u, v = (xx - x0) / w, (yy - y0) / h
    corner = (((u < .25) | (u > .75)) & ((v < .25) | (v > .75))).astype(
        np.float32) * inb
    centre = ((u > .25) & (u < .75) & (v > .25) & (v < .75)).astype(
        np.float32) * inb
    big = ((xx >= x0 - .1 * w) & (xx < x1 + .1 * w) &
           (yy >= y0 - .1 * h) & (yy < y1 + .1 * h)).astype(np.float32)
    mb = (prob >= 0.5).astype(np.float32)
    return {
        "fill": (mb * inb).sum() / max(inb.sum(), 1),
        "corner": (prob * corner).sum() / max(corner.sum(), 1),
        "centre": (prob * centre).sum() / max(centre.sum(), 1),
        "leak": (mb * (1 - big)).sum() / max(mb.sum(), 1),
    }


def eval_em(model, split, boxes_all):
    stats = {k: [] for k in ("fill", "corner", "centre", "leak")}
    for p, t in split["tiles"].items():
        if t["partition"] != "test" or t["n_boxes"] == 0:
            continue
        zn = model.project_tile(p)
        W0, H0 = Image.open(p).size
        for b in boxes_all.get(p, []):
            s = proxy_stats(prob_map_128(model, zn, b, H0, W0), b)
            for k, v in s.items():
                stats[k].append(v)
    return {k: round(float(np.mean(v)), 3) for k, v in stats.items()}, \
        len(stats["fill"])


def render_gt_prompted(model, split, boxes_all, per_domain=2, tag=""):
    od = out_dir(tag)
    chosen = []
    test = [p for p, t in split["tiles"].items() if t["partition"] == "test"]
    for dom in ("WON", "BRU", "NEON"):
        cand = [p for p in test if split["tiles"][p]["domain"] == dom]
        cand.sort(key=lambda p: -split["tiles"][p]["n_boxes"])
        chosen += cand[:per_domain]
    for p in chosen:
        zn = model.project_tile(p)
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8 * H0 / W0), dpi=140)
        ax.imshow(img)
        draw_grid(ax, H0, W0, model.s)
        for b in boxes_all.get(p, []):
            idx, r = model.box_posterior(zn, b, H0, W0)
            draw_active_cells(ax, idx, r, model.grid, model.s)
            prob = prob_map_128(model, zn, b, H0, W0)
            rs = np.array(Image.fromarray(prob).resize(
                (MASK_RES * MASK_STRIDE,) * 2, Image.BILINEAR))[:H0, :W0]
            if (rs >= 0.5).any():
                ax.contour(rs, levels=[0.5], colors=["#22dd22"],
                           linewidths=1.4)
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, ec="white", lw=0.6,
                                       alpha=0.8))
        ax.axis("off")
        name = cache_key(p)
        ax.set_title("EM posterior boundaries, GT-box prompted (all masks "
                     "green: no detection, so no TP/FP/FN here; white=GT box, "
                     f"purple=active {model.s}px cells, grid={model.s}px)  "
                     f"{name}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(od, f"gtprompt_{name}.png"))
        plt.close(fig)
        print(f"rendered {od}/gtprompt_{name}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="*", default=[],
                    help="boxinst .pt checkpoints for same-protocol comparison")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    split, boxes_all = load_split()
    model = EMModel(args.model)
    print(f"model={os.path.basename(args.model)} "
          f"feat_dir={os.path.basename(model.feat_dir)}", flush=True)

    rows = []
    s, n = eval_em(model, split, boxes_all)
    rows.append(("EM posterior (this work)", s, n))
    for c in args.baseline:
        import torch
        from dapt.backbone import pick_device
        from boxinst.eval_masks import eval_ckpt
        torch.manual_seed(0)
        sb, nb = eval_ckpt(os.path.join(REPO, c), pick_device(args.device))
        rows.append((os.path.basename(c).replace(".pt", "") + " head", sb, nb))

    print(f"\n{'method':34s} {'fill':>6} {'corner':>7} {'centre':>7} {'leak':>6}")
    for name, s, n in rows:
        print(f"{name:34s} {s['fill']:6.3f} {s['corner']:7.3f} "
              f"{s['centre']:7.3f} {s['leak']:6.3f}   (n={n})")
    print("(leak is structurally ~0 for the EM: masks only exist inside "
          "box+half-cell)")
    rep_name = os.path.basename(args.model).replace("em_model", "proxy_report"
                                                    ).replace(".npz", ".json")
    json.dump({name: {**s, "n": n} for name, s, n in rows},
              open(os.path.join(ART, rep_name), "w"), indent=2)

    if not args.no_render:
        tag = os.path.basename(args.model).replace(
            "em_model", "").replace(".npz", "")
        render_gt_prompted(model, split, boxes_all, tag=tag)


if __name__ == "__main__":
    main()

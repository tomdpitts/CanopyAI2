"""Concept figure for the Introduction: inferring a subject from a list of words (top) is like inferring a
crown from a set of box crops (bottom).

  .venv/bin/python boxinst_commonality_tcd_04/ablation/results/figs/make_concept.py [--sheet OUT.png]

Every crop and the crown are picked by a fixed rule from OAM-TCD TRAIN tiles (no test tile is read):
  candidates = tree annotations (category_id 2) whose GT box is 80-200 px a side, aspect 0.75-1.33, at least
  8 px from the tile edge, and overlapping no other annotation's box (tree or canopy). Per candidate, on its
  box crop: fill = GT polygon area / box area; exg_in = mean excess green 2G-R-B inside the polygon;
  grey_out = fraction of box pixels that are outside the polygon AND grey (HSV S < 0.12, 0.30 < V < 0.92:
  asphalt, concrete, roofs, cars); edge_out = fraction of box pixels outside the polygon with luminance
  gradient > 30 (hard man-made edges); sharp = variance of the luminance Laplacian (image sharpness).
  Cached in concept_commonality_features.json (2 unreadable tiles skipped and listed there).
  confounders = N_CONF candidates from distinct tiles with the highest grey_out * edge_out, among those with
  fill >= 0.5, exg_in above the candidate median and sharp above the candidate median (the crown is clearly a
  vegetated tree that dominates its box, so the box is an unambiguous GT tree box, and the crop is not
  blurred), placed at grid positions CONF_POS. (grey_out alone was tried first and rejected: its top hits
  were trees on grey bare ground, not man-made confounders.);
  others = N_CROPS - N_CONF candidates drawn with numpy seed SEED from the rest (box crop, no margin: what box
  supervision sees);
  crown = among candidates with >= 40 vertices and isoperimetric quotient 4*pi*A/P^2 within Q_BAND (rounded
  but lobed), the one with the highest exg_in, shown as an RGB cutout of its own tile clipped to its GT
  polygon low-passed to FOURIER_K Fourier harmonics (removes the pixel-grid staircase of the annotation;
  schematic only), alpha rendered at SS x supersampling for an anti-aliased edge.
  2026-11-28 (Tom) overrides: grid positions OVERRIDE are replaced BY EYE for diversity (a star-shaped palm and a
  leafless crown, from contact sheets of the lowest-isoperimetric-quotient and lowest-exg_in candidates); the crown
  is shown as a SQUARE crop centred on its box (side = longer box side + 2 x 4 px pad, i.e. the old cutout's extent),
  background dimmed to DIM, GT polygon (Fourier-smoothed as above) outlined in TEAL.
--sheet writes a contact sheet of the top 12 confounder candidates (for checking the rule, not for selection).
Writes concept_commonality.{pdf,png} and concept_commonality_meta.json (chosen tiles and annotation ids).
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TRAIN = os.path.join(REPO, "data", "tcd", "train")
FEAT = os.path.join(HERE, "concept_commonality_features.json")
FEAT_V = 2
N_CROPS, N_CONF, CONF_POS, SEED = 10, 2, (2, 6), 0
OVERRIDE = {7: 88903, 8: 167953}   # 2026-11-28 Tom: by eye, star-shaped palm (was 206439), leafless crown (was 297828)
DIM, TEAL = 0.85, "#11a3a3"
Q_BAND, FOURIER_K, UPS, SS = (0.83, 0.87), 24, 4, 4
GROUPS = [["ash", "trunk", "dirt"], ["car", "leaves", "birch"], ["canopy", "branches", "grass"]]
INK, INK2 = "#1a1a1a", "#555555"
# Layout is in units of a 6.0-wide canvas; the figure is SAVED at its printed width (MDPI journal \textwidth
# = 210 - 58.7 - 12.7 mm = 13.86 cm, included at width=\linewidth), so font sizes below are the printed sizes.
PHYS_W = 13.86 / 2.54
XOUT = 5.45                      # x centre of the right-hand outputs ("tree crown", crown cutout)
# whitening sphere, as in Fig. 4 panel (1) (lace_masker_fit.tex): colours from arch/arch_preamble.tex
FRZ, CROWN_C, SOIL, INK_T = "#3B6EA8", "#5E9E4E", "#C9A26B", "#2B2B2B"
# top-row intermediate: the same words, tree words gathered (crown colour), the rest adrift (soil colour)
CLOUD = [("canopy", 0.10, 0.30, CROWN_C), ("leaves", -0.12, 0.17, CROWN_C), ("ash", 0.25, 0.17, CROWN_C),
         ("branches", 0.10, 0.04, CROWN_C), ("birch", -0.10, -0.09, CROWN_C), ("trunk", 0.25, -0.09, CROWN_C),
         ("dirt", -0.32, -0.24, SOIL), ("car", 0.0, -0.34, SOIL), ("grass", 0.30, -0.26, SOIL)]
SPHERE_CROWN = [(20, 0.35), (40, 0.5), (55, 0.3), (75, 0.45), (30, 0.6)]
SPHERE_SOIL = [(200, -0.3), (230, -0.5), (250, -0.2), (215, 0.1), (260, -0.6), (190, 0.25)]


def _mix(c, w, other="#ffffff"):
    """TikZ c!w: w% of colour c, rest other."""
    a = np.array(matplotlib.colors.to_rgb(c)); b = np.array(matplotlib.colors.to_rgb(other))
    return tuple(w / 100 * a + (1 - w / 100) * b)


def sphere(ax, cx, cy, r, phys_r_in=None):
    """Fig. 4 (1) sphere: ball shading of frz!10, ink!45 rim, dashed ink!30 equator, crown/soil points."""
    from matplotlib.patches import Circle, Ellipse
    n = 400
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(1, -1, n))
    t = np.clip(np.hypot(u + 0.35, v - 0.35) / 1.35, 0, 1)            # highlight up-left, as TikZ ball shading
    base, white = np.array(_mix(FRZ, 10)), np.ones(3)
    dark = base * 0.62
    img = np.where(t[..., None] < 0.55, white + (base - white) * (t[..., None] / 0.55),
                   base + (dark - base) * ((t[..., None] - 0.55) / 0.45))
    clip = Circle((cx, cy), r, transform=ax.transData)
    im = ax.imshow(img, extent=(cx - r, cx + r, cy - r, cy + r), interpolation="bilinear", zorder=2)
    im.set_clip_path(clip)
    ax.add_patch(Circle((cx, cy), r, fc="none", ec=_mix(INK_T, 45), lw=0.5, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * 0.16 / 0.55, fc="none", ec=_mix(INK_T, 30), lw=0.4,
                         ls=(0, (2.5, 1.5)), zorder=3))
    phys_r_in = r * PHYS_W / 6.0 if phys_r_in is None else phys_r_in
    ms = 2 * (1.2 / (0.55 / 2.54 * 72)) * (phys_r_in * 72)        # dot radius 1.2 pt per 0.55 cm of radius
    for pts, col in ((SPHERE_CROWN, CROWN_C), (SPHERE_SOIL, SOIL)):
        for a, b in pts:
            px, py = sphere_xy(cx, cy, r, a, b)
            # 2026-11-28 Tom: thin vectors from the origin to each unit-sphere point
            ax.plot([cx, px], [cy, py], color=col, lw=0.4, solid_capstyle="round", zorder=3.5)
            ax.plot(px, py, "o", ms=ms, color=col, mec="none", zorder=4)


SPHERE_R_CM, SPHERE_PAD = 0.55, 1.06
# 2026-11-28 (Tom): two 16 px patches (one DINOv3 patch) in grid crop LINK_POS, each joined by a flat dashed line to a
# sphere point: background patch (bottom right of the crop) -> soil point LINK_SOIL, crown patch -> nearest crown point
# LINK_CROWN. Patch rows are set so the patch centre sits at the point's height; columns picked by eye.
LINK_POS, PATCH_PX = 4, 16
LINK_SOIL, LINK_CROWN = (190, 0.25), (75, 0.45)
LINK_COL = {"soil": 128, "crown": 112}     # patch-centre column (crop px)
LINK_BRIGHT = {"soil": "#FFC97A", "crown": "#9BFF6E"}   # 2026-11-28 Tom: brighter square outlines on the crop


def sphere_xy(cx, cy, r, a, b):
    return cx + r * np.cos(np.radians(a)) * np.sqrt(1 - b * b), cy + r * b


def write_sphere_standalone(path):
    """2026-11-28 (Tom): the same sphere (with origin vectors) as a standalone transparent PDF for Fig. 4 panel (1)
    (lace_masker_fit.tex includes it at width 2 * SPHERE_R_CM * SPHERE_PAD cm, replacing its TikZ sphere)."""
    d = 2 * SPHERE_R_CM * SPHERE_PAD / 2.54
    f = plt.figure(figsize=(d, d))
    a = f.add_axes([0, 0, 1, 1]); a.set_xlim(-SPHERE_PAD, SPHERE_PAD); a.set_ylim(-SPHERE_PAD, SPHERE_PAD); a.axis("off")
    sphere(a, 0, 0, 1, phys_r_in=SPHERE_R_CM / 2.54)
    f.savefig(path, transparent=True); plt.close(f)


def boxes_overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def ring(seg):
    return np.asarray(max(seg, key=len), np.float64).reshape(-1, 2)


def iso_quotient(p):
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])))
    return 4 * np.pi * area / per ** 2 if per > 0 else 0.0


def fourier_smooth(p, k, n=1024):
    """Resample the closed polygon to n points by arc length and keep harmonics |f| <= k."""
    d = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    t = np.concatenate([[0], np.cumsum(d)])
    u = np.linspace(0, t[-1], n, endpoint=False)
    z = np.interp(u, t, np.append(p[:, 0], p[0, 0])) + 1j * np.interp(u, t, np.append(p[:, 1], p[0, 1]))
    F = np.fft.fft(z)
    f = np.fft.fftfreq(n, 1.0 / n)
    F[np.abs(f) > k] = 0
    z = np.fft.ifft(F)
    return np.stack([z.real, z.imag], 1)


def candidates():
    out = []
    for mf in sorted(glob.glob(os.path.join(TRAIN, "*_meta.json"))):
        m = json.load(open(mf))
        anns = json.loads(m["coco_annotations"]) if isinstance(m["coco_annotations"], str) else m["coco_annotations"]
        W, H = m["width"], m["height"]
        for a in anns:
            if a["category_id"] != 2 or not isinstance(a.get("segmentation"), list) or not a["segmentation"]:
                continue
            x, y, w, h = a["bbox"]
            if not (80 <= w <= 200 and 80 <= h <= 200 and 0.75 <= w / h <= 1.33):
                continue
            if x < 8 or y < 8 or x + w > W - 8 or y + h > H - 8:
                continue
            if any(o is not a and boxes_overlap(a["bbox"], o["bbox"]) for o in anns):
                continue
            out.append(dict(tile=os.path.basename(mf)[:-10], ann_id=a["id"], bbox=[float(v) for v in a["bbox"]],
                            poly=ring(a["segmentation"])))
    return out


def box_xywh(c):
    return tuple(int(round(v)) for v in c["bbox"])


def poly_mask(c, size):
    x, y, _, _ = box_xywh(c)
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(v) for v in c["poly"] - [x, y]], fill=1)
    return np.asarray(m, bool)


def features(cands):
    """fill / exg_in / grey_out per candidate (cached; keyed by ann_id)."""
    if os.path.exists(FEAT):
        f = json.load(open(FEAT))
        if f.get("n") == len(cands) and f.get("v") == FEAT_V:  # n counts candidates before unreadable tiles are dropped
            return {int(k): v for k, v in f["feat"].items()}
    by_tile = defaultdict(list)
    for c in cands:
        by_tile[c["tile"]].append(c)
    feat, bad = {}, []
    for i, (t, cs) in enumerate(sorted(by_tile.items())):
        try:
            im = np.asarray(Image.open(os.path.join(TRAIN, t + ".tif")).convert("RGB"))
        except Exception:
            bad.append(t)
            continue
        for c in cs:
            x, y, w, h = box_xywh(c)
            a = im[y:y + h, x:x + w].astype(np.float64)
            m = poly_mask(c, (a.shape[1], a.shape[0]))
            mx, mn = a.max(2), a.min(2)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
            grey = (sat < 0.12) & (mx / 255 > 0.30) & (mx / 255 < 0.92)
            exg = 2 * a[..., 1] - a[..., 0] - a[..., 2]
            L = a.mean(2)
            gy, gx = np.gradient(L)
            mag = np.hypot(gx, gy)
            lap = L[2:, 1:-1] + L[:-2, 1:-1] + L[1:-1, 2:] + L[1:-1, :-2] - 4 * L[1:-1, 1:-1]
            feat[c["ann_id"]] = dict(fill=float(m.mean()), exg_in=float(exg[m].mean()) if m.any() else -1e9,
                                     grey_out=float((grey & ~m).mean()), edge_out=float(((mag > 30) & ~m).mean()),
                                     sharp=float(lap.var()))
        if i % 200 == 0:
            print(f"features: tile {i + 1}/{len(by_tile)}", flush=True)
    print(f"features: {len(bad)} unreadable tiles skipped", flush=True)
    json.dump(dict(n=len(cands), v=FEAT_V, unreadable_tiles=bad, feat=feat), open(FEAT, "w"))
    return feat


def crop(c):
    x, y, w, h = box_xywh(c)
    return np.asarray(Image.open(os.path.join(TRAIN, c["tile"] + ".tif")).convert("RGB").crop((x, y, x + w, y + h)))


def cutout(c):
    """RGBA crop of c's box at UPS x, alpha = Fourier-smoothed GT polygon rendered at UPS*SS and downsampled."""
    x, y, w, h = box_xywh(c)
    pad = 4
    im = Image.open(os.path.join(TRAIN, c["tile"] + ".tif")).convert("RGB").crop((x - pad, y - pad, x + w + pad, y + h + pad))
    im = im.resize((im.width * UPS, im.height * UPS), Image.LANCZOS)
    p = (fourier_smooth(c["poly"], FOURIER_K) - [x - pad, y - pad]) * UPS * SS
    m = Image.new("L", (im.width * SS, im.height * SS), 0)
    ImageDraw.Draw(m).polygon([tuple(v) for v in p], fill=255)
    out = im.convert("RGBA"); out.putalpha(m.resize(im.size, Image.LANCZOS))
    return np.asarray(out)


def tile_highlight(c, pad=4):
    """Square crop around c's box at UPS x, background brightness x DIM, crown (smoothed GT polygon) at full
    brightness; returns (rgb, polygon in output pixel coords)."""
    x, y, w, h = box_xywh(c)
    side = max(w, h) + 2 * pad
    x0, y0 = x + w // 2 - side // 2, y + h // 2 - side // 2
    im = Image.open(os.path.join(TRAIN, c["tile"] + ".tif")).convert("RGB").crop((x0, y0, x0 + side, y0 + side))
    im = im.resize((side * UPS, side * UPS), Image.LANCZOS)
    p = (fourier_smooth(c["poly"], FOURIER_K) - [x0, y0]) * UPS
    m = Image.new("L", (im.width * SS, im.height * SS), 0)
    ImageDraw.Draw(m).polygon([tuple(v) for v in p * SS], fill=255)
    a = np.asarray(m.resize(im.size, Image.LANCZOS), np.float64)[..., None] / 255
    rgb = np.asarray(im, np.float64) * (DIM + (1 - DIM) * a)
    return np.clip(rgb, 0, 255).astype(np.uint8), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()

    cands = candidates()
    print(f"candidates: {len(cands)}", flush=True)
    feat = features(cands)
    cands = [c for c in cands if c["ann_id"] in feat]          # drop candidates on unreadable tiles
    med = float(np.median([feat[c["ann_id"]]["exg_in"] for c in cands]))
    sharp_med = float(np.median([feat[c["ann_id"]]["sharp"] for c in cands]))
    F = lambda c: feat[c["ann_id"]]
    pool = sorted((c for c in cands if F(c)["fill"] >= 0.5 and F(c)["exg_in"] > med and F(c)["sharp"] > sharp_med),
                  key=lambda c: -F(c)["grey_out"] * F(c)["edge_out"])
    conf, seen = [], set()
    for c in pool:
        if c["tile"] not in seen:
            conf.append(c); seen.add(c["tile"])
        if len(conf) == 12:
            break
    if args.sheet:
        fig, axs = plt.subplots(2, 6, figsize=(12, 4.4))
        for ax, c in zip(axs.flat, conf):
            ax.imshow(crop(c)); ax.axis("off")
            ax.set_title(f"{c['tile']} {c['ann_id']}\ng {feat[c['ann_id']]['grey_out']:.2f} e {feat[c['ann_id']]['edge_out']:.2f}", fontsize=7)
        fig.savefig(args.sheet, dpi=110, bbox_inches="tight"); print("sheet", args.sheet, flush=True)
    conf = conf[:N_CONF]
    used = {c["ann_id"] for c in conf}
    rest = [c for c in cands if c["ann_id"] not in used]
    rng = np.random.default_rng(SEED)
    others = [rest[i] for i in rng.choice(len(rest), N_CROPS - N_CONF, replace=False)]
    picks, it = [], iter(others)
    for i in range(N_CROPS):
        picks.append(conf[CONF_POS.index(i)] if i in CONF_POS else next(it))
    byid = {c["ann_id"]: c for c in cands}
    replaced = {i: picks[i]["ann_id"] for i in OVERRIDE}
    for i, a in OVERRIDE.items():
        picks[i] = byid[a]

    band = [c for c in cands if len(c["poly"]) >= 40 and Q_BAND[0] <= iso_quotient(c["poly"]) <= Q_BAND[1]]
    crown = max(band, key=lambda c: feat[c["ann_id"]]["exg_in"])
    print(f"crown: {crown['tile']} ann {crown['ann_id']} (band {len(band)})", flush=True)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42})
    FW, FH = 6.0, 2.92
    fig = plt.figure(figsize=(PHYS_W, FH * PHYS_W / FW))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, FW); ov.set_ylim(0, FH); ov.axis("off")

    def arrow(y, x0=3.35, x1=4.85):
        ov.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.0, color=INK))

    def label(x, y, txt):
        ov.text(x, y, txt, ha="center", va="center", fontsize=7, color=INK2)

    def grey_box(yc, h):
        ov.add_patch(FancyBboxPatch((0.25, yc - h / 2), 2.9, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f5f5f5", ec=INK2, lw=0.8))

    # top: words -> subject
    ty = 2.36
    grey_box(ty, 0.96)                 # 2026-11-28: 0.84 -> 0.96 for the 4th (ellipsis) row
    # one white card per group of words, mirroring one box crop per tree below
    gw, gh, gg = 0.70, 0.74, 0.10
    gx0 = 1.7 - (len(GROUPS) * gw + (len(GROUPS) - 1) * gg + 0.22) / 2
    for i, grp in enumerate(GROUPS):
        cx = gx0 + i * (gw + gg)
        ov.add_patch(FancyBboxPatch((cx, ty - gh / 2), gw, gh, boxstyle="round,pad=0.0,rounding_size=0.04",
                                    fc="white", ec=INK2, lw=0.4))
        # 2026-11-28 Tom: an ellipsis under each group's words (longer lists); 4 rows fitted inside the same card
        for j, w in enumerate(grp + ["…"]):
            ov.text(cx + gw / 2, ty + (1.5 - j) * 0.155, w, ha="center", va="center", fontsize=8, color=INK,
                    style="italic" if w != "…" else "normal")
    ov.text(gx0 + len(GROUPS) * (gw + gg) + 0.05, ty, "…", ha="center", va="center", fontsize=8, color=INK)
    arrow(ty, 3.30, 3.70)
    for w, dx, dy, col in CLOUD:      # free-floating: no space is implied, only that the tree words gather
        ov.text(4.17 + dx, ty + dy, w, ha="center", va="center", fontsize=7, color=col, style="italic")
    arrow(ty, 4.62, 4.99)
    ov.text(XOUT, ty, "“tree”", ha="center", va="center", fontsize=9, color=INK, style="italic")
    label(1.7, ty - 0.48 - 0.11, "bags of words")
    label(4.17, ty - 0.48 - 0.11, "semantic clusters")
    label(XOUT, ty - 0.48 - 0.11, "shared subject")

    ov.plot([0.25, FW - 0.25], [1.66, 1.66], color="#9e9e9e", lw=0.5)

    # bottom: 2 x 5 box crops (then an ellipsis, as in the word list) -> crown cutout
    by, t, g, ncol = 0.92, 0.44, 0.06, 5
    grey_box(by, 1.16)
    x0 = 1.7 - (ncol * t + (ncol - 1) * g + 0.22) / 2
    for i, c in enumerate(picks):
        r, k = divmod(i, ncol)
        yb = by + g / 2 - r * (t + g)
        ax = fig.add_axes([(x0 + k * (t + g)) / FW, yb / FH, t / FW, t / FH])
        ax.imshow(crop(c), interpolation="lanczos", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK2); sp.set_linewidth(0.4)
    ov.text(x0 + ncol * (t + g) + 0.05, by, "…", ha="center", va="center", fontsize=8, color=INK)
    arrow(by, 3.30, 3.70)
    SX, SR = 4.17, 0.36
    sphere(ov, SX, by, SR)  # 2026-11-28 Tom: a touch larger (was 0.30)
    # patch -> sphere-point links, on a transparent top layer so they run over the crop and the sphere
    top = fig.add_axes([0, 0, 1, 1]); top.set_xlim(0, FW); top.set_ylim(0, FH); top.axis("off"); top.patch.set_alpha(0)
    lr, lk = divmod(LINK_POS, ncol)
    cW, cH = crop(picks[LINK_POS]).shape[1], crop(picks[LINK_POS]).shape[0]
    cx0, cy0 = x0 + lk * (t + g), by + g / 2 - lr * (t + g)
    side = PATCH_PX * t / ((cW + cH) / 2)
    dot_r = (2 * (1.2 / (0.55 / 2.54 * 72)) * (SR * PHYS_W / 6.0 * 72)) / 2 / 72 * FW / PHYS_W
    links = []
    for kind, (a, b), col in (("soil", LINK_SOIL, SOIL), ("crown", LINK_CROWN, CROWN_C)):
        dx, dy = sphere_xy(SX, by, SR, a, b)
        px = cx0 + (LINK_COL[kind] + 0.5) / cW * t
        top.add_patch(plt.Rectangle((px - side / 2, dy - side / 2), side, side, fc="none", ec=LINK_BRIGHT[kind], lw=0.9))
        top.plot([px + side / 2, dx - dot_r], [dy, dy], color=col, lw=0.6, ls=(0, (3, 2)))
        links.append(dict(kind=kind, point=(a, b), patch_col=LINK_COL[kind], patch_row=round((1 - (dy - cy0) / t) * cH - 0.5, 1)))
    label(1.7, by - 0.58 - 0.12, "GT boxes")
    label(4.17, by - 0.58 - 0.12, "DINOv3 features")
    label(XOUT, by - 0.58 - 0.12, "isolated crown")
    arrow(by, 4.62, 4.99)
    s = 0.72
    rgb, poly = tile_highlight(crown)
    h, w = rgb.shape[:2]
    cw, ch = (s, s * h / w) if w >= h else (s * w / h, s)
    ax = fig.add_axes([(XOUT - cw / 2) / FW, (by - ch / 2) / FH, cw / FW, ch / FH])
    ax.imshow(rgb, interpolation="lanczos", aspect="auto")
    ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), color=TEAL, lw=0.6)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(INK2); sp.set_linewidth(0.4)

    write_sphere_standalone(os.path.join(HERE, "arch", "concept_sphere.pdf"))
    stem = os.path.join(HERE, "concept_commonality")
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300)
    rec = lambda c: dict(tile=c["tile"], ann_id=c["ann_id"], bbox=c["bbox"], **feat[c["ann_id"]])
    json.dump(dict(rule=__doc__.split("--sheet")[0].strip(), n_candidates=len(cands), seed=SEED,
                   conf_pos=CONF_POS, links=links, link_pos=LINK_POS, override=OVERRIDE, override_replaced=replaced, dim=DIM, teal=TEAL, q_band=Q_BAND, fourier_k=FOURIER_K, exg_in_median=med,
                   crops=[dict(pos=i, confounder=i in CONF_POS, **rec(c)) for i, c in enumerate(picks)],
                   crown=dict(iso_quotient=iso_quotient(crown["poly"]), **rec(crown))),
              open(stem + "_meta.json", "w"), indent=1)
    print("wrote", stem + ".{pdf,png}", flush=True)


if __name__ == "__main__":
    main()

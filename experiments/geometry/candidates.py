"""Training-free crown candidate generator.

The filter under test needs a candidate pool that (a) recovers most real crowns and
(b) is polluted with shadow-induced false positives -- dark blobs that are actually
cast shadows, not trees. We get both from a multi-scale blob detector run at a low
threshold, with ZERO learned parameters, so nothing in the pipeline can overfit.

Cues (both standardised per image, combined by max):
  * darkness   = -luminance : arid shrubs AND their shadows are dark blobs on bright
                 soil, so this cue proposes crowns *and* shadows -- exactly the
                 confusable pool we want.
  * greenness  = 2G-R-B     : recovers lit green crowns (NEON forest) that darkness
                 misses.
Blobs are found as local maxima of a Difference-of-Gaussians response at a few
scales; a light greedy NMS removes duplicates. Score = DoG response (the detector
"confidence" used for PR curves). None of these knobs is tuned on labels; they are
set once from the crown-size prior (median ~56 px) and frozen.
"""
from __future__ import annotations

import os, sys
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import luminance

# Frozen from the crown-size prior (median sqrt-area ~56 px, p10-p90 37-84 px).
SCALES_SIGMA = (5.0, 8.0, 12.0)     # DoG inner sigma per scale
DOG_RATIO = 1.6                     # outer/inner sigma
BOX_FACTOR = 4.0                    # candidate box side = BOX_FACTOR * sigma
NMS_DIST_FACTOR = 1.2               # suppress maxima within this * sigma
PEAK_MIN = 0.15                     # low absolute floor on standardised response
MAX_PER_IMAGE = 400                 # cap (low thresh -> many); keeps FP pool rich


@dataclass
class Candidate:
    cx: float
    cy: float
    box: np.ndarray      # xyxy
    score: float
    sigma: float


def _standardize(f):
    med = np.median(f)
    mad = np.median(np.abs(f - med)) * 1.4826
    scale = mad if mad > 1e-6 else (f.std() + 1e-6)
    return (f - med) / scale


def _dog(field, sigma):
    return gaussian_filter(field, sigma) - gaussian_filter(field, sigma * DOG_RATIO)


def _peaks(resp, sigma, valid):
    """Local maxima of resp above PEAK_MIN, inside valid mask."""
    fp = maximum_filter(resp, size=int(max(3, NMS_DIST_FACTOR * sigma)))
    ys, xs = np.where((resp == fp) & (resp > PEAK_MIN) & valid)
    return xs, ys, resp[ys, xs]


def generate(rgb: np.ndarray, valid: np.ndarray) -> list[Candidate]:
    lum = luminance(rgb)
    darkness = _standardize(-lum)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    greenness = _standardize(2.0 * g - r - b)

    raw = []
    for sigma in SCALES_SIGMA:
        for cue in (darkness, greenness):
            resp = _dog(cue, sigma)
            xs, ys, sc = _peaks(resp, sigma, valid)
            for x, y, s in zip(xs, ys, sc):
                raw.append((float(s), float(x), float(y), float(sigma)))
    raw.sort(reverse=True)   # highest response first for greedy NMS

    kept: list[Candidate] = []
    for s, x, y, sigma in raw:
        dup = False
        for c in kept:
            if abs(c.cx - x) < NMS_DIST_FACTOR * max(sigma, c.sigma) and \
               abs(c.cy - y) < NMS_DIST_FACTOR * max(sigma, c.sigma):
                dup = True
                break
        if dup:
            continue
        half = BOX_FACTOR * sigma / 2.0
        kept.append(Candidate(cx=x, cy=y,
                              box=np.array([x - half, y - half, x + half, y + half],
                                           dtype=np.float32),
                              score=s, sigma=sigma))
        if len(kept) >= MAX_PER_IMAGE:
            break
    return kept


def match_to_gt(cands: list[Candidate], gt_boxes: np.ndarray):
    """Greedy center-in-box matching, highest score first. Returns is_tp (bool array
    aligned to cands) and n_matched_gt. A candidate is a TP if its centre lies in an
    as-yet-unmatched GT box (robust to approximate candidate box scale)."""
    order = sorted(range(len(cands)), key=lambda i: -cands[i].score)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    is_tp = np.zeros(len(cands), dtype=bool)
    for i in order:
        c = cands[i]
        inside = [(j, gt_boxes[j]) for j in range(len(gt_boxes)) if not matched[j]
                  and gt_boxes[j][0] <= c.cx <= gt_boxes[j][2]
                  and gt_boxes[j][1] <= c.cy <= gt_boxes[j][3]]
        if inside:
            # nearest-center among containing boxes
            j = min(inside, key=lambda t: (c.cx - (t[1][0] + t[1][2]) / 2) ** 2
                    + (c.cy - (t[1][1] + t[1][3]) / 2) ** 2)[0]
            matched[j] = True
            is_tp[i] = True
    return is_tp, int(matched.sum())


if __name__ == "__main__":
    import os, sys, collections
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.geometry.gdata import load_records, cohort, load_rgb, valid_mask
    recs = cohort(load_records())
    by = collections.defaultdict(lambda: [0, 0, 0, 0])  # ncand, ntp, ngt, ndark_fp
    for r in recs:
        rgb = load_rgb(r.path); vm = valid_mask(rgb)
        cands = generate(rgb, vm)
        is_tp, nm = match_to_gt(cands, r.boxes)
        lum = luminance(rgb)
        zl = _standardize(lum)
        # a shadow-like FP: false positive whose centre pixel is dark
        ndark = 0
        for c, tp in zip(cands, is_tp):
            if not tp:
                yy, xx = int(round(c.cy)), int(round(c.cx))
                if 0 <= yy < lum.shape[0] and 0 <= xx < lum.shape[1] and zl[yy, xx] < -0.3:
                    ndark += 1
        agg = by[r.domain]
        agg[0] += len(cands); agg[1] += int(is_tp.sum()); agg[2] += len(r.boxes); agg[3] += ndark
    print(f"{'dom':5} {'cand':>6} {'TP':>5} {'GT':>5} {'recall':>7} {'prec':>6} {'darkFP':>7}")
    for d in ("WON", "BRU", "NEON"):
        nc, ntp, ngt, ndf = by[d]
        print(f"{d:5} {nc:6d} {ntp:5d} {ngt:5d} {ntp/max(ngt,1):7.2f} {ntp/max(nc,1):6.2f} {ndf:7d}")

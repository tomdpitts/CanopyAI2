"""Parameter-free directional-contrast probe: does the shadow signal exist?

For every ground-truth crown we sample standardised luminance along the annotated
shadow-displacement ray (crown -> +u) and along the opposite sun-ward ray
(crown -> -u), over a bracket of offsets, staying inside the valid (non-padding)
mask. The signed contrast

    delta = lum(shadow side) - lum(sun side)

should be **negative** if the convention is right and cast shadows are real (the
anti-solar side is darker). Three azimuth conditions give the built-in controls:

  correct  -> expect delta << 0
  shuffled -> expect delta ~ 0   (direction decorrelated from the scene)
  flipped  -> expect delta >> 0  (mirror: samples the lit side as if shadow)

This is the whole experiment in miniature and with zero tunable classifier: if
correct != shuffled clears the noise band, a geometric filter can work; if not,
no filter will. Reported per acquisition and with a bootstrap CI over crowns.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.geometry.gdata import load_records, cohort, load_rgb, luminance, valid_mask
from shadow_prior.geometry import azimuth_to_vector

D_MIN, D_MAX, D_STEPS = 15.0, 60.0, 10   # offset bracket (px), marginalise magnitude
SEED = 0


def standardize(lum):
    med = np.median(lum)
    mad = np.median(np.abs(lum - med)) * 1.4826
    scale = mad if mad > 1e-6 else (lum.std() + 1e-6)
    return (lum - med) / scale


def sample_bilinear(field, rr, cc, valid):
    """Bilinear sample at float (rr,cc); return (value, ok) with ok False if the
    2x2 support touches an invalid/out-of-bounds pixel."""
    H, W = field.shape
    r0, c0 = int(np.floor(rr)), int(np.floor(cc))
    if r0 < 0 or c0 < 0 or r0 + 1 >= H or c0 + 1 >= W:
        return 0.0, False
    if not (valid[r0, c0] and valid[r0 + 1, c0] and valid[r0, c0 + 1] and valid[r0 + 1, c0 + 1]):
        return 0.0, False
    fr, fc = rr - r0, cc - c0
    v = (field[r0, c0] * (1 - fr) * (1 - fc) + field[r0 + 1, c0] * fr * (1 - fc)
         + field[r0, c0 + 1] * (1 - fr) * fc + field[r0 + 1, c0 + 1] * fr * fc)
    return float(v), True


def crown_delta(zlum, valid, cx, cy, u_row, u_col):
    offs = np.linspace(D_MIN, D_MAX, D_STEPS)
    ds, us = [], []
    for d in offs:
        vs, oks = sample_bilinear(zlum, cy + d * u_row, cx + d * u_col, valid)
        vu, oku = sample_bilinear(zlum, cy - d * u_row, cx - d * u_col, valid)
        if oks and oku:
            ds.append(vs); us.append(vu)
    if not ds:
        return None
    return float(np.mean(ds) - np.mean(us))


def run(condition, recs, rng):
    """condition in {'correct','shuffled','flipped'}. Returns per-crown deltas with
    domain tags."""
    # within-acquisition azimuth shuffle keyed by image index
    az = np.array([r.azimuth for r in recs])
    dom = np.array([r.domain for r in recs])
    az_use = az.copy()
    if condition == "shuffled":
        for d in np.unique(dom):
            idx = np.flatnonzero(dom == d)
            az_use[idx] = az[idx][rng.permutation(idx.size)]
    elif condition == "flipped":
        az_use = az + np.pi

    deltas, domains, sizes = [], [], []
    for r, a in zip(recs, az_use):
        rgb = load_rgb(r.path)
        z = standardize(luminance(rgb))
        vm = valid_mask(rgb)
        u_row, u_col = azimuth_to_vector(a)
        for (x0, y0, x1, y1) in r.boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            dv = crown_delta(z, vm, cx, cy, u_row, u_col)
            if dv is not None:
                deltas.append(dv); domains.append(r.domain)
                sizes.append(float(np.sqrt((x1 - x0) * (y1 - y0))))
    return np.array(deltas), np.array(domains), np.array(sizes)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    recs = cohort(load_records())
    rng = np.random.default_rng(SEED)
    print(f"cohort {len(recs)} images, {sum(len(r.boxes) for r in recs)} crowns")
    print(f"offset bracket d in [{D_MIN},{D_MAX}] px x{D_STEPS}\n")
    results = {}
    for cond in ("correct", "shuffled", "flipped"):
        dvals, doms, szs = run(cond, recs, np.random.default_rng(SEED))
        results[cond] = (dvals, doms, szs)
        m, lo, hi = boot_ci(dvals)
        print(f"[{cond:8}] ALL   n={len(dvals):4d}  mean delta={m:+.4f}  "
              f"CI=({lo:+.4f},{hi:+.4f})  frac<0={np.mean(dvals<0):.2f}")
        for d in ("WON", "BRU", "NEON"):
            sel = dvals[doms == d]
            if len(sel):
                md, l, h = boot_ci(sel)
                print(f"           {d:5} n={len(sel):4d}  mean delta={md:+.4f}  CI=({l:+.4f},{h:+.4f})")
        print()

    # decisive contrast: correct vs shuffled (paired is not possible-different crowns kept;
    # both use same crowns actually -> can pair)
    dc = results["correct"][0]; ds = results["shuffled"][0]
    if len(dc) == len(ds):
        diff = dc - ds
        m, lo, hi = boot_ci(diff)
        print(f">>> correct - shuffled (paired): mean={m:+.4f} CI=({lo:+.4f},{hi:+.4f})  "
              f"{'SIGNAL (CI excludes 0)' if hi < 0 else 'no clear signal'}")


if __name__ == "__main__":
    main()

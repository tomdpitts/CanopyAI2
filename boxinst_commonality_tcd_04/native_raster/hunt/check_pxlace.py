"""Gate: PixelLACE at pixels lying on cell centres == TCDMasker.box_mask cell posterior (same box, same knobs)."""
import json, numpy as np
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.native_raster.hunt.pxlace import PixelLACE
m = E.TCDMasker("/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"); PL = PixelLACE(m)
t = "tcd_tile_1012"; feat = np.load(f"/Users/tompitts/dphil/feat_cache/feat_4p_val/{t}.npy").astype(np.float32)
g = feat.shape[-1]; zn = m.project(feat); Z = zn.reshape(g, g, -1)
B = np.asarray(json.load(open(f"/Users/tompitts/dphil/feat_cache/boxfix_ap_val_tiles/{t}.json"))["boxes_2048"]).reshape(-1, 4)
# cell centres sit at 8i+8 (a pixel boundary), so compare against the pixel just below-right of each centre
d = []
for b in B[:60]:
    idx, r = m.box_mask(zn, g, b); clip, P = PL.pixel_post(Z, g, b); py0, py1, px0, px1 = clip
    for c, rv in zip(idx, r):
        i, j = divmod(int(c), g); yc, xc = i * m.s + m.origin, j * m.s + m.origin
        py, px = int(yc), int(xc)
        if py0 <= py < py1 and px0 <= px < px1:
            d.append(abs(P[py - py0, px - px0] - rv))
d = np.asarray(d); print(f"{len(d)} cells: |pixel(centre+0.5px) - cell| median {np.median(d):.4f}, p95 {np.percentile(d,95):.4f}, max {d.max():.4f}")

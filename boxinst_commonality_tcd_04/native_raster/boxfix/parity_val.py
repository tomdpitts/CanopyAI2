"""Gate: deployed render on cached windows == smallcrown/val_crowns 'lace' IoUs (diag_val.py, full-raster path)."""
import json, os, sys
import numpy as np
from boxinst_commonality_tcd_04.native_raster.boxfix import render_lib as R
NR = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
n_t = n_c = 0; worst = 0.0
for fp in R.tile_files("val")[:int(sys.argv[1]) if len(sys.argv) > 1 else None]:
    t = os.path.basename(fp)[:-4]; ref = os.path.join(NR, "smallcrown", "val_crowns", t + ".json")
    if not os.path.exists(ref): continue
    ref = json.load(open(ref))["lace"]
    got = [round(R.iou(c, R.render(c, {})), 5) for c in R.load_tile(fp)]
    worst = max(worst, float(np.abs(np.asarray(got) - np.asarray(ref)).max())); n_t += 1; n_c += len(got)
print(f"parity: {n_t} tiles {n_c} crowns max|diff| {worst:.2e}")

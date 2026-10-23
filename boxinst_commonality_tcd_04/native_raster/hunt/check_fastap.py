"""Gate for fastap: legacy-arm AP50 of the deployed val masks vs the COCO canopy-crowd 0.6636 (ap_val_s0_r2048.json)."""
import json, numpy as np
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
g = F.gt_cache("val"); tids = sorted(g); St, Sc = [], []
for t in tids:
    L = json.load(open(f"{F.FC}/boxfix_ap_val_tiles/{t}.json")); St.append(F.tile_stats(L["masks_rle"]["deployed"], g[t])); Sc.append(L["scores_product"])
print("legacy AP50", F.ap_halves(tids, St, Sc, {t: len(g[t]["trees"]) for t in tids}), "vs COCO crowd all 0.6636 A 0.6699 B 0.6567")

"""Paired tile-bootstrap 95% CI (2000) of legacy AP50(A) - AP50(B) on the 108 val tiles, same boxes and LACE product
ranking. Specs DIR:KEY as diag_pair. -> printed + diag/val_boot_<A>_vs_<B>.json"""
import json, os, sys
import numpy as np
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F
from boxinst_commonality_tcd_04.native_raster.hunt.diag_pair import load
HERE = os.path.dirname(os.path.abspath(__file__))
sa, sb = sys.argv[1], sys.argv[2]
gtc = F.gt_cache("val"); tids = sorted(gtc); SA, SB, SC = [], [], []
for t in tids:
    ra, ma = load(sa, t); rb, mb = load(sb, t); assert ra["boxes_2048"] == rb["boxes_2048"]
    SA.append(F.tile_stats(ma, gtc[t])); SB.append(F.tile_stats(mb, gtc[t])); SC.append(ra["scores_product"])
ng = np.array([len(gtc[t]["trees"]) for t in tids]); rng = np.random.default_rng(0); d = []
for _ in range(2000):
    ix = rng.integers(0, len(tids), len(tids))
    d.append(F.ap([SA[i] for i in ix], [SC[i] for i in ix], ng[ix].sum()) - F.ap([SB[i] for i in ix], [SC[i] for i in ix], ng[ix].sum()))
pt = F.ap(SA, SC, ng.sum()) - F.ap(SB, SC, ng.sum())
out = {"A": sa, "B": sb, "diff": round(pt, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "p_A_le_B": round(float(np.mean(np.asarray(d) <= 0)), 3)}
json.dump(out, open(os.path.join(HERE, "diag", f"val_boot_{sa.split('__')[-1].split(':')[0]}_vs_{sb.split(':')[0]}.json"), "w"), indent=1); print(out)

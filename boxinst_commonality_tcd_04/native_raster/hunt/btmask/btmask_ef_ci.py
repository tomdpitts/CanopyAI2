"""Paired tile bootstrap D - A for AP50, AP75, AP50:95 (BTMASK-EF arm D vs A), same resampling as dino_det/s2_final.boot
(2000, seed 0, sorted indices). Gate: AP50 and AP50:95 reproduce compare_ef_d.json exactly. -> ci_ef.json
  cd ~/dphil/CanopyAI2 && PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.hunt.btmask.btmask_ef_ci
"""
import json, os
import numpy as np
from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
from boxinst_commonality_tcd_04.dino_det import rerank as R, s2_final as F, fasteval as FE

HERE, FC = BM.HERE, BM.FC
ref = json.load(open(os.path.join(HERE, "compare_ef_d.json"))); out = {}
for split in ("test",):
    PA = json.load(open(os.path.join(HERE, f"preds_A_ef_{split}_r2048.json")))["preds"]
    PD = json.load(open(os.path.join(HERE, f"preds_D_ef_{split}_r2048.json")))["preds"]; tids = sorted(PA)
    EA = FE.build(None, None, f"{FC}/btmask_ef_fasteval_{split}_A.pkl"); EB = FE.build(None, None, f"{FC}/btmask_ef_fasteval_{split}_B.pkl")
    DA = F.per_tile(EA, {t: PA[t]["scores"] for t in tids}); DD = F.per_tile(EB, {t: PD[t]["scores"] for t in tids})
    rng = np.random.default_rng(R.SEED); d = []
    for _ in range(R.NBOOT):
        s = [tids[i] for i in np.sort(rng.integers(0, len(tids), len(tids)))]; a, b = F.ap_from(DD, s), F.ap_from(DA, s)
        d.append((a[0] - b[0], a[5] - b[5], a.mean() - b.mean()))
    d = np.asarray(d); a, b = F.ap_from(DD, tids), F.ap_from(DA, tids); full = (a[0] - b[0], a[5] - b[5], a.mean() - b.mean())
    r = lambda v: round(float(v), 4)
    out[split] = {m: {"diff": r(full[j]), "ci95": [r(np.percentile(d[:, j], 2.5)), r(np.percentile(d[:, j], 97.5))]}
                  for j, m in enumerate(("AP50", "AP75", "AP50_95"))}
    for m in ("AP50", "AP50_95"):
        g = ref[split]["boot"]["D_minus_A"][m]; assert out[split][m]["diff"] == g["diff"] and out[split][m]["ci95"] == g["ci95"], (split, m, out[split][m], g)
    print(split, out[split], "gate PASS (AP50, AP50:95 == compare_ef_d.json)", flush=True)
json.dump(out, open(os.path.join(HERE, "ci_ef.json"), "w"), indent=1)

"""Paired GT-box comparison: lace4px_simple (hunt/test/gtbox4_lace4px_simple_percrown.jsonl) vs the SAM 3 arms of
audit_2026-10-14/selvabox_evidence/gtbox_sam_persub_tiles/ (A4: zero-shot / SelvaMask-FT, whole-tile / per-subtile).
Crowns are aligned per tile by index; validity must agree. Statistic = mean over valid crowns of (LACE - SAM) IoU;
95% CI by tile bootstrap (2000 resamples, seed 0), as in gtbox_sam_persub.py. 2026-11-01, for the tab:gtbox audit.
Out: hunt/test/gtbox4_vs_sam.json
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
lace = {json.loads(l)["tile"]: json.loads(l) for l in open(os.path.join(HERE, "test/gtbox4_lace4px_simple_percrown.jsonl"))}
SD = os.path.join(NR, "audit_2026-10-14/selvabox_evidence/gtbox_sam_persub_tiles")
arms = ["base_sub", "base_whole", "ft_sub", "ft_whole"]
per = {a: [] for a in arms}; L = []; tiles = []
for f in sorted(os.listdir(SD)):
    t = f[:-5]; s = json.load(open(os.path.join(SD, f))); l = lace[t]
    assert len(l["iou"]) == len(s["valid"]) and l["valid"] == s["valid"], t
    v = np.array(s["valid"])
    tiles.append(t); L.append(np.array(l["iou"])[v])
    for a in arms: per[a].append(np.array(s[a])[v])
n = sum(len(x) for x in L)
out = {"n_tiles": len(tiles), "n_crowns": int(n), "lace4px_mean_iou": round(float(np.concatenate(L).mean()), 4)}
rng = np.random.default_rng(0); B = 2000; idx = rng.integers(0, len(tiles), (B, len(tiles)))
for a in arms:
    d = [li - si for li, si in zip(L, per[a])]
    sums = np.array([x.sum() for x in d]); cnt = np.array([len(x) for x in d])
    boot = sums[idx].sum(1) / cnt[idx].sum(1)
    out[a] = {"sam_mean_iou": round(float(np.concatenate(per[a]).mean()), 4),
              "lace_minus_sam": round(float(sums.sum() / cnt.sum()), 4),
              "ci95_tile_boot": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]}
json.dump(out, open(os.path.join(HERE, "test/gtbox4_vs_sam.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

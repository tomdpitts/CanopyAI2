"""Mine hunt_tau_table_val: what predicts the IoU-best tau? Proxy objective = number of detections with IoU >= 0.5
at their crown (the AP50-relevant event). Reports: deployed count; oracle count; and for each single signal, a
binned lookup tau(bin) (quintiles of the signal; tau per bin = the one maximising the count in that bin), scored
2-fold by random halves (fit on one half, count on the other). -> diag/tau_mine.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
X = np.load("/Users/tompitts/dphil/feat_cache/hunt_tau_table_val.npz")["X"]; cols = json.load(open(os.path.join(HERE, "diag", "tau_table_cols.json")))
c = {k: i for i, k in enumerate(cols)}
taus = [float(k[4:]) for k in cols if k.startswith("iou_")]
I = X[:, [c[f"iou_{t}"] for t in taus]]; hit = I >= 0.5
dep = taus.index(0.35)
out = {"n": len(X), "deployed_hits": int(hit[:, dep].sum()), "oracle_hits": int(hit.any(1).sum()),
       "best_single_tau": {str(t): int(hit[:, k].sum()) for k, t in enumerate(taus)}}
rng = np.random.default_rng(0); half = rng.random(len(X)) < 0.5
sig = ["short", "area_box", "bimod", "pfg_mean", "q_peak", "q_mean", "p_mean", "score", "fill_0.35", "fill_0.25"]
res = {}
for s in sig:
    v = X[:, c[s]]; edges = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(edges, v)
    tot = 0; table = {}
    for fit, ev in ((half, ~half), (~half, half)):
        for k in range(5):
            f = fit & (b == k); e = ev & (b == k)
            kk = int(hit[f].sum(0).argmax()) if f.any() else dep
            tot += int(hit[e, kk].sum()); table.setdefault(k, []).append(taus[kk])
    res[s] = {"heldout_hits": tot, "gain_vs_deployed": tot - out["deployed_hits"], "tau_by_quintile": table, "edges": np.round(edges, 3).tolist()}
out["single_signal_lookup"] = res
json.dump(out, open(os.path.join(HERE, "diag", "tau_mine.json"), "w"), indent=1)
print({k: out[k] for k in ("n", "deployed_hits", "oracle_hits")}); print(out["best_single_tau"])
for s, r in sorted(res.items(), key=lambda x: -x[1]["heldout_hits"]): print(f"{s:10s} {r['gain_vs_deployed']:+5d}  taus {r['tau_by_quintile']}")

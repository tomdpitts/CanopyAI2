"""Per-tile counts, score distributions, cap binding, density-stratified box recall. Serial."""
import json
import numpy as np

S = "/private/tmp/claude-501/-Users-tompitts-dphil-CanopyAI2/b085cec2-62cb-4a30-a5e2-6f6b537b156d/scratchpad/restor_audit"
R = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04"
A = json.load(open(S + "/arms_light.json"))
gt = json.load(open(R + "/test_gt.json"))
tids = sorted(A["1000"])
assert set(tids) == set(gt)


def polybox(p):
    if p and isinstance(p[0], list):
        xs = [x for q in p for x in q[0::2]]; ys = [y for q in p for y in q[1::2]]
    else:
        xs = p[0::2]; ys = p[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


gtb = {t: np.array([polybox(p) for p in gt[t]["trees"]], float).reshape(-1, 4) for t in tids}
ngt = np.array([len(gtb[t]) for t in tids])
print("GT trees total", ngt.sum(), "max/tile", ngt.max(), "median", np.median(ngt))

out = {}
for a in ("512", "1000", "2000"):
    tot = np.array([len(A[a][t]["scores"]) for t in tids])
    tree = np.array([sum(1 for c in A[a][t]["cls"] if c == 1) for t in tids])
    sc_all = np.array([s for t in tids for s in A[a][t]["scores"]])
    sc_tree = np.array([s for t in tids for s, c in zip(A[a][t]["scores"], A[a][t]["cls"]) if c == 1])
    mx = np.array([max(A[a][t]["scores"]) if A[a][t]["scores"] else -1 for t in tids])
    o = {"dets_total": int(tot.sum()), "dets_tree": int(tree.sum()), "tree_per_tile": round(tree.mean(), 1),
         "max_total_per_tile": int(tot.max()), "max_tree_per_tile": int(tree.max()),
         "tiles_at_600cap": int((tot >= 600).sum()), "tiles_total_ge_500": int((tot >= 500).sum()),
         "min_score": float(sc_all.min()), "max_score": float(sc_all.max()),
         "tree_score_q": {q: round(float(np.quantile(sc_tree, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "tree_n_ge": {th: int((sc_tree >= th).sum()) for th in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)}}
    out[a] = o
    out[a]["_tot"] = tot; out[a]["_tree"] = tree; out[a]["_mx"] = mx
    print(a, {k: v for k, v in o.items() if not k.startswith("_")})

# per-tile max score identity across arms (paper claim)
for a, b in (("512", "1000"), ("1000", "2000"), ("512", "2000")):
    ma, mb = out[a]["_mx"], out[b]["_mx"]
    print(f"max score per tile {a} vs {b}: identical on {(ma == mb).sum()}/{len(tids)}; "
          f"b>a {(mb > ma).sum()}, b<a {(mb < ma).sum()}; global max {ma.max()} vs {mb.max()}")

# ---- per-tile tree det deltas vs GT density ----
d = out["2000"]["_tree"] - out["1000"]["_tree"]
print("tree dets 2000-1000: sum", d.sum(), "tiles with increase", (d > 0).sum(), "decrease", (d < 0).sum(), "same", (d == 0).sum())
bins = [(0, 0), (1, 10), (11, 30), (31, 60), (61, 100), (101, 200), (201, 500)]


def iou_mat(P, G):
    if len(P) == 0 or len(G) == 0:
        return np.zeros((len(P), len(G)))
    x0 = np.maximum(P[:, None, 0], G[None, :, 0]); y0 = np.maximum(P[:, None, 1], G[None, :, 1])
    x1 = np.minimum(P[:, None, 2], G[None, :, 2]); y1 = np.minimum(P[:, None, 3], G[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ap = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1]); ag = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    return inter / (ap[:, None] + ag[None, :] - inter + 1e-9)


def greedy_tp(P, sc, G, thr=0.5):
    """COCO-style greedy matching by score; returns (#tp, #gt, list of (score, is_tp))."""
    order = np.argsort(-np.asarray(sc), kind="mergesort")
    P = P[order]; sc = np.asarray(sc)[order]
    M = iou_mat(P, G); used = np.zeros(len(G), bool); rec = []
    for i in range(len(P)):
        best, bj = thr, -1
        for j in np.nonzero(M[i] >= thr)[0]:
            if not used[j] and M[i, j] >= best:
                best, bj = M[i, j], j
        if bj >= 0:
            used[bj] = True; rec.append((sc[i], 1))
        else:
            rec.append((sc[i], 0))
    return int(used.sum()), len(G), rec


box = {}
for a in ("512", "1000", "2000"):
    per = []
    for t in tids:
        sel = [i for i, c in enumerate(A[a][t]["cls"]) if c == 1]
        P = np.array([A[a][t]["boxes"][i] for i in sel], float).reshape(-1, 4)
        sc = [A[a][t]["scores"][i] for i in sel]
        per.append(greedy_tp(P, sc, gtb[t]))
    box[a] = per

print("\nbox-level greedy TP@0.5 (no crowd handling; trees only) by GT-density bin")
print("bin | n_tiles | GT | rec512 | rec1000 | rec2000 | treedets1000/tile | treedets2000/tile | totdets1000 max | totdets2000 max")
for lo, hi in bins:
    m = (ngt >= lo) & (ngt <= hi)
    G = ngt[m].sum()
    r = {a: sum(box[a][i][0] for i in np.nonzero(m)[0]) / max(G, 1) for a in box}
    print(f"{lo}-{hi} | {m.sum()} | {G} | {r['512']:.3f} | {r['1000']:.3f} | {r['2000']:.3f} | "
          f"{out['1000']['_tree'][m].mean():.1f} | {out['2000']['_tree'][m].mean():.1f} | "
          f"{out['1000']['_tot'][m].max()} | {out['2000']['_tot'][m].max()}")
for a in box:
    print(a, "overall box recall@0.5", round(sum(x[0] for x in box[a]) / ngt.sum(), 4))

# where do the extra TPs (2000 vs 1000) come from, by score band
print("\nTP / FP by score band (box@0.5, tree only)")
bands = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
for a in box:
    rec = [x for per in box[a] for x in per[2]]
    s = np.array([x[0] for x in rec]); tp = np.array([x[1] for x in rec])
    row = []
    for lo, hi in bands:
        mm = (s >= lo) & (s < hi)
        row.append(f"[{lo},{hi}) tp={tp[mm].sum()} fp={(1 - tp[mm]).sum()}")
    print(a, " | ".join(row))

# contribution of dense tiles to the 1000->2000 TP gain
gain = np.array([box["2000"][i][0] - box["1000"][i][0] for i in range(len(tids))])
order = np.argsort(-ngt)
for k in (10, 20, 44, 88):
    print(f"top-{k} densest tiles (>= {ngt[order[k-1]]} GT): share of box TP gain {gain[order[:k]].sum()}/{gain.sum()} = {gain[order[:k]].sum()/gain.sum():.2f}; share of GT {ngt[order[:k]].sum()/ngt.sum():.2f}")

# total dets (both classes) vs 2*topk? and fraction of tiles where total dets is a large fraction of topk
for a, K in (("512", 512), ("1000", 1000), ("2000", 2000)):
    tot = out[a]["_tot"]
    print(f"rpn{a}: total dets/tile max {tot.max()}, tiles with total >= 0.5*K: {(tot >= 0.5*K).sum()}, tiles at 600 cap: {(tot >= 600).sum()}")
json.dump({a: {k: v for k, v in o.items() if not k.startswith('_')} for a, o in out.items()},
          open(S + "/dist_summary.json", "w"), indent=1)

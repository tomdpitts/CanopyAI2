"""Can an INTRINSIC per-detection signal choose among saved LACE mask variants (b24_save_variants) better than any
single variant? Rules are lookups: signal -> bins (quantiles over all val dets) -> per-bin variant chosen to maximise
legacy AP50 on the FIT half (greedy coordinate ascent over bins, 2 sweeps), evaluated on the other half; both
directions (A->B, B->A). Signals (no GT): box short side; product score; mask area ratios between variants;
agreement IoU between pairs of variants; posterior-free.
-> diag/variant_mine.json

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.hunt.variant_mine
"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import score_coco as SC
from boxinst_commonality_tcd_04.native_raster.hunt import fastap as F

HERE = os.path.dirname(os.path.abspath(__file__)); FC = F.FC
V = ["g30", "deployed", "p4_t0.3", "px4_c0.4", "e8", "p4_c0.28"]
OUT = os.path.join(HERE, "diag", "variant_mine.json")


def main():
    assert not os.path.exists(OUT)
    gtc = F.gt_cache("val"); tids = sorted(gtc)
    ST = {v: [] for v in V}; SCORES = []; FEAT = []
    for t in tids:
        R = {v: json.load(open(f"{FC}/hunt_masks/b24_save_variants__{v}/{t}.json")) for v in V}
        bx = np.asarray(R[V[0]]["boxes_2048"]).reshape(-1, 4); SCORES.append(np.asarray(R[V[0]]["scores_product"]))
        enc = {v: F.enc(R[v]["masks_rle"]) for v in V}
        for v in V:
            ST[v].append(F.tile_stats(R[v]["masks_rle"], gtc[t]))
        n = len(bx); f = np.zeros((n, 6))
        if n:
            area = {v: np.asarray(SC._areas(enc[v]), float) for v in V}
            agree = lambda a, b: np.asarray([float(mu.iou([x], [y], [0])[0][0]) if (area[a][i] + area[b][i]) else 1.0 for i, (x, y) in enumerate(zip(enc[a], enc[b]))])
            f[:, 0] = np.minimum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1])
            f[:, 1] = SCORES[-1]
            f[:, 2] = agree("g30", "deployed")
            f[:, 3] = agree("g30", "e8")
            f[:, 4] = area["g30"] / np.maximum(area["deployed"], 1)
            f[:, 5] = area["g30"] / np.maximum((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]), 1)
        FEAT.append(f)
    names = ["short", "score", "agree_g30_dep", "agree_g30_e8", "area_g30/dep", "fill_g30"]
    n_gt = {t: len(gtc[t]["trees"]) for t in tids}
    half = {"A": tids[0::2], "B": tids[1::2]}; idx = {h: [tids.index(t) for t in ts] for h, ts in half.items()}

    def ap_of(choice, ix):
        st = []
        for i in ix:
            c = choice[i]; base = ST[V[0]][i]
            if not len(c) or base[0].shape[1] == 0:
                st.append(base); continue
            st.append((np.stack([ST[V[k]][i][0][r] for r, k in enumerate(c)]), np.array([ST[V[k]][i][1][r] for r, k in enumerate(c)], bool)))
        return F.ap(st, [SCORES[i] for i in ix], sum(n_gt[tids[i]] for i in ix))

    single = {v: {h: round(ap_of([np.full(len(SCORES[i]), k) for i in range(len(tids))], idx[h]), 4) for h in ("A", "B")} for k, v in enumerate(V)}
    out = {"single": single, "rules": {}}
    allf = np.concatenate(FEAT)
    for fi, nm in enumerate(names):
        edges = np.quantile(allf[:, fi], [0.2, 0.4, 0.6, 0.8])
        bins = [np.searchsorted(edges, f[:, fi]) for f in FEAT]
        res = {}
        for fit, ev in (("A", "B"), ("B", "A")):
            tab = np.zeros(5, int)                              # start: all bins -> g30 (index 0)
            for _ in range(2):
                for b in range(5):
                    best, bs = tab[b], -1
                    for k in range(len(V)):
                        tab[b] = k
                        a = ap_of([tab[bb] for bb in bins], idx[fit])
                        if a > bs + 1e-9:
                            bs, best = a, k
                    tab[b] = best
            res[f"{fit}->{ev}"] = {"table": [V[k] for k in tab], "fit_AP": round(ap_of([tab[bb] for bb in bins], idx[fit]), 4),
                                   "eval_AP": round(ap_of([tab[bb] for bb in bins], idx[ev]), 4),
                                   "eval_g30": single["g30"][ev]}
        out["rules"][nm] = {"edges": np.round(edges, 3).tolist(), **res}
        print(nm, {k: (v["eval_AP"], v["eval_g30"], v["table"]) for k, v in res.items()}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(single))


if __name__ == "__main__":
    main()

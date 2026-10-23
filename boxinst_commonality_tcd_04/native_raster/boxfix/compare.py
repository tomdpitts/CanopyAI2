"""Compare sweep variants against SAM 3 zero-shot per-subtile on the same crowns.

val: SAM IoUs from smallcrown/val_crowns/<tile>.json ('sam_zs_sub', diag_val.py).
test: SAM IoUs from audit_2026-10-14/selvabox_evidence/gtbox_sam_persub_tiles/<tile>.json ('base_sub').
Paired difference per crown (valid crowns, tiles where both exist), tile-bootstrap 95% CI, area strata
at the fixed test tercile edges (622, 2456 px).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.boxfix.compare SWEEP.json [OUT.json]
"""
import json, os, sys
import numpy as np

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = (622.0, 2456.3)


def sam_iou(split, tile):
    if split == "val":
        p = os.path.join(NR, "smallcrown", "val_crowns", tile + ".json"); k = "sam_zs_sub"
    else:
        p = os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub_tiles", tile + ".json"); k = "base_sub"
    return np.asarray(json.load(open(p))[k]) if os.path.exists(p) else None


def boot(d_by_tile, n=2000, seed=0):
    rng = np.random.default_rng(seed); T = len(d_by_tile)
    s = np.array([x.sum() for x in d_by_tile]); c = np.array([len(x) for x in d_by_tile])
    bs = []
    for _ in range(n):
        i = rng.integers(0, T, T); bs.append(s[i].sum() / max(c[i].sum(), 1))
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    sw = json.load(open(sys.argv[1])); split = sw["split"]
    rows = []
    for r in sw["tiles"]:
        s = sam_iou(split, r["tile"])
        if s is None:
            continue
        rows.append((r, s))
    out = {"sweep": sys.argv[1], "split": split, "n_tiles": len(rows), "variants": {}}
    for name in sw["variants"]:
        res = {}
        for st, lo, hi in (("all", 0, 1e12), ("small", 0, EDGES[0]), ("mid", EDGES[0], EDGES[1]), ("large", EDGES[1], 1e12)):
            dl, L, S = [], [], []
            for r, s in rows:
                v = np.asarray(r["valid"], bool); a = np.asarray(r["area"], float)
                k = v & (a >= lo) & (a < hi)
                if not k.any():
                    continue
                li = np.asarray(r["iou"][name])[k]; si = s[k]
                dl.append(li - si); L.append(li); S.append(si)
            L, S = np.concatenate(L), np.concatenate(S)
            res[st] = {"n": int(len(L)), "lace": round(float(L.mean()), 4), "sam": round(float(S.mean()), 4),
                       "diff": round(float((L - S).mean()), 4), "ci95": boot(dl), "empty_frac": None}
        out["variants"][name] = res
    if len(sys.argv) > 2:
        assert not os.path.exists(sys.argv[2]); json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(f"{split} {len(rows)} tiles; diff = variant - SAM zs per-subtile")
    print(f"{'variant':32s} " + " ".join(f"{s:>22s}" for s in ("all", "small", "mid", "large")))
    for name, res in out["variants"].items():
        print(f"{name:32s} " + " ".join(f"{res[s]['lace']:.4f} {res[s]['diff']:+.4f}       " for s in ("all", "small", "mid", "large")))


if __name__ == "__main__":
    main()

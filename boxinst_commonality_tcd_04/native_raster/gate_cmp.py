"""Exact-equality gate: a locally re-stitched / re-derived preds file vs its published one.
Compares tile set, per-tile counts, boxes, scores, RLE counts and RLE size. Exit 1 on any diff.

  python -m boxinst_commonality_tcd_04.native_raster.gate_cmp <published> <new>
"""
import json, sys


def compare(pub, loc, label=""):
    P = json.load(open(pub))["preds"]; L = json.load(open(loc))["preds"]
    bad = {"tiles": int(set(P) != set(L)), "count": 0, "box": 0, "score": 0, "rle": 0, "size": 0}
    n = 0
    for t in sorted(set(P) & set(L)):
        a, b = P[t], L[t]
        if len(a["scores"]) != len(b["scores"]):
            bad["count"] += 1; continue
        n += len(a["scores"])
        bad["box"] += a["boxes_2048"] != b["boxes_2048"]
        bad["score"] += a["scores"] != b["scores"]
        bad["rle"] += [r["counts"] for r in a["masks_rle"]] != [r["counts"] for r in b["masks_rle"]]
        bad["size"] += [list(r["size"]) for r in a["masks_rle"]] != [list(r["size"]) for r in b["masks_rle"]]
    ok = not any(bad.values())
    print(f"[{label or 'gate'}] tiles={len(set(P)&set(L))} dets={n} mismatching-tiles={bad} -> "
          f"{'PASS byte-identical' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "") else 1)

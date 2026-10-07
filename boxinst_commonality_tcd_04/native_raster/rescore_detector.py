"""2048 masks + DETECTOR-score ranking: the `s` rows of tab:scorer / tab:factors at native raster.

Takes the 2048 LACE render (whose `scores` carry the product key s*pbar*m, copied from the
*_product.json source) and swaps in the detector scores from the published detector-ranked
512 file. Gate: boxes_2048 must be byte-identical and in the same order on every tile, and
the detection counts equal -- i.e. the two published files describe the same detection set,
so only the ranking key changes. Masks and canopy_ignore stay the 2048 ones. Refuses to overwrite.
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks2048", required=True); ap.add_argument("--det512", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    M = json.load(open(a.masks2048)); D = json.load(open(a.det512))
    assert set(M["preds"]) == set(D["preds"]), "tile sets differ"
    bad_box = bad_n = 0; n = 0; preds = {}
    for t in sorted(M["preds"]):
        m, d = M["preds"][t], D["preds"][t]
        if len(m["boxes_2048"]) != len(d["boxes_2048"]): bad_n += 1; continue
        if m["boxes_2048"] != d["boxes_2048"]: bad_box += 1
        n += len(d["scores"])
        preds[t] = {"boxes_2048": m["boxes_2048"], "scores": d["scores"],
                    "canopy_ignore": m["canopy_ignore"], "masks_rle": m["masks_rle"]}
    ok = not (bad_box or bad_n)
    print(f"GATE boxes-identical: count-mismatch tiles {bad_n}, box-mismatch tiles {bad_box}, dets {n} -> {'PASS' if ok else 'FAIL'}")
    if not ok: sys.exit(1)
    meta = dict(M["meta"]); meta.update({"ranking": "detector score s (from %s)" % a.det512,
                                         "masks_from": a.masks2048})
    json.dump({"meta": meta, "preds": preds}, open(a.out, "w")); print("->", a.out)


if __name__ == "__main__":
    main()

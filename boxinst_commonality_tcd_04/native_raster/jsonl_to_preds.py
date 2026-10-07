"""JSONL tile checkpoints (lace_multitau_modal / lace_masks_sparse_modal) -> standard preds json,
with the pairing gate: boxes_2048 and scores must be byte-identical to the published 512 source
row for every tile, and the tile set must equal the GT's. Refuses to overwrite.

  python -m boxinst_commonality_tcd_04.native_raster.jsonl_to_preds --jsonl X.jsonl \
      --src <published 512 preds> --gt <gt json> --tau 0.40 --out Y.json --model "..."
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--gt", required=True); ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--res", type=int, default=2048)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    gt = json.load(open(a.gt)); src = json.load(open(a.src))
    recs, bad = {}, 0
    for raw in open(a.jsonl, "rb"):
        try:
            r = json.loads(raw)
        except Exception:
            bad += 1; continue
        recs[r["tile"]] = r
    tids = sorted(gt)
    missing = [t for t in tids if t not in recs]; extra = [t for t in recs if t not in gt]
    print(f"tiles: gt={len(tids)} jsonl={len(recs)} missing={len(missing)} extra={len(extra)} corrupt_lines={bad}")
    assert not missing and not extra, (missing[:3], extra[:3])
    n_box_bad = n_sc_bad = n_size_bad = n = 0
    preds = {}
    for t in tids:
        r = recs[t]; s = src["preds"][t]
        if r["boxes_2048"] != s["boxes_2048"]: n_box_bad += 1
        if r["scores"] != s["scores"]: n_sc_bad += 1
        if any(list(m["size"]) != [a.res, a.res] for m in r["masks_rle"]): n_size_bad += 1
        assert len(r["masks_rle"]) == len(r["scores"]) == len(r["boxes_2048"])
        n += len(r["scores"])
        preds[t] = {"boxes_2048": r["boxes_2048"], "scores": r["scores"],
                    "canopy_ignore": r["canopy_ignore"], "masks_rle": r["masks_rle"]}
    ok = not (n_box_bad or n_sc_bad or n_size_bad)
    print(f"GATE boxes: {n_box_bad} tiles differ | scores: {n_sc_bad} | rle size: {n_size_bad} | dets {n} "
          f"(src {sum(len(v['scores']) for v in src['preds'].values())}) -> {'PASS' if ok else 'FAIL'}")
    meta = dict(src["meta"]); meta.update({"mask_res": a.res, "scale_box_to_mask": 2048 / a.res,
                                           "mask_thr": a.tau, "model": a.model, "source_512_preds": a.src,
                                           "source_jsonl": a.jsonl, "n_tiles": len(preds), "pairing_gate": ok})
    json.dump({"meta": meta, "preds": preds}, open(a.out, "w"))
    print(f"-> {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

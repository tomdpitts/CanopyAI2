"""Post-run for restor_native_modal.py: (1) gate the run against the published rpn1000 arm,
(2) derive the tree-only file from the native preds, CPU only.

Gate A  in-container 512 arm  vs  published preds_restor_rpn1000.json   -> GPU determinism
Gate B  local 512 (decode 2048 RLE -> PIL BILINEAR>=128)  vs  in-container 512 -> local Pillow
Both must be byte-identical for the 2048 row to inherit the published row's provenance.

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.restor_gate_derive --dir <dl dir>
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

PUB = "boxinst_commonality_tcd_04/restor_baseline/preds_restor_rpn1000.json"
NR = "boxinst_commonality_tcd_04/native_raster"


def cmp(P, L, label, keys=("scores", "boxes_2048", "pred_classes")):
    bad = {"tiles": 0, "count": 0, "rle": 0, "size": 0, **{k: 0 for k in keys}}
    if set(P) != set(L):
        print(f"[{label}] TILE SET DIFFERS pub-only={len(set(P)-set(L))} new-only={len(set(L)-set(P))}")
        bad["tiles"] = 1
    n = 0
    for t in sorted(set(P) & set(L)):
        a, b = P[t], L[t]
        if len(a["scores"]) != len(b["scores"]):
            bad["count"] += 1; continue
        n += len(a["scores"])
        for k in keys:
            if k in a and k in b and a[k] != b[k]:
                bad[k] += 1
        if [r["counts"] for r in a["masks_rle"]] != [r["counts"] for r in b["masks_rle"]]:
            bad["rle"] += 1
        if [list(r["size"]) for r in a["masks_rle"]] != [list(r["size"]) for r in b["masks_rle"]]:
            bad["size"] += 1
    ok = not any(bad.values())
    print(f"[{label}] dets={n}  mismatching tiles: {bad}  -> {'PASS byte-identical' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="local download of OUT_DIR")
    a = ap.parse_args()
    n2048 = json.load(open(os.path.join(a.dir, "preds_restor_rpn1000_r2048.json")))
    n512 = json.load(open(os.path.join(a.dir, "preds_restor_rpn1000_r512_incontainer.json")))
    pub = json.load(open(PUB))
    print("published meta:", {k: pub["meta"][k] for k in ("rpn_topk_test", "tree_class", "n_crowns", "n_canopy")})
    print("new meta      :", {k: n2048["meta"][k] for k in ("rpn_topk_test", "tree_class", "n_crowns", "n_canopy")})

    okA = cmp(pub["preds"], n512["preds"], "GATE A  container-512 vs published")

    loc = {}
    for t, rec in n2048["preds"].items():
        rles = []
        for r in rec["masks_rle"]:
            m = maskUtils.decode({"size": r["size"], "counts": r["counts"].encode("ascii")}).astype(bool)
            small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((512, 512), Image.BILINEAR)) >= 128
            e = maskUtils.encode(np.asfortranarray(small.astype(np.uint8)))
            rles.append({"size": [512, 512], "counts": e["counts"].decode("ascii")})
        loc[t] = {"scores": rec["scores"], "boxes_2048": rec["boxes_2048"],
                  "pred_classes": rec["pred_classes"], "masks_rle": rles}
    okB = cmp(n512["preds"], loc, "GATE B  local-512-from-2048 vs container-512")

    # tree-only derivation (== how preds_restor_rpn1000_treeonly.json was made; verified 439/439)
    out = os.path.join(NR, "preds_restor_rpn1000_r2048_treeonly.json")
    assert not os.path.exists(out), f"REFUSING to overwrite {out}"
    tc = n2048["meta"]["tree_class"]; preds = {}; n = 0
    for t, rec in n2048["preds"].items():
        sel = [i for i, c in enumerate(rec["pred_classes"]) if c == tc]
        preds[t] = {"boxes_2048": [rec["boxes_2048"][i] for i in sel],
                    "scores": [rec["scores"][i] for i in sel],
                    "masks_rle": [rec["masks_rle"][i] for i in sel],
                    "pred_classes": [tc] * len(sel)}
        n += len(sel)
    json.dump({"meta": dict(n2048["meta"], variant="tree-class only"), "preds": preds}, open(out, "w"))
    print(f"wrote {out}: {n} tree dets")
    print("GATES:", "PASS" if (okA and okB) else "FAIL")
    sys.exit(0 if (okA and okB) else 1)


if __name__ == "__main__":
    main()

"""Read-only audit of Restor rpn arms. Serial, no pools."""
import json, gc
import numpy as np

R = "/Users/tompitts/dphil/CanopyAI2/boxinst_commonality_tcd_04"
NR = R + "/native_raster"
RB = R + "/restor_baseline"


def load(fp):
    d = json.load(open(fp))
    return d.get("meta"), d["preds"]


def cmp(A, B, label, keys=("scores", "boxes_2048"), rle=True, cls=True):
    bad = {"tileset": int(set(A) != set(B)), "count": 0, "rle": 0, "cls": 0, **{k: 0 for k in keys}}
    n = 0
    for t in sorted(set(A) & set(B)):
        a, b = A[t], B[t]
        if len(a["scores"]) != len(b["scores"]):
            bad["count"] += 1; continue
        n += len(a["scores"])
        for k in keys:
            if a[k] != b[k]:
                bad[k] += 1
        if cls and "pred_classes" in a and "pred_classes" in b and a["pred_classes"] != b["pred_classes"]:
            bad["cls"] += 1
        if rle and [r["counts"] for r in a["masks_rle"]] != [r["counts"] for r in b["masks_rle"]]:
            bad["rle"] += 1
    ok = not any(bad.values())
    print(f"[{label}] dets={n} bad={bad} -> {'PASS' if ok else 'FAIL'}")
    return ok


def treefilter(P, tc=1):
    out = {}
    for t, r in P.items():
        sel = [i for i, c in enumerate(r["pred_classes"]) if c == tc]
        out[t] = {"scores": [r["scores"][i] for i in sel],
                  "boxes_2048": [r["boxes_2048"][i] for i in sel],
                  "masks_rle": [r["masks_rle"][i] for i in sel]}
    return out


res = {}
# ---------- gates at 512 vs Aug-27 restor_baseline files ----------
print("== Gate A re-check (native in-container 512 vs restor_baseline Aug-27 files)")
m, n1000_512 = load(NR + "/preds_restor_rpn1000_r512_incontainer.json"); print(" meta n1000_512", m)
_, b1000 = load(RB + "/preds_restor_rpn1000.json")
res["gateA_rpn1000"] = cmp(n1000_512, b1000, "rpn1000 native512 vs baseline rpn1000 (all classes)")
del b1000; gc.collect()

m, n2000_512 = load(NR + "/preds_restor_rpn2000_r512_incontainer.json"); print(" meta n2000_512", m)
_, b2000 = load(RB + "/preds_restor_rpn2000.json")
res["gateA_rpn2000"] = cmp(treefilter(n2000_512), b2000, "rpn2000 native512 tree-filter vs baseline rpn2000 (tree-only)", cls=False)
del b2000; gc.collect()

m, n512_512 = load(NR + "/preds_restor_rpn512_r512_incontainer.json"); print(" meta n512_512", m)
_, b512 = load(RB + "/preds_restor_allcls.json")
res["gateA_rpn512"] = cmp(n512_512, b512, "rpn512 native512 vs baseline allcls")
del b512; gc.collect()
_, b512t = load(RB + "/preds_restor.json")
res["gateA_rpn512_tree"] = cmp(treefilter(n512_512), b512t, "rpn512 native512 tree-filter vs baseline preds_restor.json", cls=False)
del b512t; gc.collect()

# ---------- per arm: 2048 vs 512 detections identical; treeonly derivation ----------
arms = {}
for a, n512 in (("512", n512_512), ("1000", n1000_512), ("2000", n2000_512)):
    m, P = load(NR + f"/preds_restor_rpn{a}_r2048.json")
    print(f"== rpn{a} r2048 meta", m)
    cmp(P, n512, f"rpn{a} r2048 vs r512_incontainer (scores/boxes/classes)", rle=False)
    mt, T = load(NR + f"/preds_restor_rpn{a}_r2048_treeonly.json")
    print(f"   treeonly meta", mt)
    cmp(treefilter(P), T, f"rpn{a} r2048 class-1 filter vs treeonly file", cls=False)
    # strip masks to save memory
    arms[a] = {t: {"scores": r["scores"], "boxes": r["boxes_2048"], "cls": r["pred_classes"],
                   "rle_sizes": sorted({tuple(x["size"]) for x in r["masks_rle"]})}
               for t, r in P.items()}
    del P, T; gc.collect()
del n512_512, n1000_512, n2000_512; gc.collect()

json.dump({a: {t: {"scores": v["scores"], "boxes": v["boxes"], "cls": v["cls"]} for t, v in A.items()}
           for a, A in arms.items()},
          open("/private/tmp/claude-501/-Users-tompitts-dphil-CanopyAI2/b085cec2-62cb-4a30-a5e2-6f6b537b156d/scratchpad/restor_audit/arms_light.json", "w"))
sizes = {a: sorted({s for v in A.values() for s in v["rle_sizes"]}) for a, A in arms.items()}
print("RLE sizes per arm (r2048 files):", sizes)
print("GATES:", res)

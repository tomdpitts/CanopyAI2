"""What do D1's false positives at AP50 consist of? (2026-11-03) Diagnostic only.

Frozen s' preds (test 439). Per tile, greedy COCO-style matching in s' order at mask IoU >= 0.5:
  best unmatched tree -> TP; else >= 50% of the det inside the canopy union -> ignored (crowd); else FP.
FP types, in priority order. m = max mask IoU with any tree; cov = intersection / tree area; pin = intersection / det area.
  dup   m >= 0.5, tree already taken
  merge >= 2 trees with cov >= 0.5
  part  pin >= 0.8 against one tree (a fragment of a crown)
  near  0.3 <= m < 0.5
  poor  0.1 <= m < 0.3
  bg    m < 0.1
For near/poor, `boxok` flags a det whose box IoU with that same tree's GT box is >= 0.5 (box fine, mask failed).
AP cost per type = AP50 with that type's FPs deleted, minus base AP50 (TP matching is unchanged by deleting FPs).
101-point COCO AP over the pooled 439 tiles; checked against the frozen scorer's base 0.7035.
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dino_det.fp_breakdown
"""
import json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results"); TAG = "dino_swinl_s0"
TYPES = ["dup", "merge", "part", "near", "poor", "bg"]


def _box_iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / np.maximum(aa[:, None] + bb[None, :] - i, 1e-9)


def _tile(args):
    t, g, r = args
    from pycocotools import mask as M
    from boxinst_commonality_tcd_04 import score_coco as SC
    tr = SC._polys_to_rle(g["trees"], 2048); can = SC._union(SC._polys_to_rle(g["canopy"], 2048), 2048)
    d = [SC._as_bytes(x) for x in r["masks_rle"]]; s = np.asarray(r["scores"], np.float64)
    nd, ng = len(d), len(tr)
    if nd == 0:
        return t, s, np.zeros(0, int), np.zeros(0, bool), ng
    iou = np.asarray(M.iou(d, tr, [0] * ng), np.float64).reshape(nd, ng) if ng else np.zeros((nd, 0))
    ad = SC._areas(d); ag = SC._areas(tr) if ng else np.zeros(0)
    inter = iou * (ad[:, None] + ag[None, :]) / (1.0 + iou)
    cov = inter / np.maximum(ag[None, :], 1); pin = inter / np.maximum(ad[:, None], 1)
    crowd = SC._frac_inside(d, can) >= 0.5
    gb = np.asarray([M.toBbox(x) for x in tr], np.float64).reshape(-1, 4); gb[:, 2:] += gb[:, :2]
    biou = _box_iou(np.asarray(r["boxes_2048"], np.float64).reshape(-1, 4), gb) if ng else np.zeros((nd, 0))
    lab = np.full(nd, -1, int); boxok = np.zeros(nd, bool); taken = np.zeros(ng, bool)   # -1 TP, -2 ignored, else TYPES index
    for i in np.argsort(-s, kind="mergesort"):
        if ng:
            c = np.where(~taken & (iou[i] >= 0.5), iou[i], -1.0); j = int(c.argmax())
            if c[j] >= 0.5:
                taken[j] = True; continue
        if crowd[i]:
            lab[i] = -2; continue
        m = iou[i].max() if ng else 0.0; jm = int(iou[i].argmax()) if ng else -1
        if m >= 0.5: k = 0
        elif ng and (cov[i] >= 0.5).sum() >= 2: k = 1
        elif ng and pin[i].max() >= 0.8: k = 2
        elif m >= 0.3: k = 3
        elif m >= 0.1: k = 4
        else: k = 5
        lab[i] = k; boxok[i] = k in (3, 4) and biou[i, jm] >= 0.5
    return t, s, lab, boxok, ng


def ap101(s, tp, fp, n_gt):
    o = np.argsort(-s, kind="mergesort"); tp = np.cumsum(tp[o]); fp = np.cumsum(fp[o])
    rc = tp / n_gt; pr = tp / np.maximum(tp + fp, 1e-12)
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    q = [pr[k] if k < len(pr) else 0.0 for k in np.searchsorted(rc, np.linspace(0, 1, 101), side="left")]
    return float(np.mean(q))


def main():
    t0 = time.time()
    gt = json.load(open(os.path.join(PKG, "test_gt.json"))); tids = sorted(gt)
    P = json.load(open(os.path.join(OUT, f"preds_{TAG}_test_sprime_r2048.json")))["preds"]
    R = []
    with mp.get_context("spawn").Pool(12) as pool:
        for k, x in enumerate(pool.imap_unordered(_tile, [(t, gt[t], P[t]) for t in tids]), 1):
            R.append(x)
            if k % 40 == 0 or k == len(tids):
                print(f"  tile {k}/439 {time.time()-t0:.0f}s", flush=True)
    s = np.concatenate([x[1] for x in R]); lab = np.concatenate([x[2] for x in R]); bok = np.concatenate([x[3] for x in R])
    n_gt = sum(x[4] for x in R)
    keep = lab != -2; tp = (lab == -1).astype(float); fp = (lab >= 0).astype(float)
    base = ap101(s[keep], tp[keep], fp[keep], n_gt)
    # where the ranking hurts: FPs above the s' of the median TP
    med = np.median(s[lab == -1])
    out = {"n_gt": n_gt, "n_tp": int(tp.sum()), "n_fp": int(fp.sum()), "n_ignored": int((~keep).sum()),
           "base_AP50_own": round(base, 4), "base_AP50_frozen": 0.7035, "median_tp_sprime": round(float(med), 4), "types": {}}
    for k, name in enumerate(TYPES):
        f = lab == k; kk = keep & ~f
        out["types"][name] = {"n": int(f.sum()), "n_above_median_tp": int((f & (s >= med)).sum()),
                              "AP50_if_deleted": round(ap101(s[kk], tp[kk], fp[kk], n_gt), 4),
                              "gain": round(ap101(s[kk], tp[kk], fp[kk], n_gt) - base, 4),
                              "boxok": int((f & bok).sum()), "boxok_above_median_tp": int((f & bok & (s >= med)).sum())}
    json.dump(out, open(os.path.join(OUT, f"fp_breakdown_{TAG}_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()

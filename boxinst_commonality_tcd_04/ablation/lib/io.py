"""Read-only access to the artefacts this ablation scores, plus the streaming loader.

Nothing in `ablation/` writes outside `ablation/`. Every path below is an INPUT; the only
writable roots are `RESULTS` and `FIGURES`.

The streaming pattern in `per_tile` is ported from
`modal_tcd_multiseed/phase4/no_ignore_sensitivity.py` rather than imported, so that file
stays untouched. The streaming is not an optimisation: 439 tiles x ~370 preds at 512^2
is ~40 GB if the decoded masks are held at once.
"""
import hashlib
import json
import os

import numpy as np
from pycocotools import mask as maskUtils

from boxinst_commonality_tcd_04 import evaluate as E

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../boxinst_commonality_tcd_04
REPO = os.path.dirname(HERE)
ABL = os.path.join(HERE, "ablation")

RESULTS = os.path.join(ABL, "results")          # writable
FIGURES = os.path.join(ABL, "figures")          # writable

# ---- inputs (read-only) ---------------------------------------------------------------
PHASE4 = os.path.join(HERE, "modal_tcd_multiseed", "phase4")
PREDS = os.path.join(PHASE4, "preds")
GT = os.path.join(HERE, "test_gt.json")
DT2 = os.path.join(HERE, "detectree2_baseline")
SPARSE = os.path.join(HERE, "modal_sparse_tcd_multiseed")
MASKER_LAB = os.path.join(HERE, "mps_tcd_multiseed_4phase", "masker_lab")
MPS = os.path.join(HERE, "mps_multiseed")

SEEDS = (0, 1, 2)

def knobbed(s):
    return os.path.join(PREDS, f"knobbed_s{s}.json")

def ref_knobbed(s):
    return os.path.join(PREDS, f"ref_knobbed_s{s}.json")


def sha1(path, _buf=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def record_inputs(paths, out=None):
    """Provenance lock: sha1 + size + mtime of every input a script consumed.

    Merges into ablation/inputs.json so each script contributes its own inputs without
    clobbering the others'.
    """
    out = out or os.path.join(ABL, "inputs.json")
    db = json.load(open(out)) if os.path.exists(out) else {}
    for p in paths:
        st = os.stat(p)
        db[os.path.relpath(p, REPO)] = {"sha1": sha1(p), "bytes": st.st_size,
                                        "mtime": int(st.st_mtime)}
    json.dump(dict(sorted(db.items())), open(out, "w"), indent=2)
    return db


def decode(r):
    """pycocotools RLE -> bool array. Counts arrive as str from JSON."""
    c = r["counts"]
    return maskUtils.decode({"size": r["size"],
                             "counts": c.encode("ascii") if isinstance(c, str) else c}
                            ).astype(bool)


def load_gt(path=GT):
    return json.load(open(path))


def load_preds(path):
    return json.load(open(path))["preds"]


def canopy_mask(gt_tile):
    can = np.array(E.raster(gt_tile["canopy"]))
    return can.any(0) if len(can) else np.zeros((E.RES, E.RES), bool)


def per_tile(preds_path, gt, want=("iou", "score", "ignore"), progress=100):
    """Stream tiles, yielding one dict per tile. Masks are freed as we go.

    Yields: tid, pm (Npred,RES,RES bool), gm (Ngt,RES,RES bool), boxes (Npred,4 @2048),
            scores (Npred,), ignore (Npred,) bool.
    Callers reduce each tile to whatever they need and must not retain `pm`/`gm`.

    `ignore` comes from the preds file's own `canopy_ignore` field where present (our runs
    write it at prediction time). Baseline preds that lack it -- DetecTree2 -- get it
    recomputed here under the identical >50%-in-canopy rule its own scorer applies
    (`detectree2_baseline/score_detectree2.py:59-61`), so both arms are scored alike.
    """
    preds = load_preds(preds_path)
    tids = sorted(gt)
    for k, tid in enumerate(tids):
        p = preds[tid]
        pm = (np.stack([decode(r) for r in p["masks_rle"]]) if p["masks_rle"]
              else np.zeros((0, E.RES, E.RES), bool))
        gm = np.array(E.raster(gt[tid]["trees"]))
        if "canopy_ignore" in p:
            ign = np.asarray(p["canopy_ignore"], bool)
        else:
            can = canopy_mask(gt[tid])
            ign = np.array([bool(m.sum()) and (m & can).sum() / m.sum() > 0.5 for m in pm],
                           dtype=bool)
            del can
        yield {"tid": tid, "i": k, "n": len(tids), "pm": pm, "gm": gm,
               "boxes": np.asarray(p["boxes_2048"], np.float32).reshape(-1, 4),
               "scores": np.asarray(p["scores"], np.float32),
               "ignore": ign}
        del pm, gm
        if progress and ((k + 1) % progress == 0 or k + 1 == len(tids)):
            print(f"    {k + 1}/{len(tids)}", flush=True)


def band(vals, nd=4):
    """mean, sample std. ddof=1 is the repo convention -- the published 0.630 +/- 0.005
    reproduces only with ddof=1; numpy's default ddof=0 would print 0.004."""
    v = np.asarray(vals, float)
    return [round(float(v.mean()), nd),
            round(float(v.std(ddof=1)), nd) if len(v) > 1 else 0.0]

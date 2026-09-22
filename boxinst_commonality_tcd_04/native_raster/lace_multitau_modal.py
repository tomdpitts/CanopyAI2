"""Modal CPU app: LACE's 439 masks at the native 2048 raster, at tau = 0.35 AND 0.40.

Built to be run ONCE. Assume it drops out halfway.

WHAT IT DOES. Re-renders seed 0's masks at 2048 from the SAME frozen detections (boxes and
scores copied verbatim from the published row, so the detection set and its ranking are
byte-identical and the raster comparison is paired). The EM posterior is rendered ONCE per box
and thresholded at BOTH taus, because only `prob >= mask_thr` depends on tau -- measured cost
1.5x a single-tau pass rather than 2x.

WHY THESE TAUS. tau=0.25 was selected on the 108 val tiles AT 512. Re-selecting on the same
tiles at 2048 (val_knobs_local.py, 2026-09-07) moves the AP50 optimum to 0.40, with 0.35 the
runner-up and the AP50:95 optimum. Both are carried so the choice is visible rather than
asserted. tau=0.25 already exists (preds_lace_s0_r2048.json) and is NOT recomputed.

    tau   val AP50@2048   val AP50:95@2048
    0.25    0.5936          0.2364
    0.35    0.6032          0.2445   <- AP50:95 optimum
    0.40    0.6045          0.2436   <- AP50 optimum (the pre-registered criterion)

ROBUSTNESS (the whole point of this file)
  * PER-TILE CHECKPOINTS. Output is JSONL, one line per tile, appended and committed to the
    volume every COMMIT_EVERY tiles. A drop costs at most COMMIT_EVERY tiles of work.
  * RESUME. On start, both JSONL files are read and only tiles present in EVERY tau file are
    skipped. A truncated final line (killed mid-write) is detected and discarded.
  * NO OVERWRITE. Everything lands under out/native_raster/multitau/, a directory that does
    not exist (checked 2026-09-07). Existing artifacts -- preds_lace_s0_r2048.json (tau=0.25),
    val_knobs_*.json -- are never opened for writing. Append mode only; a resumed run adds
    lines, it does not rewrite the file.
  * CORRECTNESS GATE. This file renders the posterior itself in order to share it across taus,
    which risks silently diverging from evaluate.pred_instance_masks (an earlier prototype of
    mine dropped the `delta` half-cell shift -- for this masker delta = 4 px at 2048, so the
    masks would have been misregistered and nothing would have complained). So on EVERY start,
    before any real work, the first tile's masks are computed BOTH ways and asserted equal for
    both taus. Mismatch aborts before a cent is spent.
  * PROGRESS. Per-tile logging with flush, so `modal app logs` shows live progress.
  * A failing tile is recorded and skipped rather than killing the run; the run reports a
    non-empty failure list loudly at the end.

Usage (detached, survives disconnection):
    python detach.py deploy lace_multitau_modal.py
    python detach.py spawn tcd04-lace-multitau render --seed 0 --note "..."
    python detach.py status
"""
import json
import os

import modal

APP_NAME = "tcd04-lace-multitau"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PH4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")
STUBS = os.path.join(PH4, "stubs")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"

RES = 2048
TAUS = (0.35, 0.40)
PRIOR_WEIGHT = 0.30                 # frozen at the 512 selection; see NATIVE_RASTER.md
KAPPA_SCALE = 1.60
COMMIT_EVERY = 10                   # tiles between volume commits -> max work lost on a drop

FEAT_DIR = "/vol/feat_4p_test"
EM_PATH = "/vol/out/em_model_4p_fix.npz"
OUT_DIR = "/vol/out/native_raster/multitau"          # NEW directory, no collision
SRC_DIR = "/vol/out/native_raster/src"               # staged published detections

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6",
                 "transformers==4.57.1", "pillow", "contourpy", "pycocotools")
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py",
            "dapt/decode.py", "dapt/eval.py", "dapt/head.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py",
            "em.py", "prepare_test.py", "cache_test.py", "cache_train_tiles.py",
            "test_gt.json"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
STUB_FILES = {
    "boxinst": ("__init__.py", "cache_feats.py"),
    "boxinst_commonality": ("__init__.py", "em.py"),
    "boxinst_tcd": ("__init__.py", "build_canopy.py", "cache.py", "prepare.py"),
}
for _pkg, _files in STUB_FILES.items():
    for _f in _files:
        image = image.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}")


def _jsonl_done(path):
    """Tile ids already written. Tolerates a truncated final line from a hard kill."""
    if not os.path.exists(path):
        return set(), 0
    done, good_bytes = set(), 0
    with open(path, "rb") as f:
        for raw in f:
            try:
                rec = json.loads(raw.decode("utf-8"))
                done.add(rec["tile"])
                good_bytes += len(raw)
            except Exception:
                break                      # truncated/corrupt tail: stop here
    return done, good_bytes


def _truncate_to(path, nbytes):
    """Drop a partial trailing line so appends stay valid JSONL."""
    if os.path.exists(path) and os.path.getsize(path) != nbytes:
        with open(path, "r+b") as f:
            f.truncate(nbytes)
        return True
    return False


@app.function(image=image, volumes={"/vol": vol}, timeout=12 * 3600,
              cpu=8, memory=32768)
def render(seed: int = 0, limit: int = 0):
    import sys
    import time

    import numpy as np
    from PIL import Image
    from pycocotools import mask as maskUtils

    sys.path.insert(0, P)
    from boxinst_commonality_tcd_04 import evaluate as E

    src = f"{SRC_DIR}/lace_s{seed}.json"
    assert os.path.exists(src), f"{src} not staged"
    published = json.load(open(src))
    meta_in, preds_in = published["meta"], published["preds"]
    assert meta_in["mask_res"] == 512 and meta_in["tile_px"] == 2048

    masker = E.TCDMasker(EM_PATH)
    gt = json.load(open(f"{PKG_R}/test_gt.json"))
    tiles = sorted(preds_in)
    if limit:
        tiles = tiles[:limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {t: f"{OUT_DIR}/preds_lace_s{seed}_r{RES}_tau{int(t*100):03d}.jsonl"
             for t in TAUS}

    # ---- resume: a tile counts as done only if present in EVERY tau file ---------------
    done = None
    for t, p in paths.items():
        d, nb = _jsonl_done(p)
        if _truncate_to(p, nb):
            print(f"[multitau] truncated partial tail of {os.path.basename(p)}", flush=True)
        done = d if done is None else (done & d)
    done = done or set()
    todo = [t for t in tiles if t not in done]
    print(f"[multitau] seed {seed}: {len(tiles)} tiles, {len(done)} already done, "
          f"{len(todo)} to do; taus={TAUS}", flush=True)

    # ---- the render path, character-for-character from evaluate.pred_instance_masks ----
    origin = getattr(masker, "origin", getattr(masker, "s", 2 * (2048.0 / RES)) / 2.0)
    s = getattr(masker, "s", 2 * (2048.0 / RES))
    scale = 2048.0 / RES
    delta = int(round((origin - s / 2.0) / scale))

    def masks_for_box(zn, g, b):
        """-> (box_mask, prob) with the delta shift applied, exactly as the library does."""
        idx, r = masker.box_mask(zn, g, b, prior_weight=PRIOR_WEIGHT,
                                 kappa_scale=KAPPA_SCALE)
        grid = np.zeros(g * g, np.float32)
        grid[idx] = r
        prob = np.array(Image.fromarray(grid.reshape(g, g)).resize(
            (RES, RES), Image.BILINEAR))
        if delta:
            shifted = np.zeros_like(prob)
            src_s = slice(max(0, -delta), RES - max(0, delta))
            dst_s = slice(max(0, delta), RES - max(0, -delta))
            shifted[dst_s, dst_s] = prob[src_s, src_s]
            prob = shifted
        x0, y0, x1, y1 = (np.asarray(b) / scale)
        box_m = np.zeros((RES, RES), bool)
        box_m[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
        return box_m, prob

    def rle(m):
        return {"size": [RES, RES],
                "counts": maskUtils.encode(
                    np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")}

    # ---- CORRECTNESS GATE: abort before spending anything if we diverge ---------------
    gate_tile = next((t for t in tiles if len(preds_in[t]["boxes_2048"])), None)
    assert gate_tile, "no tile has detections"
    gfeat = np.load(os.path.join(FEAT_DIR, gate_tile + ".npy")).astype(np.float32)
    gg = gfeat.shape[-1]
    gzn = masker.project(gfeat)
    gbx = np.asarray(preds_in[gate_tile]["boxes_2048"], np.float32).reshape(-1, 4)
    for tau in TAUS:
        ref = E.pred_instance_masks(masker, gzn, gg, gbx, res=RES, scale=scale,
                                    mask_thr=tau, prior_weight=PRIOR_WEIGHT,
                                    kappa_scale=KAPPA_SCALE)
        mine = np.array([(p >= tau) & bm for bm, p in
                         (masks_for_box(gzn, gg, b) for b in gbx)])
        assert ref.shape == mine.shape and np.array_equal(ref, mine), (
            f"GATE FAILED at tau={tau} on {gate_tile}: this file's render path diverges "
            f"from evaluate.pred_instance_masks. Aborting before spending compute.")
    print(f"[multitau] correctness gate PASSED on {gate_tile} "
          f"({len(gbx)} boxes, delta={delta}) for taus {TAUS}", flush=True)
    del gfeat, gzn

    # ---- main loop --------------------------------------------------------------------
    fhs = {t: open(p, "a") for t, p in paths.items()}
    t0, n_det, failed = time.time(), 0, []
    try:
        for k, tid in enumerate(todo):
            try:
                rec_in = preds_in[tid]
                bx = np.asarray(rec_in["boxes_2048"], np.float32).reshape(-1, 4)
                if len(bx) == 0:
                    for tau in TAUS:
                        fhs[tau].write(json.dumps(
                            {"tile": tid, "boxes_2048": [], "scores": [],
                             "canopy_ignore": [], "masks_rle": []}) + "\n")
                else:
                    feat = np.load(os.path.join(FEAT_DIR, tid + ".npy")).astype(np.float32)
                    g = feat.shape[-1]
                    zn = masker.project(feat)
                    can = np.array(E.raster(gt[tid]["canopy"], res=RES, scale=scale))
                    can = can.any(0) if len(can) else np.zeros((RES, RES), bool)
                    per_tau = {tau: {"rles": [], "ci": []} for tau in TAUS}
                    for b in bx:
                        box_m, prob = masks_for_box(zn, g, b)   # rendered ONCE
                        for tau in TAUS:
                            m = (prob >= tau) & box_m
                            per_tau[tau]["rles"].append(rle(m))
                            tot = int(m.sum())
                            per_tau[tau]["ci"].append(
                                bool(tot) and float((m & can).sum()) / tot > 0.5)
                    for tau in TAUS:
                        fhs[tau].write(json.dumps(
                            {"tile": tid, "boxes_2048": rec_in["boxes_2048"],
                             "scores": rec_in["scores"],
                             "canopy_ignore": per_tau[tau]["ci"],
                             "masks_rle": per_tau[tau]["rles"]}) + "\n")
                    n_det += len(bx)
                    del feat, zn
            except Exception as e:                       # one bad tile must not kill the run
                failed.append({"tile": tid, "error": f"{type(e).__name__}: {e}"})
                print(f"  !! FAILED {tid}: {type(e).__name__}: {e}", flush=True)

            if (k + 1) % COMMIT_EVERY == 0 or k + 1 == len(todo):
                for fh in fhs.values():
                    fh.flush()
                    os.fsync(fh.fileno())
                vol.commit()
                el = time.time() - t0
                rate = el / max(1, k + 1)
                print(f"  {k+1}/{len(todo)} tiles  {n_det} dets  {el:.0f}s  "
                      f"eta {rate*(len(todo)-k-1)/60:.0f} min  [committed]", flush=True)
    finally:
        for fh in fhs.values():
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
        vol.commit()

    summary = {"seed": seed, "res": RES, "taus": list(TAUS), "n_tiles_written": len(tiles),
               "n_det": n_det, "prior_weight": PRIOR_WEIGHT, "kappa_scale": KAPPA_SCALE,
               "failed": failed, "secs": round(time.time() - t0, 1),
               "files": {str(t): paths[t] for t in TAUS},
               "provenance": "boxes + scores verbatim from the published 512 row; posterior "
                             "rendered once per box and thresholded at each tau; render path "
                             "asserted equal to evaluate.pred_instance_masks on tile "
                             f"{gate_tile} at every tau before the run."}
    with open(f"{OUT_DIR}/summary_s{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    if failed:
        print(f"[multitau] *** {len(failed)} TILES FAILED *** {failed[:5]}", flush=True)
    print(f"[multitau] done in {summary['secs']}s, {n_det} dets", flush=True)
    return summary


@app.local_entrypoint()
def main(seed: int = 0, limit: int = 0):
    print(json.dumps(render.remote(seed, limit), indent=2)[:3000])

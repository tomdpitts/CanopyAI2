"""SelvaBox -> SAM 3 with the RELEASED SAM placement (2026-10-14, local MPS, provisional).

Selection = the seam-fixed one (`selvabox_seam.select(raw_dets, EDGE_FRAC, stored_keys)`, identical
boxes and scores to seamfix/preds_selvabox_seamfix_on_r2048.json). Change vs that file: each kept
box is prompted on its OWN source 1024 subtile (as CanopyRS `segmenter.py:137-147` /
`models/segmenter/sam3.py`): subtile resized to 1777 (wrapper default), box scaled, box padding 0,
`multimask_output=False`, mask resized back to 1024 with NEAREST (sam3.py:227) and pasted into the
2048 frame. NOT applied: CanopyRS mask post-processing (512 px downscale, ring removal, <50 px^2
parts) and its mask-stage aggregator. Source subtile = the raw_dets entry whose (2-dp box, 4-dp
score) key equals the kept box's.

Tile order: every 5th test tile first (88, the estimate sample), then the rest. Per-tile JSON
checkpoint, resume by listing.

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.selvabox_persub.selvabox_persub_local
"""
import json
import os
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mu

from boxinst_commonality_tcd_04.native_raster import selvabox_seam as SB
from boxinst_commonality_tcd_04.native_raster import selvabox_seamfix_local as L

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
TILES = os.path.join(HERE, "tiles_mps")
SAM_T, SUB, BATCH, DEV = 1777, 1024, 30, "mps"


def order():
    tids = sorted(json.load(open(os.path.join(PKG, "test_gt.json"))))
    first = tids[::5]
    return first + [t for t in tids if t not in set(first)], len(first)


def prompt_subtile(sam, proc, crop, boxes):
    import torch
    img = crop.resize((SAM_T, SAM_T), Image.BILINEAR)
    f = SAM_T / SUB
    inputs = proc(images=img, input_boxes=[(boxes * f).tolist()], return_tensors="pt")
    with torch.no_grad():
        emb = sam.get_image_embeddings(inputs["pixel_values"].to(DEV))
    masks, scores = [], []
    for i in range(0, len(boxes), BATCH):
        with torch.no_grad():
            so = sam(input_boxes=inputs["input_boxes"][:, i:i + BATCH].to(DEV), image_embeddings=emb,
                     multimask_output=False)
        m = proc.post_process_masks(so.pred_masks.cpu(), inputs["original_sizes"])[0].squeeze(1).numpy()
        sc = np.asarray(so.iou_scores.cpu()).reshape(len(m), -1)[:, 0]
        for mm, s in zip(m.astype(np.uint8), sc):
            masks.append(np.asarray(Image.fromarray(mm).resize((SUB, SUB), Image.NEAREST)).astype(bool))
            scores.append(round(float(s), 4))
    return masks, scores


def main():
    os.makedirs(TILES, exist_ok=True)
    tids, n_first = order()
    done = {f[:-5] for f in os.listdir(TILES) if f.endswith(".json")}
    todo = [t for t in tids if t not in done]
    print(f"[persub] {len(tids)} tiles ({n_first} estimate sample first), {len(done)} done, "
          f"{len(todo)} to do -> {TILES}", flush=True)
    sam, proc = L.load_sam(DEV)
    t0 = time.time()
    for n, t in enumerate(todo):
        r = json.load(open(os.path.join(L.SRC_TILES, t + ".json")))
        old = {SB.key(b, s) for b, s in zip(r["boxes_2048"], r["scores"])}
        B, S, _ = SB.select(r["raw_dets"], SB.EDGE_FRAC, old)
        src = {}
        for d in r["raw_dets"]:
            for b, s in zip(d["boxes_2048"], d["scores"]):
                src.setdefault(SB.key(b, s), (d["oy"], d["ox"]))
        keys = [SB.key(b, s) for b, s in zip(B, S)]
        origin = [src[k] for k in keys]
        rgb = L.load_rgb(t) if len(B) else None
        rles, sams = [None] * len(B), [None] * len(B)
        for oy, ox in sorted(set(origin)):
            idx = [i for i, o in enumerate(origin) if o == (oy, ox)]
            loc = B[idx] - np.array([ox, oy, ox, oy], np.float64)
            assert (loc >= -0.01).all() and (loc <= SUB + 0.01).all(), (t, oy, ox)
            ms, sc = prompt_subtile(sam, proc, rgb.crop((ox, oy, ox + SUB, oy + SUB)), loc)
            for i, m, s in zip(idx, ms, sc):
                full = np.zeros((2048, 2048), np.uint8); full[oy:oy + SUB, ox:ox + SUB] = m
                e = mu.encode(np.asfortranarray(full))
                rles[i] = {"size": [2048, 2048], "counts": e["counts"].decode("ascii")}
                sams[i] = s
        rec = {"boxes_2048": np.round(B, 2).tolist(), "scores": np.round(S, 4).tolist(),
               "sam_scores": sams, "masks_rle": rles, "src_subtile": [list(o) for o in origin],
               "device": DEV}
        tmp = os.path.join(TILES, t + ".json.tmp")
        json.dump(rec, open(tmp, "w")); os.replace(tmp, os.path.join(TILES, t + ".json"))
        k = len(done) + n + 1
        el = time.time() - t0
        print(f"  {k}/{len(tids)} {t} {len(B)} boxes  {el:.0f}s  eta {el/(n+1)*(len(todo)-n-1):.0f}s"
              + ("  [ESTIMATE SAMPLE COMPLETE]" if k == n_first else ""), flush=True)
    print("[persub] done", flush=True)


if __name__ == "__main__":
    main()

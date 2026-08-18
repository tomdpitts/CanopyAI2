"""Render 3 random unseen tiles per biome, with GT crowns drawn, for a human sparse/not call.

Crown boxes come from each annotation's COCO `bbox`, which is correct for BOTH polygon and
RLE segmentations -- unlike the `area` field, which is 0 on every RLE ann in this dataset
(see tile_index.seg_to_rle). 647 of the 266k ITC anns are RLE-encoded.

`biome` is a GEOGRAPHIC label (which WWF/Olson polygon the tile centroid falls in), not a
canopy-density measurement -- Desert & Xeric averages 99.5 crowns/tile in this corpus, denser
than Temperate Broadleaf. So the sparse/not-sparse line has to be drawn by eye, on real tiles,
before the slice is built. This renders the evidence.

Panels are drawn from the UNSEEN pool only (the 900 the detector trained on are excluded), so
what you judge is what the slice would actually contain.

Products:
    claude_outputs/tcd_biome_contact/b{code}_{n}_{tid}.jpg
    claude_outputs/tcd_biome_contact/panels.json     (per-panel captions + stats)

Usage:
    .venv/bin/python -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.contact_sheet
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.tile_index import (
    BIOME, DIRS, REPO, SPARSE_DEFAULT, load)

Image.MAX_IMAGE_PIXELS = None

OUT = os.path.join(REPO, "claude_outputs/tcd_biome_contact")
PER_BIOME = 3
SIDE = 640                      # 2048 -> 640 keeps small crowns visible at ~3.2x downscale
SEED = 0


def crown_boxes(meta):
    """xyxy @2048 for every individual-tree-crown ann. COCO bbox is xywh and always present;
    fall back to the polygon extent if an ann somehow lacks it."""
    out = []
    for a in json.loads(meta["coco_annotations"]):
        if a["category_id"] != 2:
            continue
        b = a.get("bbox")
        if b and b[2] > 0 and b[3] > 0:
            out.append((b[0], b[1], b[0] + b[2], b[1] + b[3]))
            continue
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg:
            p = np.asarray(max(seg, key=len), np.float32).reshape(-1, 2)
            out.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
    return out


def render(rec):
    """Returns (annotated thumbnail, valid_frac). OAM orthos are rotated/clipped mosaics, so
    many tiles carry a black no-data wedge; valid_frac = fraction of non-black pixels, and a
    low value means the tile is mostly empty regardless of what its biome says."""
    tid, split = rec["tid"], rec["split"]
    img = Image.open(os.path.join(DIRS[split], tid + ".tif")).convert("RGB")
    meta = json.load(open(os.path.join(DIRS[split], tid + "_meta.json")))
    a = np.asarray(img)[::8, ::8]
    valid = float((a.max(2) > 0).mean())
    s = SIDE / img.width
    img = img.resize((SIDE, int(img.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in crown_boxes(meta):
        d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 214, 10), width=1)
    return img, valid


def main():
    idx = load()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    unseen = [r for r in idx["tiles"] if not r["seen"]]
    panels = []
    for b in sorted({r["biome"] for r in unseen},
                    key=lambda b: (b not in SPARSE_DEFAULT,
                                   -sum(1 for r in unseen if r["biome"] == b))):
        pool = [r for r in unseen if r["biome"] == b]
        pick = [pool[i] for i in rng.choice(len(pool), min(PER_BIOME, len(pool)),
                                            replace=False)]
        for n, rec in enumerate(pick):
            fn = f"b{b}_{n}_{rec['tid']}.jpg"
            im, valid = render(rec)
            im.save(os.path.join(OUT, fn), quality=82, optimize=True)
            panels.append({
                "file": fn, "biome": b,
                "biome_label": BIOME.get(b, f"code {b} (not a WWF biome)"),
                "proposed_sparse": b in SPARSE_DEFAULT,
                "tid": rec["tid"], "split": rec["split"], "image_id": rec["image_id"],
                "ecoregion": rec["biome_name"] or "—", "country": rec["country"] or "—",
                "n_crowns": rec["n_crowns"],
                "canopy_pct": round(rec["canopy_frac"] * 100, 1),
                "crown_pct": round(rec["crown_frac"] * 100, 1),
                "valid_pct": round(valid * 100, 1),
                "scene_clean": rec["scene_clean"],
            })
            print(f"  {fn}  {rec['n_crowns']} crowns  valid={valid*100:.0f}%",
                  flush=True)
    json.dump({"seed": SEED, "per_biome": PER_BIOME, "side": SIDE, "panels": panels},
              open(os.path.join(OUT, "panels.json"), "w"), indent=1)
    print(f"[contact_sheet] {len(panels)} panels -> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()

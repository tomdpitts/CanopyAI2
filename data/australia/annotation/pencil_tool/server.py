"""Interactive magic-wand crown editor (SAM point-prompt + Pencil touch-up).

Tap inside a crown -> SAM segments it. Among SAM's 3 candidate masks we pick the
**greenest** (max excess-green) so it grabs the canopy, not the cast shadow. Then
paint/erase with the Pencil. CPU-only (GPU busy with finetune_v2); each tile's SAM
embedding is computed once on load and cached, so taps are fast after that.

Run:  .venv/bin/python server.py   (binds 0.0.0.0:8082)  Open: http://100.124.137.30:8082
"""
import base64
import glob
import io
import json
import os
import threading

import numpy as np
import torch
from scipy import ndimage as ndi
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # repo root (4 up)
TILE_DIR = os.path.join(HERE, "..", "tiles", "WON")
SAM_CKPT = os.path.join(ROOT, "tmp/samtest/sam_vit_b_01ec64.pth")
EDIT = os.path.join(HERE, "edited"); os.makedirs(EDIT, exist_ok=True)
MASK0 = os.path.join(HERE, "mask0"); os.makedirs(MASK0, exist_ok=True)   # preloaded proposals
PROP = os.path.join(ROOT, "tmp/samtest/sam_polygons_v3.json")
DEVICE = "cpu"   # GPU in use by finetune_v2

sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(DEVICE)
predictor = SamPredictor(sam)
_emb = {}                       # tid -> (features, original_size, input_size)
_lock = threading.Lock()
TILES = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(TILE_DIR, "*.png")))]


def _init_proposals():
    """Rasterize SAM v3 proposals -> a preloaded mask per tile (once)."""
    d = json.load(open(PROP))
    by = {os.path.splitext(os.path.basename(x["image_path"]))[0]: x
          for x in d["images"] if x["domain"] == "WON"}
    for tid in TILES:
        mp = os.path.join(MASK0, tid + ".png")
        if os.path.exists(mp) or tid not in by:
            continue
        im = by[tid]
        m = Image.new("L", (im["width"], im["height"]), 0)
        dr = ImageDraw.Draw(m)
        for ann in im["annotations"]:
            for poly in ann.get("polygons", []):
                if len(poly) >= 3:
                    dr.polygon([tuple(p) for p in poly], fill=255)
        m.save(mp)


_init_proposals()


def rgb_of(tid):
    return np.array(Image.open(os.path.join(TILE_DIR, tid + ".png")).convert("RGB"))


def ensure(tid):
    """Set predictor to tile tid, computing+caching its embedding once."""
    if tid in _emb:
        f, o, i = _emb[tid]
        predictor.features, predictor.original_size, predictor.input_size = f, o, i
        predictor.is_image_set = True
        return
    predictor.set_image(rgb_of(tid))
    _emb[tid] = (predictor.features, predictor.original_size, predictor.input_size)


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/tiles")
def api_tiles():
    return jsonify([{"id": t, "done": os.path.exists(os.path.join(EDIT, t + ".png"))} for t in TILES])


@app.route("/img/<tid>")
def img(tid):
    return send_file(os.path.join(TILE_DIR, tid + ".png"))


@app.route("/mask/<tid>")
def mask(tid):
    for p in (os.path.join(EDIT, tid + ".png"), os.path.join(MASK0, tid + ".png")):
        if os.path.exists(p):
            return send_file(p)
    return ("", 204)


@app.route("/save/<tid>", methods=["POST"])
def save(tid):
    b = request.get_json()["png"].split(",", 1)[1]
    Image.open(io.BytesIO(base64.b64decode(b))).convert("L").save(os.path.join(EDIT, tid + ".png"))
    return jsonify(ok=True)


@app.route("/embed/<tid>", methods=["POST"])
def embed(tid):
    with _lock:
        ensure(tid)
    return jsonify(ok=True)


@app.route("/segment/<tid>", methods=["POST"])
def segment(tid):
    body = request.get_json()
    pts = np.array(body["points"], dtype=np.float32)
    lbl = np.array(body["labels"], dtype=np.int32)
    with _lock:
        ensure(tid)
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    rgb = rgb_of(tid).astype(np.float32)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    H, W = exg.shape
    px = int(np.clip(pts[0][0], 0, W - 1)); py = int(np.clip(pts[0][1], 0, H - 1))
    # keep ONLY the connected component under the tapped point (drop disconnected
    # look-alike patches), and among the 3 SAM masks pick the greenest such blob.
    best_comp, best_s = None, -1e9
    for m in masks:
        lab, _ = ndi.label(m)
        cid = lab[py, px]
        if cid == 0:
            continue
        comp = lab == cid
        if comp.sum() < 15:
            continue
        s = float(exg[comp].mean())
        if s > best_s:
            best_s, best_comp = s, comp
    if best_comp is None:                       # tap landed off every mask
        best_comp = masks[int(np.argmax(scores))]
    out = best_comp.astype(np.uint8) * 255
    buf = io.BytesIO(); Image.fromarray(out, "L").save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/smooth", methods=["POST"])
def smooth():
    """Round jaggy edges per connected component (won't bridge/merge separate blobs)."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    b = request.get_json()["png"].split(",", 1)[1]
    m = np.array(Image.open(io.BytesIO(base64.b64decode(b))).convert("L")) > 127
    lab, n = ndi.label(m)
    out = np.zeros(m.shape, dtype=bool)
    for i in range(1, n + 1):
        comp = lab == i
        others = binary_dilation((lab > 0) & (lab != i), iterations=2)   # keep blobs apart
        out |= (gaussian_filter(comp.astype(np.float32), 1.5) > 0.5) & ~others
    res = out.astype(np.uint8) * 255
    buf = io.BytesIO(); Image.fromarray(res, "L").save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    print(f"[wand] {len(TILES)} WON tiles, SAM vit_b on {DEVICE}; http://100.124.137.30:8082", flush=True)
    app.run(host="0.0.0.0", port=8082, threaded=True)

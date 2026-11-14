"""Shared paths, GT and small helpers for the PhenoBench zero-shot run. Self-contained: nothing outside this folder is
imported except the repo itself (read only; _verbatim.verify and gate_parity check against repo source).

LACE code here (lace_core.py, _verbatim.py, gate_parity.py) is a byte-identical copy of the parity-gated port first
written for the LIVECell probe (2026-11-13); the gate is re-run in this folder (results/gate_parity*.json).

Runs: PB_RUN env var selects the run directory under results/ and its image list:
  sample60  60 val images, seed 0, 20 per acquisition date (results/sample60/selection.json; the first, provisional run)
  val       all 772 val images (results/val/selection.json, written by fetch.py)
"""
import json, os, sys

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "val")
RUN = os.environ.get("PB_RUN", "sample60")
RES = os.path.join(ROOT, "results", RUN)
CLS = {1: "crop", 2: "weed", 3: "crop_partial", 4: "weed_partial"}
SHIFTS = (0, 4, 8, 12)


def stems():
    return [f[:-4] for f in json.load(open(os.path.join(RES, "selection.json")))["images"]]


def have_data(stem):
    return all(os.path.exists(os.path.join(DATA, d, stem + ".png")) for d in ("images", "plant_instances", "semantics"))


def load_rgbf(stem):
    return np.asarray(Image.open(os.path.join(DATA, "images", stem + ".png")).convert("RGB"), np.float32) / 255.0


def gt_image(stem):
    """-> boxes (N,4) xyxy float (pixel extent [min, max+1)), gt RLEs, valid (N,), classes, ids. Instances sorted by id."""
    lab = np.asarray(Image.open(os.path.join(DATA, "plant_instances", stem + ".png"))).astype(np.int64)
    sem = np.asarray(Image.open(os.path.join(DATA, "semantics", stem + ".png"))).astype(np.int64)
    ids = [int(i) for i in np.unique(lab) if i > 0]
    boxes, rles, cls = [], [], []
    for i in ids:
        m = lab == i; ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        rles.append(maskUtils.encode(np.asfortranarray(m.astype(np.uint8))))
        c = np.bincount(sem[m], minlength=5); c[0] = 0; cls.append(CLS.get(int(c.argmax()), "bg"))
    valid = np.ones(len(ids), bool)                      # every id has >= 1 px and w, h >= 1 by construction
    return np.asarray(boxes, np.float32).reshape(-1, 4), rles, valid, cls, ids


def enc(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def rle_json(r):
    return {"size": [int(x) for x in r["size"]], "counts": r["counts"].decode("ascii") if isinstance(r["counts"], bytes) else r["counts"]}


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(H,W,3) -> content shifted up-left by (dy,dx), zero-filled, so a normal patch
    grid on the result samples patches starting at (dy,dx). A <=8px bottom/right strip
    is zero; TCD tiles are dense so this loses at most one patch row/col at the far
    edge (dropped at eval as pad anyway)."""
    out = np.zeros_like(arr)
    H, W = arr.shape[:2]
    out[:H - dy, :W - dx] = arr[dy:, dx:]
    return out


def check_shift():
    """_shift must be the verbatim repo function (phase4_features_tcd._shift)."""
    import ast, inspect
    f = os.path.join(os.path.dirname(ROOT), "modal_tcd_multiseed/phase4/phase4_features_tcd.py")
    text = open(f).read()
    orig = [ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_shift"][0]
    assert inspect.getsource(_shift).rstrip() == orig.rstrip(), "_shift drifted"


def features16(net, rgbf):
    """16-phase DINOv3 L24 cells on the 4 px lattice for a 1024^2 image (one deployed 1024 window per pass), returned as
    float32 after an fp16 round-trip (= the fp16 storage of the sample60 run and of feat_4p)."""
    import torch
    assert rgbf.shape == (1024, 1024, 3), rgbf.shape
    Z = np.zeros((256, 256, 1024), np.float16)
    for dy in SHIFTS:
        for dx in SHIFTS:
            x = torch.from_numpy(np.ascontiguousarray(_shift(rgbf, dy, dx))).permute(2, 0, 1)[None]
            Z[dy // 4::4, dx // 4::4] = net.extract(x.float())[0].float().cpu().numpy().transpose(1, 2, 0).astype(np.float16)
    torch.mps.synchronize()
    return Z

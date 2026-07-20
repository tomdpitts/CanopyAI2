"""Layers-trap-safe DINOv3-web feature extraction for NEON 400px tiles/patches.

Shared by the local MPS smoke and the Modal H100 run. Uses dapt.backbone.load_tile
(pads a <=512px RGB tile to a 512 canvas -> 32x32 patch grid at patch16) + net.extract
with the DETECTOR layers (21,22,23,24). A bare FrozenDinoV3Features falls back to
dapt.backbone's (3,6,9,12) — same 4096 out_dim, silently wrong features
([[dinov3-layers-default-trap]]); we hardcode the tuple + assert, and ship a parity
reference (ref_feat.npz) so a CUDA run can prove cosine>0.98 vs the local MPS features.

RGB-ONLY: reads only RGB tiles. No LiDAR/CHM/HSI.
"""
from __future__ import annotations

import os

import numpy as np
import torch

LAYERS_WEB = (21, 22, 23, 24)     # last 4 blocks of ViT-L/24 (== boxinst.cache_feats.LAYERS)
PARITY_TILE = "2018_SJER_3_252000_4104000_image_628"   # canonical NEON eval tile
PARITY_COS = 0.98


def build_net(device=None):
    from dapt.backbone import FrozenDinoV3Features
    net = FrozenDinoV3Features("web", layers=LAYERS_WEB, device=device)
    assert net.out_dim == 4096, f"expected 4096-dim web features, got {net.out_dim}"
    assert net.layers == LAYERS_WEB, f"WRONG LAYERS {net.layers} (trap!)"
    # cross-check against the canonical constant when importable (local only)
    try:
        from boxinst.cache_feats import LAYERS as REF
        assert tuple(REF) == LAYERS_WEB, (REF, LAYERS_WEB)
    except Exception:
        pass
    return net


def feat_for_image(net, path):
    """RGB tile path -> (4096,32,32) fp16 (pad-to-512, 32x32 patch grid)."""
    from dapt.backbone import load_tile
    x, (H0, W0) = load_tile(path)
    g = net.extract(x)[0].to(torch.float16).cpu().numpy()   # (4096,32,32)
    return g, (H0, W0)


def feat_for_pil(net, pil, size=512):
    """In-memory RGB PIL image (<=size on each side) -> (4096,32,32) fp16 + (H0,W0).
    Same zero-pad-to-512 as dapt.backbone.load_tile, but no file (for upscaled crops)."""
    pil = pil.convert("RGB")
    W0, H0 = pil.size
    if W0 > size or H0 > size:
        raise ValueError(f"crop {W0}x{H0} exceeds pad {size}")
    arr = np.asarray(pil, np.float32) / 255.0
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:H0, :W0] = arr
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    g = net.extract(x)[0].to(torch.float16).cpu().numpy()
    return g, (H0, W0)


def extract_dir(net, img_paths, out_dir, log_every=25):
    """Idempotent: writes out_dir/<stem>.npy per image, skips existing. Returns count."""
    import time
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in img_paths
            if not os.path.exists(os.path.join(
                out_dir, os.path.splitext(os.path.basename(p))[0] + ".npy"))]
    print(f"[extract] {len(todo)}/{len(img_paths)} -> {out_dir}", flush=True)
    t0 = time.time()
    for k, p in enumerate(todo):
        stem = os.path.splitext(os.path.basename(p))[0]
        g, _ = feat_for_image(net, p)
        np.save(os.path.join(out_dir, stem + ".npy"), g)
        if (k + 1) % log_every == 0 or k + 1 == len(todo):
            dt = time.time() - t0
            eta = (len(todo) - k - 1) * dt / (k + 1) / 60
            print(f"  {k+1}/{len(todo)}  {dt/(k+1):.2f}s/img  ETA {eta:.1f} min",
                  flush=True)
    return len(img_paths)


def make_parity_ref(net, tile_path, out_npz):
    """Compute + save the parity reference (local MPS)."""
    g, wh = feat_for_image(net, tile_path)
    np.savez(out_npz, feat=g, tile=os.path.basename(tile_path), wh=np.array(wh))
    print(f"[parity] wrote {out_npz} feat{g.shape} from {os.path.basename(tile_path)}")


def cosine_to_ref(net, tile_path, ref_npz):
    """Cosine between freshly-extracted features and a stored reference (no assert)."""
    ref = np.load(ref_npz)["feat"].astype(np.float32).ravel()
    cur, _ = feat_for_image(net, tile_path)
    cur = cur.astype(np.float32).ravel()
    return float(np.dot(cur, ref) / (np.linalg.norm(cur) * np.linalg.norm(ref) + 1e-9))


def parity_check(net, tile_path, ref_npz, gate=PARITY_COS, hard=True):
    """Cosine vs a reference. NOTE: the definitive layers-trap guard is the
    layers==(21,22,23,24)/out_dim==4096 assert in build_net (a trap gives cos~0.17).
    A cross-ENVIRONMENT ref (different transformers version / MPS vs CUDA) legitimately
    drifts to ~0.85, so compare only against a SAME-environment ref when hard=True."""
    cos = cosine_to_ref(net, tile_path, ref_npz)
    print(f"[parity] cosine(current, ref) = {cos:.4f} (gate >{gate}, hard={hard})",
          flush=True)
    if hard:
        assert cos > gate, f"PARITY FAIL cos={cos:.4f} vs same-env ref"
    return cos

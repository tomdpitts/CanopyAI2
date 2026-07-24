"""4-phase ("center-registration") real-8px feature extraction for 2048px TCD tiles.

Native TCD samples the frozen DINOv3-web backbone on ONE 16px patch grid ->
(4096,128,128), and Detector8 bilinearly UPSAMPLES that to an 8px grid (256x256) —
interpolated, no new sub-patch info. Here we run the backbone FOUR times, on the tile
shifted by every (dy,dx) in {0,8}px, and interleave the four 16px grids into a REAL
8px grid (256x256) where each cell is the phase whose patch is centered nearest it
("pick-one", stationary). The ONLY change vs native is real-8px vs interpolated-8px.

We keep the single-layer L24 slice (dims [3072:4096] of the (21,22,23,24) stack) as
the detector feature — the layer probe found L24 (1024-dim) ties the full 4096 for
detection, so real-8px L24 is the cheap head-to-head vs interp-8px L24.

Per 2048 tile, each phase is the SAME 2x2-of-1024-windows extraction as native
(cache_test.tile_feature), just on the shifted image. Products:
  - feat_4phase_L24 : (1024,256,256) fp16  REAL-8px L24  (detector, train+test)
  - native_4096     : (4096,128,128) fp16  = phase (0,0) unsliced = native feat_test,
                      kept for TEST tiles only, because the FIXED 4096-dim EM masker
                      needs full-4096 features for the box->mask step (only the
                      detector consumes the 1024-dim slice).

Geometry (see test_interleave_tcd.py, no GPU): phase (dy,dx) patch (gy,gx) center =
pixel (dx+8+16gx, dy+8+16gy). Union of the four phases' centers = every multiple of 8
-> a clean 8px lattice. Interleave: asm[:, dy//8::2, dx//8::2] = phase_feat.

Isolated: imports cache_test window constants read-only; writes nowhere shared.
"""
from __future__ import annotations

import os

import numpy as np

PHASES = ((0, 0), (0, 8), (8, 0), (8, 8))       # (dy=row shift, dx=col shift) px
CANVAS = 2048
PATCH = 16
GRID16 = CANVAS // PATCH                          # 128
GRID8 = GRID16 * 2                                # 256
L24_LO, L24_HI = 3072, 4096                       # layer 24 slice of (21,22,23,24)


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(H,W,3) -> content shifted up-left by (dy,dx), zero-filled, so a normal patch
    grid on the result samples patches starting at (dy,dx). A <=8px bottom/right strip
    is zero; TCD tiles are dense so this loses at most one patch row/col at the far
    edge (dropped at eval as pad anyway)."""
    out = np.zeros_like(arr)
    H, W = arr.shape[:2]
    out[:H - dy, :W - dx] = arr[dy:, dx:]
    return out


def _extract_2048(net, arr: np.ndarray) -> np.ndarray:
    """arr (2048,2048,3) float[0,1] -> (C,128,128) fp32, stitched from 2x2 1024
    windows. Byte-identical windowing to cache_test.tile_feature (edge-to-edge
    (0,1024) windows, each contributing its own 64x64 block)."""
    import torch
    from boxinst_commonality_tcd_04.cache_test import GRID2K, STARTS, WGRID, WIN
    out = None
    for gy, oy in enumerate(STARTS):
        for gx, ox in enumerate(STARTS):
            w = arr[oy:oy + WIN, ox:ox + WIN]
            x = torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None]
            f = net.extract(x.float())[0]                 # (C,64,64)
            if out is None:
                out = torch.zeros(f.shape[0], GRID2K, GRID2K)
            out[:, gy * WGRID:(gy + 1) * WGRID,
                gx * WGRID:(gx + 1) * WGRID] = f.float().cpu()
    return out.numpy()


def interleave(phase_feats: dict) -> np.ndarray:
    """{(dy,dx): (C,128,128)} for the 4 PHASES -> (C,256,256) real 8px grid.
    asm cell (Y,X) filled by the phase whose 16px patch is centered nearest pixel
    (8X+8, 8Y+8); by construction one phase per cell (pixel-unshuffle inverse)."""
    C = next(iter(phase_feats.values())).shape[0]
    dt = next(iter(phase_feats.values())).dtype
    asm = np.zeros((C, GRID8, GRID8), dtype=dt)
    for (dy, dx), f in phase_feats.items():
        asm[:, dy // 8::2, dx // 8::2] = f
    return asm


def feat_4phase(net, img, want_native=False):
    """PIL 2048 RGB -> ((1024,256,256) fp16 real-8px L24 [, (4096,128,128) fp16
    native-4096]). want_native returns the phase-(0,0) full-4096 grid for the masker."""
    arr = np.asarray(img.convert("RGB"), np.float32) / 255.0
    assert arr.shape[:2] == (CANVAS, CANVAS), f"expected 2048 tile, got {arr.shape}"
    native4096 = None
    phases = {}
    for (dy, dx) in PHASES:
        f = _extract_2048(net, _shift(arr, dy, dx))       # (4096,128,128) fp32
        if (dy, dx) == (0, 0) and want_native:
            native4096 = f.astype(np.float16)
        phases[(dy, dx)] = np.ascontiguousarray(f[L24_LO:L24_HI])   # (1024,128,128)
    asm = interleave(phases).astype(np.float16)            # (1024,256,256)
    return (asm, native4096) if want_native else asm


def registration_self_test(net, img, atol=1e-3):
    """Abort-fast guard on REAL DINO before the full extract:
      (1) invertibility: asm[:, 0::2, 0::2] == phase-(0,0) L24 (no shift).
      (2) real != interp: asm != bilinear x2 upsample of phase-(0,0) L24 (cos<0.999),
          i.e. the shifted passes carry genuinely new sub-patch info.
    Returns the metrics; raises on failure."""
    import torch
    asm, native = feat_4phase(net, img, want_native=True)
    phase0 = native[L24_LO:L24_HI].astype(np.float32)      # (1024,128,128)
    ee = asm[:, 0::2, 0::2].astype(np.float32)
    err = float(np.abs(ee - phase0).max())
    assert err < atol, f"REG FAIL: (even,even) != phase0 (maxabs {err:.4g})"
    up = torch.nn.functional.interpolate(
        torch.from_numpy(phase0)[None], scale_factor=2, mode="bilinear",
        align_corners=False)[0].numpy()
    a = asm.astype(np.float32).ravel(); b = up.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cos < 0.999, f"REG FAIL: assembled==bilinear (cos {cos:.5f}); no new info"
    print(f"[reg-test] OK invertible (maxabs {err:.2g}); real!=interp cos={cos:.4f}",
          flush=True)
    return {"invert_maxabs": err, "cos_vs_bilinear": cos}


def extract_split(net, img_iter, out_l24_dir, out_native_dir=None, log_every=25):
    """Idempotent per-tile extract. img_iter yields (tid, PIL_img). Writes
    out_l24_dir/<tid>.npy (1024,256,256); if out_native_dir, also the (4096,128,128)
    native grid there (test tiles). Skips tiles already written."""
    import time
    os.makedirs(out_l24_dir, exist_ok=True)
    if out_native_dir:
        os.makedirs(out_native_dir, exist_ok=True)
    items = list(img_iter)
    todo = [(t, im) for (t, im) in items
            if not os.path.exists(os.path.join(out_l24_dir, t + ".npy"))]
    print(f"[extract4p] {len(todo)}/{len(items)} -> {out_l24_dir}"
          f"{' (+native)' if out_native_dir else ''}", flush=True)
    t0 = time.time()
    for k, (tid, im) in enumerate(todo):
        if out_native_dir:
            asm, native = feat_4phase(net, im, want_native=True)
            np.save(os.path.join(out_native_dir, tid + ".npy"), native)
        else:
            asm = feat_4phase(net, im)
        np.save(os.path.join(out_l24_dir, tid + ".npy"), asm)
        if (k + 1) % log_every == 0 or k + 1 == len(todo):
            dt = time.time() - t0
            eta = (len(todo) - k - 1) * dt / (k + 1) / 60
            print(f"  {k+1}/{len(todo)}  {dt/(k+1):.2f}s/tile  ETA {eta:.1f} min",
                  flush=True)
    return len(items)

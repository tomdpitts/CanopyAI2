"""4-phase ("center-registration") DINOv3-web feature extraction for NEON.

Idea: the frozen backbone samples the image on ONE 16px patch grid, then the native
detector bilinearly UPSAMPLES those 16px features to an 8px grid (`DetectorS(up=2)`).
That 8px grid is *interpolated* — no new sub-patch information. Here we instead run the
backbone FOUR times, on the image shifted by every (dy,dx) in {0,8}px, and interleave
the four 16px feature grids into a REAL 8px grid (64x64) where each cell is filled by the
phase whose patch is centered nearest that cell ("pick-one", stationary). Same 4096 dim,
same targets/decode/stride-8 as native — the ONLY change vs native is real-8px vs
interpolated-8px features. See ../phase4/README.md.

Geometry (verified by test_interleave.py, no GPU):
    phase (dy,dx) patch (gy,gx) center = pixel (dx+8+16gx, dy+8+16gy).
    Union of the four phases' centers = every multiple of 8 -> a clean 8px lattice.
    Interleave into asm[:, 2gy+ry, 2gx+rx]: ry=dy//8 (rows), rx=dx//8 (cols).

Isolated: imports load_tile read-only; writes nowhere shared. Deletable with the folder.
"""
from __future__ import annotations

import os

import numpy as np

# (dy, dx) in pixels. Row shift = dy (feature axis 1), col shift = dx (feature axis 2).
PHASES = ((0, 0), (0, 8), (8, 0), (8, 8))
CANVAS = 512
PATCH = 16
GRID16 = CANVAS // PATCH          # 32
GRID8 = GRID16 * 2               # 64


def _shifted(canvas: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """canvas (H,W,3) -> content shifted up-left by (dy,dx) with zero fill, so a normal
    patch grid on the result samples patches starting at (dy,dx). Real content lives at
    the top-left; NEON 400px tiles have >=112px bottom-right pad, so an 8px shift is
    lossless there (a <=8px top/left strip is dropped only on ~500px tiles -> negligible,
    and the pad region is dropped at eval anyway)."""
    out = np.zeros_like(canvas)
    H, W = canvas.shape[:2]
    out[:H - dy, :W - dx] = canvas[dy:, dx:]
    return out


def interleave(phase_feats: dict) -> np.ndarray:
    """{(dy,dx): (C,32,32)} for the 4 PHASES -> (C,64,64) real 8px grid.

    asm cell (Y,X) is filled by the phase whose 16px patch is centered nearest pixel
    (8X+8, 8Y+8); by construction each cell maps to exactly one phase (pixel-unshuffle
    inverse: four sub-lattices, one per phase)."""
    C = next(iter(phase_feats.values())).shape[0]
    asm = np.zeros((C, GRID8, GRID8), dtype=next(iter(phase_feats.values())).dtype)
    for (dy, dx), f in phase_feats.items():
        asm[:, dy // 8::2, dx // 8::2] = f
    return asm


def four_phase_from_canvas(canvas: np.ndarray, extract_fn) -> np.ndarray:
    """canvas (512,512,3) float[0,1], extract_fn: (B,3,512,512)->(B,C,32,32) numpy.
    Returns (C,64,64). Batches the 4 phases into ONE extract call."""
    shifted = [_shifted(canvas, dy, dx) for (dy, dx) in PHASES]      # 4 x (512,512,3)
    batch = np.stack([s.transpose(2, 0, 1) for s in shifted])        # (4,3,512,512)
    feats = extract_fn(batch)                                        # (4,C,32,32)
    return interleave({p: feats[i] for i, p in enumerate(PHASES)})


def _net_extract_fn(net):
    """Wrap a FrozenDinoV3Features into a numpy (B,3,512,512)->(B,C,32,32) fn."""
    import torch

    def fn(batch_np):
        x = torch.from_numpy(np.ascontiguousarray(batch_np)).float()
        return net.extract(x).to(torch.float16).cpu().numpy()
    return fn


def feat_4phase(net, path):
    """RGB tile path -> (4096,64,64) fp16 real-8px feature + (H0,W0)."""
    from dapt.backbone import load_tile
    x, (H0, W0) = load_tile(path)                     # x (1,3,512,512) [0,1]
    canvas = x[0].permute(1, 2, 0).numpy()            # (512,512,3)
    asm = four_phase_from_canvas(canvas, _net_extract_fn(net))
    return asm.astype(np.float16), (H0, W0)


def registration_self_test(net, tile_path, atol=1e-3):
    """CUDA-side abort-fast guard, run before the full extract. Asserts the interleave is
    correct against REAL DINO features:
      (1) invertibility: the (even,even) sub-lattice of the assembled grid == a direct
          phase-(0,0) extraction (no shift) -> the interleave indexing is consistent.
      (2) real != interpolated: the (odd,·)/(·,odd) cells differ from a bilinear x2
          upsample of the phase-(0,0) grid (cosine < 0.999) -> the shifted passes carry
          genuinely new sub-patch information (else the experiment is a no-op).
    """
    import torch
    from dapt.backbone import load_tile
    x, _ = load_tile(tile_path)
    canvas = x[0].permute(1, 2, 0).numpy()
    ex = _net_extract_fn(net)
    asm = four_phase_from_canvas(canvas, ex)                         # (C,64,64)
    phase0 = ex(canvas.transpose(2, 0, 1)[None])[0]                  # (C,32,32) no shift
    # (1) invertibility
    ee = asm[:, 0::2, 0::2]
    err = float(np.abs(ee.astype(np.float32) - phase0.astype(np.float32)).max())
    assert err < atol, f"REG FAIL: (even,even) != phase0 (maxabs {err:.4g})"
    # (2) real != interpolated
    up = torch.nn.functional.interpolate(
        torch.from_numpy(phase0.astype(np.float32))[None], scale_factor=2,
        mode="bilinear", align_corners=False)[0].numpy()            # (C,64,64)
    a = asm.astype(np.float32).ravel(); b = up.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cos < 0.999, f"REG FAIL: assembled==bilinear (cos {cos:.5f}); no new info"
    print(f"[reg-test] OK invertible (maxabs {err:.2g}); real!=interp cos={cos:.4f}",
          flush=True)
    return {"invert_maxabs": err, "cos_vs_bilinear": cos}


def extract_dir_4phase(net, img_paths, out_dir, log_every=25):
    """Idempotent per-tile: out_dir/<stem>.npy = (4096,64,64) fp16. Skips existing."""
    import time
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in img_paths
            if not os.path.exists(os.path.join(
                out_dir, os.path.splitext(os.path.basename(p))[0] + ".npy"))]
    print(f"[extract4p] {len(todo)}/{len(img_paths)} -> {out_dir}", flush=True)
    t0 = time.time()
    for k, p in enumerate(todo):
        stem = os.path.splitext(os.path.basename(p))[0]
        g, _ = feat_4phase(net, p)
        np.save(os.path.join(out_dir, stem + ".npy"), g)
        if (k + 1) % log_every == 0 or k + 1 == len(todo):
            dt = time.time() - t0
            eta = (len(todo) - k - 1) * dt / (k + 1) / 60
            print(f"  {k+1}/{len(todo)}  {dt/(k+1):.2f}s/tile  ETA {eta:.1f} min",
                  flush=True)
    return len(img_paths)

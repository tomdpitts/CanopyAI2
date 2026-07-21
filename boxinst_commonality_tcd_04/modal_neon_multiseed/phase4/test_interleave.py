"""Local, NO-GPU registration test for the 4-phase interleave.

The interleave is pure geometry (which phase-patch fills which 8px cell) and is
INDEPENDENT of what maps a patch -> vector. So a mock "encoder" = per-patch mean of the
image is sufficient to validate the exact production interleave code before spending a
cent on Modal. Checks:
  (1) invertibility  - the (even,even) sub-lattice == a direct no-shift extraction.
  (2) blob registration - a bright blob at pixel (py,px) produces its max assembled
      response in the cell within 1 of encode()'s target cell int(px//8) (native
      stride-8 convention; a benign <=half-cell offset is absorbed by the offset head).
  (3) monotonic ordering - a left->right ramp yields left->right increasing responses
      (guards against an axis swap / flipped sub-lattice).

Run:  python phase4/test_interleave.py    (from modal_neon_multiseed/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4.phase4_features import (CANVAS, GRID8, four_phase_from_canvas,  # noqa: E402
                                    interleave, _shifted)


def mock_extract(batch):
    """(B,3,512,512) -> (B,1,32,32): mean intensity per 16x16 patch (channel-averaged)."""
    B = batch.shape[0]
    p = batch.reshape(B, 3, 32, 16, 32, 16).mean(axis=(1, 3, 5))     # (B,32,32)
    return p[:, None, :, :].astype(np.float32)


def test_invertibility():
    rng = np.random.default_rng(0)
    canvas = rng.random((CANVAS, CANVAS, 3), np.float32)
    asm = four_phase_from_canvas(canvas, mock_extract)               # (1,64,64)
    phase0 = mock_extract(canvas.transpose(2, 0, 1)[None])[0]        # (1,32,32)
    ee = asm[:, 0::2, 0::2]
    err = np.abs(ee - phase0).max()
    assert err < 1e-6, f"(even,even) != phase0: maxabs {err}"
    print(f"  [1] invertibility OK (maxabs {err:.2g})")


def test_blob_registration():
    worst = 0
    rng = np.random.default_rng(1)
    for _ in range(200):
        py, px = int(rng.integers(24, 376)), int(rng.integers(24, 376))  # inside 400px
        canvas = np.zeros((CANVAS, CANVAS, 3), np.float32)
        canvas[py - 3:py + 3, px - 3:px + 3] = 1.0                   # 6x6 bright blob
        asm = four_phase_from_canvas(canvas, mock_extract)[0]        # (64,64)
        Y, X = np.unravel_index(int(asm.argmax()), asm.shape)
        gx, gy = px // 8, py // 8                                    # encode target cell
        d = max(abs(X - gx), abs(Y - gy))
        worst = max(worst, d)
        assert d <= 1, f"blob ({px},{py}): max cell ({X},{Y}) vs target ({gx},{gy}) d={d}"
    print(f"  [2] blob registration OK (worst |cell offset| = {worst} <= 1)")


def test_monotonic_ordering():
    # Realistic 400px content (bottom-right is pad on real NEON tiles); an x-ramp over
    # the content. Check the INTERIOR (cols covering pixels well inside 400px, away from
    # the <=8px shift zero-fill edge) is monotonic left->right -> no axis swap.
    C = 400
    canvas = np.zeros((CANVAS, CANVAS, 3), np.float32)
    ramp = np.tile(np.linspace(0, 1, C, dtype=np.float32), (C, 1))
    canvas[:C, :C] = np.stack([ramp] * 3, -1)
    asm = four_phase_from_canvas(canvas, mock_extract)[0]            # (64,64)
    inner = asm[:C // 8 - 2, :C // 8 - 2].mean(0)                   # interior cols
    assert np.all(np.diff(inner) > -1e-6), "interior cols not monotonic in x -> axis bug"
    assert inner.argmax() == len(inner) - 1, "brightest interior x-cell not rightmost"
    print(f"  [3] monotonic x-ordering OK over interior {len(inner)} cols "
          f"(no axis swap / sub-lattice flip)")


def test_sublattice_assignment():
    """Each phase must land on its own sub-lattice and nowhere else."""
    feats = {}
    for i, p in enumerate(((0, 0), (0, 8), (8, 0), (8, 8))):
        f = np.full((1, 32, 32), i + 1, np.float32)
        feats[p] = f
    asm = interleave(feats)[0]
    assert (asm[0::2, 0::2] == 1).all() and (asm[0::2, 1::2] == 2).all()
    assert (asm[1::2, 0::2] == 3).all() and (asm[1::2, 1::2] == 4).all()
    print("  [4] sub-lattice assignment OK ((dy,dx) -> (row,col) parity)")


if __name__ == "__main__":
    print("4-phase interleave registration test (no GPU):")
    test_invertibility()
    test_blob_registration()
    test_monotonic_ordering()
    test_sublattice_assignment()
    print("ALL PASS")

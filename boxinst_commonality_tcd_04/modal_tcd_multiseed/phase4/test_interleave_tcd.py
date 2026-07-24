"""Pure-numpy geometry test for the 2048px 4-phase interleave (no GPU, free).

Checks the interleave indexing is a correct pixel-unshuffle inverse: the four phases
tile the 256x256 grid exactly, their sub-lattices are disjoint, and the (even,even)
sub-lattice recovers phase (0,0) (invertibility). Run before any GPU spend.

    .venv/bin/python -m boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.test_interleave_tcd
"""
import numpy as np

from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_features_tcd import (
    GRID8, GRID16, PHASES, interleave)


def test():
    C = 3
    # constant-per-phase feats so we can check exactly where each lands
    feats = {p: np.full((C, GRID16, GRID16), i + 1, np.float32)
             for i, p in enumerate(PHASES)}
    asm = interleave(feats)
    assert asm.shape == (C, GRID8, GRID8), asm.shape
    for i, (dy, dx) in enumerate(PHASES):
        sub = asm[:, dy // 8::2, dx // 8::2]
        assert np.all(sub == i + 1), f"phase {(dy,dx)} misplaced"
    assert asm.min() > 0, "some cell unfilled — phases don't tile the grid"
    assert np.all(asm[:, 0::2, 0::2] == 1), "invertibility: (even,even) != phase(0,0)"

    # positional check: distinct value per (phase, cell) round-trips to the right cell
    feats2 = {}
    for i, p in enumerate(PHASES):
        a = np.arange(GRID16 * GRID16, dtype=np.float32).reshape(1, GRID16, GRID16)
        feats2[p] = a + i * 1e6
    asm2 = interleave(feats2)
    for i, (dy, dx) in enumerate(PHASES):
        got = asm2[:, dy // 8::2, dx // 8::2]
        want = np.arange(GRID16 * GRID16, dtype=np.float32).reshape(1, GRID16, GRID16) \
            + i * 1e6
        assert np.array_equal(got, want), f"phase {(dy,dx)} cell order wrong"
    print(f"geometry OK: 4 phases tile {GRID8}x{GRID8}, sub-lattices disjoint, "
          f"invertible, cell order preserved")


if __name__ == "__main__":
    test()

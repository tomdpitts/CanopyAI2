"""Single source of truth for the image/azimuth orientation convention.

Both the feature computation (:mod:`shadow_prior.shadow_feature`) and the
augmentation (:mod:`shadow_prior.dataset`) must agree on how an azimuth angle maps
to a displacement in pixels and how a rotation of the tile updates the azimuth.
Keeping that convention in one place is what makes the *recompute-not-rotate*
invariant of design decision #2 testable: rotate the tile and the scalar with the
same convention, recompute, and the response is the rotation of the original.

Convention
----------
* Arrays are indexed ``[row, col]`` with ``row`` increasing **downward** (standard
  image/raster layout).
* ``azimuth_rad`` is the direction along which a shadow is displaced from the
  object that casts it. Its unit vector in pixel space is::

      u_col = cos(azimuth_rad)        # along +columns (image x, to the right)
      u_row = sin(azimuth_rad)        # along +rows    (image y, downward)

  The absolute zero of the azimuth is arbitrary for the experiment; only the
  *consistency* between feature computation and the rotation update matters.

Rotation update (derived, not guessed)
--------------------------------------
We rotate tiles with :func:`scipy.ndimage.rotate(tile, deg, reshape=False)`. That
routine maps an output pixel ``o`` to the input sample ``M @ o`` with
``M = [[c, s], [-s, c]]`` and ``c, s = cos(rad), sin(rad)`` (``rad`` = the degrees
in radians). Hence content at input ``p`` lands at output ``Mᵀ p``, so a direction
vector ``u`` rotates to ``Mᵀ u``. Substituting ``u = (sin φ, cos φ)`` gives
``(sin(φ - rad), cos(φ - rad))``; i.e. the azimuth updates as ``φ -> φ - rad``.
See :func:`rotate_azimuth`. (The unit test for the recompute invariant is the
empirical check on this sign.)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def azimuth_to_vector(azimuth_rad: float) -> Tuple[float, float]:
    """Return the unit displacement ``(u_row, u_col)`` for ``azimuth_rad``.

    Row-first to match NumPy ``[row, col]`` indexing.
    """
    return (math.sin(azimuth_rad), math.cos(azimuth_rad))


def vector_to_azimuth(shadow_x: float, shadow_y: float) -> float:
    """Map an annotation ``(shadow_x, shadow_y)`` shadow vector to this module's azimuth.

    The finetune CSV encodes the shadow direction **compass-style** (North = 0):
    ``shadow_x = sin(shadow_angle)`` is the East / +column component and
    ``shadow_y = cos(shadow_angle)`` is the **North** component. North points *up*
    (toward decreasing rows), so the row (y-down) displacement is ``-shadow_y``.
    We therefore return ``atan2(-shadow_y, shadow_x)`` so that
    ``azimuth_to_vector(phi) = (sin phi, cos phi)`` reproduces the displacement
    ``(u_row, u_col) = (-shadow_y, shadow_x)`` in the image array's row-down frame.

    Confirmed empirically 2026-06-30: the previous ``atan2(shadow_y, shadow_x)``
    treated North as row-down -- a vertical reflection that made the shadow arrow
    point at the mirror image of the true shadow (an angle-dependent error the
    180-degree-only :func:`shadow_prior.verify.recommend_convention` could not
    catch). Verified against annotated crowns with known base azimuths
    WON = 215 deg, BRU = 118 deg.
    """
    return math.atan2(-shadow_y, shadow_x)


def rotate_azimuth(azimuth_rad: float, rotation_deg: float) -> float:
    """Update an azimuth to stay consistent with a tile rotation of ``rotation_deg``.

    ``rotation_deg`` is the angle passed to :func:`scipy.ndimage.rotate`. The
    azimuth transforms as ``φ -> φ - radians(rotation_deg)`` (see module
    docstring). Result is wrapped into ``[0, 2π)``.
    """
    updated = azimuth_rad - math.radians(rotation_deg)
    return updated % (2.0 * math.pi)


def wrap_2pi(azimuth_rad: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into ``[0, 2π)``."""
    two_pi = 2.0 * math.pi
    return np.mod(azimuth_rad, two_pi)

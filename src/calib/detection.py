"""Checkerboard corner detection.

Uses the sector-based detector (``findChessboardCornersSB``) which is robust to
blur and uneven illumination (typical of accumulated/low-contrast images), with
a fallback to the classic detector.  For very high-resolution images the
search can be run on a downscaled copy and then refined to sub-pixel accuracy at
full resolution, which keeps detection fast without losing accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_SUBPIX_WINDOW = (11, 11)


def auto_downscale(width: int, target_width: int = 1500) -> int:
    """Integer downscale factor so the search image is ~``target_width`` wide."""
    return max(1, int(round(width / float(target_width))))


def _detect(gray: np.ndarray, pattern_size: tuple[int, int]):
    """Return (found, corners, method) using SB first, then the classic detector."""
    try:
        flags_sb = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
        if found:
            return True, corners, "sb"
    except cv2.error:
        pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    return bool(found), corners, "classic"


def find_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    downscale: int | str = "auto",
) -> tuple[bool, np.ndarray | None]:
    """Detect checkerboard corners in a grayscale image.

    Args:
        gray: Single-channel image.
        pattern_size: (cols, rows) inner-corner count.
        downscale: Integer factor for the initial search, or "auto" to pick one
            from the image width. Corners are always refined at full resolution.

    Returns:
        (found, corners) where corners has shape ``(N, 1, 2)`` in full-resolution
        pixel coordinates, or (False, None).
    """
    if gray is None or gray.ndim != 2:
        return False, None
    h, w = gray.shape[:2]
    ds = auto_downscale(w) if downscale == "auto" else max(1, int(downscale))

    search = gray
    if ds > 1:
        search = cv2.resize(gray, (w // ds, h // ds), interpolation=cv2.INTER_AREA)

    found, corners, method = _detect(search, pattern_size)
    if not found or corners is None:
        return False, None

    corners = corners.astype(np.float32)
    if ds > 1:
        corners *= float(ds)

    # Refine at full resolution when we searched on a downscaled image or used
    # the classic (non-subpixel) detector.
    if ds > 1 or method == "classic":
        corners = cv2.cornerSubPix(
            gray, corners, _SUBPIX_WINDOW, (-1, -1), _SUBPIX_CRITERIA
        )
    return True, corners


def draw_corners(
    image: np.ndarray, pattern_size: tuple[int, int], corners: np.ndarray
) -> np.ndarray:
    """Return a BGR copy of ``image`` with the detected corners drawn."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    return vis

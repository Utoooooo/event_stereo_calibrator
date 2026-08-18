"""Checkerboard specification used for calibration.

The board is described by its number of *inner corners* (the quantity OpenCV's
``findChessboardCorners`` expects), which equals ``(squares_x - 1, squares_y - 1)``
for a printed/displayed board, plus the physical size of one square.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoardSpec:
    """Checkerboard geometry.

    Attributes:
        cols: Number of inner corners along the horizontal axis.
        rows: Number of inner corners along the vertical axis.
        square_mm: Physical side length of one square, in millimetres. The
            resulting stereo translation vector is expressed in these units.
    """

    cols: int = 9
    rows: int = 6
    square_mm: float = 25.0

    @property
    def pattern_size(self) -> tuple[int, int]:
        """(cols, rows) tuple as expected by OpenCV chessboard functions."""
        return (self.cols, self.rows)

    @property
    def num_corners(self) -> int:
        return self.cols * self.rows

    def object_points(self) -> np.ndarray:
        """3D coordinates of the corners on the board plane (z = 0), in mm.

        Ordering matches OpenCV's corner ordering for ``pattern_size``.
        """
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        objp *= float(self.square_mm)
        return objp

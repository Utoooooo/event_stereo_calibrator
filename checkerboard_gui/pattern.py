"""Checkerboard pattern model and rendering.

The board is described by the number of *squares* along each axis and the size
of one square in screen pixels.  OpenCV calibration usually works with the
number of *inner corners*, so we expose that as a convenience property.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap


@dataclass
class CheckerboardConfig:
    """Configuration of a flashing checkerboard pattern.

    The display cycles through two phases:
      1. the checkerboard, shown for ``board_ms`` milliseconds;
      2. the "off" phase, shown for ``off_ms`` milliseconds, which is either an
         all-black frame or the colour-inverted checkerboard (see ``inverted``).

    Attributes:
        rows: Number of squares along the vertical axis.
        cols: Number of squares along the horizontal axis.
        square_px: Side length of a single square, in screen pixels.
        board_ms: How long the checkerboard is shown each cycle.
        off_ms: How long the off phase (black or inverted board) is shown.
        inverted: When True the off phase shows the inverted checkerboard
            instead of black, giving event cameras stronger, symmetric edges.
    """

    rows: int = 7
    cols: int = 10
    square_px: int = 100
    board_ms: float = 100.0
    off_ms: float = 20.0
    inverted: bool = False

    @property
    def width_px(self) -> int:
        return self.cols * self.square_px

    @property
    def height_px(self) -> int:
        return self.rows * self.square_px

    @property
    def inner_corners(self) -> tuple[int, int]:
        """Inner-corner count (cols-1, rows-1) used by OpenCV findChessboardCorners."""
        return (max(self.cols - 1, 0), max(self.rows - 1, 0))

    @property
    def cycle_ms(self) -> float:
        """Total duration of one board+off cycle, in milliseconds."""
        return self.board_ms + self.off_ms


def render_pixmap(
    config: CheckerboardConfig,
    light: QColor | None = None,
    dark: QColor | None = None,
    invert: bool = False,
) -> QPixmap:
    """Render the checkerboard to a QPixmap at full (1 px == 1 px) resolution.

    The top-left square is the ``dark`` colour, matching the most common
    calibration-target convention.  When ``invert`` is True the colours are
    swapped (top-left becomes ``light``); the inner-corner positions are
    identical, so an inverted board calibrates the frame camera the same way.
    """
    light = light or QColor(255, 255, 255)
    dark = dark or QColor(0, 0, 0)

    background, foreground = (light, dark) if invert else (dark, light)

    width = max(config.width_px, 1)
    height = max(config.height_px, 1)

    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(background)

    painter = QPainter(image)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(foreground)
    s = config.square_px
    for r in range(config.rows):
        for c in range(config.cols):
            # Foreground square when (r + c) is odd.
            if (r + c) % 2 == 1:
                painter.drawRect(c * s, r * s, s, s)
    painter.end()

    return QPixmap.fromImage(image)

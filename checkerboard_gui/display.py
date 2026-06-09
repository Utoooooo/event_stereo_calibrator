"""Fullscreen flashing checkerboard display.

The window cycles through two phases:

  * the checkerboard, shown for ``board_ms`` (the long, stable window the normal
    frame camera should expose within);
  * the "off" phase, shown for ``off_ms`` -- either an all-black frame or, when
    ``inverted`` is enabled, the colour-inverted checkerboard.

Event cameras only report brightness *changes*, so the phase transitions are
what make the target observable.  Flashing against the inverted board produces
an event at every square edge in both directions (stronger, symmetric events);
flashing against black produces events only at the light squares.

Timing notes
------------
Each transition is scheduled against an absolute start time so timer overshoot
does not accumulate.  Phase durations still cannot be resolved finer than one
monitor refresh interval (~16.7 ms at 60 Hz), so a high-refresh display helps
for short ``off_ms`` values.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QElapsedTimer, QTimer
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

from .pattern import CheckerboardConfig, render_pixmap


class FlashingCheckerboard(QWidget):
    """Borderless fullscreen widget that cycles board <-> (black | inverted)."""

    def __init__(self, config: CheckerboardConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._board = render_pixmap(config)
        self._inverted = render_pixmap(config, invert=True) if config.inverted else None

        self._phase = 0  # 0 = board, 1 = off (black or inverted)
        self._paused = False

        # Drift-corrected scheduling state.
        self._clock = QElapsedTimer()
        self._target_ms = 0.0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_tick)

        self.setWindowTitle("Flashing Checkerboard")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setStyleSheet("background-color: black;")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._phase = 0
        self._target_ms = 0.0
        self._clock.start()
        self.update()
        self._schedule_next()

    def _phase_duration(self, phase: int) -> float:
        return self._config.board_ms if phase == 0 else self._config.off_ms

    def _schedule_next(self) -> None:
        if self._paused:
            return
        duration = self._phase_duration(self._phase)
        if duration <= 0:
            return
        self._target_ms += duration
        delay = self._target_ms - self._clock.elapsed()
        self._timer.start(max(0, round(delay)))

    def _on_tick(self) -> None:
        if self._paused:
            return
        self._phase ^= 1
        self.update()
        self._schedule_next()

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        pixmap = self._current_pixmap()
        if pixmap is not None and not pixmap.isNull():
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)
        painter.end()

    def _current_pixmap(self) -> QPixmap | None:
        if self._phase == 0:
            return self._board
        # Off phase: inverted board if enabled, otherwise black (no pixmap).
        return self._inverted

    # -- input -------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif key == Qt.Key.Key_Space:
            self._toggle_pause()
        else:
            super().keyPressEvent(event)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._timer.stop()
            # Freeze on the board so it can be inspected.
            self._phase = 0
            self.update()
        else:
            self._target_ms = 0.0
            self._clock.start()
            self._schedule_next()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().closeEvent(event)

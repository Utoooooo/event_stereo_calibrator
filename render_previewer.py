#!/usr/bin/env python3
"""Simple previewer for event-camera ``.npz`` captures.

Loads an ``.npz`` file containing an ``events`` array (shape ``(N, 6)`` with
columns ``[type, t_s, t_ms, t_us, x, y]``), accumulates the events into a frame,
and displays the rendered grayscale image.

By default the image is **normalized** so the busiest pixel maps to 255.  An
optional fixed saturation cap (>= N events -> 255) is available too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from checkerboard_gui.camera import (
    SENSOR_H,
    SENSOR_W,
    accumulate_events,
    frame_to_uint8,
)


def load_events(path: str) -> np.ndarray:
    """Load the event array from an ``.npz`` file.

    Uses the ``events`` key if present, otherwise the first array in the archive.
    Raises ValueError if the array is not shaped ``(N, 6)``.
    """
    with np.load(path, allow_pickle=False) as data:
        key = "events" if "events" in data.files else (data.files[0] if data.files else None)
        if key is None:
            raise ValueError("No arrays found in the .npz file.")
        events = data[key]
    if events.ndim != 2 or events.shape[1] != 6:
        raise ValueError(
            f"Expected an (N, 6) event array, got shape {events.shape}."
        )
    return events


def render_events(events: np.ndarray, saturation: int | None) -> np.ndarray:
    """Accumulate events into a frame and convert to an 8-bit image."""
    accum = np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint32)
    accumulate_events(accum, events)
    return frame_to_uint8(accum, saturation=saturation)


def _to_qpixmap(gray: np.ndarray) -> QPixmap:
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    image = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image.copy())


class PreviewerWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Event .npz Render Previewer")
        self._events: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.load_btn = QPushButton("Load .npz...")
        self.load_btn.clicked.connect(self._load)
        self.file_label = QLabel("No file loaded.")
        self.file_label.setStyleSheet("color: gray;")
        top.addWidget(self.load_btn)
        top.addWidget(self.file_label, 1)
        root.addLayout(top)

        options = QHBoxLayout()
        self.normalize_check = QCheckBox("Normalize (brightest pixel \u2192 255)")
        self.normalize_check.setChecked(True)
        self.normalize_check.toggled.connect(self._on_normalize_toggled)
        options.addWidget(self.normalize_check)

        options.addWidget(QLabel("Saturation:"))
        self.sat_spin = QSpinBox()
        self.sat_spin.setRange(1, 65535)
        self.sat_spin.setValue(20)
        self.sat_spin.setSuffix(" events")
        self.sat_spin.setEnabled(False)
        self.sat_spin.valueChanged.connect(self._render)
        options.addWidget(self.sat_spin)
        options.addStretch(1)
        root.addLayout(options)

        self.image_label = QLabel("Load an .npz file to preview.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet("background-color: black; color: gray;")
        root.addWidget(self.image_label, 1)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: gray;")
        root.addWidget(self.stats_label)

    # -- actions ----------------------------------------------------------
    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open event capture", "", "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            self._events = load_events(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.file_label.setText(Path(path).name)
        self.file_label.setStyleSheet("")
        self._render()

    def _on_normalize_toggled(self, checked: bool) -> None:
        self.sat_spin.setEnabled(not checked)
        self._render()

    def _render(self) -> None:
        if self._events is None:
            return
        saturation = None if self.normalize_check.isChecked() else self.sat_spin.value()
        frame = render_events(self._events, saturation)
        self._pixmap = _to_qpixmap(frame)
        self._update_pixmap()

        n = self._events.shape[0]
        peak = int(frame.max())
        self.stats_label.setText(
            f"{n:,} events  |  {SENSOR_W}x{SENSOR_H}  |  peak intensity: {peak}/255"
        )

    def _update_pixmap(self) -> None:
        if getattr(self, "_pixmap", None) is None:
            return
        self.image_label.setPixmap(
            self._pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_pixmap()


def main() -> int:
    app = QApplication(sys.argv)
    window = PreviewerWindow()
    window.resize(620, 680)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

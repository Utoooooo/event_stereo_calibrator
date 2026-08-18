#!/usr/bin/env python3
"""Batch previewer/renderer for event-camera ``.npz`` captures.

Same look & feel as ``render_previewer.py``: load one ``.npz`` to preview the
rendered image and confirm the rendering settings (normalize by default, or a
fixed saturation cap).  A *Batch generate* button then converts **all** ``.npz``
files in the loaded file's folder to JPGs using those settings, saving them into
a ``rendered`` subfolder.

Supported ``.npz`` layouts:
  * structured ``events`` array with fields ``x, y, p, t`` (+ ``width``/``height``
    keys), e.g. Prophesee/Metavision exports;
  * plain ``(N, 6)`` arrays ``[type, t_s, t_ms, t_us, x, y]`` (OpenMV GENX320);
  * plain ``(N, 4)`` arrays ``[x, y, p, t]`` or ``(N, 2)`` ``[x, y]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

RENDER_SUBFOLDER = "rendered"


# -- rendering core --------------------------------------------------------
def load_events(path: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load events from an ``.npz`` and return ``(xs, ys, width, height)``.

    ``xs``/``ys`` are int64 pixel coordinates. Width/height come from the file
    when available, otherwise they are inferred from the coordinate ranges.
    """
    with np.load(path, allow_pickle=False) as data:
        files = list(data.files)
        if not files:
            raise ValueError("Empty .npz file.")
        events = data["events"] if "events" in files else data[files[0]]
        width = int(data["width"]) if "width" in files else None
        height = int(data["height"]) if "height" in files else None

        if events.dtype.names:  # structured array (x, y, p, t)
            names = events.dtype.names
            if "x" not in names or "y" not in names:
                raise ValueError(f"Structured events lack x/y fields: {names}")
            xs = events["x"].astype(np.int64)
            ys = events["y"].astype(np.int64)
        else:
            arr = np.asarray(events)
            if arr.ndim != 2:
                raise ValueError(f"Unsupported events array shape {arr.shape}.")
            ncol = arr.shape[1]
            if ncol >= 6:  # OpenMV GENX320: x=col4, y=col5
                xs, ys = arr[:, 4].astype(np.int64), arr[:, 5].astype(np.int64)
            elif ncol >= 4:  # x, y, p, t
                xs, ys = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
            elif ncol == 2:  # x, y
                xs, ys = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
            else:
                raise ValueError(f"Unsupported events array with {ncol} columns.")

    if width is None:
        width = int(xs.max()) + 1 if xs.size else 1
    if height is None:
        height = int(ys.max()) + 1 if ys.size else 1
    return xs, ys, width, height


def accumulate(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    """Count events per pixel into an ``(height, width)`` int64 buffer."""
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = xs[valid], ys[valid]
    idx = ys * width + xs
    counts = np.bincount(idx, minlength=width * height)
    return counts.reshape(height, width)


def counts_to_uint8(accum: np.ndarray, saturation: int | None) -> np.ndarray:
    """Convert a count buffer to 8-bit. None => normalize to max; else cap."""
    if saturation is not None and saturation > 0:
        scale = 255.0 / saturation
        return np.clip(accum * scale, 0, 255).astype(np.uint8)
    peak = int(accum.max())
    if peak <= 0:
        return np.zeros(accum.shape, dtype=np.uint8)
    return np.clip(accum * (255.0 / peak), 0, 255).astype(np.uint8)


def render_file(path: str | Path, saturation: int | None) -> np.ndarray:
    xs, ys, width, height = load_events(path)
    return counts_to_uint8(accumulate(xs, ys, width, height), saturation)


def _to_qpixmap(gray: np.ndarray) -> QPixmap:
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    image = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image.copy())


# -- batch worker ----------------------------------------------------------
class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(int, int, str)  # saved, failed, out_dir

    def __init__(self, files: list[Path], out_dir: Path, saturation: int | None) -> None:
        super().__init__()
        self._files = files
        self._out_dir = out_dir
        self._saturation = saturation
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        saved = failed = 0
        total = len(self._files)
        for i, path in enumerate(self._files):
            if self._cancel:
                break
            self.progress.emit(i + 1, total, path.name)
            try:
                frame = render_file(path, self._saturation)
                out_path = self._out_dir / f"{path.stem}.jpg"
                if cv2.imwrite(str(out_path), frame):
                    saved += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        self.finished_ok.emit(saved, failed, str(self._out_dir))


# -- GUI -------------------------------------------------------------------
class BatchRendererWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Event .npz Batch Renderer")
        self._path: Path | None = None
        self._pixmap: QPixmap | None = None
        self._worker: BatchWorker | None = None
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

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Bottom row: batch generate at the right.
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.batch_btn = QPushButton("Batch generate \u2192 JPG")
        self.batch_btn.setMinimumHeight(36)
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self._batch)
        bottom.addWidget(self.batch_btn)
        root.addLayout(bottom)

    # -- actions ----------------------------------------------------------
    def _saturation(self) -> int | None:
        return None if self.normalize_check.isChecked() else self.sat_spin.value()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open event capture", "", "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            self._xs, self._ys, self._w, self._h = load_events(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._path = Path(path)
        self.file_label.setText(self._path.name)
        self.file_label.setStyleSheet("")
        self.batch_btn.setEnabled(True)
        self._render()

    def _on_normalize_toggled(self, checked: bool) -> None:
        self.sat_spin.setEnabled(not checked)
        self._render()

    def _render(self) -> None:
        if self._path is None:
            return
        accum = accumulate(self._xs, self._ys, self._w, self._h)
        frame = counts_to_uint8(accum, self._saturation())
        self._pixmap = _to_qpixmap(frame)
        self._update_pixmap()
        self.stats_label.setText(
            f"{self._xs.size:,} events  |  {self._w}x{self._h}  |  "
            f"peak intensity: {int(frame.max())}/255"
        )

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
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

    # -- batch ------------------------------------------------------------
    def _batch(self) -> None:
        if self._path is None:
            return
        folder = self._path.parent
        files = sorted(folder.glob("*.npz"))
        if not files:
            QMessageBox.warning(self, "Batch", "No .npz files found in the folder.")
            return
        out_dir = folder / RENDER_SUBFOLDER
        out_dir.mkdir(parents=True, exist_ok=True)

        self.batch_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)

        self._worker = BatchWorker(files, out_dir, self._saturation())
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished_ok.connect(self._on_batch_done)
        self._worker.start()

    def _on_batch_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}  {name}")

    def _on_batch_done(self, saved: int, failed: int, out_dir: str) -> None:
        self.progress.setVisible(False)
        self.batch_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        msg = f"Saved {saved} JPG(s) to:\n{out_dir}"
        if failed:
            msg += f"\n\n{failed} file(s) failed."
        QMessageBox.information(self, "Batch complete", msg)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = BatchRendererWindow()
    window.resize(640, 760)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

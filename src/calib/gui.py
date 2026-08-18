"""GUI for two-camera checkerboard stereo calibration.

Workflow:
  1. Set the checkerboard geometry (inner corners + square size).
  2. Load the two image folders (camera A and camera B).
  3. Detect corners (runs on a worker thread, with progress).
  4. Inspect per-pair detection overlays in the preview.
  5. Calibrate, view the results, and save them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from calib.board import BoardSpec
from calib.calibration import run_full_calibration, to_payload
from calib.detection import draw_corners, find_corners
from calib.io import find_pairs, save_results
from calib.viz3d import plot_scene


@dataclass
class PairDetection:
    name: str
    path_a: Path
    path_b: Path
    found_a: bool = False
    found_b: bool = False
    corners_a: np.ndarray | None = None
    corners_b: np.ndarray | None = None
    size_a: tuple[int, int] | None = None  # (w, h)
    size_b: tuple[int, int] | None = None

    @property
    def usable(self) -> bool:
        return self.found_a and self.found_b


def cv_to_qpixmap(image: np.ndarray, max_width: int = 520) -> QPixmap:
    """Convert a BGR or grayscale OpenCV image to a (downscaled) QPixmap."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class DetectionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal()

    def __init__(self, pairs, board: BoardSpec) -> None:
        super().__init__()
        self._pairs = pairs
        self._board = board
        self.results: list[PairDetection] = []
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        ps = self._board.pattern_size
        total = len(self._pairs)
        for i, pair in enumerate(self._pairs):
            if self._cancel:
                break
            self.progress.emit(i + 1, total, pair.name)
            det = PairDetection(pair.name, pair.path_a, pair.path_b)

            gray_a = cv2.imread(str(pair.path_a), cv2.IMREAD_GRAYSCALE)
            if gray_a is not None:
                det.size_a = (gray_a.shape[1], gray_a.shape[0])
                det.found_a, det.corners_a = find_corners(gray_a, ps, "auto")

            gray_b = cv2.imread(str(pair.path_b), cv2.IMREAD_GRAYSCALE)
            if gray_b is not None:
                det.size_b = (gray_b.shape[1], gray_b.shape[0])
                det.found_b, det.corners_b = find_corners(gray_b, ps, "auto")

            self.results.append(det)
        self.finished_ok.emit()


class ResultsDialog(QDialog):
    """Popup showing the numeric calibration results."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Results")
        self.resize(620, 540)
        layout = QVBoxLayout(self)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        view.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class CalibrationWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Camera Calibration")
        self._detections: list[PairDetection] = []
        self._worker: DetectionWorker | None = None
        self._payload: dict | None = None
        self._stereo = None
        self._results_text = ""
        self._build_ui()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        board_box = QGroupBox("Checkerboard")
        board_form = QFormLayout(board_box)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 100)
        self.cols_spin.setValue(9)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 100)
        self.rows_spin.setValue(6)
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(0.1, 1000.0)
        self.square_spin.setDecimals(3)
        self.square_spin.setSuffix(" mm")
        self.square_spin.setValue(25.0)
        board_form.addRow("Inner corners X:", self.cols_spin)
        board_form.addRow("Inner corners Y:", self.rows_spin)
        board_form.addRow("Square size:", self.square_spin)
        layout.addWidget(board_box)

        folders_box = QGroupBox("Image folders")
        folders_form = QFormLayout(folders_box)
        self.rgb_edit = QLineEdit()
        rgb_btn = QPushButton("Browse...")
        rgb_btn.clicked.connect(lambda: self._browse(self.rgb_edit))
        rgb_row = QHBoxLayout()
        rgb_row.addWidget(self.rgb_edit)
        rgb_row.addWidget(rgb_btn)
        rgb_w = QWidget()
        rgb_w.setLayout(rgb_row)

        self.evt_edit = QLineEdit()
        evt_btn = QPushButton("Browse...")
        evt_btn.clicked.connect(lambda: self._browse(self.evt_edit))
        evt_row = QHBoxLayout()
        evt_row.addWidget(self.evt_edit)
        evt_row.addWidget(evt_btn)
        evt_w = QWidget()
        evt_w.setLayout(evt_row)

        folders_form.addRow("Camera A:", rgb_w)
        folders_form.addRow("Camera B:", evt_w)
        layout.addWidget(folders_box)

        self.detect_btn = QPushButton("Detect corners")
        self.detect_btn.clicked.connect(self._detect)
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self._calibrate)
        self.calibrate_btn.setEnabled(False)
        self.save_btn = QPushButton("Save results...")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.save_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #444;")
        layout.addWidget(self.status_label)

        self.pair_list = QListWidget()
        self.pair_list.currentRowChanged.connect(self._show_pair)
        layout.addWidget(QLabel("Pairs (\u2713 = corners found):"))
        layout.addWidget(self.pair_list, 1)

        return panel

    def _build_preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        images_row = QHBoxLayout()
        self.preview_a = self._make_preview_label("Camera A")
        self.preview_b = self._make_preview_label("Camera B")
        images_row.addWidget(self._wrap_preview(self.preview_a, "Camera A"))
        images_row.addWidget(self._wrap_preview(self.preview_b, "Camera B"))
        layout.addLayout(images_row, 1)

        header = QHBoxLayout()
        header.addWidget(QLabel("3D calibration scene:"))
        header.addStretch(1)
        self.show_numbers_btn = QPushButton("Show numeric results")
        self.show_numbers_btn.setEnabled(False)
        self.show_numbers_btn.clicked.connect(self._open_results_dialog)
        header.addWidget(self.show_numbers_btn)
        layout.addLayout(header)

        self._fig = Figure(figsize=(5, 4))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_title("Calibrate to view the camera/target 3D scene")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(300)
        toolbar = NavigationToolbar(self._canvas, panel)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, 2)

        return panel

    @staticmethod
    def _make_preview_label(_title: str) -> QLabel:
        label = QLabel("No image")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFrameShape(QFrame.Shape.Box)
        label.setMinimumSize(360, 360)
        label.setStyleSheet("background-color: #202020; color: gray;")
        return label

    @staticmethod
    def _wrap_preview(label: QLabel, title: str) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.addWidget(QLabel(title))
        v.addWidget(label, 1)
        return wrap

    # -- helpers ----------------------------------------------------------
    def _board(self) -> BoardSpec:
        return BoardSpec(
            cols=self.cols_spin.value(),
            rows=self.rows_spin.value(),
            square_mm=self.square_spin.value(),
        )

    def _browse(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", edit.text())
        if folder:
            edit.setText(folder)

    def _set_busy(self, busy: bool) -> None:
        for w in (self.detect_btn, self.calibrate_btn, self.save_btn):
            w.setEnabled(not busy)
        if busy:
            self.calibrate_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

    # -- detection --------------------------------------------------------
    def _detect(self) -> None:
        rgb_dir = self.rgb_edit.text().strip()
        evt_dir = self.evt_edit.text().strip()
        if not rgb_dir or not evt_dir:
            QMessageBox.warning(self, "Folders", "Select both image folders first.")
            return
        pairs, method = find_pairs(rgb_dir, evt_dir)
        if not pairs:
            QMessageBox.warning(
                self, "No pairs", "No matching image pairs found in those folders."
            )
            return

        self._payload = None
        self.status_label.setText(
            f"Found {len(pairs)} image pairs (matched by {method}). Detecting corners..."
        )
        self.pair_list.clear()
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(pairs))
        self.progress.setValue(0)

        self._worker = DetectionWorker(pairs, self._board())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_detection_done)
        self._worker.start()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}  {name}")

    def _on_detection_done(self) -> None:
        assert self._worker is not None
        self._detections = self._worker.results
        self.progress.setVisible(False)
        self._set_busy(False)

        usable = 0
        self.pair_list.clear()
        for det in self._detections:
            mark_a = "\u2713" if det.found_a else "\u2717"
            mark_b = "\u2713" if det.found_b else "\u2717"
            item = QListWidgetItem(f"{det.name}   A:{mark_a}  B:{mark_b}")
            if not det.usable:
                item.setForeground(Qt.GlobalColor.gray)
            self.pair_list.addItem(item)
            usable += int(det.usable)

        self.calibrate_btn.setEnabled(usable >= 3)
        self.status_label.setText(
            f"Detected corners in {usable} / {len(self._detections)} pairs "
            f"(both cameras). "
            + ("Ready to calibrate." if usable >= 3 else "Need at least 3 usable pairs.")
        )
        if self._detections:
            self.pair_list.setCurrentRow(0)

    # -- preview ----------------------------------------------------------
    def _show_pair(self, row: int) -> None:
        if not (0 <= row < len(self._detections)):
            return
        det = self._detections[row]
        ps = self._board().pattern_size
        self._render_preview(self.preview_a, det.path_a, ps, det.corners_a, det.found_a)
        self._render_preview(self.preview_b, det.path_b, ps, det.corners_b, det.found_b)

    def _render_preview(self, label, path, pattern_size, corners, found) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            label.setText("Could not load image")
            return
        if found and corners is not None:
            image = draw_corners(image, pattern_size, corners)
        label.setPixmap(cv_to_qpixmap(image, max_width=label.width() or 520))

    # -- calibration ------------------------------------------------------
    def _calibrate(self) -> None:
        usable = [d for d in self._detections if d.usable]
        if len(usable) < 3:
            QMessageBox.warning(self, "Calibrate", "Need at least 3 usable pairs.")
            return
        size_a = usable[0].size_a
        size_b = usable[0].size_b
        if size_a is None or size_b is None:
            QMessageBox.warning(self, "Calibrate", "Missing image sizes.")
            return

        img_a = [d.corners_a for d in usable]
        img_b = [d.corners_b for d in usable]
        board = self._board()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stereo = run_full_calibration(board, img_a, img_b, size_a, size_b)
            self._payload = to_payload(stereo)
            self._stereo = stereo
        except cv2.error as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Calibration failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        self._results_text = self._format_results(self._payload)
        self._draw_scene()
        self.save_btn.setEnabled(True)
        self.show_numbers_btn.setEnabled(True)
        self.status_label.setText(
            f"Calibration done. Stereo RMS {self._payload['stereo']['rms_px']:.3f} px, "
            f"baseline {self._payload['stereo']['baseline_mm']:.1f} mm."
        )
        self._open_results_dialog()

    def _draw_scene(self) -> None:
        if self._stereo is None:
            return
        plot_scene(self._ax, self._stereo)
        self._canvas.draw_idle()

    def _open_results_dialog(self) -> None:
        if not self._results_text:
            return
        dialog = ResultsDialog(self._results_text, self)
        dialog.exec()

    @staticmethod
    def _format_results(payload: dict) -> str:
        np.set_printoptions(precision=4, suppress=True)

        def cam_block(cam: dict) -> str:
            K = np.asarray(cam["K"])
            dist = np.asarray(cam["dist"]).ravel()
            return (
                f"{cam['name']}  ({cam['image_size'][0]}x{cam['image_size'][1]})\n"
                f"  reprojection RMS: {cam['reprojection_rms_px']:.4f} px\n"
                f"  K =\n{K}\n"
                f"  dist = {dist}\n"
            )

        s = payload["stereo"]
        R = np.asarray(s["R"])
        T = np.asarray(s["T"]).ravel()
        lines = [
            f"Pairs used: {payload['num_pairs']}",
            f"Board: {payload['board']['cols']}x{payload['board']['rows']} inner "
            f"corners, {payload['board']['square_mm']} mm squares",
            "",
            cam_block(payload["camera_a"]),
            cam_block(payload["camera_b"]),
            f"Stereo RMS: {s['rms_px']:.4f} px",
            f"Baseline |T|: {s['baseline_mm']:.2f} mm",
            f"R (A->B) =\n{R}",
            f"T (A->B, mm) = {T}",
            "",
            s["note"],
        ]
        return "\n".join(lines)

    def _save(self) -> None:
        if self._payload is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        json_path, npz_path = save_results(folder, self._payload)
        QMessageBox.information(
            self, "Saved", f"Saved:\n{json_path}\n{npz_path}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.resize(1180, 760)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

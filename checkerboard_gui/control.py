"""Control panel for live event monitoring and on-demand stereo capture.

Shows a real-time stitched view of the left/right event cameras, a *Capture*
button that accumulates events over the configured window into a PNG per camera
(saved under ``captured/``), and an *Exit* button that disconnects both ports.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .camera import (
    CAPTURE_SATURATION,
    SENSOR_H,
    SENSOR_W,
    EventCameraWorker,
    frame_to_uint8,
)

LIVE_FPS = 30
DIVIDER_W = 4


def _stitch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((SENSOR_H, DIVIDER_W), 128, dtype=np.uint8)
    return np.hstack((left, divider, right))


def _to_qpixmap(gray: np.ndarray) -> QPixmap:
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    image = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image.copy())


class ControlWindow(QWidget):
    def __init__(
        self,
        left: EventCameraWorker,
        right: EventCameraWorker,
        accumulation_ms: float,
        captured_dir: Path,
        toggle_flash_pause: Callable[[], bool | None] | None = None,
        flash_pause_state: Callable[[], bool | None] | None = None,
        save_callback: Callable[[], None] | None = None,
        normalize: bool = True,
        on_normalize_changed: Callable[[bool], None] | None = None,
        target_pairs: int = 0,
        on_session_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._left = left
        self._right = right
        self._accumulation_ms = accumulation_ms
        self._captured_dir = captured_dir
        self._toggle_flash_pause = toggle_flash_pause
        self._flash_pause_state = flash_pause_state
        self._save_callback = save_callback
        self._normalize = normalize
        self._on_normalize_changed = on_normalize_changed
        self._target_pairs = max(0, int(target_pairs))
        self._on_session_complete = on_session_complete
        self._capture_count = 0
        self._capture_in_progress = False
        self._completed = False
        self._pending: dict[str, np.ndarray | None] = {}

        # One session folder per control-panel session: calib_<timestamp>/CAM_L|CAM_R
        self._session_stamp = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = captured_dir / f"calib_{self._session_stamp}"
        self._cam_dirs = {
            "L": self._session_dir / "CAM_L",
            "R": self._session_dir / "CAM_R",
        }

        self.setWindowTitle("Event Camera Control Panel")
        self._build_ui()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(int(1000 / LIVE_FPS))
        self._live_timer.timeout.connect(self._update_live)
        self._live_timer.start()

        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(15)
        self._capture_timer.timeout.connect(self._poll_capture)

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        view_box = QGroupBox("Live event monitor  (Left | Right)")
        view_layout = QVBoxLayout(view_box)
        self.view_label = QLabel()
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setFrameShape(QFrame.Shape.Box)
        self.view_label.setMinimumSize(660, 340)
        self.view_label.setStyleSheet("background-color: black;")
        view_layout.addWidget(self.view_label)
        root.addWidget(view_box, 1)

        status_row = QHBoxLayout()
        self.left_status = QLabel()
        self.right_status = QLabel()
        for lbl in (self.left_status, self.right_status):
            lbl.setTextFormat(Qt.TextFormat.RichText)
        status_row.addWidget(self.left_status)
        status_row.addStretch(1)
        status_row.addWidget(self.right_status)
        root.addLayout(status_row)

        self.normalize_check = QCheckBox(
            "Normalize rendering (brightest pixel \u2192 255)"
        )
        self.normalize_check.setChecked(self._normalize)
        self.normalize_check.setToolTip(
            f"On: each capture is scaled so its brightest pixel becomes 255.\n"
            f"Off: a fixed saturation cap is used (\u2265 {CAPTURE_SATURATION} "
            f"events \u2192 255)."
        )
        self.normalize_check.toggled.connect(self._on_normalize_toggled)
        root.addWidget(self.normalize_check)

        target_txt = (
            f"{self._target_pairs} pairs" if self._target_pairs else "unlimited"
        )
        self.capture_info = QLabel(
            f"Accumulation window: <b>{self._accumulation_ms:.0f} ms</b> &nbsp;|&nbsp; "
            f"Target: <b>{target_txt}</b><br>"
            f"Saving to <b>{self._session_dir}</b>"
        )
        self.capture_info.setTextFormat(Qt.TextFormat.RichText)
        self.capture_info.setWordWrap(True)
        root.addWidget(self.capture_info)

        button_row = QHBoxLayout()
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.clicked.connect(self._start_capture)
        self.pause_btn = QPushButton("Pause Flashing")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.clicked.connect(self._toggle_flash)
        self.exit_btn = QPushButton("Exit (abort session)")
        self.exit_btn.setMinimumHeight(40)
        self.exit_btn.clicked.connect(self.close)
        button_row.addWidget(self.capture_btn, 2)
        button_row.addWidget(self.pause_btn, 1)
        button_row.addWidget(self.exit_btn, 1)
        root.addLayout(button_row)
        self._refresh_pause_button()

    # -- live view --------------------------------------------------------
    def _camera_status(self, worker: EventCameraWorker) -> str:
        if worker.error:
            return (
                f"<span style='color:#c0392b;'><b>{worker.name}</b> "
                f"({worker.port}): ERROR - {worker.error}</span>"
            )
        if not worker.connected:
            return (
                f"<span style='color:#d68910;'><b>{worker.name}</b> "
                f"({worker.port}): connecting...</span>"
            )
        return (
            f"<span style='color:#1e8449;'><b>{worker.name}</b> "
            f"({worker.port}): {worker.event_rate:,.0f} ev/s, "
            f"{worker.total_events:,} total</span>"
        )

    def _update_live(self) -> None:
        left = self._left.get_live_frame()
        right = self._right.get_live_frame()
        pixmap = _to_qpixmap(_stitch(left, right))
        self.view_label.setPixmap(
            pixmap.scaled(
                self.view_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.left_status.setText(self._camera_status(self._left))
        self.right_status.setText(self._camera_status(self._right))
        self._refresh_pause_button()

    # -- rendering option -------------------------------------------------
    def _on_normalize_toggled(self, checked: bool) -> None:
        self._normalize = checked
        if self._on_normalize_changed is not None:
            self._on_normalize_changed(checked)

    # -- flashing display pause -------------------------------------------
    def _toggle_flash(self) -> None:
        if self._toggle_flash_pause is None:
            return
        self._toggle_flash_pause()
        self._refresh_pause_button()

    def _refresh_pause_button(self) -> None:
        state = self._flash_pause_state() if self._flash_pause_state else None
        if state is None:
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("Pause Flashing (no display)")
            return
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Resume Flashing" if state else "Pause Flashing")

    # -- capture ----------------------------------------------------------
    def _start_capture(self) -> None:
        if self._capture_in_progress:
            return
        if not (self._left.connected and self._right.connected):
            self.capture_info.setText(
                "<span style='color:#c0392b;'>Cannot capture: both cameras must "
                "be connected.</span>"
            )
            return
        self._capture_in_progress = True
        self._pending = {"L": None, "R": None}
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Capturing...")
        duration_s = self._accumulation_ms / 1000.0
        self._left.request_capture(duration_s)
        self._right.request_capture(duration_s)
        self._capture_timer.start()

    def _poll_capture(self) -> None:
        if self._pending.get("L") is None:
            result = self._left.poll_capture_result()
            if result is not None:
                self._pending["L"] = result
        if self._pending.get("R") is None:
            result = self._right.poll_capture_result()
            if result is not None:
                self._pending["R"] = result

        if self._pending.get("L") is not None and self._pending.get("R") is not None:
            self._capture_timer.stop()
            self._finish_capture(self._pending["L"], self._pending["R"])

    def _finish_capture(self, left_accum: np.ndarray, right_accum: np.ndarray) -> None:
        for cam_dir in self._cam_dirs.values():
            cam_dir.mkdir(parents=True, exist_ok=True)
        self._capture_count += 1
        idx = self._capture_count
        saved = []
        saturation = None if self._normalize else CAPTURE_SATURATION
        for key, accum in (("L", left_accum), ("R", right_accum)):
            frame = frame_to_uint8(accum, saturation=saturation)
            path = self._cam_dirs[key] / f"{idx:03d}.png"
            cv2.imwrite(str(path), frame)
            saved.append((key, accum, path))

        details = "  |  ".join(
            f"{key}: {int(accum.sum()):,} events -> {path.parent.name}/{path.name}"
            for key, accum, path in saved
        )
        progress = f"#{idx}" + (f" / {self._target_pairs}" if self._target_pairs else "")
        self.capture_info.setText(
            f"<span style='color:#1e8449;'>Captured {progress}</span> &nbsp; {details}"
        )
        self._capture_in_progress = False

        if self._target_pairs and self._capture_count >= self._target_pairs:
            self._complete_session()
            return
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("Capture")

    def _complete_session(self) -> None:
        self._completed = True
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Done")
        self.capture_info.setText(
            f"<span style='color:#1e8449;'><b>Completed {self._capture_count} "
            f"pairs.</b></span> Saved to {self._session_dir}. Closing..."
        )
        QTimer.singleShot(1200, self.close)

    # -- shutdown ---------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._save_callback is not None:
            self._save_callback()
        self._live_timer.stop()
        self._capture_timer.stop()
        self._left.stop()
        self._right.stop()
        super().closeEvent(event)
        # On a completed session, hand control back so the app can close.
        if self._completed and self._on_session_complete is not None:
            QTimer.singleShot(0, self._on_session_complete)

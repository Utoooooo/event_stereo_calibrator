"""Configuration GUI for the event-camera calibration tool.

Provides the flashing-checkerboard generator (geometry, phase timing, fullscreen
display on a chosen monitor) plus the event-camera capture setup: serial-port
selection for the left/right cameras, an accumulation window, a monitor selector
for the control panel, and a launcher for that panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import EventCameraWorker, list_serial_ports
from .control import ControlWindow
from .display import FlashingCheckerboard
from .pattern import CheckerboardConfig, render_pixmap
from .settings import load_settings, save_settings

CAM_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "event_streaming_cam.py"
CAPTURED_DIR = Path(__file__).resolve().parent.parent / "captured"


class ConfigWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Event Camera Calibration - Control")
        self._display: FlashingCheckerboard | None = None
        self._control: ControlWindow | None = None
        self._workers: list[EventCameraWorker] = []
        self._normalize = True

        self._build_ui()
        self._apply_settings(load_settings())
        self._refresh_preview()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # Left: form controls.
        form_box = QGroupBox("Pattern settings")
        form = QFormLayout(form_box)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 100)
        self.rows_spin.setValue(7)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 100)
        self.cols_spin.setValue(10)

        self.square_spin = QSpinBox()
        self.square_spin.setRange(2, 2000)
        self.square_spin.setSuffix(" px")
        self.square_spin.setValue(100)

        self.board_spin = QDoubleSpinBox()
        self.board_spin.setRange(1.0, 100000.0)
        self.board_spin.setDecimals(1)
        self.board_spin.setSuffix(" ms")
        self.board_spin.setValue(100.0)

        self.off_spin = QDoubleSpinBox()
        self.off_spin.setRange(1.0, 100000.0)
        self.off_spin.setDecimals(1)
        self.off_spin.setSuffix(" ms")
        self.off_spin.setValue(20.0)

        self.invert_check = QCheckBox("Use inverted checkerboard for off phase")
        self.invert_check.setToolTip(
            "Off phase shows the colour-inverted checkerboard instead of black.\n"
            "Corner positions are unchanged, so frame-camera calibration is\n"
            "unaffected; event cameras get stronger, symmetric edges."
        )

        self.screen_combo = QComboBox()
        self._populate_screens()

        form.addRow("Rows (squares):", self.rows_spin)
        form.addRow("Columns (squares):", self.cols_spin)
        form.addRow("Square size:", self.square_spin)
        form.addRow("Checkerboard time:", self.board_spin)
        form.addRow("Off-phase time:", self.off_spin)
        form.addRow("", self.invert_check)
        form.addRow("Display on:", self.screen_combo)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(self.info_label)

        self.start_btn = QPushButton("Start Flashing Display")
        self.start_btn.clicked.connect(self._start_display)
        self.save_btn = QPushButton("Save Pattern as PNG...")
        self.save_btn.clicked.connect(self._save_png)

        form.addRow(self.start_btn)
        form.addRow(self.save_btn)

        hint = QLabel(
            "In the display window: <b>Esc</b>/<b>Q</b> to exit, "
            "<b>Space</b> to pause."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        # Right column: preview on top, event-camera setup below.
        right = QVBoxLayout()

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.Box)
        self.preview_label.setMinimumSize(360, 260)
        self.preview_label.setStyleSheet("background-color: #303030;")
        preview_layout.addWidget(self.preview_label)
        right.addWidget(preview_box, 1)

        right.addWidget(self._build_camera_box())

        root.addWidget(form_box)
        root.addLayout(right, 1)

        for w in (self.rows_spin, self.cols_spin, self.square_spin):
            w.valueChanged.connect(self._refresh_preview)
        self.board_spin.valueChanged.connect(self._refresh_preview)
        self.off_spin.valueChanged.connect(self._refresh_preview)
        self.invert_check.toggled.connect(self._refresh_preview)
        self.screen_combo.currentIndexChanged.connect(self._refresh_preview)

    def _populate_screens(self) -> None:
        self.screen_combo.clear()
        for i, screen in enumerate(QApplication.screens()):
            geo = screen.geometry()
            rate = screen.refreshRate()
            self.screen_combo.addItem(
                f"{i}: {screen.name()} ({geo.width()}x{geo.height()} @ {rate:.0f}Hz)"
            )

    # -- event-camera setup ------------------------------------------------
    def _build_camera_box(self) -> QGroupBox:
        box = QGroupBox("Event cameras (capture)")
        form = QFormLayout(box)

        self.left_port_combo = QComboBox()
        self.right_port_combo = QComboBox()
        self._populate_ports()

        refresh_btn = QPushButton("Refresh ports")
        refresh_btn.clicked.connect(self._populate_ports)

        self.accum_spin = QDoubleSpinBox()
        self.accum_spin.setRange(1.0, 10000.0)
        self.accum_spin.setDecimals(0)
        self.accum_spin.setSuffix(" ms")
        self.accum_spin.setValue(200.0)

        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 100000)
        self.quantity_spin.setValue(20)
        self.quantity_spin.setSuffix(" pairs")
        self.quantity_spin.setToolTip(
            "The control panel closes automatically once this many stereo pairs "
            "have been captured. Exit aborts the session early."
        )

        self.control_screen_combo = QComboBox()
        for i, screen in enumerate(QApplication.screens()):
            geo = screen.geometry()
            self.control_screen_combo.addItem(
                f"{i}: {screen.name()} ({geo.width()}x{geo.height()})"
            )

        self.open_control_btn = QPushButton("Connect && Open Control Panel")
        self.open_control_btn.clicked.connect(self._open_control_panel)

        form.addRow("Left camera port:", self.left_port_combo)
        form.addRow("Right camera port:", self.right_port_combo)
        form.addRow("", refresh_btn)
        form.addRow("Accumulation time:", self.accum_spin)
        form.addRow("Pairs to capture:", self.quantity_spin)
        form.addRow("Control GUI display:", self.control_screen_combo)
        form.addRow(self.open_control_btn)
        return box

    def _populate_ports(self) -> None:
        ports = list_serial_ports()
        for combo in (self.left_port_combo, self.right_port_combo):
            current = combo.currentData()
            combo.clear()
            for device, desc in ports:
                combo.addItem(f"{device} - {desc}", device)
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        # Default the right camera to a different port than the left, if possible.
        if len(ports) >= 2 and self.right_port_combo.currentIndex() == 0:
            self.right_port_combo.setCurrentIndex(1)

    def _open_control_panel(self) -> None:
        left_port = self.left_port_combo.currentData()
        right_port = self.right_port_combo.currentData()
        if not left_port or not right_port:
            QMessageBox.warning(self, "Ports", "Select a serial port for both cameras.")
            return
        if left_port == right_port:
            QMessageBox.warning(
                self, "Ports", "Left and right cameras must use different ports."
            )
            return
        if not CAM_SCRIPT_PATH.exists():
            QMessageBox.critical(
                self, "Missing script", f"Camera script not found:\n{CAM_SCRIPT_PATH}"
            )
            return

        if self._control is not None:
            self._control.close()

        script = CAM_SCRIPT_PATH.read_text(encoding="utf-8")
        left = EventCameraWorker(left_port, "LEFT", script)
        right = EventCameraWorker(right_port, "RIGHT", script)
        self._workers = [left, right]
        left.start()
        right.start()

        self._control = ControlWindow(
            left,
            right,
            self.accum_spin.value(),
            CAPTURED_DIR,
            toggle_flash_pause=self._toggle_flash_pause,
            flash_pause_state=self._flash_pause_state,
            save_callback=self._save_settings_now,
            normalize=self._normalize,
            on_normalize_changed=self._set_normalize,
            target_pairs=self.quantity_spin.value(),
            on_session_complete=self.close,
        )
        self._place_on_control_screen(self._control)
        self._control.show()
        self._control.raise_()
        self._control.activateWindow()

    def _place_on_control_screen(self, window: QWidget) -> None:
        screens = QApplication.screens()
        idx = self.control_screen_combo.currentIndex()
        screen = screens[idx] if 0 <= idx < len(screens) else QApplication.primaryScreen()
        geo = screen.availableGeometry()
        w, h = 1040, 720
        x = geo.x() + max(0, (geo.width() - w) // 2)
        y = geo.y() + max(0, (geo.height() - h) // 2)
        window.setScreen(screen)
        window.setGeometry(x, y, min(w, geo.width()), min(h, geo.height()))

    # -- state -------------------------------------------------------------
    def _config(self) -> CheckerboardConfig:
        return CheckerboardConfig(
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            square_px=self.square_spin.value(),
            board_ms=self.board_spin.value(),
            off_ms=self.off_spin.value(),
            inverted=self.invert_check.isChecked(),
        )

    def _refresh_preview(self) -> None:
        config = self._config()
        pixmap = render_pixmap(config)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.preview_label.setPixmap(scaled)

        cx, cy = config.inner_corners
        off_label = "inverted board" if config.inverted else "black"
        warn = ""
        screen_idx = self.screen_combo.currentIndex()
        screens = QApplication.screens()
        if 0 <= screen_idx < len(screens):
            refresh = screens[screen_idx].refreshRate()
            if refresh:
                frame_ms = 1000.0 / refresh
                shortest = min(config.board_ms, config.off_ms)
                if shortest < frame_ms:
                    warn = (
                        f"<br><span style='color:#c0392b;'>&#9888; Shortest phase "
                        f"({shortest:.1f} ms) is below one {refresh:.0f} Hz frame "
                        f"({frame_ms:.1f} ms); it cannot be shown cleanly.</span>"
                    )
        self.info_label.setText(
            f"Board: <b>{config.width_px} x {config.height_px} px</b><br>"
            f"OpenCV inner corners: <b>{cx} x {cy}</b><br>"
            f"Cycle: <b>{config.board_ms:.0f} ms</b> board + "
            f"<b>{config.off_ms:.0f} ms</b> {off_label} "
            f"= {config.cycle_ms:.0f} ms ({1000.0 / config.cycle_ms:.1f} Hz)"
            f"{warn}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview()

    # -- actions -----------------------------------------------------------
    def _start_display(self) -> None:
        config = self._config()
        screens = QApplication.screens()
        idx = self.screen_combo.currentIndex()
        screen = screens[idx] if 0 <= idx < len(screens) else QApplication.primaryScreen()

        if self._display is not None:
            self._display.close()

        self._display = FlashingCheckerboard(config)
        self._display.setScreen(screen)
        self._display.setGeometry(screen.geometry())
        self._display.showFullScreen()
        self._display.start()
        self._display.activateWindow()
        self._display.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings_now()
        if self._control is not None:
            self._control.close()
        if self._display is not None:
            self._display.close()
        for worker in self._workers:
            worker.stop()
        super().closeEvent(event)

    # -- settings persistence ---------------------------------------------
    def _apply_settings(self, data: dict) -> None:
        spin_map = {
            "rows": self.rows_spin,
            "cols": self.cols_spin,
            "square_px": self.square_spin,
            "board_ms": self.board_spin,
            "off_ms": self.off_spin,
            "accumulation_ms": self.accum_spin,
            "pairs": self.quantity_spin,
        }
        for key, spin in spin_map.items():
            value = data.get(key)
            if isinstance(value, (int, float)):
                spin.setValue(value)

        if isinstance(data.get("inverted"), bool):
            self.invert_check.setChecked(data["inverted"])

        if isinstance(data.get("normalize"), bool):
            self._normalize = data["normalize"]

        # Monitor indices: fall back to default if out of range.
        for key, combo in (
            ("checkerboard_screen", self.screen_combo),
            ("control_screen", self.control_screen_combo),
        ):
            idx = data.get(key)
            if isinstance(idx, int) and 0 <= idx < combo.count():
                combo.setCurrentIndex(idx)

        # Serial ports: select by saved device, fall back to default if absent.
        for key, combo in (
            ("left_port", self.left_port_combo),
            ("right_port", self.right_port_combo),
        ):
            device = data.get(key)
            if device:
                found = combo.findData(device)
                if found >= 0:
                    combo.setCurrentIndex(found)

    def _set_normalize(self, value: bool) -> None:
        self._normalize = bool(value)

    def _save_settings_now(self) -> None:
        save_settings(self._collect_settings())

    def _collect_settings(self) -> dict:
        return {
            "rows": self.rows_spin.value(),
            "cols": self.cols_spin.value(),
            "square_px": self.square_spin.value(),
            "board_ms": self.board_spin.value(),
            "off_ms": self.off_spin.value(),
            "inverted": self.invert_check.isChecked(),
            "checkerboard_screen": self.screen_combo.currentIndex(),
            "control_screen": self.control_screen_combo.currentIndex(),
            "left_port": self.left_port_combo.currentData(),
            "right_port": self.right_port_combo.currentData(),
            "accumulation_ms": self.accum_spin.value(),
            "pairs": self.quantity_spin.value(),
            "normalize": self._normalize,
        }

    def _active_display(self) -> FlashingCheckerboard | None:
        if self._display is not None and self._display.isVisible():
            return self._display
        return None

    def _toggle_flash_pause(self) -> bool | None:
        display = self._active_display()
        if display is None:
            return None
        return display.toggle_pause()

    def _flash_pause_state(self) -> bool | None:
        display = self._active_display()
        if display is None:
            return None
        return display.is_paused

    def _save_png(self) -> None:
        config = self._config()
        default = f"checkerboard_{config.rows}x{config.cols}_{config.square_px}px.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save checkerboard", default, "PNG image (*.png)"
        )
        if not path:
            return
        pixmap: QPixmap = render_pixmap(config)
        if pixmap.save(path, "PNG"):
            QMessageBox.information(self, "Saved", f"Saved pattern to:\n{path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save the image.")


def main() -> int:
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.resize(920, 620)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration GUI for the flashing checkerboard generator.

Lets the user set the board geometry (rows, columns, square size in pixels) and
the flash rate, preview the pattern, export it as PNG, and launch a fullscreen
flashing display on a chosen monitor.
"""

from __future__ import annotations

import sys

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

from .display import FlashingCheckerboard
from .pattern import CheckerboardConfig, render_pixmap


class ConfigWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Event Camera Calibration - Checkerboard Generator")
        self._display: FlashingCheckerboard | None = None

        self._build_ui()
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

        # Right: live preview.
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.Box)
        self.preview_label.setMinimumSize(360, 300)
        self.preview_label.setStyleSheet("background-color: #303030;")
        preview_layout.addWidget(self.preview_label)

        root.addWidget(form_box)
        root.addWidget(preview_box, 1)

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
    window.resize(820, 460)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""
CameraPanel — live feed (Hamamatsu Orca Flash 4V3) + exposure control.

Uses camera.start_live() / camera.get_live_frame() for preview,
and camera.snap() for single captures — exactly as in hyperspectral_measure.py.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QSizePolicy, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap


def frame_to_qpixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    """Convert uint16 Hamamatsu frame to a scaled 8-bit QPixmap for display."""
    if frame.dtype == np.uint16:
        f_min, f_max = float(frame.min()), float(frame.max())
        if f_max > f_min:
            frame8 = ((frame.astype(np.float32) - f_min) / (f_max - f_min) * 255).astype(np.uint8)
        else:
            frame8 = np.zeros(frame.shape, dtype=np.uint8)
    else:
        frame8 = frame.astype(np.uint8)

    rows, cols = frame8.shape[:2]
    if frame8.ndim == 2:
        qimg = QImage(frame8.data, cols, rows, cols, QImage.Format_Grayscale8)
    else:
        qimg = QImage(frame8.data, cols, rows, 3 * cols, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CameraPanel(QWidget):
    """Camera live view panel for Hamamatsu Orca Flash 4V3."""

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._live_worker = None
        self._last_frame: np.ndarray | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Live view display ──────────────────────────────────────
        self.view_label = QLabel("No signal")
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setMinimumSize(512, 384)
        self.view_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_label.setStyleSheet(
            "background:#0d0d0d; color:#555; font-size:13px; border-radius:4px;"
        )
        layout.addWidget(self.view_label, stretch=1)

        # ── Controls ───────────────────────────────────────────────
        grp = QGroupBox("Hamamatsu Orca Flash 4V3")
        grp.setObjectName("ctrlGroup")
        inner = QVBoxLayout(grp)
        inner.setSpacing(6)

        # Status
        self.cam_status = QLabel("Camera: disconnected")
        self.cam_status.setObjectName("statusLabel")
        inner.addWidget(self.cam_status)

        # Exposure — in seconds, matching HamamatsuDevice.setExposure()
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure"))
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.0001, 60.0)
        self.exposure_spin.setValue(self.camera.get_exposure())
        self.exposure_spin.setSingleStep(0.005)
        self.exposure_spin.setDecimals(4)
        self.exposure_spin.setSuffix(" s")
        self.exposure_spin.setFixedWidth(100)
        self.exposure_spin.setToolTip("Exposure time in seconds (passed to HamamatsuDevice.setExposure)")
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)
        exp_row.addWidget(self.exposure_spin)
        exp_row.addStretch()
        inner.addLayout(exp_row)

        # Trigger source selector
        trig_row = QHBoxLayout()
        trig_row.addWidget(QLabel("Trigger"))
        self.trig_combo = QComboBox()
        self.trig_combo.addItems(["internal", "external"])
        self.trig_combo.setFixedWidth(100)
        self.trig_combo.currentTextChanged.connect(self._on_trigger_changed)
        trig_row.addWidget(self.trig_combo)
        trig_row.addStretch()
        inner.addLayout(trig_row)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setObjectName("btnConnect")
        self.btn_connect.clicked.connect(self._toggle_connect)

        self.btn_live = QPushButton("▶  Live")
        self.btn_live.setCheckable(True)
        self.btn_live.setObjectName("btnLive")
        self.btn_live.setEnabled(False)
        self.btn_live.clicked.connect(self._toggle_live)

        self.btn_snap = QPushButton("📷  Snap")
        self.btn_snap.setObjectName("btnSnap")
        self.btn_snap.setEnabled(False)
        self.btn_snap.clicked.connect(self._snap_once)

        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_live)
        btn_row.addWidget(self.btn_snap)
        inner.addLayout(btn_row)

        layout.addWidget(grp)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def connect_camera(self):
        ok = self.camera.open()
        if ok:
            self.cam_status.setText(f"Camera: connected")
            self.btn_connect.setText("Disconnect")
            self.btn_live.setEnabled(True)
            self.btn_snap.setEnabled(True)
        else:
            self.cam_status.setText("Camera: connection FAILED — check DCAM driver")
        return ok

    def disconnect_camera(self):
        self.stop_live()
        self.camera.close()
        self.cam_status.setText("Camera: disconnected")
        self.btn_connect.setText("Connect")
        self.btn_live.setEnabled(False)
        self.btn_snap.setEnabled(False)

    def start_live(self):
        if self._live_worker and self._live_worker.isRunning():
            return
        from core.acquisition import LiveViewWorker
        self._live_worker = LiveViewWorker(self.camera)
        self._live_worker.new_frame.connect(self.on_new_frame)
        self._live_worker.error.connect(self._on_camera_error)
        self._live_worker.start()
        self.btn_live.setChecked(True)
        self.btn_live.setText("⏹  Stop")

    def stop_live(self):
        if self._live_worker:
            self._live_worker.stop()
            self._live_worker = None
        self.btn_live.setChecked(False)
        self.btn_live.setText("▶  Live")

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    @pyqtSlot(np.ndarray)
    def on_new_frame(self, frame: np.ndarray):
        self._last_frame = frame
        pw = self.view_label.width()
        ph = self.view_label.height()
        self.view_label.setPixmap(frame_to_qpixmap(frame, pw, ph))

    def _toggle_connect(self):
        if self.camera.is_open:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def _toggle_live(self, checked: bool):
        if checked:
            self.start_live()
        else:
            self.stop_live()

    def _snap_once(self):
        self.stop_live()
        frame = self.camera.snap()
        if frame is not None:
            self.on_new_frame(frame)
        else:
            self.cam_status.setText("Snap failed — check camera.")

    def _on_exposure_changed(self, value: float):
        self.camera.set_exposure(value)

    def _on_trigger_changed(self, source: str):
        self.camera.set_trigger_source(source)

    def _on_camera_error(self, msg: str):
        self.cam_status.setText(f"Camera error: {msg}")
        self.stop_live()

"""
CameraPanel — Hamamatsu Orca Flash 4V3 controls (left column subpanel).

Controls only — the live image lives in the shared LiveViewPanel. This panel
forwards to HamamatsuCamera (i.e. HamamatsuDevice):
    Exposure (s)         -> setExposure
    Binning (1/2/4)      -> setBinning
    ROI / subarray       -> setSubarrayH/V + setSubArrayMode + setSubarrayHpos/Vpos
    Trigger source       -> setTriggerSource
Read-back: model, internal frame rate, sensor temperature.

Binning and ROI can only change while the camera is idle, so they are disabled
while the shared live view is running (the main window calls set_live_active()).
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal


class CameraPanel(QWidget):
    """Camera parameter panel for the Hamamatsu Orca Flash 4V3 (controls only)."""

    connection_changed = pyqtSignal(bool)     # True = connected

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        # Set by the main window so disconnect can stop the shared live view first.
        self.live_panel = None
        self._live_active = False
        self._setup_ui()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Camera — Hamamatsu Orca Flash 4V3")
        inner = QVBoxLayout(grp)
        inner.setSpacing(6)

        self.cam_status = QLabel("Camera: disconnected")
        self.cam_status.setObjectName("statusLabel")
        inner.addWidget(self.cam_status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        grid.addWidget(QLabel("Exposure"), 0, 0)
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.0001, 60.0)
        self.exposure_spin.setDecimals(4)
        self.exposure_spin.setSingleStep(0.005)
        self.exposure_spin.setSuffix(" s")
        self.exposure_spin.setValue(self.camera.get_exposure())
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)
        grid.addWidget(self.exposure_spin, 0, 1)

        grid.addWidget(QLabel("Binning"), 0, 2)
        self.binning_combo = QComboBox()
        self.binning_combo.addItems(["1", "2", "4"])
        self.binning_combo.currentTextChanged.connect(self._on_binning_changed)
        grid.addWidget(self.binning_combo, 0, 3)

        grid.addWidget(QLabel("Trigger"), 1, 0)
        self.trig_combo = QComboBox()
        self.trig_combo.addItems(["internal", "external"])
        self.trig_combo.currentTextChanged.connect(self._on_trigger_changed)
        grid.addWidget(self.trig_combo, 1, 1)
        inner.addLayout(grid)

        # ── ROI / subarray ─────────────────────────────────────────
        roi_grp = QGroupBox("ROI (subarray, sensor px)")
        roi_grid = QGridLayout(roi_grp)
        roi_grid.setHorizontalSpacing(8)
        roi_grid.setVerticalSpacing(4)

        def _mk_spin(maximum, value):
            s = QSpinBox()
            s.setRange(0, maximum)
            s.setSingleStep(4)
            s.setValue(value)
            return s

        roi_grid.addWidget(QLabel("H size"), 0, 0)
        self.roi_hsize = _mk_spin(self.camera.sensor_h, self.camera.sensor_h)
        roi_grid.addWidget(self.roi_hsize, 0, 1)
        roi_grid.addWidget(QLabel("V size"), 0, 2)
        self.roi_vsize = _mk_spin(self.camera.sensor_v, self.camera.sensor_v)
        roi_grid.addWidget(self.roi_vsize, 0, 3)

        roi_grid.addWidget(QLabel("H pos"), 1, 0)
        self.roi_hpos = _mk_spin(self.camera.sensor_h, 0)
        roi_grid.addWidget(self.roi_hpos, 1, 1)
        roi_grid.addWidget(QLabel("V pos"), 1, 2)
        self.roi_vpos = _mk_spin(self.camera.sensor_v, 0)
        roi_grid.addWidget(self.roi_vpos, 1, 3)

        roi_btn_row = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full_frame = QPushButton("Full frame")
        self.btn_full_frame.clicked.connect(self._full_frame)
        roi_btn_row.addWidget(self.btn_apply_roi)
        roi_btn_row.addWidget(self.btn_full_frame)
        roi_btn_row.addStretch()
        roi_grid.addLayout(roi_btn_row, 2, 0, 1, 4)
        inner.addWidget(roi_grp)

        self.readback = QLabel("frame rate: —    temp: —")
        self.readback.setObjectName("statusLabel")
        inner.addWidget(self.readback)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setObjectName("btnConnect")
        self.btn_connect.clicked.connect(self._toggle_connect)
        inner.addWidget(self.btn_connect)

        # Mirror of the live-view Live/Snap buttons (wired by the main window
        # via live_panel.register_buttons). They drive the single shared view.
        live_row = QHBoxLayout()
        self.btn_live = QPushButton("▶  Live")
        self.btn_live.setObjectName("btnLive")
        self.btn_snap = QPushButton("Snap")
        self.btn_snap.setObjectName("btnSnap")
        live_row.addWidget(self.btn_live)
        live_row.addWidget(self.btn_snap)
        inner.addLayout(live_row)

        layout.addWidget(grp)
        self._set_settings_enabled(False)

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def connect_camera(self):
        ok = self.camera.open()
        if ok:
            self.cam_status.setText(f"Camera: connected — {self.camera.get_model_info()}")
            self.btn_connect.setText("Disconnect")
            self._set_settings_enabled(True)
            self._sync_from_camera()
        else:
            self.cam_status.setText("Camera: connection FAILED — check DCAM driver")
        self.connection_changed.emit(self.camera.is_open)
        return ok

    def disconnect_camera(self):
        # Stop the shared live view before closing the device.
        if self.live_panel is not None:
            self.live_panel.stop_live()
        self.camera.close()
        self.cam_status.setText("Camera: disconnected")
        self.btn_connect.setText("Connect")
        self._set_settings_enabled(False)
        self.connection_changed.emit(False)

    def _toggle_connect(self):
        if self.camera.is_open:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def _sync_from_camera(self):
        widgets = (self.exposure_spin, self.binning_combo, self.trig_combo,
                   self.roi_hsize, self.roi_vsize, self.roi_hpos, self.roi_vpos)
        for w in widgets:
            w.blockSignals(True)
        self.exposure_spin.setValue(self.camera.get_exposure())
        self.binning_combo.setCurrentText(str(self.camera.get_binning()))
        self.trig_combo.setCurrentText(self.camera.get_trigger_source())
        roi = self.camera.get_roi()
        self.roi_hsize.setValue(roi['hsize'])
        self.roi_vsize.setValue(roi['vsize'])
        self.roi_hpos.setValue(roi['hpos'])
        self.roi_vpos.setValue(roi['vpos'])
        for w in widgets:
            w.blockSignals(False)
        self._update_readback()

    def _update_readback(self):
        if not self.camera.is_open:
            return
        fr = self.camera.get_internal_frame_rate()
        t = self.camera.get_temperature()
        self.readback.setText(f"frame rate: {fr:.2f} Hz    temp: {t:.1f} °C")

    # ------------------------------------------------------------------ #
    #  Live-state coupling (set by main window)
    # ------------------------------------------------------------------ #

    def set_live_active(self, active: bool):
        """ROI/binning are only settable while idle, so lock them during live."""
        self._live_active = active
        if self.camera.is_open:
            self._set_settings_enabled(True)

    # ------------------------------------------------------------------ #
    #  Setting slots
    # ------------------------------------------------------------------ #

    def _on_exposure_changed(self, value: float):
        self.camera.set_exposure(value)
        self._update_readback()

    def _on_binning_changed(self, text: str):
        self.camera.set_binning(int(text))
        self._update_readback()

    def _on_trigger_changed(self, source: str):
        self.camera.set_trigger_source(source)

    def _apply_roi(self):
        self.camera.set_roi(self.roi_hsize.value(), self.roi_vsize.value(),
                            self.roi_hpos.value(), self.roi_vpos.value())
        self._sync_from_camera()

    def _full_frame(self):
        self.camera.set_full_frame()
        self._sync_from_camera()

    # ------------------------------------------------------------------ #
    #  Enable/disable
    # ------------------------------------------------------------------ #

    def _set_settings_enabled(self, enabled: bool):
        idle = enabled and not self._live_active
        for w in (self.roi_hsize, self.roi_vsize, self.roi_hpos, self.roi_vpos,
                  self.btn_apply_roi, self.btn_full_frame, self.binning_combo):
            w.setEnabled(idle)
        for w in (self.exposure_spin, self.trig_combo):
            w.setEnabled(enabled)

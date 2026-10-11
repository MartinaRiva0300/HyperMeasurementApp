"""
CameraPanel — Teledyne Retiga E7 controls (left column subpanel).

Controls only — the live image lives in the shared LiveViewPanel. This panel
forwards to RetigaCamera (i.e. PVcamDevice / CameraHW.py settings):
    Exposure (ms, integer)  -> set_exposure
    Binning (1/2/4)         -> set_binning
    Readout port (0/1/2)    -> set_readout (resets gain to 1)
    Gain (1/2)              -> set_gain (2 only on readout 1 and 2)
    Trigger mode            -> set_trigger_mode (PVCAM exposure mode)
    ROI (h0, v0, w, h)      -> set_roi
Read-back: camera name, sensor temperature (+ setpoint), estimated maximum
frame rate for the current ROI/exposure (readout-time model of measure()).

Binning, ROI, readout, gain and trigger can only change while the camera is
idle, so they are disabled while the shared live view is running (the main
window calls set_live_active()). Changing the exposure during live restarts
the live acquisition, because PVCAM latches exp_time at start_live().
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal

from hardware.camera import TRIGGER_MODES, READOUT_PORTS, GAINS


class CameraPanel(QWidget):
    """Camera parameter panel for the Teledyne Retiga E7 (controls only)."""

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

        grp = QGroupBox("Camera — Teledyne Retiga E7")
        inner = QVBoxLayout(grp)
        inner.setSpacing(6)

        self.cam_status = QLabel("Camera: disconnected")
        self.cam_status.setObjectName("statusLabel")
        inner.addWidget(self.cam_status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        grid.addWidget(QLabel("Exposure"), 0, 0)
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(1, 3600000)          # ms; >~ s only on readout 1/2
        self.exposure_spin.setSingleStep(1)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setValue(int(self.camera.get_exposure()))
        self.exposure_spin.setToolTip(
            "Exposure time in integer milliseconds (PVCAM exp_time).\n"
            "Exposures above a few seconds need readout port 1 or 2 (Long Exposure).")
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)
        grid.addWidget(self.exposure_spin, 0, 1)

        grid.addWidget(QLabel("Binning"), 0, 2)
        self.binning_combo = QComboBox()
        self.binning_combo.addItems(["1", "2", "4"])
        self.binning_combo.currentTextChanged.connect(self._on_binning_changed)
        grid.addWidget(self.binning_combo, 0, 3)

        grid.addWidget(QLabel("Readout"), 1, 0)
        self.readout_combo = QComboBox()
        for idx, label in READOUT_PORTS.items():
            self.readout_combo.addItem(f"{idx} – {label}", idx)
        self.readout_combo.setToolTip(
            "PVCAM readout port: 0 Speed, 1 Long Exposure, 2 Long Exposure EDR.\n"
            "Changing it resets the gain to 1.")
        self.readout_combo.currentIndexChanged.connect(self._on_readout_changed)
        grid.addWidget(self.readout_combo, 1, 1)

        grid.addWidget(QLabel("Gain"), 1, 2)
        self.gain_combo = QComboBox()
        for idx, label in GAINS.items():
            self.gain_combo.addItem(f"{idx} – {label}", idx)
        self.gain_combo.setToolTip("Gain index: 1 Full Well; 2 Sensitivity (readout 1 and 2 only).")
        self.gain_combo.currentIndexChanged.connect(self._on_gain_changed)
        grid.addWidget(self.gain_combo, 1, 3)

        grid.addWidget(QLabel("Trigger"), 2, 0)
        self.trig_combo = QComboBox()
        self.trig_combo.addItems(TRIGGER_MODES)
        self.trig_combo.setToolTip(
            "PVCAM exposure mode. The scan sets 'Internal Trigger' or 'Edge Trigger'\n"
            "itself and restores 'Internal Trigger' when done.")
        self.trig_combo.currentTextChanged.connect(self._on_trigger_changed)
        grid.addWidget(self.trig_combo, 2, 1, 1, 3)
        inner.addLayout(grid)

        # ── ROI / subarray ─────────────────────────────────────────
        roi_grp = QGroupBox("ROI (sensor px, unbinned)")
        roi_grid = QGridLayout(roi_grp)
        roi_grid.setHorizontalSpacing(8)
        roi_grid.setVerticalSpacing(4)

        def _mk_spin(minimum, maximum, value):
            s = QSpinBox()
            s.setRange(minimum, maximum)
            s.setSingleStep(4)
            s.setValue(value)
            return s

        roi_grid.addWidget(QLabel("H size"), 0, 0)
        self.roi_hsize = _mk_spin(1, self.camera.sensor_h, self.camera.subarrayh)
        roi_grid.addWidget(self.roi_hsize, 0, 1)
        roi_grid.addWidget(QLabel("V size"), 0, 2)
        self.roi_vsize = _mk_spin(1, self.camera.sensor_v, self.camera.subarrayv)
        roi_grid.addWidget(self.roi_vsize, 0, 3)

        roi_grid.addWidget(QLabel("H pos"), 1, 0)
        self.roi_hpos = _mk_spin(0, self.camera.sensor_h - 1, self.camera.subarrayh_pos)
        roi_grid.addWidget(self.roi_hpos, 1, 1)
        roi_grid.addWidget(QLabel("V pos"), 1, 2)
        self.roi_vpos = _mk_spin(0, self.camera.sensor_v - 1, self.camera.subarrayv_pos)
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

        self.readback = QLabel("temp: —    max rate: —")
        self.readback.setObjectName("statusLabel")
        self.readback.setWordWrap(True)
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
            self.cam_status.setText("Camera: connection FAILED — check PVCAM driver / PyVCAM")
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

    def _widgets(self):
        return (self.exposure_spin, self.binning_combo, self.readout_combo,
                self.gain_combo, self.trig_combo,
                self.roi_hsize, self.roi_vsize, self.roi_hpos, self.roi_vpos)

    def _sync_from_camera(self):
        for w in self._widgets():
            w.blockSignals(True)
        self.exposure_spin.setValue(int(self.camera.get_exposure()))
        self.binning_combo.setCurrentText(str(self.camera.get_binning()))
        self._select_data(self.readout_combo, self.camera.get_readout_port())
        self._select_data(self.gain_combo, self.camera.get_gain())
        self.trig_combo.setCurrentText(self.camera.get_trigger_mode())
        roi = self.camera.get_roi()
        self.roi_hsize.setMaximum(self.camera.sensor_h)
        self.roi_vsize.setMaximum(self.camera.sensor_v)
        self.roi_hpos.setMaximum(self.camera.sensor_h - 1)
        self.roi_vpos.setMaximum(self.camera.sensor_v - 1)
        self.roi_hsize.setValue(roi['hsize'])
        self.roi_vsize.setValue(roi['vsize'])
        self.roi_hpos.setValue(roi['hpos'])
        self.roi_vpos.setValue(roi['vpos'])
        for w in self._widgets():
            w.blockSignals(False)
        self._update_gain_choices()
        self._update_readback()

    @staticmethod
    def _select_data(combo: QComboBox, data):
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_readback(self):
        if not self.camera.is_open:
            return
        t = self.camera.get_temperature()
        sp = self.camera.get_temperature_setpoint()
        fr = self.camera.get_max_frame_rate()
        roi = self.camera.get_roi()
        self.readback.setText(
            f"temp: {t:.1f} °C (setpoint {sp:.1f} °C)    "
            f"max rate ≈ {fr:.1f} Hz for ROI {roi['hsize']}×{roi['vsize']} "
            f"@ {self.camera.get_exposure()} ms")

    # ------------------------------------------------------------------ #
    #  Live-state coupling (set by main window)
    # ------------------------------------------------------------------ #

    def set_live_active(self, active: bool):
        """ROI/binning/readout/gain/trigger are only settable while idle."""
        self._live_active = active
        if self.camera.is_open:
            self._set_settings_enabled(True)

    # ------------------------------------------------------------------ #
    #  Setting slots
    # ------------------------------------------------------------------ #

    def _on_exposure_changed(self, value: int):
        self.camera.set_exposure(int(value))
        self._update_readback()
        # PVCAM latches exp_time at start_live(): restart the live view to apply it.
        if self._live_active and self.live_panel is not None:
            self.live_panel.restart_live()

    def _on_binning_changed(self, text: str):
        self.camera.set_binning(int(text))
        self._update_readback()

    def _on_readout_changed(self, _index: int):
        port = self.readout_combo.currentData()
        self.camera.set_readout_port(int(port))
        self.gain_combo.blockSignals(True)
        self._select_data(self.gain_combo, self.camera.get_gain())
        self.gain_combo.blockSignals(False)
        self._update_gain_choices()

    def _update_gain_choices(self):
        """Gain 2 exists only on readout ports 1 and 2."""
        port = int(self.camera.readout_port)
        idx2 = self.gain_combo.findData(2)
        model = self.gain_combo.model()
        item = model.item(idx2) if idx2 >= 0 else None
        if item is not None:
            item.setEnabled(port != 0)

    def _on_gain_changed(self, _index: int):
        gain = self.gain_combo.currentData()
        self.camera.set_gain(int(gain))
        self.gain_combo.blockSignals(True)
        self._select_data(self.gain_combo, self.camera.get_gain())
        self.gain_combo.blockSignals(False)

    def _on_trigger_changed(self, mode: str):
        self.camera.set_trigger_mode(mode)

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
                  self.btn_apply_roi, self.btn_full_frame, self.binning_combo,
                  self.readout_combo, self.gain_combo, self.trig_combo):
            w.setEnabled(idle)
        self.exposure_spin.setEnabled(enabled)

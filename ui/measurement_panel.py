"""
MeasurementPanel — hyperspectral line-scan controls (left column subpanel).

Holds every parameter from hyperspectral_measure.py: scan axis, start_pos (mm),
step (um), step_num, camera trigger (internal/external), motor velocity (auto or
manual), refresh period, the interferogram probe pixel, and HDF5 saving.

Controls only — there is no image/plot here. During a scan, frames go to the
shared live view via image_sink, and interferogram points go to point_sink /
clear_plot (wired by the main window to the LiveViewPanel and InterferogramPanel).
The scan runs in HyperSpectralWorker (background QThread).
"""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QLineEdit
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal


class MeasurementPanel(QWidget):
    started = pyqtSignal()      # a scan began (camera busy)
    stopped = pyqtSignal()      # scan ended (camera free)

    def __init__(self, stage, camera, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.camera = camera
        self._worker = None
        self._save_dir = os.path.expanduser("~/hyper_data")
        # Wired by the main window:
        self.image_sink = None      # callable(frame) -> show in live view
        self.point_sink = None      # callable(x, y)  -> add interferogram point
        self.clear_plot = None      # callable()       -> clear interferogram
        self.probe_provider = None  # callable() -> (row, col) probe pixel
        self._setup_ui()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Measurement — Hyperspectral Scan")
        pv = QVBoxLayout(grp)
        pv.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        r = 0

        grid.addWidget(QLabel("Scan axis"), r, 0)
        self.axis_combo = QComboBox()
        grid.addWidget(self.axis_combo, r, 1)
        r += 1

        grid.addWidget(QLabel("Start pos"), r, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-500, 500)
        self.start_spin.setDecimals(4)
        self.start_spin.setSuffix(" mm")
        self.start_spin.setValue(2.3)
        grid.addWidget(self.start_spin, r, 1)
        r += 1

        grid.addWidget(QLabel("Step"), r, 0)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.01, 100000)
        self.step_spin.setDecimals(2)
        self.step_spin.setSuffix(" µm")
        self.step_spin.setValue(40.0)
        grid.addWidget(self.step_spin, r, 1)
        r += 1

        grid.addWidget(QLabel("Steps (frames)"), r, 0)
        self.stepnum_spin = QSpinBox()
        self.stepnum_spin.setRange(1, 1000000)
        self.stepnum_spin.setValue(50)
        grid.addWidget(self.stepnum_spin, r, 1)
        r += 1

        grid.addWidget(QLabel("Trigger"), r, 0)
        self.trig_combo = QComboBox()
        self.trig_combo.addItems(["internal", "external"])
        grid.addWidget(self.trig_combo, r, 1)
        r += 1

        grid.addWidget(QLabel("Motor velocity"), r, 0)
        self.vel_spin = QDoubleSpinBox()
        self.vel_spin.setRange(0.000001, 250.0)
        self.vel_spin.setDecimals(4)
        self.vel_spin.setSuffix(" mm/s")
        self.vel_spin.setValue(0.125)
        self.vel_spin.setEnabled(False)
        grid.addWidget(self.vel_spin, r, 1)
        r += 1

        self.auto_vel_chk = QCheckBox("Auto velocity (from exposure/ROI)")
        self.auto_vel_chk.setChecked(True)
        self.auto_vel_chk.toggled.connect(lambda on: self.vel_spin.setEnabled(not on))
        grid.addWidget(self.auto_vel_chk, r, 0, 1, 2)
        r += 1

        grid.addWidget(QLabel("Refresh"), r, 0)
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(1, 5000)
        self.refresh_spin.setValue(50)
        self.refresh_spin.setSuffix(" ms")
        grid.addWidget(self.refresh_spin, r, 1)
        r += 1

        pv.addLayout(grid)

        # ── Saving ─────────────────────────────────────────────────
        save_grp = QGroupBox("Save")
        sv = QVBoxLayout(save_grp)
        self.save_chk = QCheckBox("Save to HDF5")
        sv.addWidget(self.save_chk)
        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample"))
        self.sample_edit = QLineEdit()
        sample_row.addWidget(self.sample_edit)
        sv.addLayout(sample_row)
        dir_row = QHBoxLayout()
        self.path_label = QLabel(self._save_dir)
        self.path_label.setObjectName("pathLabel")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        dir_row.addWidget(self.path_label, stretch=1)
        dir_row.addWidget(self.btn_browse)
        sv.addLayout(dir_row)
        pv.addWidget(save_grp)

        # ── Run controls ───────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Start scan")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.clicked.connect(self._run)
        self.btn_abort = QPushButton("⏹ Interrupt")
        self.btn_abort.setObjectName("btnHalt")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_abort)
        pv.addLayout(run_row)

        self.progress = QProgressBar()
        pv.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        pv.addWidget(self.status_label)

        layout.addWidget(grp)

    # ------------------------------------------------------------------ #
    #  Axis list (called when the stage connects)
    # ------------------------------------------------------------------ #

    def refresh_axes(self):
        current = self.axis_combo.currentText()
        self.axis_combo.clear()
        try:
            axes = self.stage.axes
        except Exception:
            axes = []
        self.axis_combo.addItems(axes)
        if current in axes:
            self.axis_combo.setCurrentText(current)

    # ------------------------------------------------------------------ #
    #  Save dir
    # ------------------------------------------------------------------ #

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select save directory", self._save_dir)
        if d:
            self._save_dir = d
            self.path_label.setText(d)

    # ------------------------------------------------------------------ #
    #  Run / abort
    # ------------------------------------------------------------------ #

    def _run(self):
        if not self.camera.is_open:
            self.status_label.setText("Camera not connected.")
            return
        if not self.stage.is_connected:
            self.status_label.setText("Stage not connected.")
            return
        axis = self.axis_combo.currentText()
        if not axis:
            self.status_label.setText("No scan axis available.")
            return
        try:
            scan = self.stage.get_wrapper(axis)
        except Exception as e:
            self.status_label.setText(f"Cannot get axis {axis}: {e}")
            return
        if scan is None:
            self.status_label.setText(f"Axis {axis} not available for scanning.")
            return

        probe = self.probe_provider() if callable(self.probe_provider) else (0, 0)

        params = dict(
            start_pos=self.start_spin.value(),
            step_um=self.step_spin.value(),
            step_num=self.stepnum_spin.value(),
            trigger=self.trig_combo.currentText(),
            save_h5=self.save_chk.isChecked(),
            save_dir=self._save_dir,
            sample=self.sample_edit.text().strip(),
            posx=probe[0],
            posy=probe[1],
            refresh_ms=self.refresh_spin.value(),
            fast_velocity=5.0,
            auto_velocity=self.auto_vel_chk.isChecked(),
            motor_velocity=self.vel_spin.value(),
            xsampling=1.0, ysampling=1.0, zsampling=1.0,
        )

        if callable(self.clear_plot):
            self.clear_plot()

        from core.acquisition import HyperSpectralWorker
        self._worker = HyperSpectralWorker(scan, self.camera, params)
        self._worker.new_image.connect(self._on_image)
        self._worker.point.connect(self._on_point)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self.status_label.setText)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        total = params['step_num'] + (1 if params['trigger'] == 'external' else 0)
        self.progress.setMaximum(total)
        self.progress.setValue(0)
        self._set_running(True)
        self.status_label.setText("Running…")
        self.started.emit()
        self._worker.start()

    def _abort(self):
        if self._worker:
            self._worker.abort()
        self.status_label.setText("Interrupting…")

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_abort.setEnabled(running)

    def setEnabled_run(self, enabled: bool):
        """Enable/disable the Start button (main window locks it during live)."""
        if not (self._worker and self._worker.isRunning()):
            self.btn_run.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    #  Worker slots
    # ------------------------------------------------------------------ #

    @pyqtSlot(np.ndarray)
    def _on_image(self, frame: np.ndarray):
        if callable(self.image_sink):
            self.image_sink(frame)

    @pyqtSlot(float, float)
    def _on_point(self, pos_mm: float, intensity: float):
        if callable(self.point_sink):
            self.point_sink(pos_mm, intensity)

    @pyqtSlot(int, int)
    def _on_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    @pyqtSlot(str)
    def _on_finished(self, path: str):
        self._set_running(False)
        self.status_label.setText(f"Done — saved {path}" if path else "Done.")
        self.stopped.emit()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._set_running(False)
        self.status_label.setText(f"Error: {msg}")
        self.stopped.emit()

    # ------------------------------------------------------------------ #
    def shutdown(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(3000)

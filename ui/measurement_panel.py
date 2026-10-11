"""
MeasurementPanel — hyperspectral line-scan controls (left column subpanel).

Holds every parameter from hyperspectral_measure.py (RetigaE7_PI): scan axis,
start_pos (mm), step (um), step_num, camera trigger (Internal / Edge), optional
list of positions loaded from a text file (first column, mm), motor velocity
(auto from exposure/ROI, or manual), refresh period, the interferogram probe
pixel, and HDF5 saving.

Controls only — there is no image/plot here. During a scan, frames go to the
shared live view via image_sink, and interferogram points go to point_sink /
clear_plot (wired by the main window to the LiveViewPanel and InterferogramPanel).
The scan runs in HyperSpectralWorker (background QThread).
"""

from __future__ import annotations

import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QLineEdit
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal


# Measurement trigger choices (measure() supports 'Internal' and 'Edge').
TRIGGER_CHOICES = [("Internal Trigger (step & snap)", 'internal'),
                   ("Edge Trigger (PI position trigger)", 'edge')]


class MeasurementPanel(QWidget):
    started = pyqtSignal()      # a scan began (camera busy)
    stopped = pyqtSignal()      # scan ended (camera free)

    def __init__(self, stage, camera, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.camera = camera
        self._worker = None
        self._save_dir = os.path.expanduser("~/hyper_data")
        self._positions: np.ndarray | None = None     # loaded targets (mm)
        self._positions_file = ''
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
        for label, key in TRIGGER_CHOICES:
            self.trig_combo.addItem(label, key)
        self.trig_combo.currentIndexChanged.connect(self._on_trigger_changed)
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
        self.auto_vel_chk.setToolTip(
            "Edge Trigger only: velocity = step / (4·exposure), capped at 75 % of the\n"
            "maximum frame rate for the current ROI (readout-time model of measure()).")
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

        # ── Positions from file (Load_positions in measure()) ──────────
        pos_grp = QGroupBox("Positions from file (Internal Trigger)")
        pl = QVBoxLayout(pos_grp)
        pos_row = QHBoxLayout()
        self.btn_load_pos = QPushButton("Load positions…")
        self.btn_load_pos.setToolTip(
            "Text/CSV file; the first column is the list of absolute targets in mm.\n"
            "When enabled, it replaces Start pos / Step / Steps (Internal Trigger only).")
        self.btn_load_pos.clicked.connect(self._load_positions)
        self.btn_clear_pos = QPushButton("Clear")
        self.btn_clear_pos.setEnabled(False)
        self.btn_clear_pos.clicked.connect(self._clear_positions)
        pos_row.addWidget(self.btn_load_pos)
        pos_row.addWidget(self.btn_clear_pos)
        pos_row.addStretch()
        pl.addLayout(pos_row)
        self.use_pos_chk = QCheckBox("Use loaded positions")
        self.use_pos_chk.setEnabled(False)
        pl.addWidget(self.use_pos_chk)
        self.pos_label = QLabel("No position file loaded")
        self.pos_label.setObjectName("pathLabel")
        self.pos_label.setWordWrap(True)
        pl.addWidget(self.pos_label)
        pv.addWidget(pos_grp)

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
        self._on_trigger_changed()

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
    #  Trigger / positions
    # ------------------------------------------------------------------ #

    def trigger_key(self) -> str:
        return self.trig_combo.currentData() or 'internal'

    def _on_trigger_changed(self, *_):
        edge = self.trigger_key() == 'edge'
        # Auto/manual velocity only matters for the continuous (Edge) scan.
        self.auto_vel_chk.setEnabled(edge)
        self.vel_spin.setEnabled(edge and not self.auto_vel_chk.isChecked())
        # Loaded positions only apply to the Internal-trigger scan.
        has_pos = self._positions is not None
        self.use_pos_chk.setEnabled(has_pos and not edge)

    def _load_positions(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select position file", self._save_dir,
            "Text files (*.txt);;CSV files (*.csv);;All files (*)")
        if not filename:
            return
        try:
            data = np.loadtxt(filename, dtype=float, ndmin=2)
            positions = data[:, 0]               # first column only (measure())
        except Exception:
            try:
                data = np.loadtxt(filename, dtype=float, ndmin=2, delimiter=',')
                positions = data[:, 0]
            except Exception as e:
                self.status_label.setText(f"Could not read positions: {e}")
                return
        if positions.size == 0:
            self.status_label.setText("Position file is empty.")
            return
        self._positions = positions
        self._positions_file = filename
        self.pos_label.setText(
            f"{os.path.basename(filename)}: {positions.size} positions, "
            f"{positions.min():.4f} → {positions.max():.4f} mm")
        self.use_pos_chk.setChecked(True)
        self.btn_clear_pos.setEnabled(True)
        self._on_trigger_changed()

    def _clear_positions(self):
        self._positions = None
        self._positions_file = ''
        self.pos_label.setText("No position file loaded")
        self.use_pos_chk.setChecked(False)
        self.btn_clear_pos.setEnabled(False)
        self._on_trigger_changed()

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
        trigger = self.trigger_key()
        use_positions = (trigger == 'internal' and self._positions is not None
                         and self.use_pos_chk.isChecked())

        params = dict(
            start_pos=self.start_spin.value(),
            step_um=self.step_spin.value(),
            step_num=self.stepnum_spin.value(),
            trigger=trigger,
            positions=self._positions if use_positions else None,
            positions_file=self._positions_file if use_positions else '',
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
            camera_settings=self.camera.settings_dict(),
        )
        total = len(self._positions) if use_positions else params['step_num']
        if use_positions:
            self.stepnum_spin.setValue(total)

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

"""
AcquisitionPanel — build a list of positions, run the sequence, save results.

Mirrors the internal-trigger branch of hyperspectral_measure.measure():
  stage.motor.move_absolute(pos) + wait_on_target() -> camera snap (fixed_length, 1 frame).

Results saved as .npy files (one per position).
"""

import os
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QProgressBar,
    QFileDialog, QSpinBox, QDoubleSpinBox, QHeaderView,
    QAbstractItemView, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSlot


class AcquisitionPanel(QWidget):
    def __init__(self, stage, camera, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.camera = camera
        self._worker = None
        self._results = []
        self._save_dir = os.path.expanduser("~/microscope_data")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        grp = QGroupBox("Acquisition Sequence  (move → snap)")
        grp.setObjectName("ctrlGroup")
        inner = QVBoxLayout(grp)
        inner.setSpacing(6)

        # ── Position table ────────────────────────────────────────
        # Columns: #, Axis 1 (mm), Axis 2 (mm), Axis 3 (mm)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Axis 1 (mm)", "Axis 2 (mm)", "Axis 3 (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(160)
        self.table.setObjectName("posTable")
        inner.addWidget(self.table)

        # ── Position list buttons ─────────────────────────────────
        pos_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Add current position")
        self.btn_add.setObjectName("btnConnect")
        self.btn_add.setToolTip("Read current stage position and add it to the list")
        self.btn_add.clicked.connect(self._add_current)

        self.btn_remove = QPushButton("− Remove")
        self.btn_remove.clicked.connect(self._remove_selected)

        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(lambda: self.table.setRowCount(0))

        pos_row.addWidget(self.btn_add)
        pos_row.addWidget(self.btn_remove)
        pos_row.addWidget(self.btn_clear)
        pos_row.addStretch()
        inner.addLayout(pos_row)

        # ── Settings ──────────────────────────────────────────────
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Settle after move"))
        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(0, 10000)
        self.settle_spin.setValue(300)
        self.settle_spin.setSuffix(" ms")
        self.settle_spin.setFixedWidth(80)
        self.settle_spin.setToolTip(
            "Wait time after stage reaches target before triggering camera\n"
            "(accounts for mechanical vibration settling)")
        settings_row.addWidget(self.settle_spin)

        settings_row.addSpacing(16)
        settings_row.addWidget(QLabel("Save to"))
        self.path_label = QLabel(self._save_dir)
        self.path_label.setObjectName("pathLabel")
        settings_row.addWidget(self.path_label)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        settings_row.addWidget(self.btn_browse)
        settings_row.addStretch()
        inner.addLayout(settings_row)

        # ── Run row ────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run sequence")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.setFixedHeight(34)
        self.btn_run.clicked.connect(self._run)

        self.btn_abort = QPushButton("⏹  Abort")
        self.btn_abort.setObjectName("btnHalt")
        self.btn_abort.setFixedHeight(34)
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)

        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_abort)
        inner.addLayout(run_row)

        # ── Progress ──────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setObjectName("acqProgress")
        inner.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        inner.addWidget(self.status_label)

        layout.addWidget(grp)

    # ------------------------------------------------------------------ #
    #  Position list management
    # ------------------------------------------------------------------ #

    def _add_current(self):
        if not self.stage.is_connected:
            self.status_label.setText("Stage not connected.")
            return
        pos = self.stage.get_position()   # dict {axis: mm}
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        for col, ax in enumerate(['1', '2', '3'], start=1):
            val = pos.get(ax, 0.0)
            self.table.setItem(row, col, QTableWidgetItem(f"{val:.6f}"))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))

    def _get_positions(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            pos = {}
            for col, ax in enumerate(['1', '2', '3'], start=1):
                item = self.table.item(r, col)
                if item and item.text().strip():
                    pos[ax] = float(item.text())
            if pos:
                out.append(pos)
        return out

    # ------------------------------------------------------------------ #
    #  Save directory
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
        positions = self._get_positions()
        if not positions:
            self.status_label.setText("No positions — use '+ Add current position'.")
            return
        if not self.stage.is_connected:
            self.status_label.setText("Stage not connected.")
            return
        if not self.camera.is_open:
            self.status_label.setText("Camera not connected.")
            return

        os.makedirs(self._save_dir, exist_ok=True)
        self._results = []

        from core.acquisition import AcquisitionWorker
        self._worker = AcquisitionWorker(
            self.stage, self.camera, positions,
            settle_ms=self.settle_spin.value()
        )
        self.progress.setMaximum(len(positions))
        self.progress.setValue(0)
        self._worker.progress.connect(self._on_progress)
        self._worker.new_image.connect(self._on_new_image)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.status_label.setText("Running…")
        self._worker.start()

    def _abort(self):
        if self._worker:
            self._worker.abort()
        self.status_label.setText("Aborted.")
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)

    # ------------------------------------------------------------------ #
    #  Worker slots
    # ------------------------------------------------------------------ #

    @pyqtSlot(int, int)
    def _on_progress(self, current, total):
        self.progress.setValue(current)
        self.status_label.setText(f"Step {current} / {total}")

    @pyqtSlot(dict, np.ndarray)
    def _on_new_image(self, pos: dict, frame: np.ndarray):
        self._results.append((pos, frame))
        ts = time.strftime("%Y%m%d_%H%M%S")
        pos_str = "_".join(f"ax{k}_{v:.4f}mm" for k, v in sorted(pos.items()))
        fname = os.path.join(self._save_dir, f"{ts}_{pos_str}.npy")
        np.save(fname, frame)

    @pyqtSlot(list)
    def _on_finished(self, results):
        self._results = results
        n = len(results)
        self.status_label.setText(
            f"Done — {n} frame(s) saved to {self._save_dir}"
        )
        self.progress.setValue(self.progress.maximum())
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)

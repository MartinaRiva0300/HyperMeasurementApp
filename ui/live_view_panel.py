"""
LiveViewPanel — the single live image window (right side, top 3/4).

Owns the shared pyqtgraph ImageView, the display options (auto levels + manual
min/max, auto range), the primary Live/Snap buttons, and a draggable crosshair
cursor with live intensity profiles (intensity along the row and the column
through the cursor). The camera live worker lives here; measurement frames are
pushed in via display_frame(), so there is exactly ONE image window in the app.

Other panels (e.g. the camera control panel) can mirror the Live/Snap buttons by
calling register_buttons(); all copies stay in sync and drive the same worker.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QCheckBox,
    QLabel, QSpinBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from ui.widgets import make_image_view, show_frame


class LiveViewPanel(QWidget):
    live_started = pyqtSignal()
    live_stopped = pyqtSignal()

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._live_worker = None
        self._last_frame: np.ndarray | None = None
        self._live_buttons: list[QPushButton] = []
        self._snap_buttons: list[QPushButton] = []
        self._setup_ui()
        self._live_buttons.append(self.btn_live)
        self._snap_buttons.append(self.btn_snap)
        self.set_camera_available(False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── Image over profile plot (profile hidden until cursor on) ──
        self.imv = make_image_view()
        self.imv.setMinimumSize(480, 300)

        self.profile_plot = pg.PlotWidget(title="Cursor profile")
        self.profile_plot.addLegend(offset=(10, 10))
        self.profile_plot.setLabel('bottom', 'pixel')
        self.profile_plot.setLabel('left', 'intensity')
        self.h_curve = self.profile_plot.plot([], [], pen=pg.mkPen('c', width=1),
                                              name='row (horizontal)')
        self.v_curve = self.profile_plot.plot([], [], pen=pg.mkPen('m', width=1),
                                              name='col (vertical)')
        self.profile_plot.setMinimumHeight(120)
        self.profile_plot.hide()

        self._img_splitter = QSplitter(Qt.Vertical)
        self._img_splitter.addWidget(self.imv)
        self._img_splitter.addWidget(self.profile_plot)
        self._img_splitter.setStretchFactor(0, 3)
        self._img_splitter.setStretchFactor(1, 1)
        layout.addWidget(self._img_splitter, stretch=1)

        # ── Crosshair cursor (added to the image view, hidden initially) ──
        pen = pg.mkPen((255, 220, 0), width=1)
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pen)   # selects column (x)
        self.hline = pg.InfiniteLine(angle=0, movable=True, pen=pen)    # selects row (y)
        view = self.imv.getView()
        view.addItem(self.vline, ignoreBounds=True)
        view.addItem(self.hline, ignoreBounds=True)
        self.vline.hide()
        self.hline.hide()
        self.vline.sigPositionChanged.connect(self._update_profiles)
        self.hline.sigPositionChanged.connect(self._update_profiles)

        # ── Button / option row ────────────────────────────────────
        row = QHBoxLayout()
        self.btn_live = QPushButton("▶  Live")
        self.btn_live.setObjectName("btnLive")
        self.btn_live.setCheckable(True)
        self.btn_live.clicked.connect(self._on_live_clicked)

        self.btn_snap = QPushButton("Snap")
        self.btn_snap.setObjectName("btnSnap")
        self.btn_snap.clicked.connect(self._on_snap_clicked)

        self.btn_cursor = QPushButton("⌖ Cursor profile")
        self.btn_cursor.setCheckable(True)
        self.btn_cursor.setToolTip(
            "Show a draggable crosshair and the intensity profile along its\n"
            "row and column (updates live and while you drag the cursor)")
        self.btn_cursor.toggled.connect(self._toggle_cursor)

        self.chk_auto_levels = QCheckBox("Auto levels")
        self.chk_auto_levels.setChecked(True)
        self.chk_auto_levels.toggled.connect(self._on_auto_levels_toggled)

        self.level_min = QSpinBox()
        self.level_min.setRange(0, 65535)
        self.level_min.setValue(60)
        self.level_min.setPrefix("min ")
        self.level_min.setFixedWidth(100)
        self.level_min.valueChanged.connect(self._reapply_levels)

        self.level_max = QSpinBox()
        self.level_max.setRange(0, 65535)
        self.level_max.setValue(4000)
        self.level_max.setPrefix("max ")
        self.level_max.setFixedWidth(100)
        self.level_max.valueChanged.connect(self._reapply_levels)

        self.chk_auto_range = QCheckBox("Auto range")
        self.chk_auto_range.setChecked(True)

        def _sep():
            f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFrameShadow(QFrame.Sunken)
            return f

        row.addWidget(self.btn_live)
        row.addWidget(self.btn_snap)
        row.addWidget(self.btn_cursor)
        row.addWidget(_sep())
        row.addWidget(self.chk_auto_levels)
        row.addWidget(self.level_min)
        row.addWidget(self.level_max)
        row.addWidget(_sep())
        row.addWidget(self.chk_auto_range)
        row.addStretch()
        layout.addLayout(row)

        self._on_auto_levels_toggled(self.chk_auto_levels.isChecked())

    # ------------------------------------------------------------------ #
    #  Mirror buttons (registered by other panels)
    # ------------------------------------------------------------------ #

    def register_buttons(self, live_btn: QPushButton, snap_btn: QPushButton):
        live_btn.setCheckable(True)
        live_btn.clicked.connect(self._on_live_clicked)
        snap_btn.clicked.connect(self._on_snap_clicked)
        self._live_buttons.append(live_btn)
        self._snap_buttons.append(snap_btn)
        self._sync_live_buttons()

    # ------------------------------------------------------------------ #
    #  External control (used by the main window)
    # ------------------------------------------------------------------ #

    def set_camera_available(self, available: bool):
        if not available:
            self.stop_live()
        self._set_buttons_enabled(available)

    def set_controls_enabled(self, enabled: bool):
        self._set_buttons_enabled(enabled and self.camera.is_open)

    def _set_buttons_enabled(self, ok: bool):
        for b in self._live_buttons + self._snap_buttons:
            b.setEnabled(ok)

    def _sync_live_buttons(self):
        running = bool(self._live_worker and self._live_worker.isRunning())
        for b in self._live_buttons:
            b.setChecked(running)
            b.setText("⏹  Stop" if running else "▶  Live")

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    # ------------------------------------------------------------------ #
    #  Display
    # ------------------------------------------------------------------ #

    def display_frame(self, frame: np.ndarray):
        self._last_frame = frame
        auto = self.chk_auto_levels.isChecked()
        show_frame(self.imv, frame,
                   auto_levels=auto,
                   auto_range=self.chk_auto_range.isChecked(),
                   level_min=self.level_min.value(),
                   level_max=self.level_max.value())
        if self.profile_plot.isVisible():
            self._update_profiles()

    def _on_auto_levels_toggled(self, auto: bool):
        self.level_min.setEnabled(not auto)
        self.level_max.setEnabled(not auto)
        self._reapply_levels()

    def _reapply_levels(self):
        if self._last_frame is not None:
            self.display_frame(self._last_frame)

    # ------------------------------------------------------------------ #
    #  Cursor / intensity profiles
    # ------------------------------------------------------------------ #

    def _toggle_cursor(self, on: bool):
        self.vline.setVisible(on)
        self.hline.setVisible(on)
        self.profile_plot.setVisible(on)
        if on:
            # Centre the crosshair on the current frame the first time.
            if self._last_frame is not None:
                v, h = self._last_frame.shape          # (rows, cols)
                self.vline.setValue(h / 2)             # x = column
                self.hline.setValue(v / 2)             # y = row
            self._update_profiles()

    def _update_profiles(self):
        frame = self._last_frame
        if frame is None or not self.profile_plot.isVisible():
            return
        v, h = frame.shape                              # (rows, cols)
        col = int(np.clip(round(self.vline.value()), 0, h - 1))
        row = int(np.clip(round(self.hline.value()), 0, v - 1))
        # Horizontal profile: intensity along the selected row, vs column.
        self.h_curve.setData(np.arange(h), frame[row, :])
        # Vertical profile: intensity along the selected column, vs row.
        self.v_curve.setData(np.arange(v), frame[:, col])
        self.profile_plot.setTitle(
            f"Cursor (row {row}, col {col}) = {int(frame[row, col])}")

    # ------------------------------------------------------------------ #
    #  Live worker
    # ------------------------------------------------------------------ #

    def start_live(self):
        if not self.camera.is_open:
            return
        if self._live_worker and self._live_worker.isRunning():
            return
        from core.acquisition import LiveViewWorker
        self._live_worker = LiveViewWorker(self.camera)
        self._live_worker.new_frame.connect(self._on_new_frame)
        self._live_worker.error.connect(self._on_error)
        self._live_worker.start()
        self._sync_live_buttons()
        self.live_started.emit()

    def stop_live(self):
        was_running = self._live_worker is not None
        if self._live_worker:
            self._live_worker.stop()
            self._live_worker = None
        self._sync_live_buttons()
        if was_running:
            self.live_stopped.emit()

    @pyqtSlot(np.ndarray)
    def _on_new_frame(self, frame: np.ndarray):
        self.display_frame(frame)

    def _on_live_clicked(self, *_):
        if self._live_worker and self._live_worker.isRunning():
            self.stop_live()
        else:
            self.start_live()

    def _on_snap_clicked(self, *_):
        self.stop_live()
        frame = self.camera.snap()
        if frame is not None:
            self.display_frame(frame)

    def _on_error(self, msg: str):
        print(f"[LiveView] camera error: {msg}")
        self.stop_live()

"""Stage control panel for the GUI: the TWINS wedge stage (SmarAct MCS2).

Connect and manual jog/move for the interferometer wedge positioner. Blocking
operations (connect, reference, move) run in background threads so the UI never
freezes; position is polled on a timer. A "Simulate" checkbox lets you exercise
the controls with no hardware (default ON until the stage is wired up).

The actual TWINS interferogram scan (sweep stage + read camera ROI -> spectrum)
lives in ui/twins_scan.py and ui/measure_kspace.py; this panel covers the
connection + manual control.
"""
from __future__ import annotations

import threading

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from instruments.twins_stage import TwinsStage, TRAVEL_MM


class StageController(QtCore.QObject):
    """Wraps a stage driver; runs blocking ops on a thread, emits updates."""

    status = QtCore.pyqtSignal(str)
    position = QtCore.pyqtSignal(str)
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, driver, fmt) -> None:
        super().__init__()
        self.driver = driver
        self._fmt = fmt
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def run(self, fn, done_msg: str = "") -> None:
        if self._busy:
            return
        self._busy = True
        self.busy_changed.emit(True)

        def _work():
            try:
                fn()
                if done_msg:
                    self.status.emit(done_msg)
            except Exception as e:  # noqa: BLE001
                self.status.emit(f"error: {e}")
            finally:
                self._busy = False
                self.busy_changed.emit(False)
                self._emit_pos()

        threading.Thread(target=_work, daemon=True).start()

    def poll(self) -> None:
        if not self._busy:
            self._emit_pos()

    def _emit_pos(self) -> None:
        try:
            if getattr(self.driver, "is_connected", False):
                self.position.emit(self._fmt(self.driver))
        except Exception:  # noqa: BLE001
            pass


class StagesPanel(QWidget):
    """Owns the TWINS-stage driver, its controller and the position-poll timer.

    ``twins_group`` is exposed so it can be placed into the TWINS tab; this
    widget itself stays hidden and just keeps the timer/threads alive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.twins = TwinsStage()
        self.twins_ctl = StageController(
            self.twins, lambda d: f"{d.get_position():.4f} mm")

        self.twins_group = self._build_twins_group()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)

    # -- TWINS (SmarAct MCS2 / SLC-1750) stage -------------------------------
    def _build_twins_group(self) -> QGroupBox:
        g = QGroupBox("TWINS Stage (SmarAct MCS2 / SLC-1750)")
        v = QVBoxLayout(g)

        self.t_sim = QCheckBox("Simulate (no hardware)")
        self.t_sim.setChecked(True)
        v.addWidget(self.t_sim)

        self.t_connect = QPushButton("Connect")
        self.t_connect.clicked.connect(self._twins_connect)
        v.addWidget(self.t_connect)

        self.t_pos = QLabel("-- mm")
        self.t_pos.setStyleSheet("font-weight:600;")
        v.addWidget(self.t_pos)

        go = QHBoxLayout()
        go.addWidget(QLabel("Go to"))
        self.t_goto = QDoubleSpinBox()
        # Absolute range follows the positioner's closed-loop travel.
        self.t_goto.setRange(0.0, TRAVEL_MM)
        self.t_goto.setDecimals(3)
        self.t_goto.setValue(min(24.0, TRAVEL_MM))
        self.t_goto.setSuffix(" mm")
        go.addWidget(self.t_goto)
        self.t_go = QPushButton("Go")
        self.t_go.clicked.connect(
            lambda: self.twins_ctl.run(
                lambda: (self.twins.move_to(self.t_goto.value()),
                         self.twins.wait_for_stop()),
                f"moved to {self.t_goto.value():.3f} mm"))
        go.addWidget(self.t_go)
        v.addLayout(go)

        # Jog by +/- a step in micrometres (relative move).
        jog = QHBoxLayout()
        jog.addWidget(QLabel("Jog"))
        self.t_step = QDoubleSpinBox()
        self.t_step.setRange(0.1, 5000.0)
        self.t_step.setDecimals(1)
        self.t_step.setSingleStep(1.0)
        self.t_step.setValue(10.0)
        self.t_step.setSuffix(" µm")
        jog.addWidget(self.t_step)
        self.t_minus = QPushButton("−")
        self.t_minus.clicked.connect(lambda: self._twins_jog(-1.0))
        self.t_plus = QPushButton("+")
        self.t_plus.clicked.connect(lambda: self._twins_jog(+1.0))
        jog.addWidget(self.t_minus)
        jog.addWidget(self.t_plus)
        v.addLayout(jog)

        self.t_status = QLabel("offline")
        self.t_status.setStyleSheet("color:#888; font-size:11px;")
        self.t_status.setWordWrap(True)
        v.addWidget(self.t_status)

        self.twins_ctl.position.connect(self.t_pos.setText)
        self.twins_ctl.status.connect(self.t_status.setText)
        self.twins_ctl.busy_changed.connect(lambda *_: self._refresh_twins())
        self._refresh_twins()
        return g

    def _twins_connect(self) -> None:
        if self.twins.is_connected:
            # Leave the wedge where it is on disconnect (no move to SAFE).
            self.twins_ctl.run(lambda: self.twins.disconnect(safe=False), "disconnected")
        else:
            # Reference (so absolute positions are valid) then park at HOME,
            # matching the reference repo. Connecting moves the wedge.
            sim = self.t_sim.isChecked()
            self.twins_ctl.run(
                lambda: self.twins.connect(simulate=sim, home=True), "connected")

    def _twins_jog(self, sign: float) -> None:
        # micrometres -> mm; refresh the cached position first so the relative
        # move is taken from the stage's actual current position.
        dmm = sign * self.t_step.value() / 1000.0
        self.twins_ctl.run(
            lambda: (self.twins.get_position(),
                     self.twins.move_by(dmm),
                     self.twins.wait_for_stop()),
            f"jogged {sign * self.t_step.value():+.1f} µm")

    def _refresh_twins(self) -> None:
        conn = self.twins.is_connected
        busy = self.twins_ctl.busy
        self.t_connect.setText("Disconnect" if conn else "Connect")
        self.t_connect.setEnabled(not busy)
        self.t_sim.setEnabled(not conn and not busy)
        for w in (self.t_go, self.t_goto, self.t_step, self.t_minus, self.t_plus):
            w.setEnabled(conn and not busy)
        if not conn:
            self.t_pos.setText("-- mm")

    def _poll(self) -> None:
        self.twins_ctl.poll()

    def freeze(self, frozen: bool) -> None:
        """Pause position polling and lock the manual controls during a Measure
        scan that drives the stage from its own thread (avoids DLL contention)."""
        if frozen:
            self.timer.stop()
        else:
            self.timer.start(500)
        self.twins_group.setEnabled(not frozen)

    def shutdown(self) -> None:
        try:
            if self.twins.is_connected:
                self.twins.disconnect()
        except Exception:  # noqa: BLE001
            pass

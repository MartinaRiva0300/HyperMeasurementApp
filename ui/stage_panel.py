"""
StagePanel — position readout (mm), manual jog controls (µm steps),
velocity setting, home and halt — all matching PI_VC_Device's API.

Per axis: current position (encoder read-back), an independent "Go to" target
(to-go position), and a relative jog step (µm).
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSlot


class AxisJogWidget(QWidget):
    """
    One row of controls for a single PI axis.

    Jog step is in µm (PI_VC_Device.move_relative expects µm).
    Go-to target is in mm (PI_VC_Device.move_absolute expects mm).
    Position read-back is in mm.
    """

    def __init__(self, axis_label: str, on_jog, parent=None):
        super().__init__(parent)
        self.axis_label = axis_label
        self.on_jog = on_jog
        self._target_initialised = False
        self._setup_ui()

    def _setup_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)

        # ── Axis label ──────────────────────────────────────────────
        lbl = QLabel(f"Axis {self.axis_label}")
        lbl.setFixedWidth(48)
        lbl.setObjectName("axisLabel")
        row.addWidget(lbl)

        # ── Position display (mm) — encoder read-back ───────────────
        self.pos_label = QLabel("0.000000 mm")
        self.pos_label.setFixedWidth(110)
        self.pos_label.setObjectName("posDisplay")
        self.pos_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.pos_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        row.addWidget(sep)

        # ── Jog (µm) ───────────────────────────────────────────────
        row.addWidget(QLabel("Jog"))

        self.btn_neg = QPushButton("−")
        self.btn_neg.setFixedSize(28, 28)
        self.btn_neg.setObjectName("jogBtn")
        self.btn_neg.clicked.connect(lambda: self._jog(-1))

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.001, 10000.0)
        self.step_spin.setValue(10.0)
        self.step_spin.setDecimals(3)
        self.step_spin.setSuffix(" µm")
        self.step_spin.setFixedWidth(100)
        self.step_spin.setToolTip("Relative jog step size in micrometres")

        self.btn_pos = QPushButton("+")
        self.btn_pos.setFixedSize(28, 28)
        self.btn_pos.setObjectName("jogBtn")
        self.btn_pos.clicked.connect(lambda: self._jog(+1))

        row.addWidget(self.btn_neg)
        row.addWidget(self.step_spin)
        row.addWidget(self.btn_pos)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        row.addWidget(sep2)

        # ── Go to absolute (mm) — the to-go position ────────────────
        row.addWidget(QLabel("Go to"))
        self.goto_spin = QDoubleSpinBox()
        self.goto_spin.setRange(-500.0, 500.0)
        self.goto_spin.setDecimals(6)
        self.goto_spin.setSuffix(" mm")
        self.goto_spin.setFixedWidth(120)
        self.goto_spin.setToolTip("Absolute target (to-go) position in mm")

        self.btn_goto = QPushButton("Go")
        self.btn_goto.setFixedWidth(38)
        self.btn_goto.setObjectName("btnGo")
        self.btn_goto.clicked.connect(self._goto)

        row.addWidget(self.goto_spin)
        row.addWidget(self.btn_goto)
        row.addStretch()

    def update_position(self, value_mm: float):
        # Encoder read-back only. The "Go to" target is independent — it is the
        # user's to-go position and must not be overwritten by polling.
        self.pos_label.setText(f"{value_mm:.6f} mm")
        if not self._target_initialised:
            # Seed the to-go field with the current position the first time only.
            self.goto_spin.blockSignals(True)
            self.goto_spin.setValue(value_mm)
            self.goto_spin.blockSignals(False)
            self._target_initialised = True

    def set_enabled(self, enabled: bool):
        for w in [self.btn_neg, self.btn_pos, self.btn_goto,
                  self.step_spin, self.goto_spin]:
            w.setEnabled(enabled)

    def _jog(self, direction: int):
        step_um = self.step_spin.value() * direction
        # 'relative' -> µm (PI_VC_Device.move_relative expects µm)
        self.on_jog('relative', {self.axis_label: step_um})

    def _goto(self):
        target_mm = self.goto_spin.value()
        # 'absolute' -> mm (PI_VC_Device.move_absolute expects mm)
        self.on_jog('absolute', {self.axis_label: target_mm})


class StagePanel(QWidget):
    """Stage control panel: per-axis readout, jog, velocity, home, halt."""

    def __init__(self, stage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self._poller = None
        self._axis_widgets: dict[str, AxisJogWidget] = {}
        # Optional hook the main window sets to be notified when the stage connects.
        self.on_connected = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        grp = QGroupBox("Motorised Stage  (PI)")
        grp.setObjectName("ctrlGroup")
        inner = QVBoxLayout(grp)
        inner.setSpacing(8)

        # ── Top bar: status + connect + home + halt ───────────────
        top = QHBoxLayout()
        self.stage_status = QLabel("Stage: disconnected")
        self.stage_status.setObjectName("statusLabel")
        top.addWidget(self.stage_status)
        top.addStretch()

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setObjectName("btnConnect")
        self.btn_connect.setFixedWidth(90)
        self.btn_connect.clicked.connect(self._toggle_connect)

        self.btn_home = QPushButton("⌂ Home")
        self.btn_home.setObjectName("btnHome")
        self.btn_home.setEnabled(False)
        self.btn_home.setToolTip("Move all axes to their home/reference position")
        self.btn_home.clicked.connect(self._go_home)

        self.btn_set_home = QPushButton("Set Home")
        self.btn_set_home.setObjectName("btnHome")
        self.btn_set_home.setEnabled(False)
        self.btn_set_home.setToolTip("Set current position as home (zero reference)")
        self.btn_set_home.clicked.connect(self._set_home)

        self.btn_halt = QPushButton("⏹ HALT")
        self.btn_halt.setObjectName("btnHalt")
        self.btn_halt.setEnabled(False)
        self.btn_halt.clicked.connect(self._halt)

        for w in [self.btn_connect, self.btn_home, self.btn_set_home, self.btn_halt]:
            top.addWidget(w)
        inner.addLayout(top)

        # ── Velocity row ──────────────────────────────────────────
        vel_row = QHBoxLayout()
        vel_row.addWidget(QLabel("Velocity"))
        self.vel_spin = QDoubleSpinBox()
        self.vel_spin.setRange(0.000001, 250.0)
        self.vel_spin.setValue(5.0)
        self.vel_spin.setDecimals(4)
        self.vel_spin.setSuffix(" mm/s")
        self.vel_spin.setFixedWidth(110)
        self.vel_spin.setToolTip("Stage velocity (applied to all axes via set_velocity)")
        self.btn_set_vel = QPushButton("Apply")
        self.btn_set_vel.setFixedWidth(55)
        self.btn_set_vel.setEnabled(False)
        self.btn_set_vel.clicked.connect(self._set_velocity)
        vel_row.addWidget(self.vel_spin)
        vel_row.addWidget(self.btn_set_vel)
        vel_row.addStretch()
        inner.addLayout(vel_row)

        # ── Per-axis rows (populated after connect) ───────────────
        self.axes_container = QWidget()
        self.axes_layout = QVBoxLayout(self.axes_container)
        self.axes_layout.setContentsMargins(0, 0, 0, 0)
        self.axes_layout.setSpacing(2)
        inner.addWidget(self.axes_container)
        inner.addStretch()

        layout.addWidget(grp)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def connect_stage(self):
        ok = self.stage.connect()
        if ok:
            self._build_axis_rows()
            self._start_poller()
            self.stage_status.setText(
                f"Stage: connected  ·  {len(self.stage.axes)} axis/axes"
            )
            self.btn_connect.setText("Disconnect")
            for b in [self.btn_home, self.btn_set_home, self.btn_halt, self.btn_set_vel]:
                b.setEnabled(True)
            # Read and display current velocity (first axis)
            try:
                ax0 = self.stage.axes[0]
                w = self.stage.get_wrapper(ax0)
                if w:
                    self.vel_spin.setValue(w.get_velocity())
            except Exception:
                pass
            if callable(self.on_connected):
                self.on_connected()
        else:
            self.stage_status.setText("Stage: connection FAILED — check serial number")
        return ok

    def disconnect_stage(self):
        self._stop_poller()
        self.stage.disconnect()
        self.stage_status.setText("Stage: disconnected")
        self.btn_connect.setText("Connect")
        for b in [self.btn_home, self.btn_set_home, self.btn_halt, self.btn_set_vel]:
            b.setEnabled(False)
        for w in self._axis_widgets.values():
            w.deleteLater()
        self._axis_widgets.clear()

    def set_axes_enabled(self, enabled: bool):
        for w in self._axis_widgets.values():
            w.set_enabled(enabled)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _build_axis_rows(self):
        for ax in self.stage.axes:
            row = AxisJogWidget(ax, self._on_jog)
            self._axis_widgets[ax] = row
            self.axes_layout.addWidget(row)

    def _start_poller(self):
        from core.acquisition import StagePoller
        self._poller = StagePoller(self.stage, interval_ms=250)
        self._poller.position_updated.connect(self._on_position_updated)
        self._poller.start()

    def _stop_poller(self):
        if self._poller:
            self._poller.stop()
            self._poller = None

    def _toggle_connect(self):
        if self.stage.is_connected:
            self.disconnect_stage()
        else:
            self.connect_stage()

    def _go_home(self):
        from core.acquisition import JogWorker
        # Absolute move to 0.0 mm on each axis — same as PI_VC_Device.go_home()
        pos = {ax: 0.0 for ax in self.stage.axes}
        worker = JogWorker(self.stage, 'absolute', pos)
        self.set_axes_enabled(False)
        worker.move_done.connect(lambda: self.set_axes_enabled(True))
        worker.start()
        self._home_worker = worker  # keep a reference so it isn't GC'd

    def _set_home(self):
        """Calls motor.set_home() on each axis wrapper."""
        for ax in self.stage.axes:
            try:
                w = self.stage.get_wrapper(ax)
                if w:
                    w.set_home()
            except Exception as e:
                print(f"[Stage] set_home error on axis {ax}: {e}")

    def _halt(self):
        self.stage.halt()

    def _set_velocity(self):
        """Apply velocity to all axis wrappers via PI_VC_Device.set_velocity()."""
        v = self.vel_spin.value()
        for ax in self.stage.axes:
            try:
                w = self.stage.get_wrapper(ax)
                if w:
                    w.set_velocity(v)
            except Exception as e:
                print(f"[Stage] set_velocity error on axis {ax}: {e}")

    def _on_jog(self, move_type: str, params: dict):
        from core.acquisition import JogWorker
        worker = JogWorker(self.stage, move_type, params)
        self.set_axes_enabled(False)
        worker.move_done.connect(lambda: self.set_axes_enabled(True))
        worker.error.connect(lambda msg: print(f"[Jog error] {msg}"))
        worker.start()
        self._jog_worker = worker  # keep a reference so it isn't GC'd

    @pyqtSlot(dict)
    def _on_position_updated(self, pos: dict):
        for ax, val in pos.items():
            if ax in self._axis_widgets:
                self._axis_widgets[ax].update_position(val)

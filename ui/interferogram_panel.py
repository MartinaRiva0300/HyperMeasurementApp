"""
InterferogramPanel — the interferogram plot (right side, bottom 1/4).

Plots intensity at the probe pixel versus stage position, updated live during a
measurement. The probe-pixel selector (row, col) sits in the top-right corner of
this panel; the measurement reads it via get_probe() at scan start.
"""

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox
)


class InterferogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._x: list[float] = []
        self._y: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Top bar: probe-pixel selector, pushed to the right ──────
        top = QHBoxLayout()
        top.addStretch()
        top.addWidget(QLabel("Probe px (row, col)"))
        self.posx_spin = QSpinBox(); self.posx_spin.setRange(0, 100000); self.posx_spin.setValue(800)
        self.posy_spin = QSpinBox(); self.posy_spin.setRange(0, 100000); self.posy_spin.setValue(600)
        self.posx_spin.setFixedWidth(80)
        self.posy_spin.setFixedWidth(80)
        top.addWidget(self.posx_spin)
        top.addWidget(self.posy_spin)
        layout.addLayout(top)

        # ── Plot ───────────────────────────────────────────────────
        self.plot = pg.PlotWidget(title="Interferogram")
        self.plot.setLabel('bottom', 'position', units='mm')
        self.plot.setLabel('left', 'intensity')
        self._curve = self.plot.plot([], [], pen='r')
        layout.addWidget(self.plot, stretch=1)

    # ------------------------------------------------------------------ #

    def get_probe(self) -> tuple[int, int]:
        """Returns the (row, col) probe pixel for the interferogram."""
        return self.posx_spin.value(), self.posy_spin.value()

    def clear(self):
        self._x.clear()
        self._y.clear()
        self._curve.setData([], [])

    def add_point(self, x: float, y: float):
        self._x.append(x)
        self._y.append(y)
        self._curve.setData(self._x, self._y)

    @property
    def point_count(self) -> int:
        return len(self._x)

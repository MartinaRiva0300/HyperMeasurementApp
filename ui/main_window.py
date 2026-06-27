"""
MainWindow — two main panels:

    ┌─────────────────┬──────────────────────────────────────┐
    │  CONTROLS (left) │  LIVE IMAGE  (right, top 3/4)         │
    │  • Camera        │  + Live / Snap / display options      │
    │  • Stage         │                                       │
    │  • Measurement   ├──────────────────────────────────────┤
    │                  │  INTERFEROGRAM (right, bottom 1/4)    │
    └─────────────────┴──────────────────────────────────────┘

There is a single live image window (LiveViewPanel). The camera live worker and
the measurement scan both render into it. Coordination guarantees only one of
them drives the camera at a time.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QScrollArea, QAction
)
from PyQt5.QtCore import Qt

from ui.camera_panel import CameraPanel
from ui.stage_panel import StagePanel
from ui.measurement_panel import MeasurementPanel
from ui.live_view_panel import LiveViewPanel
from ui.interferogram_panel import InterferogramPanel


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}
QScrollArea { border: none; }
QGroupBox {
    border: 1px solid #2d2d4e;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    background-color: #16213e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #7f8cff;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
QPushButton {
    background-color: #2d2d4e;
    border: 1px solid #3d3d6e;
    border-radius: 4px;
    padding: 4px 10px;
    color: #c8c8e8;
    min-height: 26px;
}
QPushButton:hover  { background-color: #3a3a60; border-color: #6060a0; }
QPushButton:pressed { background-color: #1a1a38; }
QPushButton:disabled { color: #555570; border-color: #2a2a44; }
QPushButton:checked { background-color: #3a5090; border-color: #6080d0; color: #ffffff; }
QPushButton#btnLive   { background-color: #1a4030; border-color: #2a7050; color: #80e0a0; }
QPushButton#btnLive:checked { background-color: #3a2020; border-color: #904040; color: #e08080; }
QPushButton#btnSnap   { background-color: #2a3060; border-color: #4050a0; color: #a0b0ff; }
QPushButton#btnRun    { background-color: #1a4030; border-color: #2a7050; color: #80e0a0; min-height:34px; font-weight:600; }
QPushButton#btnHalt   { background-color: #4a1a1a; border-color: #903030; color: #e08080; }
QPushButton#btnHome   { background-color: #2a2a40; border-color: #4040a0; }
QPushButton#btnConnect { background-color: #1e3050; border-color: #3060a0; color: #80b0e0; }
QPushButton#btnGo     { background-color: #2a3050; border-color: #4050a0; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
    background-color: #0d0d1e;
    border: 1px solid #303050;
    border-radius: 3px;
    padding: 2px 4px;
    color: #d0d0f0;
    min-height: 22px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border-color: #6070c0;
}
QComboBox QAbstractItemView {
    background: #16213e;
    selection-background-color: #3050a0;
}
QLabel#posDisplay {
    background: #0d0d1e;
    border: 1px solid #2d2d4e;
    border-radius: 3px;
    padding: 1px 6px;
    color: #90ffc0;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QLabel#axisLabel { color: #a0a8e0; font-weight: 600; }
QLabel#statusLabel { color: #8888aa; font-size: 11px; }
QLabel#pathLabel { color: #7070a0; font-size: 11px; }
QProgressBar {
    background-color: #0d0d1e;
    border: 1px solid #2d2d4e;
    border-radius: 3px;
    text-align: center;
    color: #d0d0f0;
    height: 18px;
}
QProgressBar::chunk { background-color: #3060c0; border-radius: 2px; }
QSplitter::handle { background: #2d2d4e; }
QStatusBar { background-color: #0d0d1e; color: #6868a0; font-size: 11px; }
QMenuBar { background-color: #0d0d1e; color: #c0c0e0; }
QMenuBar::item:selected { background-color: #2d2d4e; }
QMenu { background-color: #16213e; border: 1px solid #2d2d4e; color: #c0c0e0; }
QMenu::item:selected { background-color: #2d4080; }
"""


class MainWindow(QMainWindow):
    def __init__(self, camera, stage):
        super().__init__()
        self.camera = camera
        self.stage = stage
        self._live_active = False
        self._meas_active = False

        self.setWindowTitle("Hyperspectral Control")
        self.resize(1500, 900)
        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_ui()
        self._wire_coordination()
        self.statusBar().showMessage(
            "Ready — connect hardware via the Hardware menu or the Connect buttons")

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("File")
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        hw_menu = mb.addMenu("Hardware")
        conn_cam_act = QAction("Connect Camera", self)
        conn_cam_act.triggered.connect(lambda: self.camera_panel.connect_camera())
        conn_stage_act = QAction("Connect Stage", self)
        conn_stage_act.triggered.connect(lambda: self.stage_panel.connect_stage())
        hw_menu.addAction(conn_cam_act)
        hw_menu.addAction(conn_stage_act)

    def _build_ui(self):
        # ── Panels ─────────────────────────────────────────────────
        self.camera_panel = CameraPanel(self.camera)
        self.stage_panel = StagePanel(self.stage)
        self.measurement_panel = MeasurementPanel(self.stage, self.camera)
        self.live_panel = LiveViewPanel(self.camera)
        self.interferogram_panel = InterferogramPanel()

        # ── Left: stacked controls in a scroll area ────────────────
        controls = QWidget()
        cv = QVBoxLayout(controls)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)
        cv.addWidget(self.camera_panel)
        cv.addWidget(self.stage_panel)
        cv.addWidget(self.measurement_panel)
        cv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(380)

        # ── Right: live (top 3/4) over interferogram (bottom 1/4) ──
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.live_panel)
        right_splitter.addWidget(self.interferogram_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([660, 220])      # ~3:1

        # ── Outer: controls | display ──────────────────────────────
        outer = QSplitter(Qt.Horizontal)
        outer.addWidget(scroll)
        outer.addWidget(right_splitter)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setSizes([420, 1080])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(outer)
        self.setCentralWidget(central)

    def _wire_coordination(self):
        # Stage connects -> measurement panel learns the available axes.
        self.stage_panel.on_connected = self.measurement_panel.refresh_axes

        # Camera control panel needs to stop the shared live view on disconnect.
        self.camera_panel.live_panel = self.live_panel

        # Mirror the Live/Snap buttons at the bottom of the camera panel.
        self.live_panel.register_buttons(self.camera_panel.btn_live,
                                         self.camera_panel.btn_snap)
        self.live_panel.set_camera_available(self.camera.is_open)

        # Single shared live view + interferogram sinks for the measurement scan.
        self.measurement_panel.image_sink = self.live_panel.display_frame
        self.measurement_panel.point_sink = self.interferogram_panel.add_point
        self.measurement_panel.clear_plot = self.interferogram_panel.clear
        self.measurement_panel.probe_provider = self.interferogram_panel.get_probe

        # State coupling.
        self.camera_panel.connection_changed.connect(self._on_camera_connection)
        self.live_panel.live_started.connect(self._on_live_started)
        self.live_panel.live_stopped.connect(self._on_live_stopped)
        self.measurement_panel.started.connect(self._on_meas_started)
        self.measurement_panel.stopped.connect(self._on_meas_stopped)

    # ------------------------------------------------------------------ #
    #  Coordination
    # ------------------------------------------------------------------ #

    def _on_camera_connection(self, connected: bool):
        self.live_panel.set_camera_available(connected)
        self._recompute()

    def _on_live_started(self):
        self._live_active = True
        self._recompute()

    def _on_live_stopped(self):
        self._live_active = False
        self._recompute()

    def _on_meas_started(self):
        self._meas_active = True
        self.live_panel.stop_live()        # free the camera for the scan
        self._recompute()

    def _on_meas_stopped(self):
        self._meas_active = False
        self._recompute()

    def _recompute(self):
        # ROI/binning lock while the live view is running.
        self.camera_panel.set_live_active(self._live_active)
        # Live buttons available unless a measurement is running.
        self.live_panel.set_controls_enabled(not self._meas_active)
        # Start scan only when nothing else drives the camera.
        self.measurement_panel.setEnabled_run(
            not self._live_active and not self._meas_active)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        """Clean shutdown: stop measurement, live, disconnect hardware."""
        self.measurement_panel.shutdown()
        self.live_panel.stop_live()
        if self.camera.is_open:
            self.camera.close()
        if self.stage.is_connected:
            self.stage.disconnect()
        event.accept()

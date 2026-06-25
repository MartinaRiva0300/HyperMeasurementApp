"""
MainWindow — assembles CameraPanel, StagePanel, and AcquisitionPanel.
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar, QAction, QMenuBar
)
from PyQt5.QtCore import Qt

from ui.camera_panel import CameraPanel
from ui.stage_panel import StagePanel
from ui.acquisition_panel import AcquisitionPanel


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}

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

/* Accent buttons */
QPushButton#btnLive   { background-color: #1a4030; border-color: #2a7050; color: #80e0a0; }
QPushButton#btnLive:checked { background-color: #3a2020; border-color: #904040; color: #e08080; }
QPushButton#btnSnap   { background-color: #2a3060; border-color: #4050a0; color: #a0b0ff; }
QPushButton#btnRun    { background-color: #1a4030; border-color: #2a7050; color: #80e0a0; min-height:34px; font-weight:600; }
QPushButton#btnHalt   { background-color: #4a1a1a; border-color: #903030; color: #e08080; }
QPushButton#btnHome   { background-color: #2a2a40; border-color: #4040a0; }
QPushButton#btnConnect { background-color: #1e3050; border-color: #3060a0; color: #80b0e0; }
QPushButton#btnGo     { background-color: #2a3050; border-color: #4050a0; }

QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0d0d1e;
    border: 1px solid #3030508;
    border-radius: 3px;
    padding: 2px 4px;
    color: #d0d0f0;
    min-height: 22px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
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

QProgressBar {
    background-color: #0d0d1e;
    border: 1px solid #2d2d4e;
    border-radius: 3px;
    text-align: center;
    color: #d0d0f0;
    height: 18px;
}
QProgressBar::chunk { background-color: #3060c0; border-radius: 2px; }

QTableWidget {
    background-color: #0d0d1e;
    border: 1px solid #2d2d4e;
    gridline-color: #2d2d4e;
    color: #d0d0f0;
    alternate-background-color: #111130;
}
QTableWidget QHeaderView::section {
    background-color: #1a1a38;
    border: none;
    border-bottom: 1px solid #3030508;
    padding: 4px;
    color: #7f8cff;
    font-size: 11px;
}
QTableWidget::item:selected { background-color: #2a4080; }

QLabel#pathLabel { color: #7070a0; font-size: 11px; }

QSplitter::handle { background: #2d2d4e; width: 2px; height: 2px; }

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
        self.setWindowTitle("Microscope Control")
        self.resize(1400, 820)
        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_ui()
        self._build_status_bar()

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
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── Top splitter: camera (left) | stage (right) ───────────
        top_splitter = QSplitter(Qt.Horizontal)

        self.camera_panel = CameraPanel(self.camera)
        self.stage_panel = StagePanel(self.stage)

        top_splitter.addWidget(self.camera_panel)
        top_splitter.addWidget(self.stage_panel)
        top_splitter.setStretchFactor(0, 3)   # camera gets more width
        top_splitter.setStretchFactor(1, 2)

        # ── Bottom: acquisition panel ──────────────────────────────
        self.acq_panel = AcquisitionPanel(self.stage, self.camera)

        # ── Vertical split: top | acquisition ────────────────────
        vert_splitter = QSplitter(Qt.Vertical)
        vert_splitter.addWidget(top_splitter)
        vert_splitter.addWidget(self.acq_panel)
        vert_splitter.setStretchFactor(0, 3)
        vert_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(vert_splitter)

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.showMessage("Ready  —  connect hardware via the Hardware menu or the Connect buttons")

    def closeEvent(self, event):
        """Clean shutdown: stop live, disconnect hardware."""
        self.camera_panel.stop_live()
        if self.camera.is_open:
            self.camera.close()
        if self.stage.is_connected:
            self.stage.disconnect()
        event.accept()

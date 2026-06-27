#!/usr/bin/env python3
"""
Hyperspectral Control App
=========================
Hamamatsu Orca Flash 4V3  +  PI motorised stage (PI_VC_Device)

Three panels:
  1. Camera   — exposure, binning, ROI, trigger, live view (pyqtgraph)
  2. Stage    — per-axis encoder read-back, to-go position, relative jog,
                velocity, home/halt
  3. Measurement — the hyperspectral line scan (port of hyperspectral_measure.py):
                start_pos, step, step_num, internal/external trigger, motor
                velocity, HDF5 saving, live image + interferogram plot.

All hardware I/O runs in background QThreads (see core/acquisition.py) so the
UI and live image stay responsive.

RUN
---
    python main.py              # mock hardware (no physical devices needed)
    python main.py --real       # real Hamamatsu + PI hardware

CONFIGURE REAL HARDWARE
-----------------------
Edit make_real_hardware() below: set the PI USB serial number(s) / axis ids and
camera defaults. PI_VC_device.py and CameraDevice.py are imported from their
sibling ScopeFoundry Projects folders automatically (see hardware/*.py).
"""

import sys
import os
import argparse

# Make hardware/, core/, ui/ importable from any working directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow


# ── Configuration ─────────────────────────────────────────────────────── #

def make_mock_hardware():
    """Simulated stage + camera for UI development (no devices needed)."""
    from hardware.camera import MockCamera
    from hardware.stage import MockPIStage
    camera = MockCamera(frame_x=2048, frame_y=2048, binning=1)
    stage = MockPIStage(axes=('1',))
    return camera, stage


def make_real_hardware():
    """
    Real Hamamatsu + PI stage.
    ▶ Edit serial numbers / axis ids and camera defaults here.
    """
    from hardware.camera import HamamatsuCamera
    from hardware.stage import PIStageWrapper, StageManager

    camera = HamamatsuCamera(
        camera_id=0,
        frame_x=2048,
        frame_y=2048,
        exposure=0.01,        # seconds
        binning=1,
        trsource='internal',
        trmode='normal',
        trpolarity='positive',
        tractive='edge',
        subarrayh_pos=0,
        subarrayv_pos=0,
    )

    # ── Edit serial number(s) to match your devices ────────────── #
    stage = StageManager([
        PIStageWrapper('0185500162', axis='1'),     # ← your serial here
        # PIStageWrapper('0185500007', axis='2'),   # uncomment for more axes
    ])

    return camera, stage


# ── Entry point ────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Hyperspectral Control App")
    parser.add_argument('--real', action='store_true',
                        help="Use real hardware (default: simulation mode)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    if args.real:
        print("Starting with REAL hardware…")
        camera, stage = make_real_hardware()
    else:
        print("Starting in SIMULATION mode (use --real for hardware)")
        camera, stage = make_mock_hardware()

    window = MainWindow(camera, stage)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

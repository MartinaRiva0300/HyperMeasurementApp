#!/usr/bin/env python3
"""
Hyperspectral Control App
=========================
Teledyne Retiga E7 (PVCAM / PyVCAM)  +  PI motorised stage (PI_VC_Device)

Three panels:
  1. Camera   — exposure (ms), binning, readout port, gain, trigger mode, ROI,
                live view (pyqtgraph)
  2. Stage    — per-axis encoder read-back, to-go position, relative jog,
                velocity, home/halt
  3. Measurement — the hyperspectral line scan (port of hyperspectral_measure.py,
                RetigaE7_PI branch): start_pos, step, step_num, Internal/Edge
                trigger, positions from file, motor velocity, HDF5 saving,
                live image + interferogram plot.

All hardware I/O runs in background QThreads (see core/acquisition.py) so the
UI and live image stay responsive.

RUN
---
    python main.py              # mock hardware (no physical devices needed)
    python main.py --real       # real Retiga E7 + PI hardware

CONFIGURE REAL HARDWARE
-----------------------
Edit make_real_hardware() below: set the PI USB serial number(s) / axis ids and
camera defaults. CameraDevice.py (Teledyne_ScopeFoundry) and PI_VC_device.py
(PI_ScopeFoundry) are imported from their sibling ScopeFoundry Projects folders
automatically (see hardware/*.py).
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
    camera = MockCamera(frame_x=3200, frame_y=2200, binning=1, exposure_ms=20)
    stage = MockPIStage(axes=('1',))
    return camera, stage


def make_real_hardware():
    """
    Real Retiga E7 + PI stage.
    ▶ Edit serial numbers / axis ids and camera defaults here.
    """
    from hardware.camera import RetigaCamera
    from hardware.stage import PIStageWrapper, StageManager

    camera = RetigaCamera(
        exposure_ms=20,                 # integer milliseconds
        binning=1,
        readout_port=0,                 # 0 Speed, 1 Long Exposure, 2 Long Exposure EDR
        gain=1,                         # 1 Full Well (2 Sensitivity on readout 1/2)
        trigger_mode='Internal Trigger',
        roi=(0, 0, 3200, 2200),         # h0, v0, width, height (sensor px)
    )

    # ── Edit serial number(s) to match your devices ────────────── #
    # Same controller as hyperspectral_app.py (PI_HW, encoder='VC').
    stage = StageManager([
        PIStageWrapper('0024550348', axis='1'),     # ← your serial here
        # PIStageWrapper('0185500007', axis='2'),   # uncomment for more axes
    ])

    return camera, stage


# ── Entry point ────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Hyperspectral Control App (Retiga E7 + PI)")
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

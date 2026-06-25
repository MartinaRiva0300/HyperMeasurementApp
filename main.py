#!/usr/bin/env python3
"""
Hyperspectral Control App
======================
Hamamatsu Orca Flash 4V3  +  PI motorised stage (PI_VC_Device)

HOW TO CONFIGURE
----------------
1. Set USE_MOCK = False when running with real hardware.

2. For each real PI axis, create a PIStageWrapper with its USB serial number
   and axis identifier (as set in your PI_VC_Device calls).
   Example for a single-axis system:
       from hardware.stage import PIStageWrapper, StageManager
       stage = StageManager([PIStageWrapper('0185500006', axis='1')])

3. Camera parameters below match the ones used in your HamamatsuDevice
   constructor and hyperspectral_measure.py.

4. Make sure PI_VC_device.py and CameraDevice.py are on sys.path
   (e.g. in this folder or in hardware/).

RUN
---
    python main.py              # mock hardware (no physical devices needed)
    python main.py --real       # real hardware
"""

import sys
import os
import argparse

# Make hardware/ and core/ importable from any working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow

# ── Configuration ─────────────────────────────────────────────────────── #

def make_mock_hardware():
    """Return simulated stage + camera for UI development."""
    from hardware.camera import MockCamera
    from hardware.stage import MockPIStage
    camera = MockCamera(frame_x=2048, frame_y=2048, binning=1)
    stage  = MockPIStage(axes=('1', '2', '3'))
    return camera, stage


def make_real_hardware():
    """
    Return real Hamamatsu + PI stage.
    ▶  Edit serial numbers and camera parameters here.
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

    # ── Edit serial numbers to match your devices ──────────────── #
    stage = StageManager([
        PIStageWrapper('0185500006', axis='1'),   # ← your serial here
        # PIStageWrapper('0185500007', axis='2'), # uncomment for more axes
    ])

    return camera, stage


# ── Entry point ────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Microscope Control App")
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
